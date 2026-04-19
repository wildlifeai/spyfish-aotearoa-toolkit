"""
Tests for StageRunner orchestration behavior.

Focused on the failure-handling contract:
  - Global stage failures are logged + recorded in `failed_stages`, not re-raised
  - A failing global stage does not prevent later stages from running
  - Drop stages record per-drop errors via the section's ERROR status
"""

import argparse
from unittest.mock import MagicMock

from spyfish.config.base import MlStatus
from spyfish.orchestrator.stage import DropStage, GlobalStage, StageRunner


def _args(**flags) -> argparse.Namespace:
    """Build an argparse namespace with all known stage flags set to False by default."""
    ns = argparse.Namespace()
    # Populate every flag used in the test with the supplied value (or False)
    for flag in ["ingest", "check_arrivals", "ml", "later", "early", "only_stage"]:
        setattr(ns, flag, flags.get(flag, False))
    for k, v in flags.items():
        setattr(ns, k, v)
    return ns


def test_failing_global_stage_does_not_abort_pipeline():
    """A global stage that raises should be logged and recorded, then the
    runner should still execute later stages. Regression for the old
    behavior where _run_global re-raised and killed downstream stages.
    """
    later_ran = MagicMock()

    def early_fn():
        raise RuntimeError("simulated ingest failure")

    stages = [
        GlobalStage("early", "early stage that fails", lambda: early_fn()),
        GlobalStage("later", "later stage that should still run", later_ran),
    ]
    # Wire the first stage to the failing fn (the lambda above is just a
    # factory — simpler to pass early_fn directly):
    stages[0] = GlobalStage("early", "early stage that fails", early_fn)

    db = MagicMock()
    runner = StageRunner(stages, db)

    runner.run(_args())  # should NOT raise

    assert runner.failed_stages == ["early"]
    later_ran.assert_called_once()  # later stage still ran despite early crash


def test_successful_global_stages_leave_failed_stages_empty():
    ran_a = MagicMock()
    ran_b = MagicMock()
    stages = [
        GlobalStage("a", "stage a", ran_a),
        GlobalStage("b", "stage b", ran_b),
    ]
    db = MagicMock()
    runner = StageRunner(stages, db)

    runner.run(_args())

    assert runner.failed_stages == []
    ran_a.assert_called_once()
    ran_b.assert_called_once()


def test_drop_stage_fn_exception_sets_section_error():
    """When a drop stage function raises, the runner should set that drop's
    section column to the section's ERROR status via update_section_status.
    """
    db = MagicMock()
    # Mock the eligibility query to return one drop
    db.get_deployments_eligible.return_value = [
        {"drop_id": "KSF_20240124_BUV_KSF_085_01"}
    ]

    def boom(drop_id: str):
        raise RuntimeError("simulated processing failure")

    stages = [
        DropStage(
            flag="only-stage",
            description="stage that always fails",
            fn=boom,
            section=MlStatus.COLUMN,
            input_statuses=[MlStatus.READY],
        )
    ]
    runner = StageRunner(stages, db)
    runner.run(_args(only_stage=True))

    # Runner should have called update_section_status with the ml_error value
    db.update_section_status.assert_called_once_with(
        "KSF_20240124_BUV_KSF_085_01", MlStatus.COLUMN, MlStatus.ERROR
    )
    # Global failed_stages only tracks global stage crashes, not per-drop
    assert runner.failed_stages == []


def test_drop_stage_fn_returns_none_leaves_status_unchanged():
    """If a drop stage fn returns None, no advance_status call should happen."""
    db = MagicMock()
    db.get_deployments_eligible.return_value = [
        {"drop_id": "KSF_20240124_BUV_KSF_085_01"}
    ]

    stages = [
        DropStage(
            flag="only-stage",
            description="stage that returns None",
            fn=lambda drop_id: None,
            section=MlStatus.COLUMN,
            input_statuses=[MlStatus.READY],
        )
    ]
    runner = StageRunner(stages, db)
    runner.run(_args(only_stage=True))

    db.advance_status.assert_not_called()
    db.update_section_status.assert_not_called()
