"""The three "done" definitions: `stage_flags` and `is_complete` (both in
doc_report/data.py, under the "Three done questions" note) and
`analysed_by_status` (ecology_data.py), each next to the metric it drives.

They deliberately disagree on some deployments. These tests pin exactly where,
so a change to one of them is a visible, reviewed decision rather than drift.
"""

import sys
from pathlib import Path

import pandas as pd

APP = Path(__file__).resolve().parents[2] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from doc_report.data import is_complete, stage_flags  # noqa: E402
from ecology_data import analysed_by_status  # noqa: E402

from spyfish.config.base import (  # noqa: E402
    CitSciStatus,
    ExpertStatus,
    IngestStatus,
    MlStatus,
    ReportingStatus,
    VideoPresence,
)


def _dep(**overrides) -> dict:
    row = {
        "drop_id": "KSF_20240124_BUV_KSF_085_01",
        "ingest_status": IngestStatus.OK,
        "ml_status": MlStatus.PENDING,
        "citsci_status": CitSciStatus.PENDING,
        "expert_status": ExpertStatus.PENDING,
        "reporting_status": ReportingStatus.PENDING,
        "video_presence": VideoPresence.PRESENT,
        "ml_annotations": 0,
        "citsci_annotations": 0,
        "expert_annotations": 0,
    }
    row.update(overrides)
    return row


def test_stage_flags_counts_archived_footage_as_existing():
    df = stage_flags(pd.DataFrame([_dep(video_presence=VideoPresence.ARCHIVED)]))
    assert df.loc[0, "footage_exists"]
    assert not df.loc[0, "has_video"]


def test_stage_flags_reads_annotation_counts_not_statuses():
    # Status says ML never finished, but ML annotations exist: stage_flags
    # trusts the data.
    df = stage_flags(pd.DataFrame([_dep(ml_status=MlStatus.ERROR, ml_annotations=3)]))
    assert df.loc[0, "ml_done"]
    assert not df.loc[0, "expert_done"]


def test_stage_flags_treats_missing_counts_as_zero():
    df = stage_flags(pd.DataFrame([_dep(expert_annotations=None)]))
    assert not df.loc[0, "expert_done"]


def test_is_complete_by_expert_or_reporting_or_all_skipped():
    df = pd.DataFrame(
        [
            _dep(expert_status=ExpertStatus.COMPLETE),
            _dep(reporting_status=ReportingStatus.COMPLETE),
            _dep(
                ml_status=MlStatus.SKIPPED,
                citsci_status=CitSciStatus.SKIPPED,
                expert_status=ExpertStatus.SKIPPED,
            ),
            # Two of three skipped is not complete.
            _dep(ml_status=MlStatus.SKIPPED, citsci_status=CitSciStatus.SKIPPED),
            # Expert annotations alone, without the status, are not complete.
            _dep(expert_annotations=5),
        ]
    )
    assert is_complete(df).tolist() == [True, True, True, False, False]


def test_analysed_by_status_needs_ingest_ok():
    df = pd.DataFrame(
        [
            _dep(ml_status=MlStatus.COMPLETE),
            _dep(
                ingest_status=IngestStatus.VALIDATION_ERROR,
                ml_status=MlStatus.COMPLETE,
            ),
        ]
    )
    assert analysed_by_status(df).tolist() == [True, False]


def test_analysed_by_status_any_section():
    df = pd.DataFrame(
        [
            _dep(citsci_status=CitSciStatus.COMPLETE),
            _dep(expert_status=ExpertStatus.UPLOADED),
            _dep(expert_status=ExpertStatus.COMPLETE),
            _dep(),  # nothing finished
        ]
    )
    assert analysed_by_status(df).tolist() == [True, True, True, False]


def test_complete_and_analysed_disagree_on_ml_only():
    # The documented difference: ML complete is enough to count as effort for
    # an abundance mean, but leaves the deployment needing action.
    df = pd.DataFrame([_dep(ml_status=MlStatus.COMPLETE)])
    assert analysed_by_status(df).iloc[0]
    assert not is_complete(df).iloc[0]
