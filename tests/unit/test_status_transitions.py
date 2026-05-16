"""
Tests for the multi-section status model.

Covers:
  - All section transitions (ML, CitSci, Biigle, Reporting)
  - Invalid transition rejection
  - Error clearing from validation_errors
  - Cross-section eligibility queries with prerequisites
  - Status value uniqueness across all sections
  - SQL injection whitelist on column-name interpolation
"""

import pytest

from spyfish.config.base import (
    SECTIONS,
    CitSciStatus,
    ExpertStatus,
    IngestStatus,
    InvalidTransitionError,
    MlStatus,
    ReportingStatus,
)

DROP = "KSF_20240124_BUV_KSF_085_01"
DROP_2 = "KSF_20240124_BUV_KSF_085_02"
DROP_3 = "KSF_20240124_BUV_KSF_085_03"


# ── Status value uniqueness ──────────────────────────────────────────────────


def _all_status_values(cls):
    """Collect all string status values from a status class.

    Skips the COLUMN attribute because it's the DB column name, not a status.
    """
    return {
        v
        for k, v in vars(cls).items()
        if isinstance(v, str) and not k.startswith("_") and k != "COLUMN"
    }


def test_status_values_globally_unique():
    """Every status string is unique across all section classes — no collisions."""
    all_values = []
    for cls in [MlStatus, CitSciStatus, ExpertStatus, ReportingStatus, IngestStatus]:
        all_values.extend(_all_status_values(cls))
    assert len(all_values) == len(set(all_values)), (
        f"Duplicate status values found: "
        f"{[v for v in all_values if all_values.count(v) > 1]}"
    )


def test_status_values_have_section_prefix():
    """All processing status values carry their section prefix."""
    for val in _all_status_values(MlStatus):
        assert val.startswith("ml_"), f"MlStatus value {val!r} missing ml_ prefix"
    for val in _all_status_values(CitSciStatus):
        assert val.startswith(
            "citsci_"
        ), f"CitSciStatus value {val!r} missing citsci_ prefix"
    for val in _all_status_values(ExpertStatus):
        assert val.startswith(
            "expert_"
        ), f"ExpertStatus value {val!r} missing expert_ prefix"
    for val in _all_status_values(ReportingStatus):
        assert val.startswith(
            "reporting_"
        ), f"ReportingStatus value {val!r} missing reporting_ prefix"


def test_sections_registry_covers_all_section_classes():
    """The SECTIONS registry should be keyed by each class's COLUMN attribute."""
    for cls in [MlStatus, CitSciStatus, ExpertStatus, ReportingStatus]:
        assert cls.COLUMN in SECTIONS
        assert SECTIONS[cls.COLUMN] is cls


# ── ML transitions ───────────────────────────────────────────────────────────


def test_ml_full_happy_path(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP, ml_status=MlStatus.PENDING)
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.READY)
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.RUNNING)
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.COMPLETE)
    assert temp_db.get_deployment(DROP)["ml_status"] == MlStatus.COMPLETE


def test_ml_error_and_retry(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP, ml_status=MlStatus.READY)
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.RUNNING)
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.ERROR)
    assert temp_db.get_deployment(DROP)["ml_status"] == MlStatus.ERROR
    # Retry: error → ready
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.READY)
    assert temp_db.get_deployment(DROP)["ml_status"] == MlStatus.READY


def test_ml_invalid_transition_raises(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP, ml_status=MlStatus.PENDING)
    with pytest.raises(InvalidTransitionError):
        temp_db.advance_status(
            DROP, MlStatus.COLUMN, MlStatus.COMPLETE
        )  # pending → complete not allowed


# ── CitSci transitions ───────────────────────────────────────────────────────


def test_citsci_happy_path(temp_db):
    """Happy path: pending → clips_uploaded → complete (frame subjects removed)."""
    temp_db.add_or_update_deployment(drop_id=DROP)
    temp_db.advance_status(DROP, CitSciStatus.COLUMN, CitSciStatus.CLIPS_UPLOADED)
    temp_db.advance_status(DROP, CitSciStatus.COLUMN, CitSciStatus.COMPLETE)
    assert temp_db.get_deployment(DROP)["citsci_status"] == CitSciStatus.COMPLETE


def test_citsci_invalid_transition_raises(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP)
    with pytest.raises(InvalidTransitionError):
        temp_db.advance_status(
            DROP, CitSciStatus.COLUMN, CitSciStatus.COMPLETE
        )  # pending → complete not allowed


# ── Biigle transitions ───────────────────────────────────────────────────────


def test_biigle_full_happy_path(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP)
    temp_db.advance_status(DROP, ExpertStatus.COLUMN, ExpertStatus.UPLOADED)
    temp_db.advance_status(DROP, ExpertStatus.COLUMN, ExpertStatus.COMPLETE)
    assert temp_db.get_deployment(DROP)["expert_status"] == ExpertStatus.COMPLETE


def test_biigle_error_and_retry(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP)
    temp_db.advance_status(DROP, ExpertStatus.COLUMN, ExpertStatus.UPLOADED)
    temp_db.advance_status(DROP, ExpertStatus.COLUMN, ExpertStatus.ERROR)
    temp_db.advance_status(DROP, ExpertStatus.COLUMN, ExpertStatus.PENDING)
    assert temp_db.get_deployment(DROP)["expert_status"] == ExpertStatus.PENDING


def test_expert_invalid_transition_raises(temp_db):
    """COMPLETE → PENDING isn't a valid expert_status move.

    Note: PENDING → COMPLETE *is* allowed (it's how non-BIIGLE direct paths,
    e.g. legacy CSV ingest, advance without going through UPLOADED). What
    isn't allowed is rewinding from COMPLETE back to PENDING.
    """
    temp_db.add_or_update_deployment(drop_id=DROP)
    temp_db.advance_status(DROP, ExpertStatus.COLUMN, ExpertStatus.COMPLETE)
    with pytest.raises(InvalidTransitionError):
        temp_db.advance_status(DROP, ExpertStatus.COLUMN, ExpertStatus.PENDING)


# ── Reporting transitions ────────────────────────────────────────────────────


def test_reporting_happy_path(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP)
    temp_db.advance_status(DROP, ReportingStatus.COLUMN, ReportingStatus.COMPLETE)
    assert temp_db.get_deployment(DROP)["reporting_status"] == ReportingStatus.COMPLETE


def test_reporting_invalid_transition_raises(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP)
    with pytest.raises(InvalidTransitionError):
        temp_db.advance_status(
            DROP, ReportingStatus.COLUMN, ReportingStatus.ERROR
        )  # pending → error not allowed


# ── Error clearing ───────────────────────────────────────────────────────────


def test_error_clearing_on_ml_retry(temp_db):
    """Moving out of ml_error should clear ml_error rows from validation_errors."""
    temp_db.add_or_update_deployment(drop_id=DROP, ml_status=MlStatus.READY)
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.RUNNING)
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.ERROR)

    temp_db.add_validation_error(
        survey_id="KSF_20240124_BUV",
        drop_id=DROP,
        error_type=MlStatus.ERROR,
        column_name="ml_inference",
        error_message="Inference failed",
    )

    errors_before = temp_db.get_all_validation_errors()
    ml_errors = [
        e
        for e in errors_before
        if e["ErrorType"] == MlStatus.ERROR and e["DropID"] == DROP
    ]
    assert len(ml_errors) == 1

    # Retry clears the error
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.READY)
    errors_after = temp_db.get_all_validation_errors()
    ml_errors_after = [
        e
        for e in errors_after
        if e["ErrorType"] == MlStatus.ERROR and e["DropID"] == DROP
    ]
    assert len(ml_errors_after) == 0


def test_error_clearing_preserves_other_sections(temp_db):
    """Clearing ML errors should not touch BIIGLE errors on the same drop."""
    temp_db.add_or_update_deployment(drop_id=DROP, ml_status=MlStatus.READY)
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.RUNNING)
    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.ERROR)

    temp_db.add_validation_error(
        survey_id="KSF_20240124_BUV",
        drop_id=DROP,
        error_type=MlStatus.ERROR,
        column_name="",
        error_message="ML failed",
    )
    temp_db.add_validation_error(
        survey_id="KSF_20240124_BUV",
        drop_id=DROP,
        error_type=ExpertStatus.ERROR,
        column_name="",
        error_message="Upload failed",
    )

    temp_db.advance_status(DROP, MlStatus.COLUMN, MlStatus.READY)

    errors = temp_db.get_all_validation_errors()
    types = [e["ErrorType"] for e in errors if e["DropID"] == DROP]
    assert MlStatus.ERROR not in types
    assert ExpertStatus.ERROR in types


# ── Cross-section eligibility ────────────────────────────────────────────────


def test_eligibility_with_ml_prerequisite(temp_db):
    """Zooniverse-clips requires ml_status = ml_complete."""
    temp_db.add_or_update_deployment(drop_id=DROP, ml_status=MlStatus.COMPLETE)
    temp_db.add_or_update_deployment(drop_id=DROP_2, ml_status=MlStatus.RUNNING)

    eligible = temp_db.get_deployments_eligible(
        CitSciStatus.COLUMN,
        [CitSciStatus.PENDING],
        prerequisites={MlStatus.COLUMN: MlStatus.COMPLETE},
    )
    drop_ids = [r["drop_id"] for r in eligible]
    assert DROP in drop_ids
    assert DROP_2 not in drop_ids


def test_eligibility_filters_ingest_status(temp_db):
    """Only ingest_status='ok' deployments are eligible."""
    temp_db.add_or_update_deployment(drop_id=DROP, ml_status=MlStatus.READY)
    temp_db.add_or_update_deployment(
        drop_id=DROP_2, ml_status=MlStatus.READY, ingest_status=IngestStatus.EXCLUDED
    )

    eligible = temp_db.get_deployments_eligible(MlStatus.COLUMN, [MlStatus.READY])
    drop_ids = [r["drop_id"] for r in eligible]
    assert DROP in drop_ids
    assert DROP_2 not in drop_ids


# ── Default statuses on insert ───────────────────────────────────────────────


def test_default_statuses_on_insert(temp_db):
    """New deployment gets section-prefixed pending values by default."""
    temp_db.add_or_update_deployment(drop_id=DROP)
    dep = temp_db.get_deployment(DROP)
    assert dep["ml_status"] == MlStatus.PENDING
    assert dep["citsci_status"] == CitSciStatus.PENDING
    assert dep["expert_status"] == ExpertStatus.PENDING
    assert dep["reporting_status"] == ReportingStatus.PENDING


# ── SQL injection whitelist ──────────────────────────────────────────────────


def test_update_section_status_rejects_unknown_column(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP)
    with pytest.raises(ValueError, match="Invalid column name"):
        temp_db.update_section_status(DROP, "drop_id; DROP TABLE deployments--", "x")


def test_get_deployments_eligible_rejects_unknown_section(temp_db):
    with pytest.raises(ValueError, match="Invalid column name"):
        temp_db.get_deployments_eligible("bogus_col", [MlStatus.READY])


def test_get_deployments_eligible_rejects_unknown_prerequisite_key(temp_db):
    temp_db.add_or_update_deployment(drop_id=DROP, ml_status=MlStatus.READY)
    with pytest.raises(ValueError, match="Invalid column name"):
        temp_db.get_deployments_eligible(
            MlStatus.COLUMN,
            [MlStatus.READY],
            prerequisites={"drop_id; DROP TABLE--": "x"},
        )


def test_get_deployments_by_section_status_rejects_unknown_column(temp_db):
    with pytest.raises(ValueError, match="Invalid column name"):
        temp_db.get_deployments_by_section_status("nonexistent_column", 0)


def test_advance_status_rejects_unknown_section(temp_db):
    """advance_status has its own SECTIONS check separate from validate_column."""
    temp_db.add_or_update_deployment(drop_id=DROP)
    with pytest.raises(ValueError, match="Unknown section"):
        temp_db.advance_status(DROP, "nonexistent_section", "some_value")
