"""--db-refresh pass for the Zooniverse validation sections (zoo_frames / zoo_crops)."""

from unittest.mock import MagicMock, patch

from spyfish.config.base import ZooCropsStatus, ZooFramesStatus
from spyfish.orchestrator import db_refresh

DROP_A = "KSF_20240124_BUV_KSF_085_01"
DROP_B = "KSF_20240124_BUV_KSF_085_02"


@patch("spyfish.orchestrator.db_refresh._ok_drops_not_at")
@patch("spyfish.zooniverse.upload.existing_subject_count")
def test_set_on_zooniverse_marks_pending_drop_uploaded(count, candidates):
    """DB says pending, Zooniverse holds the set: the DB was behind, not the upload."""
    db = MagicMock()
    candidates.side_effect = lambda _db, section, _excl: (
        [{"drop_id": DROP_A}, {"drop_id": DROP_B}]
        if section == ZooFramesStatus.COLUMN
        else []
    )
    count.side_effect = lambda kind, drop_id: 12 if drop_id == DROP_A else None

    assert db_refresh.refresh_zooniverse_validation(db) == 1
    db.update_section_status.assert_called_once_with(
        DROP_A, ZooFramesStatus.COLUMN, ZooFramesStatus.UPLOADED
    )


@patch("spyfish.orchestrator.db_refresh._ok_drops_not_at")
@patch("spyfish.zooniverse.upload.existing_subject_count")
def test_settled_statuses_are_excluded_from_the_query(count, candidates):
    """Uploaded, complete and (for crops) skipped drops are never asked about."""
    candidates.return_value = []
    db_refresh.refresh_zooniverse_validation(MagicMock())
    excludes = {call.args[1]: call.args[2] for call in candidates.call_args_list}
    assert set(excludes[ZooFramesStatus.COLUMN]) == {
        ZooFramesStatus.UPLOADED,
        ZooFramesStatus.COMPLETE,
    }
    assert set(excludes[ZooCropsStatus.COLUMN]) == {
        ZooCropsStatus.UPLOADED,
        ZooCropsStatus.COMPLETE,
        ZooCropsStatus.SKIPPED,
    }
    count.assert_not_called()


@patch("spyfish.orchestrator.db_refresh._ok_drops_not_at")
@patch("spyfish.zooniverse.upload.existing_subject_count")
def test_empty_set_does_not_count_as_uploaded(count, candidates):
    candidates.side_effect = lambda _db, section, _excl: (
        [{"drop_id": DROP_A}] if section == ZooCropsStatus.COLUMN else []
    )
    count.return_value = 0
    db = MagicMock()
    assert db_refresh.refresh_zooniverse_validation(db) == 0
    db.update_section_status.assert_not_called()
