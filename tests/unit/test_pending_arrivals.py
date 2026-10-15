"""Tests for check_pending_arrivals.

Guards the dead state found 2026-08-23: ml_pending + video_presence=present
matched neither branch of the old (absent, archived) filter, so those drops
were never advanced and never processed.
"""

from unittest.mock import patch

from spyfish.config.base import MlStatus, VideoPresence
from spyfish.orchestrator.ingest import check_pending_arrivals

DROP = "KSF_20240124_BUV_KSF_085_01"
KEY = f"media/KSF_20240124_BUV/{DROP}/{DROP}.mp4"


def _pending(presence):
    return [{"drop_id": DROP, "video_path": KEY, "video_presence": presence}]


@patch("spyfish.orchestrator.ingest.DatabaseManager")
def test_present_but_pending_drop_is_advanced(mock_db_class):
    """The dead state: S3 has the video and the DB knows it, but ml_status
    never moved. It must still advance."""
    db = mock_db_class.return_value
    db.get_deployments_eligible.return_value = _pending(VideoPresence.PRESENT)

    check_pending_arrivals(known_files={KEY}, media_file_info={KEY: "STANDARD"})

    db.advance_status.assert_called_once_with(DROP, MlStatus.COLUMN, MlStatus.READY)


@patch("spyfish.orchestrator.ingest.DatabaseManager")
def test_absent_drop_that_has_arrived_is_advanced(mock_db_class):
    db = mock_db_class.return_value
    db.get_deployments_eligible.return_value = _pending(VideoPresence.ABSENT)

    check_pending_arrivals(known_files={KEY}, media_file_info={KEY: "STANDARD"})

    db.advance_status.assert_called_once_with(DROP, MlStatus.COLUMN, MlStatus.READY)
    db.update_deployment_fields.assert_called_with(
        DROP, video_presence=VideoPresence.PRESENT
    )


@patch("spyfish.orchestrator.ingest.DatabaseManager")
def test_video_still_missing_is_left_alone(mock_db_class):
    db = mock_db_class.return_value
    db.get_deployments_eligible.return_value = _pending(VideoPresence.ABSENT)

    check_pending_arrivals(known_files=set(), media_file_info={})

    db.advance_status.assert_not_called()


@patch("spyfish.orchestrator.ingest.S3Handler")
@patch("spyfish.orchestrator.ingest.DatabaseManager")
def test_deep_archive_is_marked_not_advanced(mock_db_class, mock_s3_class):
    """A restore has to finish before ML can read it, so mark and wait."""
    db = mock_db_class.return_value
    db.get_deployments_eligible.return_value = _pending(VideoPresence.ABSENT)
    mock_s3_class.return_value.is_restored.return_value = False

    check_pending_arrivals(known_files={KEY}, media_file_info={KEY: "DEEP_ARCHIVE"})

    db.advance_status.assert_not_called()
    db.update_deployment_fields.assert_called_once_with(
        DROP, video_presence=VideoPresence.ARCHIVED
    )
    mock_s3_class.return_value.is_restored.assert_called_once_with(KEY)


@patch("spyfish.orchestrator.ingest.S3Handler")
@patch("spyfish.orchestrator.ingest.DatabaseManager")
def test_restored_deep_archive_is_advanced(mock_db_class, mock_s3_class):
    """The listing still says DEEP_ARCHIVE after a restore; only head_object
    knows a readable copy exists. 118 SLI videos sat restored for a week
    (2026-10-09) while this check kept them archived."""
    db = mock_db_class.return_value
    db.get_deployments_eligible.return_value = _pending(VideoPresence.ARCHIVED)
    mock_s3_class.return_value.is_restored.return_value = True

    check_pending_arrivals(known_files={KEY}, media_file_info={KEY: "DEEP_ARCHIVE"})

    db.update_deployment_fields.assert_called_once_with(
        DROP, video_presence=VideoPresence.PRESENT
    )
    db.advance_status.assert_called_once_with(DROP, MlStatus.COLUMN, MlStatus.READY)


@patch("spyfish.orchestrator.ingest.S3Handler")
@patch("spyfish.orchestrator.ingest.DatabaseManager")
def test_standard_storage_never_heads_the_object(mock_db_class, mock_s3_class):
    """The per-object check is only paid for archived drops."""
    db = mock_db_class.return_value
    db.get_deployments_eligible.return_value = _pending(VideoPresence.ABSENT)

    check_pending_arrivals(known_files={KEY}, media_file_info={KEY: "STANDARD"})

    mock_s3_class.return_value.is_restored.assert_not_called()
    db.advance_status.assert_called_once_with(DROP, MlStatus.COLUMN, MlStatus.READY)


def test_is_restored_reads_the_restore_header():
    """Storage class alone cannot distinguish the three archive states."""
    from unittest.mock import MagicMock

    from spyfish.storage.s3_handler import S3Handler

    handler = S3Handler.__new__(S3Handler)
    handler.get_object_storage = MagicMock()

    handler.get_object_storage.return_value = {
        "storage_class": "DEEP_ARCHIVE",
        "restore": 'ongoing-request="false", expiry-date="Thu, 15 Oct 2026 00:00:00 GMT"',
    }
    assert handler.is_restored(KEY) is True

    handler.get_object_storage.return_value = {
        "storage_class": "DEEP_ARCHIVE",
        "restore": 'ongoing-request="true"',
    }
    assert handler.is_restored(KEY) is False

    handler.get_object_storage.return_value = {
        "storage_class": "DEEP_ARCHIVE",
        "restore": None,
    }
    assert handler.is_restored(KEY) is False

    handler.get_object_storage.return_value = None
    assert handler.is_restored(KEY) is False
