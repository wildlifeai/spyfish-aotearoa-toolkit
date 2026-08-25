"""Tests for check_pending_arrivals.

Guards the dead state found 2026-08-23: ml_pending + video_presence=present
matched neither branch of the old (absent, archived) filter, so those drops
were never advanced and never processed.
"""
from unittest.mock import MagicMock, patch

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


@patch("spyfish.orchestrator.ingest.DatabaseManager")
def test_deep_archive_is_marked_not_advanced(mock_db_class):
    """A restore has to finish before ML can read it, so mark and wait."""
    db = mock_db_class.return_value
    db.get_deployments_eligible.return_value = _pending(VideoPresence.ABSENT)

    check_pending_arrivals(known_files={KEY}, media_file_info={KEY: "DEEP_ARCHIVE"})

    db.advance_status.assert_not_called()
    db.update_deployment_fields.assert_called_once_with(
        DROP, video_presence=VideoPresence.ARCHIVED
    )
