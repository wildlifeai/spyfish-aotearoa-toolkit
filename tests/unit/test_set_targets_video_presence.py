"""--set-targets can mark a restored DEEP_ARCHIVE video as readable.

The ML runner refuses `archived` drops before it looks on disk, so a targets
CSV that sets ml_ready on a restored drop must also set video_presence, or the
drop is skipped with a "restore it first" warning (SLI, 2026-10-12).
"""

from unittest.mock import patch

from spyfish.config.base import MlStatus, VideoPresence
from spyfish.test_setup import process_csv_targets

DROP = "SLI_20220228_BUV_SLI_104_01"


def _csv(tmp_path, presence):
    p = tmp_path / "targets.csv"
    p.write_text(f"DropID,ml_status,video_presence\n{DROP},ml_ready,{presence}\n")
    return str(p)


@patch("spyfish.test_setup.DatabaseManager")
def test_video_presence_column_is_written(mock_db_class, tmp_path):
    db = mock_db_class.return_value
    db.get_deployments_by_ids.return_value = {DROP: {"drop_id": DROP}}
    db.get_max_priority.return_value = 0

    process_csv_targets(_csv(tmp_path, VideoPresence.PRESENT))

    db.update_deployment_fields.assert_called_once_with(
        DROP, priority=1, video_presence=VideoPresence.PRESENT
    )
    db.update_section_status.assert_called_once_with(
        DROP, MlStatus.COLUMN, MlStatus.READY
    )


@patch("spyfish.test_setup.DatabaseManager")
def test_bad_video_presence_value_is_rejected(mock_db_class, tmp_path):
    db = mock_db_class.return_value
    db.get_deployments_by_ids.return_value = {DROP: {"drop_id": DROP}}
    db.get_max_priority.return_value = 0

    process_csv_targets(_csv(tmp_path, "restored"))

    db.update_deployment_fields.assert_called_once_with(DROP, priority=1)
