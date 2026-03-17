import pytest
from unittest.mock import patch, MagicMock
from spyfish.orchestrator.ml_runner import MLRunner
from spyfish.config.base import PipelineStatus

@patch('spyfish.orchestrator.ml_runner.DatabaseManager')
@patch('spyfish.orchestrator.ml_runner.S3Handler')
def test_get_inference_targets(mock_s3_class, mock_db_class):
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value

    # Mock database to return one item ready for ML
    mock_db.get_deployments_by_status.return_value = [
        {
            "drop_id": "KSF_20240124_BUV_KSF_085_01",
            "video_path": "path/1.mp4",
            "sampling_start": 0,
            "sampling_end": 100
        }
    ]

    # Mock S3 download to succeed
    mock_s3.download_object_from_s3.return_value = True

    runner = MLRunner()
    targets = runner.get_inference_targets()

    assert len(targets) == 1
    assert targets[0]["DropID"] == "KSF_20240124_BUV_KSF_085_01"
