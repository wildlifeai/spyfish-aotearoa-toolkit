from unittest.mock import patch

from spyfish.orchestrator.ml_runner import MLRunner


def _make_records(n: int) -> list[dict]:
    return [
        {
            "drop_id": f"KSF_20240124_BUV_KSF_085_{i+1:02d}",
            "video_path": f"path/{i+1}.mp4",
            "sampling_start": 0,
            "sampling_end": 100,
            "video_storage_class": "STANDARD",
        }
        for i in range(n)
    ]


@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_get_inference_targets(mock_s3_class, mock_db_class):
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value

    # Mock database to return one item ready for ML
    mock_db.get_deployments_eligible.return_value = [
        {
            "drop_id": "KSF_20240124_BUV_KSF_085_01",
            "video_path": "path/1.mp4",
            "sampling_start": 120,
            "sampling_end": 100,
            "video_storage_class": "STANDARD",
        }
    ]

    # Mock S3 download to succeed
    mock_s3.download_object_from_s3.return_value = True

    runner = MLRunner()
    targets = runner.get_inference_targets()

    assert len(targets) == 1
    assert targets[0]["drop_id"] == "KSF_20240124_BUV_KSF_085_01"


@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_get_inference_targets_caps_at_limit(mock_s3_class, mock_db_class, mock_exists):
    """With 5 eligible drops and limit=3, only 3 downloads should be attempted."""
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value
    mock_db.get_deployments_eligible.return_value = _make_records(5)
    mock_s3.download_object_from_s3.return_value = True
    # db_path check → True; local video existence → False (so we download)
    mock_exists.side_effect = lambda p: str(p).endswith(".db")

    runner = MLRunner()
    runner.limit = 3
    targets = runner.get_inference_targets()

    assert len(targets) == 3
    # First 3 by priority order
    assert [t["drop_id"] for t in targets] == [
        "KSF_20240124_BUV_KSF_085_01",
        "KSF_20240124_BUV_KSF_085_02",
        "KSF_20240124_BUV_KSF_085_03",
    ]
    assert mock_s3.download_object_from_s3.call_count == 3


@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_get_inference_targets_backfills_on_download_failure(
    mock_s3_class, mock_db_class, mock_exists
):
    """Regression: limit used to be applied BEFORE download loop, so failures
    silently reduced the batch size. Now failures should be backfilled from
    later candidates until we have `limit` successes."""
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value
    mock_db.get_deployments_eligible.return_value = _make_records(5)
    # First 2 downloads fail, next 3 succeed
    mock_s3.download_object_from_s3.side_effect = [False, False, True, True, True]
    mock_exists.side_effect = lambda p: str(p).endswith(".db")

    runner = MLRunner()
    runner.limit = 3
    targets = runner.get_inference_targets()

    # Should still get 3 targets (drops 03, 04, 05) despite early failures
    assert len(targets) == 3
    assert [t["drop_id"] for t in targets] == [
        "KSF_20240124_BUV_KSF_085_03",
        "KSF_20240124_BUV_KSF_085_04",
        "KSF_20240124_BUV_KSF_085_05",
    ]


@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_get_inference_targets_all_downloads_fail(
    mock_s3_class, mock_db_class, mock_exists
):
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value
    mock_db.get_deployments_eligible.return_value = _make_records(3)
    mock_s3.download_object_from_s3.return_value = False
    mock_exists.side_effect = lambda p: str(p).endswith(".db")

    runner = MLRunner()
    runner.limit = 3
    targets = runner.get_inference_targets()

    assert targets == []
