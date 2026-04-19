from spyfish.config.base import MlStatus


def test_add_or_update_deployment(temp_db):
    drop_id = "KSF_20240124_BUV_KSF_085_01"
    temp_db.add_or_update_deployment(
        drop_id=drop_id,
        ml_status=MlStatus.PENDING,
        video_path="test/path.mp4",
    )

    deployment = temp_db.get_deployment(drop_id)
    assert deployment is not None
    assert deployment["ml_status"] == MlStatus.PENDING
    assert deployment["video_path"] == "test/path.mp4"


def test_advance_ml_status(temp_db):
    drop_id = "KSF_20240124_BUV_KSF_085_02"
    temp_db.add_or_update_deployment(drop_id=drop_id, ml_status=MlStatus.READY)

    temp_db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.RUNNING)
    deployment = temp_db.get_deployment(drop_id)
    assert deployment["ml_status"] == MlStatus.RUNNING
