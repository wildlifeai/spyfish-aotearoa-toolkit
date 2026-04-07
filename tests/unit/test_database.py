from spyfish.config.base import PipelineStatus


def test_add_or_update_deployment(temp_db):
    drop_id = "KSF_20240124_BUV_KSF_085_01"
    temp_db.add_or_update_deployment(
        drop_id=drop_id,
        status=PipelineStatus.PENDING_ARRIVAL,
        video_path="test/path.mp4",
    )

    deployment = temp_db.get_deployment(drop_id)
    assert deployment is not None
    assert deployment["status"] == PipelineStatus.PENDING_ARRIVAL
    assert deployment["video_path"] == "test/path.mp4"


def test_update_status(temp_db):
    drop_id = "KSF_20240124_BUV_KSF_085_02"
    temp_db.add_or_update_deployment(drop_id, PipelineStatus.READY_FOR_ML)

    temp_db.update_status(drop_id, PipelineStatus.PROCESSING_ML)
    deployment = temp_db.get_deployment(drop_id)
    assert deployment["status"] == PipelineStatus.PROCESSING_ML
