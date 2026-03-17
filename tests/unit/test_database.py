import pytest
from spyfish.config.base import PipelineStatus

def test_add_or_update_deployment(temp_db):
    drop_id = "ABC_12345678_BUV_ABC_123_01"
    temp_db.add_or_update_deployment(
        drop_id=drop_id,
        status=PipelineStatus.PENDING_ARRIVAL,
        video_path="test/path.mp4"
    )

    deployment = temp_db.get_deployment(drop_id)
    assert deployment is not None
    assert deployment['status'] == PipelineStatus.PENDING_ARRIVAL
    assert deployment['video_path'] == "test/path.mp4"

def test_update_status(temp_db):
    drop_id = "XYZ_99999999_BUV_ABC_123_01"
    temp_db.add_or_update_deployment(drop_id, PipelineStatus.READY_FOR_ML)

    temp_db.update_status(drop_id, PipelineStatus.PROCESSING_ML)
    deployment = temp_db.get_deployment(drop_id)
    assert deployment['status'] == PipelineStatus.PROCESSING_ML
