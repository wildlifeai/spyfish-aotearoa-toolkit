"""
Unit tests for S3Handler — small surface for now, focused on regression
coverage for the delete-on-upload-failure bug. Broader coverage is tracked
in todo.md under "Testing — critical gaps".
"""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import BotoCoreError

from spyfish.storage.s3_handler import S3Handler


@pytest.fixture(autouse=True)
def _reset_s3_singleton():
    """S3Handler is a singleton — reset between tests so each gets a clean state."""
    S3Handler._instance = None
    yield
    S3Handler._instance = None


def _make_handler():
    with patch("spyfish.storage.s3_handler.boto3"):
        return S3Handler(bucket="test-bucket")


def test_upload_preserves_local_file_on_failure(tmp_path):
    """Regression: `finally` block used to delete the local file even when
    upload raised BotoCoreError, causing silent data loss. The delete must
    only run after a confirmed successful upload."""
    local_file = tmp_path / "annotations.csv"
    local_file.write_text("drop_id,count\nKSF_001,5\n")

    handler = _make_handler()
    handler.s3 = MagicMock()
    handler.s3.upload_file.side_effect = BotoCoreError()

    result = handler.upload_file_to_s3(
        filename=str(local_file),
        key="test/annotations.csv",
        delete_file_after_upload=True,
    )

    assert result is False
    # The critical assertion: local file must still exist for retry
    assert (
        local_file.exists()
    ), "Local file was deleted after upload failure — data would be lost"


def test_upload_deletes_local_file_on_success(tmp_path):
    """When upload succeeds and delete_file_after_upload=True, the local
    file should be removed."""
    local_file = tmp_path / "annotations.csv"
    local_file.write_text("drop_id,count\nKSF_001,5\n")

    handler = _make_handler()
    handler.s3 = MagicMock()
    handler.s3.upload_file.return_value = None  # success

    result = handler.upload_file_to_s3(
        filename=str(local_file),
        key="test/annotations.csv",
        delete_file_after_upload=True,
    )

    assert result is True
    assert not local_file.exists()


def test_upload_keeps_local_file_when_delete_flag_false(tmp_path):
    """Default behaviour: delete_file_after_upload=False → file stays."""
    local_file = tmp_path / "annotations.csv"
    local_file.write_text("x")

    handler = _make_handler()
    handler.s3 = MagicMock()
    handler.s3.upload_file.return_value = None

    handler.upload_file_to_s3(filename=str(local_file), key="test/x.csv")

    assert local_file.exists()
