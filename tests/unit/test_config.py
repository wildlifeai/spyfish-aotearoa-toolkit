from unittest.mock import patch

import pytest

from spyfish.config.paths import config


def test_validate_drop_id_valid():
    valid_id = "KSF_20240124_BUV_KSF_085_01"
    assert config.validate_drop_id(valid_id) == valid_id


def test_validate_drop_id_invalid_traversal():
    malicious_id = "../etc/passwd"
    with pytest.raises(ValueError, match="Invalid DropID format"):
        config.validate_drop_id(malicious_id)


def test_database_path_generation():
    db_path = config.db_path
    assert "spyfish_pipeline.db" in str(db_path)
