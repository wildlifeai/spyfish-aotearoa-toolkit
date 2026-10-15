"""The slate JPEG lives in the drop's own folder, named after the drop."""

import pytest

from spyfish.config.wrapper import config


def test_slate_frame_path_sits_in_drop_dir():
    drop_id = "KSF_20240124_BUV_KSF_085_01"
    path = config.get_slate_frame_path(drop_id)
    assert path.parent == config.get_drop_dir(drop_id)
    assert path.name == f"{drop_id}{config.slate_frame_suffix}"


def test_slate_frame_path_rejects_bad_drop_id():
    with pytest.raises(ValueError):
        config.get_slate_frame_path("../etc/passwd")
