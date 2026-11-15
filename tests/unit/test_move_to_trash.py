"""move_to_trash: files and folders go to a dated trash folder, never deleted."""

from datetime import date

import pytest

from spyfish.config.wrapper import config
from spyfish.utils import move_to_trash


@pytest.fixture
def trash(tmp_path, monkeypatch):
    t = tmp_path / "trash"
    monkeypatch.setattr(type(config), "trash_dir", property(lambda self: t))
    return t / date.today().isoformat()


def test_file_moves_under_its_parent_name(tmp_path, trash):
    cdir = tmp_path / "SLI_20240124_BUV_SLI_001_01_slate_candidates"
    cdir.mkdir()
    f = cdir / "1_0012.00s.jpg"
    f.write_bytes(b"x")
    dest = move_to_trash(f)
    assert not f.exists()
    assert dest == trash / cdir.name / "1_0012.00s.jpg"
    assert dest.read_bytes() == b"x"


def test_same_name_twice_keeps_both(tmp_path, trash):
    for content in (b"first", b"second"):
        d = tmp_path / "drop"
        d.mkdir(exist_ok=True)
        (d / "a.jpg").write_bytes(content)
        move_to_trash(d / "a.jpg")
    assert (trash / "drop" / "a.jpg").read_bytes() == b"first"
    assert (trash / "drop" / "a_1.jpg").read_bytes() == b"second"


def test_folder_moves_whole(tmp_path, trash):
    d = tmp_path / "slates_rejected" / "X_slate_candidates"
    d.mkdir(parents=True)
    (d / "scores.csv").write_text("rank\n")
    dest = move_to_trash(d)
    assert not d.exists()
    assert (dest / "scores.csv").read_text() == "rank\n"


def test_missing_path_is_none(tmp_path, trash):
    assert move_to_trash(tmp_path / "nope.jpg") is None
