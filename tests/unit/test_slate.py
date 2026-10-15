"""Slate detection on synthetic frames, and the per-drop save/skip behaviour."""

import cv2
import numpy as np
import pytest

from spyfish.config.wrapper import config
from spyfish.extraction import slate
from spyfish.extraction.slate import SlateParams, extract_slate_frame, score_frame

DROP = "KSF_20240124_BUV_KSF_085_01"
SEA = (200, 120, 40)  # BGR: saturated blue, like water or a deck-side sea view


def _sea(w=640, h=360):
    return np.full((h, w, 3), SEA, dtype=np.uint8)


def _board(frame, x=150, y=60, w=340, h=250, strokes=30):
    """A white board with handwriting-sized dark marks, held roughly square-on."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 245, 245), -1)
    rng = np.random.default_rng(0)
    for _ in range(strokes):
        sx = int(rng.integers(x + 15, x + w - 25))
        sy = int(rng.integers(y + 15, y + h - 25))
        cv2.rectangle(frame, (sx, sy), (sx + 6, sy + 6), (20, 20, 20), -1)
    return frame


@pytest.fixture
def params():
    return SlateParams.from_config()


def test_params_come_from_config(params):
    assert params.stroke_area_px == tuple(config.slate_detection["stroke_area_px"])
    assert params.max_seconds == config.slate_detection["max_seconds"]


def test_written_board_scores(params):
    fs = score_frame(_board(_sea()), params)
    assert fs.score > 0
    assert fs.strokes >= params.min_strokes
    assert fs.aspect < params.max_aspect


def test_blank_board_does_not_score(params):
    """A white panel with nothing written on it is a wall or a deck, not a slate."""
    fs = score_frame(_board(_sea(), strokes=0), params)
    assert fs.score == 0


def test_underwater_frame_does_not_score(params):
    assert score_frame(_sea(), params).score == 0


def test_long_thin_white_strip_does_not_score(params):
    """Deck edges and gunwales are bright and marked, but 4:1 or longer."""
    fs = score_frame(_board(_sea(), x=20, y=100, w=600, h=100), params)
    assert fs.aspect > params.max_aspect
    assert fs.score == 0


def test_extract_saves_jpeg_to_drop_path(tmp_path, monkeypatch):
    out = tmp_path / "drop" / f"{DROP}_slate.jpg"
    monkeypatch.setattr(config, "get_slate_frame_path", lambda d: out)
    best = slate.FrameScore(2.0, 0.25, 0.9, 1.3, 0.08, 25, 0.2)
    seen = {}

    def fake_find(video, max_seconds=None):
        seen["max_seconds"] = max_seconds
        return best, _board(_sea())

    monkeypatch.setattr(slate, "find_slate_frame", fake_find)

    assert extract_slate_frame(DROP, tmp_path / "v.mp4", max_seconds=42.0) == out
    assert seen["max_seconds"] == 42.0
    assert out.exists()
    assert cv2.imread(str(out)).shape == (360, 640, 3)


def test_extract_returns_none_when_no_slate(tmp_path, monkeypatch):
    out = tmp_path / f"{DROP}_slate.jpg"
    monkeypatch.setattr(config, "get_slate_frame_path", lambda d: out)
    monkeypatch.setattr(
        slate, "find_slate_frame", lambda v, max_seconds=None: (None, None)
    )

    assert extract_slate_frame(DROP, tmp_path / "v.mp4") is None
    assert not out.exists()


def test_extract_skips_when_jpeg_already_exists(tmp_path, monkeypatch):
    out = tmp_path / f"{DROP}_slate.jpg"
    out.write_bytes(b"old")
    monkeypatch.setattr(config, "get_slate_frame_path", lambda d: out)

    def boom(_, max_seconds=None):
        raise AssertionError("video must not be opened when the JPEG exists")

    monkeypatch.setattr(slate, "find_slate_frame", boom)
    assert extract_slate_frame(DROP, tmp_path / "v.mp4") == out
    assert out.read_bytes() == b"old"


def test_extract_force_rescans_over_existing_jpeg(tmp_path, monkeypatch):
    out = tmp_path / f"{DROP}_slate.jpg"
    out.write_bytes(b"old")
    monkeypatch.setattr(config, "get_slate_frame_path", lambda d: out)
    best = slate.FrameScore(2.0, 0.25, 0.9, 1.3, 0.08, 25, 0.2)
    monkeypatch.setattr(
        slate, "find_slate_frame", lambda v, max_seconds=None: (best, _board(_sea()))
    )

    assert extract_slate_frame(DROP, tmp_path / "v.mp4", force=True) == out
    assert out.read_bytes() != b"old"
    assert cv2.imread(str(out)).shape == (360, 640, 3)


def test_extract_rejects_bad_drop_id(tmp_path):
    with pytest.raises(ValueError):
        extract_slate_frame("../oops", tmp_path / "v.mp4")
