"""Tests for canvas-aware YOLO normalisation in biigle_to_yolo.

The bug these lock in: `convert_annotations_to_yolo` used to normalise boxes by
the *on-disk* frame's pixel size. The only correct denominator is the canvas
the box was DRAWN on — Biigle records it per image in the `attributes` JSON.
When the two diverge (e.g. a 1920-wide annotation against a 1440-wide
re-extracted frame, as with the TON video-era training frames), trusting the
on-disk file mis-normalises every box. These tests prove the canvas wins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from spyfish.biigle.biigle_to_yolo import (
    _canvas_from_attributes,
    convert_annotations_to_yolo,
)

# A box centred at pixel (960, 540) spanning 192x108px. On a 1920x1080 canvas
# that is YOLO (0.5, 0.5, 0.1, 0.1). On a 1440x1080 canvas the same pixels would
# read (0.667, 0.5, 0.133, 0.1) — wrong. The point coords below are the four
# corners as Biigle stores them (8 flat floats).
_POINTS = [864.0, 486.0, 1056.0, 486.0, 1056.0, 594.0, 864.0, 594.0]


def _attrs(w: int, h: int) -> str:
    return json.dumps({"size": 1, "mimetype": "image/jpeg", "width": w, "height": h})


def _make_frame(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h)).save(path)


def _read_label(labels_dir: Path, stem: str) -> list[float]:
    line = (labels_dir / f"{stem}.txt").read_text().strip()
    return [float(x) for x in line.split()[1:]]  # drop class id


def test_canvas_from_attributes_parses_width_height():
    g = pd.DataFrame({"attributes": [_attrs(1920, 1080)]})
    assert _canvas_from_attributes(g) == (1920, 1080)


def test_canvas_from_attributes_absent_column_returns_none():
    assert _canvas_from_attributes(pd.DataFrame({"points": ["[]"]})) is None


def test_attributes_canvas_wins_over_on_disk_frame(tmp_path):
    """The TON case: frame on disk is 1440 wide but it was annotated at 1920.
    Normalisation must use 1920 (the canvas), giving cx≈0.5 not 0.667."""
    images = tmp_path / "frames"
    _make_frame(images / "f.jpg", 1440, 1080)  # divergent on-disk size
    df = pd.DataFrame(
        {
            "filename": ["f.jpg"],
            "label_name": ["fish"],
            "points": [json.dumps(_POINTS)],
            "attributes": [_attrs(1920, 1080)],
        }
    )
    convert_annotations_to_yolo(df, {"fish": 0}, tmp_path / "labels", images)
    cx, cy, w, h = _read_label(tmp_path / "labels", "f")
    assert abs(cx - 0.5) < 1e-3
    assert abs(w - 0.1) < 1e-3


def test_falls_back_to_on_disk_when_no_attributes(tmp_path):
    """No attributes column → use the on-disk frame size (1920 here → cx≈0.5)."""
    images = tmp_path / "frames"
    _make_frame(images / "f.jpg", 1920, 1080)
    df = pd.DataFrame(
        {
            "filename": ["f.jpg"],
            "label_name": ["fish"],
            "points": [json.dumps(_POINTS)],
        }
    )
    convert_annotations_to_yolo(df, {"fish": 0}, tmp_path / "labels", images)
    cx, _, w, _ = _read_label(tmp_path / "labels", "f")
    assert abs(cx - 0.5) < 1e-3
    assert abs(w - 0.1) < 1e-3


def test_canvas_mismatch_is_logged(tmp_path, caplog):
    images = tmp_path / "frames"
    _make_frame(images / "f.jpg", 1440, 1080)
    df = pd.DataFrame(
        {
            "filename": ["f.jpg"],
            "label_name": ["fish"],
            "points": [json.dumps(_POINTS)],
            "attributes": [_attrs(1920, 1080)],
        }
    )
    with caplog.at_level("WARNING"):
        convert_annotations_to_yolo(df, {"fish": 0}, tmp_path / "labels", images)
    assert any("differs from the on-disk frame" in r.message for r in caplog.records)


# --- rectangle parsing: flat (image) vs nested (video keyframe) ---


def test_rectangle_corners_flat_image_points():
    from spyfish.biigle.biigle_to_yolo import _rectangle_corners

    assert _rectangle_corners(json.dumps(_POINTS)) == _POINTS


def test_rectangle_corners_unwraps_single_video_keyframe():
    from spyfish.biigle.biigle_to_yolo import _rectangle_corners

    # video format: [[...8...]] -> must NOT be dropped
    assert _rectangle_corners(json.dumps([_POINTS])) == _POINTS


def test_rectangle_corners_skips_multi_keyframe():
    from spyfish.biigle.biigle_to_yolo import _rectangle_corners

    # a box that moves across two keyframes can't be placed on one frame here
    assert _rectangle_corners(json.dumps([_POINTS, _POINTS])) is None


def test_rectangle_corners_skips_non_rectangle():
    from spyfish.biigle.biigle_to_yolo import _rectangle_corners

    assert _rectangle_corners(json.dumps([1.0, 2.0])) is None  # a Point
    assert _rectangle_corners("not json") is None


def test_nested_video_box_is_converted_not_dropped(tmp_path):
    """Regression: video rows ([[...]] points) used to fail len==8 and vanish."""
    images = tmp_path / "frames"
    _make_frame(images / "f.jpg", 1920, 1080)
    df = pd.DataFrame(
        {
            "filename": ["f.jpg"],
            "label_name": ["fish"],
            "points": [json.dumps([_POINTS])],  # nested video keyframe
            "attributes": [_attrs(1920, 1080)],
        }
    )
    convert_annotations_to_yolo(df, {"fish": 0}, tmp_path / "labels", images)
    cx, _, w, _ = _read_label(tmp_path / "labels", "f")
    assert abs(cx - 0.5) < 1e-3 and abs(w - 0.1) < 1e-3


def test_rotated_box_collapses_to_axis_aligned_hbb(tmp_path):
    """A rotated (oriented) box becomes a correct horizontal AABB box."""
    images = tmp_path / "frames"
    _make_frame(images / "f.jpg", 1920, 1080)
    rotated = [900, 500, 1040, 560, 1000, 700, 860, 640]  # ~rotated quad
    df = pd.DataFrame(
        {
            "filename": ["f.jpg"],
            "label_name": ["fish"],
            "points": [json.dumps(rotated)],
            "attributes": [_attrs(1920, 1080)],
        }
    )
    convert_annotations_to_yolo(df, {"fish": 0}, tmp_path / "labels", images)
    cx, cy, w, h = _read_label(tmp_path / "labels", "f")
    assert 0 < cx < 1 and 0 < cy < 1 and 0 < w < 1 and 0 < h < 1
