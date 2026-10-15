"""Zooniverse validation subjects: selection override, inference grid, rendering."""

from __future__ import annotations

import csv as csv_module
import json
import re
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from spyfish.config.wrapper import config
from spyfish.extraction import select_frames as sf
from spyfish.zooniverse import validation_frames as vf

DROP_ID = "KSF_20240124_BUV_KSF_085_06"
FPS = 30
W, H = 640, 360

STRATEGY = {
    "per_species_maxn": 2,
    "per_species_confusing": 1,
    "fish_bands": 3,
    "blind_export": 3,
    "random_detection_export": 3,
    "start_export": 1,
    "effort_multiplier": 1,
    "spacing_divisor": 2,
    "peak_merge_seconds": 5.0,
    "same_frame_seconds": 1.0,
    "temporal_spacing_seconds": 30,
}


@pytest.fixture
def strategy(monkeypatch):
    """A frame_strategy for selection tests (the test config has none)."""
    strat = dict(STRATEGY)
    monkeypatch.setattr(type(config), "frame_strategy", property(lambda self: strat))
    monkeypatch.setattr(type(config), "min_frames_per_drop", property(lambda self: 1))
    return strat


@pytest.fixture
def video(tmp_path):
    """10 s synthetic video, 30 fps, each frame a different grey level."""
    path = tmp_path / f"{DROP_ID}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(10 * FPS):
        writer.write(np.full((H, W, 3), i % 255, dtype=np.uint8))
    writer.release()
    return path


def _raw_csv(path, rows):
    """rows: (frame, class, confidence, cx, cy, w, h)."""
    with open(path, "w", newline="") as fh:
        w = csv_module.writer(fh)
        w.writerow(["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"])
        for frame, cls, conf, cx, cy, bw, bh in rows:
            w.writerow([frame, frame / FPS, cls, conf, cx, cy, bw, bh])
    return path


# ── effort override on the shared BIIGLE selector ────────────────────────────


def _blind_count(sel):
    return int(sel["SelectionReason"].str.startswith("Blind").sum())


def test_effort_override_scales_blind_frames_and_leaves_config_alone(
    tmp_path, strategy
):
    raw = _raw_csv(
        tmp_path / "raw.csv",
        [(f, "Parapercis colias", 0.9, 100, 100, 80, 60) for f in (300, 600, 900)],
    )
    dep = {"sampling_start": 0, "sampling_end": 1800}
    with patch.object(sf, "DatabaseManager") as db:
        db.return_value.get_deployment.return_value = dep
        default = sf.select_frames(str(raw), str(tmp_path / "a.csv"), DROP_ID)
        boosted = sf.select_frames(
            str(raw), str(tmp_path / "b.csv"), DROP_ID, effort_multiplier=3
        )
    assert _blind_count(default) == STRATEGY["blind_export"]
    assert _blind_count(boosted) == STRATEGY["blind_export"] * 3
    # Copied, never mutated: a later BIIGLE selection must not inherit x3.
    assert strategy["effort_multiplier"] == 1


# ── inference grid ───────────────────────────────────────────────────────────


def _grid(video, start, end, dets):
    cap = cv2.VideoCapture(str(video))
    try:
        return vf.inference_grid(cap, start, end, dets)
    finally:
        cap.release()


def test_grid_steps_by_stride_inside_the_window(video):
    fps, grid = _grid(video, 1.0, 8.0, pd.DataFrame(columns=["frame"]))
    stride = round(fps / config.ml_fps)
    assert stride == 10
    assert all(b - a == stride for a, b in zip(grid, grid[1:]))
    assert grid[0] / fps >= 1.0 - 1 / fps
    assert grid[-1] / fps <= 8.0


def test_grid_reanchors_on_raw_csv_frames(video):
    """Detections are frames inference provably saw; the grid must contain them."""
    dets = pd.DataFrame({"frame": [45, 75, 105]})
    _, grid = _grid(video, 1.0, 8.0, dets)
    assert {45, 75, 105} <= set(grid)


def test_snap_moves_blind_times_onto_grid_and_dedups():
    grid = list(range(30, 241, 10))
    sel = pd.DataFrame(
        {
            config.csv_clip_max_time_column: [2.04, 2.0, 5.5],
            config.selection_reason_column: [
                "Blind (False Negative Check)",
                "MaxN (Parapercis colias)",
                "Blind (False Negative Check)",
            ],
            config.csv_scientific_name_column: [None, "Parapercis colias", None],
        }
    )
    out = vf.snap_selection_to_grid(sel, grid, FPS)
    assert set(out["frame"]) <= set(grid)
    # 2.04 s and 2.0 s land on the same frame (60): one subject, not two.
    assert list(out["frame"]) == [60, 160]


# ── shared selection reuse ───────────────────────────────────────────────────


def test_selection_reused_only_for_same_model_and_effort(tmp_path, monkeypatch):
    path = tmp_path / "sel.csv"
    monkeypatch.setattr(config, "get_validation_selection_path", lambda d: path)
    made = pd.DataFrame(
        {
            config.csv_clip_max_time_column: [10.0],
            config.selection_reason_column: ["MaxN (fish)"],
        }
    )
    with patch.object(vf, "select_frames", return_value=made.copy()) as sel:
        vf.get_or_create_selection(DROP_ID, tmp_path / "raw.csv", "model_a")
        vf.get_or_create_selection(DROP_ID, tmp_path / "raw.csv", "model_a")
        assert sel.call_count == 1  # second stage reads the first one's file
        vf.get_or_create_selection(DROP_ID, tmp_path / "raw.csv", "model_b")
        assert sel.call_count == 2  # another model's selection is stale


def test_selection_ignores_old_file_without_provenance(tmp_path, monkeypatch):
    path = tmp_path / "sel.csv"
    pd.DataFrame({config.csv_clip_max_time_column: [1.0]}).to_csv(path, index=False)
    monkeypatch.setattr(config, "get_validation_selection_path", lambda d: path)
    with patch.object(vf, "select_frames", return_value=pd.DataFrame()) as sel:
        vf.get_or_create_selection(DROP_ID, tmp_path / "raw.csv", "model_a")
    assert sel.call_count == 1


# ── rendering ────────────────────────────────────────────────────────────────


class _Registry:
    def label_for(self, name):
        return {"Parapercis colias": "Blue cod"}.get(name, name)


@pytest.fixture
def ctx(tmp_path, video, monkeypatch):
    monkeypatch.setattr(config, "get_frames_dir", lambda d: tmp_path / "frames")
    monkeypatch.setattr(
        config, "get_validation_upload_dir", lambda d: tmp_path / "upload"
    )
    monkeypatch.setattr(
        config,
        "get_validation_manifest_path",
        lambda d, k: tmp_path / "annotations" / f"{k}_manifest.csv",
    )
    monkeypatch.setattr(vf, "species_registry", lambda: _Registry())
    dets = pd.DataFrame(
        [
            # frame, class, conf, cx, cy, w, h
            (60, "Parapercis colias", 0.9, 200, 180, 120, 80),
            (60, "fish", 0.4, 500, 100, 20, 20),  # too small to crop
            (60, "Parapercis colias", 0.7, 400, 250, 100, 70),
        ],
        columns=["frame", "class", "confidence", "x", "y", "w", "h"],
    )
    dets["time_seconds"] = dets["frame"] / FPS
    dets["box_index"] = dets.groupby("frame").cumcount()
    frames = pd.DataFrame(
        {
            "frame": [60, 150],
            "time_seconds": [2.0, 5.0],
            config.selection_reason_column: [
                "MaxN (Parapercis colias)",
                "Blind (False Negative Check)",
            ],
            config.csv_scientific_name_column: ["Parapercis colias", None],
        }
    )
    return vf.ValidationContext(
        drop_id=DROP_ID,
        model_name="species_test",
        video_path=video,
        sampling_start=0.0,
        detections=dets,
        frames=frames,
    )


def test_every_box_in_a_picked_frame_is_cropped_or_explained(ctx):
    m = vf.render_validation_crops(ctx)
    assert len(m) == 3  # all three boxes on frame 60, none from the empty blind frame
    cropped = m[m[vf.CROP_PATH_COLUMN].notna()]
    assert len(cropped) == 2
    small = m[m["NotCroppedReason"].notna()]
    assert list(small[config.csv_scientific_name_column]) == ["fish"]
    for p in cropped[vf.CROP_PATH_COLUMN]:
        img = cv2.imread(p)
        assert img.shape[:2] == (800, 800)
        # The crop's name starts with its frame's name, so the two join by string.
        assert Path(p).name.startswith(f"{DROP_ID}__frame_0002.000s__crop_")
    assert set(m["DetectionKey"]) == {f"{DROP_ID}|60|{i}" for i in range(3)}
    assert list(cropped["LabelShown"]) == ["Blue cod", "Blue cod"]
    # Count of the box's own species in the frame, as clips report MaxInterval.
    assert list(cropped[config.csv_max_interval_column]) == [2, 2]


def test_box_in_name_redraws_the_box(ctx):
    m = vf.render_validation_crops(ctx)
    name = Path(m[vf.CROP_PATH_COLUMN].dropna().iloc[0]).name
    x1, y1, x2, y2 = map(
        int, re.search(r"__box_(\d+)_(\d+)_(\d+)_(\d+)", name).groups()
    )
    assert (x1, y1, x2, y2) == (140, 140, 260, 220)


def test_frames_keep_empty_blind_frames_and_record_boxes(ctx):
    m = vf.render_validation_frames(ctx)
    assert len(m) == 2
    blind = m[m[config.selection_reason_column].str.startswith("Blind")].iloc[0]
    assert blind[config.csv_max_interval_column] == 0
    assert json.loads(blind["Boxes"]) == []
    busy = m[m["FrameIndex"] == 60].iloc[0]
    assert busy[config.csv_max_interval_column] == 3
    assert len(json.loads(busy["Boxes"])) == 3
    assert busy["RenderScale"] == 1.0  # 640 px wide, under max_width_px
    assert Path(busy[vf.FRAME_PATH_COLUMN]).name.endswith("__vframe.jpg")
    assert config.get_validation_manifest_path(DROP_ID, "frames").exists()


def test_random_detection_frames_are_fair_and_reproducible(tmp_path, strategy):
    """Random detection frames always hold a box, and a re-run picks the same ones."""
    rows = [
        (f, "Notolabrus celidotus", 0.6, 100, 100, 80, 60)
        for f in range(300, 54000, 900)
    ]
    raw = _raw_csv(tmp_path / "raw.csv", rows)
    dep = {"sampling_start": 0, "sampling_end": 1800}
    with patch.object(sf, "DatabaseManager") as db:
        db.return_value.get_deployment.return_value = dep
        a = sf.select_frames(str(raw), str(tmp_path / "a.csv"), DROP_ID)
        b = sf.select_frames(str(raw), str(tmp_path / "b.csv"), DROP_ID)
    pick = a[a["SelectionReason"] == "Random detection"]
    assert len(pick) == STRATEGY["random_detection_export"]
    detection_times = {f / FPS for f, *_ in rows}
    times = pick[config.csv_clip_max_time_column]
    assert all(min(abs(t - d) for d in detection_times) < 1e-6 for t in times)
    assert list(times) == list(
        b.loc[
            b["SelectionReason"] == "Random detection", config.csv_clip_max_time_column
        ]
    )


def test_random_detection_maps_to_spot_check_label():
    from spyfish.biigle.selection_reason import canonical_reason

    assert canonical_reason("Random detection") == "spot_check"


def test_clean_frames_land_in_shared_frames_folder_and_are_reused(ctx, tmp_path):
    """Raw frames live in frames/ under the standard name, extracted once."""
    vf.render_validation_frames(ctx)
    clean = tmp_path / "frames" / f"{DROP_ID}__frame_0002.000s.jpg"
    assert clean.exists()
    stamp = clean.stat().st_mtime_ns
    m = vf.render_validation_crops(ctx)  # second stage: reuses, does not re-extract
    assert clean.stat().st_mtime_ns == stamp
    assert set(m["CleanFramePath"]) == {str(clean)}
    # Upload copies are separate from the raw frames.
    assert not any("crop" in p.name for p in (tmp_path / "frames").iterdir())


def test_drop_with_no_detections_still_gets_blind_frames(tmp_path, strategy):
    """No boxes is not "nothing to review": blind frames are the recall check."""
    raw = _raw_csv(tmp_path / "raw.csv", [])
    with patch.object(sf, "DatabaseManager") as db:
        db.return_value.get_deployment.return_value = {
            "sampling_start": 0,
            "sampling_end": 1800,
        }
        sel = sf.select_frames(str(raw), str(tmp_path / "s.csv"), DROP_ID)
    assert len(sel) == STRATEGY["blind_export"]
    assert sel["SelectionReason"].str.startswith("Blind").all()


# ── stage entry points ───────────────────────────────────────────────────────


def test_already_uploaded_set_skips_rendering(tmp_path, monkeypatch):
    from spyfish.zooniverse import upload as up

    manifest = tmp_path / "frames_manifest.csv"
    pd.DataFrame({vf.FRAME_PATH_COLUMN: ["a.jpg", "b.jpg"]}).to_csv(
        manifest, index=False
    )
    monkeypatch.setattr(config, "get_validation_manifest_path", lambda d, k: manifest)
    with (
        patch.object(up, "existing_subject_count", return_value=2),
        patch.object(vf, "prepare_validation") as prep,
    ):
        assert vf.run_validation_frames_drop(DROP_ID) == vf.ZooFramesStatus.UPLOADED
    prep.assert_not_called()


def test_existing_set_with_other_count_resumes(tmp_path, monkeypatch):
    """A partial set is not "done": the stage renders again so the uploader
    can add the missing subjects, rather than raising every run."""
    from spyfish.zooniverse import upload as up

    manifest = tmp_path / "frames_manifest.csv"
    pd.DataFrame({vf.FRAME_PATH_COLUMN: ["a.jpg", "b.jpg"]}).to_csv(
        manifest, index=False
    )
    monkeypatch.setattr(config, "get_validation_manifest_path", lambda d, k: manifest)
    with (
        patch.object(up, "existing_subject_count", return_value=1),
        patch.object(vf, "prepare_validation", return_value=None) as prep,
    ):
        assert vf.run_validation_frames_drop(DROP_ID) is None
    prep.assert_called_once_with(DROP_ID)


def test_empty_manifest_leaves_the_drop_pending(tmp_path, monkeypatch):
    """Every picked frame failed to decode: nothing to upload, no error."""
    from spyfish.zooniverse import upload as up

    with (
        patch.object(up, "existing_subject_count", return_value=None),
        patch.object(vf, "prepare_validation", return_value=object()),
        patch.object(vf, "render_validation_frames", return_value=pd.DataFrame()),
        patch.object(up, "upload_validation_frames_to_zooniverse") as upload,
    ):
        assert vf.run_validation_frames_drop(DROP_ID) is None
    upload.assert_not_called()


def test_dry_run_never_asks_zooniverse(monkeypatch):
    from spyfish.zooniverse import upload as up

    with (
        patch.object(up, "existing_subject_count") as ask,
        patch.object(vf, "prepare_validation", return_value=None),
    ):
        assert vf.run_validation_frames_drop(DROP_ID, dry_run=True) is None
    ask.assert_not_called()
