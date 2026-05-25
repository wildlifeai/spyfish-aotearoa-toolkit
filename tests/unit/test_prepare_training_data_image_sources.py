"""Regression tests for the `training_frames/` image-source wiring in
prepare_training_data.

Expert labels downloaded for survey-level Training-frames volumes
(`download_training_volume_labels`) land in `<drop>/labels/` while their JPEGs
stay in `<drop>/training_frames/`. The trainer must therefore treat
`training_frames/` as a canonical image source — alongside the normal
`frames/` — so labels (including empty-`.txt` background/negatives) get paired
with their images. These tests lock that in.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from spyfish.config.wrapper import config
from spyfish.ml.training.prepare_training_data import (
    _build_image_index,
    _sample_background_frames,
    copy_split_files,
    discover_extra_drops,
)

DROP = "KSF_20240124_BUV_KSF_085_01"
SURVEY = "KSF_20240124_BUV"
POS_STEM = f"{DROP}__frame_1.000s"
NEG_STEM = f"{DROP}__frame_2.000s"


def _touch(p: Path, content: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def _make_training_drop(root: Path, *, with_raw_csv: bool = True) -> Path:
    """A training-only drop: JPEGs in training_frames/, labels in labels/
    (one positive box + one empty negative), no MaxN CSV."""
    drop_dir = root / SURVEY / DROP
    _touch(drop_dir / "training_frames" / f"{POS_STEM}.jpg")
    _touch(drop_dir / "training_frames" / f"{NEG_STEM}.jpg")
    _touch(drop_dir / "labels" / f"{POS_STEM}.txt", "0 0.5 0.5 0.2 0.2\n")
    _touch(drop_dir / "labels" / f"{NEG_STEM}.txt", "")  # empty == background
    if with_raw_csv:
        # download_training_volume_labels writes the TRAINING suffix (not the
        # expert one) — discover_extra_drops must read it for species.
        _touch(
            drop_dir / "annotations" / f"{DROP}_biigle_training_raw.csv",
            f"filename,label_name\n{POS_STEM}.jpg,fish\n",
        )
    return drop_dir


@pytest.fixture
def stub_class_map(tmp_path, monkeypatch):
    """Point config.class_map_path at a minimal map with a 'fish' bucket —
    the real one is gitignored, so discover_extra_drops would otherwise skip
    every drop in CI."""
    cm = tmp_path / "class_map.json"
    cm.write_text(
        json.dumps(
            {
                "0": {"class_id": 0, "scientific_name": "fish", "common_name": ""},
                "1": {
                    "class_id": 1,
                    "scientific_name": "Pagrus auratus",
                    "common_name": "Snapper",
                },
            }
        )
    )
    monkeypatch.setattr(type(config), "class_map_path", property(lambda self: cm))
    return cm


class TestBuildImageIndex:
    def test_indexes_training_frames(self, tmp_path):
        _make_training_drop(tmp_path)
        index = _build_image_index(tmp_path)
        assert set(index) == {DROP}
        assert set(index[DROP]) == {POS_STEM, NEG_STEM}

    def test_indexes_both_frames_and_training_frames(self, tmp_path):
        _make_training_drop(tmp_path)
        ml_stem = f"{DROP}__frame_9.000s"
        _touch(tmp_path / SURVEY / DROP / "frames" / f"{ml_stem}.jpg")
        index = _build_image_index(tmp_path)
        assert {POS_STEM, NEG_STEM, ml_stem} <= set(index[DROP])

    def test_excludes_derivative_dirs(self, tmp_path):
        drop_dir = tmp_path / SURVEY / DROP
        _touch(drop_dir / "qa_frames" / f"{DROP}__frame_3.000s.jpg")
        _touch(drop_dir / "zooniverse_frames" / f"{DROP}__frame_4.000s.jpg")
        assert _build_image_index(tmp_path) == {}


class TestDiscoverExtraDrops:
    def test_discovers_training_only_drop(self, tmp_path, stub_class_map):
        _make_training_drop(tmp_path)
        extras, _ = discover_extra_drops(tmp_path)
        assert DROP in extras

    def test_skips_when_no_image_source_dir(self, tmp_path, stub_class_map):
        # Labels + raw CSV present, but neither frames/ nor training_frames/.
        drop_dir = tmp_path / SURVEY / DROP
        _touch(drop_dir / "labels" / f"{POS_STEM}.txt", "0 0.5 0.5 0.2 0.2\n")
        _touch(
            drop_dir / "annotations" / f"{DROP}_biigle_expert_raw.csv",
            f"filename,label_name\n{POS_STEM}.jpg,fish\n",
        )
        extras, _ = discover_extra_drops(tmp_path)
        assert DROP not in extras


class TestCopySplitFilesNegatives:
    def test_empty_negative_paired_with_training_frame(self, tmp_path):
        _make_training_drop(tmp_path)
        # copy_split_files reads staged labels from labels_dir/<drop_id>/.
        staged = tmp_path / "staged_labels"
        (staged / DROP).mkdir(parents=True)
        (staged / DROP / f"{POS_STEM}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (staged / DROP / f"{NEG_STEM}.txt").write_text("")  # negative

        out = tmp_path / "dataset"
        n_images, n_labels = copy_split_files(
            [DROP],
            images_dir=tmp_path,
            labels_dir=staged,
            output_dir=out,
            split_name="train",
        )

        assert (n_images, n_labels) == (2, 2)
        # The empty negative and its training_frames image both made it through.
        assert (out / "labels" / "train" / f"{NEG_STEM}.txt").read_text() == ""
        assert (out / "images" / "train" / f"{NEG_STEM}.jpg").exists()


class TestSampleBackgroundFrames:
    def _bg(self, n):
        return [f"bg_{i}" for i in range(n)]

    def test_disabled_at_zero_ratio(self):
        out = _sample_background_frames(
            {"d": self._bg(50)}, {"d"}, 90, 0.0, random.Random(0)
        )
        assert out == {}

    def test_target_is_r_over_1_minus_r_times_positives(self):
        # r=0.1, P=90 → B = 0.1/0.9*90 = 10. Plenty available → admit exactly 10.
        out = _sample_background_frames(
            {"d": self._bg(50)}, {"d"}, 90, 0.1, random.Random(0)
        )
        admitted = sum(len(v) for v in out.values())
        assert admitted == 10
        # And the result is 10% of the resulting train set: 10 / (90 + 10).
        assert admitted / (90 + admitted) == pytest.approx(0.1)

    def test_admits_all_when_fewer_than_target(self):
        out = _sample_background_frames(
            {"d": self._bg(3)}, {"d"}, 90, 0.1, random.Random(0)
        )
        assert sum(len(v) for v in out.values()) == 3  # only 3 available

    def test_only_train_drops_contribute(self):
        # Backgrounds live on a non-train drop → ignored.
        out = _sample_background_frames(
            {"other": self._bg(50)}, {"d"}, 90, 0.1, random.Random(0)
        )
        assert out == {}

    def test_pools_across_train_drops(self):
        out = _sample_background_frames(
            {"a": self._bg(5), "b": self._bg(5)},
            {"a", "b"},
            90,
            0.1,
            random.Random(0),
        )
        assert sum(len(v) for v in out.values()) == 10  # pooled a+b

    def test_deterministic_for_fixed_seed(self):
        args = ({"d": self._bg(50)}, {"d"}, 90, 0.1)
        a = _sample_background_frames(*args, random.Random(42))
        b = _sample_background_frames(*args, random.Random(42))
        assert a == b
