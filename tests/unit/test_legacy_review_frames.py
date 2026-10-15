"""Legacy expert moments become pipeline frames after --ml.

Clean frame in frames/ (written once, model-independent), boxed copy in
qa_frames/legacy/<model>/ (one folder per model, so re-runs sit side by side).
"""

import cv2
import numpy as np
import pandas as pd
import pytest

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.config.wrapper import config
from spyfish.ml.legacy_review_frames import (
    KIND_INTERVAL,
    KIND_PEAK,
    legacy_review_moments,
    write_legacy_review_frames,
)

DROP = "SLI_20240124_BUV_SLI_069_01"


def _rows(*triples):
    return pd.DataFrame(
        triples, columns=["scientific_name", "max_interval", "time_of_max_seconds"]
    )


@pytest.fixture(autouse=True)
def _plain_labels(monkeypatch):
    """The pipeline_env config carries no class map, so the common-name
    lookup cannot be built here; names pass through unchanged. The labeller
    itself is covered in test_draw_frames.py."""
    monkeypatch.setattr(
        "spyfish.ml.legacy_review_frames.common_name_labeller", lambda: str
    )


def test_peak_is_highest_count_earliest_tie():
    rows = _rows(
        ("Pagrus auratus", 4, 1727.0),
        ("Pagrus auratus", 6, 1988.0),
        ("Pagrus auratus", 6, 1898.0),
        ("Parapercis colias", 4, 676.0),
    )
    peaks = legacy_review_moments(rows, n_random=0, seed=DROP)
    assert peaks["kind"].eq(KIND_PEAK).all()
    snapper = peaks[peaks.scientific_name == "Pagrus auratus"].iloc[0]
    assert (snapper.max_interval, snapper.time_of_max_seconds) == (6, 1898.0)
    assert len(peaks) == 2


def test_random_intervals_are_seeded_and_never_the_peak():
    rows = _rows(*[("Pagrus auratus", 1, 30.0 * i) for i in range(1, 20)])
    rows.loc[5, "max_interval"] = 5  # the peak at 180 s
    a = legacy_review_moments(rows, n_random=3, seed=DROP)
    b = legacy_review_moments(rows, n_random=3, seed=DROP)
    pd.testing.assert_frame_equal(a, b)
    extras = a[a.kind == KIND_INTERVAL]
    assert len(extras) == 3
    assert 180.0 not in extras.time_of_max_seconds.values
    other = legacy_review_moments(rows, n_random=3, seed="SLI_20240124_BUV_SLI_093_01")
    assert set(other[other.kind == KIND_INTERVAL].time_of_max_seconds) != set(
        extras.time_of_max_seconds
    )


def test_random_count_is_capped_by_available_rows():
    rows = _rows(("Seriola lalandi", 1, 1520.0), ("Seriola lalandi", 1, 1550.0))
    out = legacy_review_moments(rows, n_random=3, seed=DROP)
    assert (out.kind == KIND_INTERVAL).sum() == 1


class _AnnDb:
    def __init__(self, rows):
        self.rows = rows

    def get_annotations_for_drop(self, drop_id, annotated_by=None):
        return [r for r in self.rows if annotated_by in (None, r["annotated_by"])]


def _legacy(species, n, seconds, external_id="legacy"):
    return {
        "drop_id": DROP,
        "scientific_name": species,
        "max_interval": n,
        "time_of_max_seconds": seconds,
        "time_of_max": None,
        "annotated_by": "expert",
        "external_id": external_id,
    }


def _write_video(path, seconds=6, fps=3):
    """A tiny colour-ramp video: frame i is a flat colour, so a frame can be
    told apart by its pixel value."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(seconds * fps):
        w.write(np.full((48, 64, 3), (i * 10) % 255, dtype=np.uint8))
    w.release()


def test_writes_clean_frame_once_and_qa_copy_per_model(pipeline_env):
    video = pipeline_env.tmp_path / "media" / f"{DROP}.mp4"
    _write_video(video)
    raw = pd.DataFrame(
        [
            # frame 6 is t=2.0 s, one snapper box
            {
                "frame": 6,
                "time_seconds": 2.0,
                "class": "Pagrus auratus",
                "confidence": 0.9,
                "x": 32,
                "y": 24,
                "w": 20,
                "h": 16,
            },
            {
                "frame": 12,
                "time_seconds": 4.0,
                "class": "fish",
                "confidence": 0.3,
                "x": 10,
                "y": 10,
                "w": 8,
                "h": 8,
            },
        ]
    )
    ann = _AnnDb(
        [
            _legacy("Pagrus auratus", 3, 2.1),
            _legacy(NULL_DEPLOYMENT, 0, None),  # absence row, never a frame
            _legacy(
                "Pagrus auratus", 9, 2.5, external_id="52605406"
            ),  # BIIGLE, not legacy
        ]
    )

    first = write_legacy_review_frames(DROP, video, raw, "model_a", ann, base_conf=0.25)
    assert len(first) == 1
    qa = first[0]
    assert qa.parent == config.get_drop_dir(DROP) / "qa_frames" / "legacy" / "model_a"
    assert "Pagrus_auratus__n3__legacy_maxn" in qa.name
    # Clean frame carries the RAW frame's second (2.0), the nearest within 0.5 s
    # of the legacy 2.1, so the boxes match the pixels.
    clean = config.get_frames_dir(DROP) / f"{DROP}__frame_0002.000s.jpg"
    assert clean.exists()
    clean_mtime = clean.stat().st_mtime

    second = write_legacy_review_frames(
        DROP, video, raw, "model_b", ann, base_conf=0.25
    )
    assert second[0].parent.name == "model_b"
    assert clean.stat().st_mtime == clean_mtime  # not re-extracted
    assert qa.exists()  # model_a's copy untouched


def test_no_legacy_rows_means_no_video_opened(pipeline_env, tmp_path):
    missing = tmp_path / "nope.mp4"
    out = write_legacy_review_frames(
        DROP, missing, pd.DataFrame(), "m", _AnnDb([]), 0.25
    )
    assert out == []


def test_moment_far_from_any_detection_is_seeked_by_time(pipeline_env):
    video = pipeline_env.tmp_path / "media" / f"{DROP}.mp4"
    _write_video(video)
    raw = pd.DataFrame(
        [
            {
                "frame": 0,
                "time_seconds": 0.0,
                "class": "fish",
                "confidence": 0.9,
                "x": 5,
                "y": 5,
                "w": 4,
                "h": 4,
            }
        ]
    )
    ann = _AnnDb([_legacy("Parapercis colias", 2, 4.0)])
    out = write_legacy_review_frames(DROP, video, raw, "m", ann, base_conf=0.25)
    assert len(out) == 1
    assert (config.get_frames_dir(DROP) / f"{DROP}__frame_0004.000s.jpg").exists()
