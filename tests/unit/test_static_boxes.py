"""scripts/two_stage/static_boxes.py: what counts as a box that never moves."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "two_stage"))

from static_boxes import static_tracks  # noqa: E402

DROP = "HOR_20251204_BUV_HOR_033_01"


def _rows(times, x, y, w=80, h=40, sway=0.0, cls="Parapercis colias", seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "frame": int(t * 30),
            "time_seconds": float(t),
            "class": cls,
            "confidence": 0.6,
            "x": x + rng.normal(0, sway),
            "y": y + rng.normal(0, sway),
            "w": w,
            "h": h,
        }
        for t in times
    ]


def test_stick_with_camera_sway_is_static_resting_fish_is_not():
    t = np.arange(0, 1800, 1.0)  # 30 min sampled at 1 s
    rows = (
        # a stick for 24 min, rocking ±6 px with the swell, missed now and then
        _rows([s for s in t[60:1500] if s % 7], 400, 300, sway=6)
        # a blue cod resting 90 s, then gone
        + _rows(t[100:190], 900, 500, seed=1)
        # a fish crossing the frame
        + [
            r | {"x": 100 + 5 * i}
            for i, r in enumerate(_rows(t[300:400], 0, 200, seed=2))
        ]
    )
    tracks, crops = static_tracks(pd.DataFrame(rows), DROP)
    assert len(tracks) == 1
    assert tracks.iloc[0]["span_min"] >= 23.5
    assert len(crops) == 5 and crops["track"].nunique() == 1


def test_stick_seen_on_one_frame_in_fifty_is_still_static():
    """HOR_065: YOLO saw the stick on ~1% of frames for 27 min. Continuity is
    not the signal, same pixels and same size are."""
    t = np.arange(0, 1800, 1 / 3)
    tracks, _ = static_tracks(
        pd.DataFrame(_rows(t[::50], 580, 280, w=160, h=80, sway=4)), DROP
    )
    assert len(tracks) == 1


def test_fish_coming_back_to_the_bait_is_not_static():
    """Different fish at the same spot: positions scatter tens of pixels and
    sizes differ, so neither spread test passes."""
    rng = np.random.default_rng(3)
    t = np.arange(0, 1800, 1.0)
    rows = [
        r | {"w": 80 * s, "h": 40 * s}
        for r, s in zip(
            _rows(t[::3], 800, 400, sway=45, seed=4), rng.uniform(0.6, 1.6, 600)
        )
    ]
    tracks, _ = static_tracks(pd.DataFrame(rows), DROP)
    assert tracks.empty


def test_crops_skip_a_fish_lying_across_the_spot():
    """The static object always has the same box; a fish passing through gets
    a different one and must not become a `not a fish` crop."""
    t = np.arange(0, 1800, 1.0)
    rows = _rows(t[::10], 500, 300, w=200, h=80, sway=2)
    fish = [
        r | {"w": 90, "h": 220, "class": "fish"}
        for r in _rows(t[905:915], 500, 300, seed=5)
    ]
    tracks, crops = static_tracks(pd.DataFrame(rows + fish), DROP)
    assert len(tracks) == 1
    assert (crops["w"] > 150).all()  # only the wide static box, never the tall fish
