"""spyfish/ml/static_boxes.flag_static: which static tracks --ml-post relabels."""

import numpy as np
import pandas as pd

from spyfish.config.wrapper import config
from spyfish.ml.static_boxes import (
    STATIC_CLASS_COLUMN,
    STATIC_TRACK_COLUMN,
    flag_static,
)

DROP = "AHE_20250513_BUV_AHE_119_01"


def _rows(times, x, y, w, h, conf, cls):
    return [
        {
            "frame": int(t * 3),
            "time_seconds": float(t),
            "class": cls,
            "confidence": conf,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }
        for t in times
    ]


def _drop():
    t = np.arange(0, 1800, 1 / 3)  # 30 min at 3 fps
    canister = _rows(
        t[::2], 900, 600, 400, 160, 0.33, "fish"
    )  # every other frame, weak
    resting_cod = _rows(
        t[1200:4800:60], 300, 300, 120, 60, 0.70, "Parapercis colias"
    )  # 20 min, sparse, sure
    return pd.DataFrame(canister + resting_cod)


def test_canister_is_flagged_resting_fish_is_not():
    out, flagged = flag_static(_drop(), DROP)
    assert len(flagged) == 1 and (flagged["fish_score"] < 0).all()
    label = config.static_box_label
    assert set(out.loc[out["x"] == 900, "class"]) == {label}
    assert set(out.loc[out["x"] == 900, STATIC_CLASS_COLUMN]) == {"fish"}
    assert set(out.loc[out["x"] == 300, "class"]) == {"Parapercis colias"}
    assert out.loc[out["x"] == 300, STATIC_TRACK_COLUMN].isna().all()


def test_rerun_is_idempotent():
    once, _ = flag_static(_drop(), DROP)
    twice, _ = flag_static(once, DROP)
    pd.testing.assert_frame_equal(once, twice)


def test_static_label_never_counts_toward_maxn():
    assert config.static_box_label in config.maxn_exclude_classes
