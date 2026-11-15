"""The MaxN timeline renders one row per species plus review tracks, from synthetic data."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spyfish.config.wrapper import config
from spyfish.visualisations import maxn_visualisation as mv
from spyfish.visualisations.maxn_visualisation import _runs, plot_maxn_timeline

DROP = "KSF_20240124_BUV_KSF_085_01"
FPS = 3.0


def _raw(visits):
    """Detections on a 3 fps grid: visits = [(class, start_s, end_s, count, conf)]."""
    rows = []
    for cls, a, b, n, conf in visits:
        for t in np.arange(a, b, 1 / FPS):
            for _ in range(n):
                rows.append(
                    {
                        "frame": int(t * 60),
                        "time_seconds": t,
                        "class": cls,
                        "confidence": conf,
                        "x": 0.5,
                        "y": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def _maxn(rows):
    tcol, ccol = config.csv_maxn_time_seconds_column, config.csv_max_interval_column
    scol, acol = (
        config.csv_scientific_name_column,
        config.csv_confidence_agreement_column,
    )
    return pd.DataFrame(
        [{tcol: t, ccol: n, scol: sp, acol: conf} for sp, t, n, conf in rows]
    )


@pytest.fixture
def data():
    raw = _raw(
        [
            ("Parapercis colias", 100, 110, 1, 0.8),  # held 10 s, survives persistence
            ("Parapercis colias", 200, 203, 3, 0.7),  # 3 fish for 3 s, the MaxN
            ("Notolabrus celidotus", 150, 150.4, 1, 0.6),  # one frame: filtered out
            (
                "Chrysophrys auratus",
                400,
                405,
                1,
                0.3,
            ),  # below maxn_conf, context row only
        ]
    )
    maxn = _maxn(
        [
            ("Parapercis colias", 201.0, 3, 0.70),
            ("Parapercis colias", 105.0, 1, 0.80),
            ("Notolabrus celidotus", 150.0, 1, 0.60),
        ]
    )
    return raw, maxn


def test_runs_merge_touching_intervals():
    assert _runs([(0, 10), (10, 20), (30, 40), (35, 50)]) == [[0, 20], [30, 50]]


def test_renders_png_with_one_row_per_species(tmp_path, data, monkeypatch):
    raw, maxn = data
    captured = {}
    real = mv.plt.subplots

    def spy(n, *a, **kw):
        captured["n_rows"] = n
        return real(n, *a, **kw)

    monkeypatch.setattr(mv.plt, "subplots", spy)
    out = plot_maxn_timeline(
        raw, maxn, DROP, tmp_path, 0.25, 0.4, 10, sampling_start=52, sampling_end=500
    )
    assert out == tmp_path / f"{DROP}_maxn_timeline.png"
    w, h = Image.open(out).size
    assert w == mv.FIG_W_IN * 150
    # context + 2 species rows (snapper is below maxn_conf) + the clip track
    assert captured["n_rows"] == 1 + 2 + 1


def test_frame_tracks_only_when_given(tmp_path, data, monkeypatch):
    raw, maxn = data
    captured = {}
    real = mv.plt.subplots
    monkeypatch.setattr(
        mv.plt,
        "subplots",
        lambda n, *a, **kw: captured.update(n=n) or real(n, *a, **kw),
    )
    frames = pd.DataFrame(
        {
            config.csv_maxn_time_seconds_column: [201.0, 300.0],
            config.csv_scientific_name_column: ["Parapercis colias", "fish"],
            config.selection_reason_column: ["MaxN (Parapercis colias)", "Spot check"],
        }
    )
    plot_maxn_timeline(
        raw, maxn, DROP, tmp_path, 0.25, 0.4, 10, frames_df=frames, zoo_frames_df=frames
    )
    assert captured["n"] == 1 + 2 + 3  # clips + BIIGLE frames + Zooniverse frames


def test_persistent_counts_drop_single_frame_visits(data):
    raw, _ = data
    counts = mv._Counts(raw, 52, 500)
    spotty = raw[raw["class"] == "Notolabrus celidotus"]
    assert counts.raw(spotty).max() == 1
    assert counts.persistent(spotty).max() == 0
    cod = raw[raw["class"] == "Parapercis colias"]
    assert counts.persistent(cod).max() == 3


def test_no_detections_returns_none(tmp_path, data):
    raw, maxn = data
    assert (
        plot_maxn_timeline(
            raw,
            maxn,
            DROP,
            tmp_path,
            base_conf=0.99,
            maxn_conf=0.99,
            interval_seconds=10,
        )
        is None
    )
