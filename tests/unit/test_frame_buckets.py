"""Unit tests for the shared frame-selection buckets in select_frames."""

from __future__ import annotations

import csv as csv_module
import math

import pytest

from spyfish.extraction.select_frames import (
    collapse_peaks,
    per_frame_species_counts,
    spread_timestamps,
)

DROP_ID = "KSF_20240124_BUV_KSF_085_01"


def _raw_csv(path, rows):
    """rows: (time_seconds, class, confidence, w, h)."""
    with open(path, "w", newline="") as fh:
        w = csv_module.writer(fh)
        w.writerow(["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"])
        for t, cls, conf, bw, bh in rows:
            w.writerow([int(t * 30), t, cls, conf, 10, 10, bw, bh])
    return str(path)


class TestPerFrameSpeciesCounts:
    def test_counts_and_geometry_per_timestamp_and_class(self, tmp_path):
        p = _raw_csv(
            tmp_path / "raw.csv",
            [
                (100.0, "fish", 0.9, 30, 40),
                (100.0, "fish", 0.9, 30, 40),
                (200.0, "Parapercis colias", 0.9, 60, 80),
            ],
        )
        out = per_frame_species_counts(p, 0.25)
        fish = out[out["class"] == "fish"].iloc[0]
        assert fish["count"] == 2
        assert round(fish["diag"]) == 50  # 3-4-5 triangle
        assert round(fish["elong"], 2) == round(40 / 30, 2)

    def test_elongation_is_orientation_symmetric(self, tmp_path):
        # The same animal swimming two ways must score the same.
        p = _raw_csv(
            tmp_path / "raw.csv",
            [(100.0, "fish", 0.9, 20, 80), (200.0, "fish", 0.9, 80, 20)],
        )
        out = per_frame_species_counts(p, 0.25)
        assert out["elong"].nunique() == 1

    def test_confidence_floor_and_empty_inputs(self, tmp_path):
        p = _raw_csv(tmp_path / "raw.csv", [(100.0, "fish", 0.1, 10, 10)])
        assert per_frame_species_counts(p, 0.25).empty
        assert per_frame_species_counts(str(tmp_path / "missing.csv"), 0.25).empty


class TestCollapsePeaks:
    def test_merges_when_the_shared_frame_keeps_both_counts(self, tmp_path):
        # Both species are at full count at t=100, so one frame serves both.
        p = _raw_csv(
            tmp_path / "raw.csv",
            [
                (100.0, "fish", 0.9, 10, 10),
                (100.0, "Pagrus auratus", 0.9, 10, 10),
                (103.0, "Pagrus auratus", 0.9, 10, 10),
            ],
        )
        counts = per_frame_species_counts(p, 0.25)
        peaks = [(100.0, "fish", 1), (103.0, "Pagrus auratus", 1)]
        out = collapse_peaks(peaks, counts, spacing=10.0)
        assert len(out) == 1
        assert sorted(sp for sp, _ in out[0][1]) == ["Pagrus auratus", "fish"]

    def test_keeps_both_when_neither_frame_preserves_the_other(self, tmp_path):
        # B peaks at 2 at t=103 but is absent at t=100 — merging would undercount.
        p = _raw_csv(
            tmp_path / "raw.csv",
            [
                (100.0, "fish", 0.9, 10, 10),
                (100.0, "fish", 0.9, 10, 10),
                (103.0, "Pagrus auratus", 0.9, 10, 10),
                (103.0, "Pagrus auratus", 0.9, 10, 10),
            ],
        )
        counts = per_frame_species_counts(p, 0.25)
        peaks = [(100.0, "fish", 2), (103.0, "Pagrus auratus", 2)]
        out = collapse_peaks(peaks, counts, spacing=10.0)
        assert len(out) == 2

    def test_symmetric_the_richer_frame_absorbs_the_other(self, tmp_path):
        # t=103 holds BOTH at full count; t=100 holds only fish. Keep t=103.
        p = _raw_csv(
            tmp_path / "raw.csv",
            [
                (100.0, "fish", 0.9, 10, 10),
                (103.0, "fish", 0.9, 10, 10),
                (103.0, "Pagrus auratus", 0.9, 10, 10),
            ],
        )
        counts = per_frame_species_counts(p, 0.25)
        peaks = [(100.0, "fish", 1), (103.0, "Pagrus auratus", 1)]
        out = collapse_peaks(peaks, counts, spacing=10.0)
        assert len(out) == 1
        assert out[0][0] == 103.0

    def test_far_apart_peaks_are_never_merged(self, tmp_path):
        p = _raw_csv(
            tmp_path / "raw.csv",
            [(100.0, "fish", 0.9, 10, 10), (500.0, "fish", 0.9, 10, 10)],
        )
        counts = per_frame_species_counts(p, 0.25)
        peaks = [(100.0, "fish", 1), (500.0, "fish", 1)]
        assert len(collapse_peaks(peaks, counts, spacing=30.0)) == 2

    def test_empty_input(self):
        import pandas as pd

        assert collapse_peaks([], pd.DataFrame(), 10.0) == []


class TestSpreadTimestamps:
    def test_basic_shape_n10_in_30min_window(self):
        ts = spread_timestamps(start=60.0, end=1800.0, n=10)
        assert len(ts) == 10
        # Strictly increasing
        assert all(b > a for a, b in zip(ts, ts[1:]))
        # Last is exactly end
        assert math.isclose(ts[-1], 1800.0, rel_tol=0, abs_tol=1e-9)
        # Every value in window
        assert all(60.0 < t <= 1800.0 for t in ts)

    def test_back_loaded_more_density_in_second_half(self):
        # Back-loading goal: at least 60% of timestamps fall in the second
        # half of the window. With power=2 and N=10 it should be 5/10.
        ts = spread_timestamps(start=60.0, end=1800.0, n=10, power=2.0)
        midpoint = (60.0 + 1800.0) / 2
        in_second_half = sum(1 for t in ts if t > midpoint)
        assert (
            in_second_half >= 5
        ), f"Expected ≥5/10 in second half (back-loaded), got {in_second_half}: {ts}"

    def test_higher_power_more_aggressive_back_loading(self):
        # power=3 should put more frames late than power=1.5
        midpoint = 0.5
        mild = spread_timestamps(0.0, 1.0, n=20, power=1.5)
        heavy = spread_timestamps(0.0, 1.0, n=20, power=3.0)
        mild_late = sum(1 for t in mild if t > midpoint)
        heavy_late = sum(1 for t in heavy if t > midpoint)
        assert (
            heavy_late >= mild_late
        ), f"Higher power should be ≥ as back-loaded (heavy={heavy_late}, mild={mild_late})"

    def test_n1_returns_endpoint(self):
        ts = spread_timestamps(0.0, 100.0, n=1)
        assert ts == [100.0]

    def test_doubling_n_subset_property(self):
        # For power=2, the formula is t_i = start + span * (1 - ((N-i)/N)^2).
        # Doubling N produces a superset where every old t_i appears as the
        # NEW t_{2i}. This makes "re-run with larger N" only add new frames.
        ts10 = spread_timestamps(0.0, 1000.0, n=10, power=2.0)
        ts20 = spread_timestamps(0.0, 1000.0, n=20, power=2.0)
        for i, old_t in enumerate(ts10, start=1):
            assert math.isclose(
                ts20[2 * i - 1], old_t, rel_tol=0, abs_tol=1e-9
            ), f"ts20[{2*i-1}]={ts20[2*i-1]} should equal ts10[{i-1}]={old_t}"

    def test_invalid_n_zero(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            spread_timestamps(0.0, 100.0, n=0)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="must be > start"):
            spread_timestamps(100.0, 100.0, n=5)
        with pytest.raises(ValueError, match="must be > start"):
            spread_timestamps(100.0, 50.0, n=5)

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="power must be > 0"):
            spread_timestamps(0.0, 100.0, n=5, power=0.0)
        with pytest.raises(ValueError, match="power must be > 0"):
            spread_timestamps(0.0, 100.0, n=5, power=-1.0)
