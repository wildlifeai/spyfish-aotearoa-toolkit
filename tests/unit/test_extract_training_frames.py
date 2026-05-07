"""Unit tests for spyfish.ml.training.extract_training_frames helpers."""

from __future__ import annotations

import math

import pytest

from spyfish.ml.training.extract_training_frames import _quadratic_timestamps


class TestQuadraticTimestamps:
    def test_basic_shape_n10_in_30min_window(self):
        ts = _quadratic_timestamps(start=60.0, end=1800.0, n=10)
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
        ts = _quadratic_timestamps(start=60.0, end=1800.0, n=10, power=2.0)
        midpoint = (60.0 + 1800.0) / 2
        in_second_half = sum(1 for t in ts if t > midpoint)
        assert (
            in_second_half >= 5
        ), f"Expected ≥5/10 in second half (back-loaded), got {in_second_half}: {ts}"

    def test_higher_power_more_aggressive_back_loading(self):
        # power=3 should put more frames late than power=1.5
        midpoint = 0.5
        mild = _quadratic_timestamps(0.0, 1.0, n=20, power=1.5)
        heavy = _quadratic_timestamps(0.0, 1.0, n=20, power=3.0)
        mild_late = sum(1 for t in mild if t > midpoint)
        heavy_late = sum(1 for t in heavy if t > midpoint)
        assert (
            heavy_late >= mild_late
        ), f"Higher power should be ≥ as back-loaded (heavy={heavy_late}, mild={mild_late})"

    def test_n1_returns_endpoint(self):
        ts = _quadratic_timestamps(0.0, 100.0, n=1)
        assert ts == [100.0]

    def test_doubling_n_subset_property(self):
        # For power=2, the formula is t_i = start + span * (1 - ((N-i)/N)^2).
        # Doubling N produces a superset where every old t_i appears as the
        # NEW t_{2i}. This makes "re-run with larger N" only add new frames.
        ts10 = _quadratic_timestamps(0.0, 1000.0, n=10, power=2.0)
        ts20 = _quadratic_timestamps(0.0, 1000.0, n=20, power=2.0)
        for i, old_t in enumerate(ts10, start=1):
            assert math.isclose(
                ts20[2 * i - 1], old_t, rel_tol=0, abs_tol=1e-9
            ), f"ts20[{2*i-1}]={ts20[2*i-1]} should equal ts10[{i-1}]={old_t}"

    def test_invalid_n_zero(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            _quadratic_timestamps(0.0, 100.0, n=0)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="must be > start"):
            _quadratic_timestamps(100.0, 100.0, n=5)
        with pytest.raises(ValueError, match="must be > start"):
            _quadratic_timestamps(100.0, 50.0, n=5)

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="power must be > 0"):
            _quadratic_timestamps(0.0, 100.0, n=5, power=0.0)
        with pytest.raises(ValueError, match="power must be > 0"):
            _quadratic_timestamps(0.0, 100.0, n=5, power=-1.0)
