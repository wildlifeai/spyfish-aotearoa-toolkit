"""A blank SamplingStart/SamplingEnd arrives as NaN, not None.

float("nan") does not raise, every NaN comparison is False, and bool(nan) is
True - so a NaN slipped past the ingest parse guard, past all three window
rules, and died on int(nan) inside ML hours later. 1046 deployments were
ingest_status=ok with no sampling window this way (2026-08-23).
"""

import math

import pytest

from spyfish.config.wrapper import config

DROP = "KSF_20240124_BUV_KSF_085_01"


def test_nan_window_is_rejected():
    errors = config.validate_sampling_window(DROP, float("nan"), float("nan"))
    assert errors, "NaN window must be an error, not silently valid"
    assert "missing" in errors[0]


def test_none_window_is_rejected():
    assert config.validate_sampling_window(DROP, None, None)


def test_nan_end_alone_is_rejected():
    assert config.validate_sampling_window(DROP, 60.0, float("nan"))


def test_valid_window_still_passes():
    expected = config.buv_video_duration_seconds
    assert config.validate_sampling_window(DROP, 60.0, 60.0 + expected) == []


def test_the_nan_semantics_that_caused_this():
    """Documents why each guard failed, so nobody 'simplifies' them back."""
    nan = float("nan")
    assert math.isnan(float(nan))  # float() does not raise on NaN
    assert not (nan == 0)  # so sampling_start == 0 never fires
    assert not (nan < 1800)  # so the short-window rule never fires
    assert bool(nan)  # so `if sampling_end` passes
    with pytest.raises(ValueError):
        int(nan)  # and this is where it finally blows up
