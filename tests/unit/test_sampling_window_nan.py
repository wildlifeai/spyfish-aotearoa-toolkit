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


def test_a_few_seconds_short_is_an_issue_not_an_error():
    """SLI_20220228_BUV_SLI_093_01 was blocked for being 8 s short of 30 min."""
    expected = config.buv_video_duration_seconds
    errors, issues = config.check_sampling_window(DROP, 60.0, 60.0 + expected - 8.0)
    assert errors == []
    assert len(issues) == 1 and "shorter than" in issues[0]


def test_far_short_is_still_an_error():
    expected = config.buv_video_duration_seconds
    tol = config.sampling_window_issue_seconds
    errors, issues = config.check_sampling_window(
        DROP, 60.0, 60.0 + expected - tol - 1.0
    )
    assert len(errors) == 1 and "shorter than" in errors[0]
    assert issues == []


def test_a_long_window_is_normal():
    """The pipeline analyses sampling_start..sampling_end, so a long window is neither error nor issue."""
    expected = config.buv_video_duration_seconds
    assert config.check_sampling_window(DROP, 60.0, 60.0 + expected + 5000) == ([], [])


def test_validate_returns_only_the_blocking_errors():
    expected = config.buv_video_duration_seconds
    assert config.validate_sampling_window(DROP, 60.0, 60.0 + expected - 8.0) == []


def test_a_late_start_with_a_short_window_is_caught():
    """End past 1800 s used to pass however late the start: the window is what counts."""
    expected = config.buv_video_duration_seconds
    tol = config.sampling_window_issue_seconds
    errors, _ = config.check_sampling_window(
        DROP, 300.0, expected + 100.0
    )  # window 1600 s
    assert len(errors) == 1 and "shorter than" in errors[0]
    assert tol < 200
