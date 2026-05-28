"""
Tests for spyfish/utils.py — time conversion, filename generation.
"""

from spyfish.utils import (
    generate_clip_filename,
    generate_frame_filename,
    seconds_to_time,
    time_to_seconds,
)

# ── seconds_to_time ──────────────────────────────────────────────────────────


def test_seconds_to_time_zero():
    assert seconds_to_time(0) == "00:00:00.000"


def test_seconds_to_time_simple():
    assert seconds_to_time(65.5) == "00:01:05.500"


def test_seconds_to_time_hours():
    assert seconds_to_time(3661.123) == "01:01:01.123"


def test_seconds_to_time_millisecond_rollover():
    """0.9999 seconds should round to 1.000, not produce ms=1000."""
    result = seconds_to_time(59.9999)
    assert result == "00:01:00.000"  # rolls up to next second → next minute


def test_seconds_to_time_nan():
    assert seconds_to_time(float("nan")) == "00:00:00.000"


def test_seconds_to_time_large():
    assert seconds_to_time(86400) == "24:00:00.000"


# ── time_to_seconds ──────────────────────────────────────────────────────────


def test_time_to_seconds_hhmmss():
    assert time_to_seconds("01:02:03.500") == 3723.5


def test_time_to_seconds_mmss():
    assert time_to_seconds("02:30") == 150.0


def test_time_to_seconds_raw_seconds():
    assert time_to_seconds("45.5") == 45.5


def test_time_to_seconds_nan():
    assert time_to_seconds(float("nan")) == 0.0


def test_time_to_seconds_roundtrip():
    """seconds_to_time → time_to_seconds should be identity (within ms precision)."""
    original = 1234.567
    converted = time_to_seconds(seconds_to_time(original))
    assert abs(converted - original) < 0.001


# ── generate_clip_filename ───────────────────────────────────────────────────


def test_generate_clip_filename():
    result = generate_clip_filename("KSF_20240124_BUV_KSF_085_01", 10.0, 445.0)
    assert result == "KSF_20240124_BUV_KSF_085_01__clip_10s_00445s.mp4"


def test_generate_clip_filename_short_duration():
    result = generate_clip_filename("DROP_01", 5.0, 60.0)
    assert result == "DROP_01__clip_05s_00060s.mp4"


# ── generate_frame_filename ──────────────────────────────────────────────────


def test_generate_frame_filename():
    # Integer seconds are zero-padded to width 4 so lex sort matches numeric order.
    result = generate_frame_filename("KSF_20240124_BUV_KSF_085_01", 125.5)
    assert result == "KSF_20240124_BUV_KSF_085_01__frame_0125.500s.jpg"


def test_generate_frame_filename_integer_seconds():
    result = generate_frame_filename("DROP_01", 60.0)
    assert result == "DROP_01__frame_0060.000s.jpg"


def test_generate_frame_filename_lex_matches_numeric_order():
    a = generate_frame_filename("D", 100.077)
    b = generate_frame_filename("D", 1011.736)
    assert a < b  # would fail without zero-padding
