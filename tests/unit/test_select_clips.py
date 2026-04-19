"""
Tests for extraction/select_clips.py — ClipSelector and strategy selection.

Covers:
  - ClipSelector: add_interval dedup, check_temporal_spacing, finalize_df
  - select_clips_with_strategy: binary strategy basics, clip_cap, empty input
"""

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.extraction.select_clips import ClipSelector, select_clips_with_strategy

DROP_ID = "KSF_20240124_BUV_KSF_085_01"
SAMPLING_START = 120.0
CLIP_LENGTH = 10


def _make_detection_df(
    times_and_counts: list[tuple[float, int, float]],
) -> pd.DataFrame:
    """Build a DataFrame matching what MaxN CSV looks like after post-processing.

    Each tuple is (time_seconds, max_interval, confidence).
    """
    return pd.DataFrame(
        [
            {
                config.csv_time_seconds_column: t,
                config.csv_max_interval_column: count,
                config.csv_confidence_agreement_column: conf,
                config.csv_scientific_name_column: "Pagrus auratus",
            }
            for t, count, conf in times_and_counts
        ]
    )


# ── ClipSelector ─────────────────────────────────────────────────────────────


def test_clip_selector_add_interval():
    sel = ClipSelector(DROP_ID, SAMPLING_START, CLIP_LENGTH)
    row = {
        config.csv_time_seconds_column: 130.0,
        config.csv_max_interval_column: 5,
        config.csv_confidence_agreement_column: 0.9,
        config.csv_scientific_name_column: "Pagrus auratus",
    }
    assert sel.add_interval(row, "MaxN") is True
    assert len(sel.selections_rows) == 1


def test_clip_selector_deduplicates_same_bucket():
    sel = ClipSelector(DROP_ID, SAMPLING_START, CLIP_LENGTH)
    row1 = {
        config.csv_time_seconds_column: 131.0,  # bucket: 130
        config.csv_max_interval_column: 5,
        config.csv_confidence_agreement_column: 0.9,
        config.csv_scientific_name_column: "Pagrus auratus",
    }
    row2 = {
        config.csv_time_seconds_column: 135.0,  # same bucket: 130
        config.csv_max_interval_column: 3,
        config.csv_confidence_agreement_column: 0.8,
        config.csv_scientific_name_column: "Pagrus auratus",
    }
    assert sel.add_interval(row1, "MaxN") is True
    assert sel.add_interval(row2, "MaxN") is False  # same 10s bucket
    assert len(sel.selections_rows) == 1


def test_clip_selector_different_buckets():
    sel = ClipSelector(DROP_ID, SAMPLING_START, CLIP_LENGTH)
    row1 = {
        config.csv_time_seconds_column: 130.0,  # bucket: 130
        config.csv_max_interval_column: 5,
        config.csv_confidence_agreement_column: 0.9,
        config.csv_scientific_name_column: "Pagrus auratus",
    }
    row2 = {
        config.csv_time_seconds_column: 145.0,  # bucket: 140
        config.csv_max_interval_column: 3,
        config.csv_confidence_agreement_column: 0.8,
        config.csv_scientific_name_column: "Pagrus auratus",
    }
    assert sel.add_interval(row1, "MaxN") is True
    assert sel.add_interval(row2, "MaxN") is True
    assert len(sel.selections_rows) == 2


def test_clip_selector_temporal_spacing():
    sel = ClipSelector(DROP_ID, SAMPLING_START, CLIP_LENGTH)
    sel.add_interval(
        {
            config.csv_time_seconds_column: 130.0,
            config.csv_max_interval_column: 5,
            config.csv_confidence_agreement_column: 0.9,
            config.csv_scientific_name_column: "Pagrus auratus",
        },
        "MaxN",
    )
    # 135 is in the same 10s bucket as 130 → spacing check with 10s spacing fails
    assert sel.check_temporal_spacing(135.0, 10) is False
    # 145 is 1 bucket away → passes with 10s spacing
    assert sel.check_temporal_spacing(145.0, 10) is True
    # 0 spacing → always passes
    assert sel.check_temporal_spacing(131.0, 0) is True


def test_clip_selector_finalize_df_empty():
    sel = ClipSelector(DROP_ID, SAMPLING_START, CLIP_LENGTH)
    df = sel.finalize_df()
    assert df.empty
    assert config.csv_clip_start_absolute_column in df.columns


def test_clip_selector_finalize_df_sorted():
    sel = ClipSelector(DROP_ID, SAMPLING_START, CLIP_LENGTH)
    for t in [150.0, 130.0, 140.0]:
        sel.add_interval(
            {
                config.csv_time_seconds_column: t,
                config.csv_max_interval_column: 1,
                config.csv_confidence_agreement_column: 0.9,
                config.csv_scientific_name_column: "Fish",
            },
            "MaxN",
        )
    df = sel.finalize_df()
    starts = df[config.csv_clip_start_absolute_column].tolist()
    assert starts == sorted(starts)


# ── select_clips_with_strategy ───────────────────────────────────────────────

BINARY_PARAMS = {
    "maxn_export": 3,
    "confusing_export": 2,
    "empty_export": 1,
    "start_export": 1,
    "temporal_spacing_seconds": 10,
}


def test_select_clips_binary_basic():
    df = _make_detection_df(
        [
            (130.0, 5, 0.9),
            (145.0, 3, 0.8),
            (160.0, 1, 0.5),
            (175.0, 2, 0.7),
        ]
    )
    result = select_clips_with_strategy(
        df,
        DROP_ID,
        SAMPLING_START,
        CLIP_LENGTH,
        BINARY_PARAMS,
        is_multiclass=False,
        video_start_threshold=120,
        clip_cap=50,
    )
    assert not result.empty
    assert config.csv_clip_start_absolute_column in result.columns
    # Should have up to 3 MaxN + 2 confusing + 1 empty + 1 start (but limited by data)
    assert len(result) <= sum(BINARY_PARAMS.values())


def test_select_clips_empty_input():
    df = _make_detection_df([])
    result = select_clips_with_strategy(
        df,
        DROP_ID,
        SAMPLING_START,
        CLIP_LENGTH,
        BINARY_PARAMS,
        is_multiclass=False,
        video_start_threshold=120,
        clip_cap=50,
    )
    assert result.empty


def test_select_clips_cap_limits_output():
    """With more data than the cap, output should be capped."""
    df = _make_detection_df(
        [(float(t), i % 5 + 1, 0.8) for i, t in enumerate(range(130, 500, 15))]
    )
    result = select_clips_with_strategy(
        df,
        DROP_ID,
        SAMPLING_START,
        CLIP_LENGTH,
        BINARY_PARAMS,
        is_multiclass=False,
        video_start_threshold=120,
        clip_cap=5,
    )
    assert len(result) <= 5


def test_select_clips_stores_absolute_times():
    """Clip start/end times should be absolute (>= sampling_start)."""
    df = _make_detection_df(
        [
            (130.0, 5, 0.9),
            (145.0, 3, 0.8),
        ]
    )
    result = select_clips_with_strategy(
        df,
        DROP_ID,
        SAMPLING_START,
        CLIP_LENGTH,
        BINARY_PARAMS,
        is_multiclass=False,
        video_start_threshold=120,
        clip_cap=50,
    )
    for _, row in result.iterrows():
        start = row[config.csv_clip_start_absolute_column]
        end = row[config.csv_clip_end_absolute_column]
        assert (
            start >= SAMPLING_START
        ), f"Clip start {start} < sampling_start {SAMPLING_START}"
        assert end > start
        assert end - start == CLIP_LENGTH
