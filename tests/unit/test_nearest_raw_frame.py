"""``nearest_raw_frame``: the one lookup from a video second to the raw-CSV frame.

Shared by the COCO builder, the selections frame extractor, the ML QA frame
picker and the legacy review frames, so the four agree on which frame a
moment maps to.
"""

import numpy as np
import pandas as pd

from spyfish.extraction.extract_frames import nearest_raw_frame

DROP_ID = "KSF_20240124_BUV_KSF_085_01"
RAW_COLUMNS = ["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"]


def _raw(*rows):
    """Raw inference CSV rows as (frame, time_seconds); the box columns are filler."""
    return pd.DataFrame(
        [[f, t, "fish", 0.9, 10, 10, 5, 5] for f, t in rows], columns=RAW_COLUMNS
    )


def test_without_tolerance_returns_the_nearest_however_far():
    raw = _raw((0, 0.0), (90, 3.0), (3600, 120.0))
    assert nearest_raw_frame(raw, 2.6) == (90, 3.0)
    # No tolerance: a moment 1580 s from any detection still snaps to one.
    assert nearest_raw_frame(raw, 1700.0) == (3600, 120.0)


def test_within_tolerance_returns_frame_and_its_time():
    raw = _raw((6, 2.0), (12, 4.0))
    # The RAW frame's second comes back (2.0), not the asked-for 2.1, so the
    # caller names the clean frame after the pixels it actually extracted.
    assert nearest_raw_frame(raw, 2.1, max_delta=0.5) == (6, 2.0)


def test_beyond_tolerance_returns_nones():
    raw = _raw((0, 0.0))
    assert nearest_raw_frame(raw, 4.0, max_delta=0.5) == (None, None)
    # The boundary itself is inside the tolerance.
    assert nearest_raw_frame(raw, 0.5, max_delta=0.5) == (0, 0.0)


def test_empty_frame_returns_nones():
    assert nearest_raw_frame(pd.DataFrame(), 2.0) == (None, None)
    assert nearest_raw_frame(pd.DataFrame(columns=RAW_COLUMNS), 2.0, 1.0) == (
        None,
        None,
    )


def test_all_nan_times_return_nones_rather_than_raising():
    raw = _raw((0, np.nan), (30, np.nan))
    assert nearest_raw_frame(raw, 2.0) == (None, None)
    assert nearest_raw_frame(raw, 2.0, max_delta=1.0) == (None, None)


def test_nan_times_are_skipped_not_matched():
    raw = _raw((0, np.nan), (30, 1.0), (60, 2.0))
    assert nearest_raw_frame(raw, 1.9) == (60, 2.0)


def test_lookup_is_positional_so_a_repeated_index_is_fine():
    # Two drops' raw CSVs concatenated without ignore_index, as a merged
    # volume might be: labels repeat, the answer must not.
    raw = pd.concat([_raw((0, 0.0), (30, 1.0)), _raw((0, 0.0), (45, 1.5))])
    assert raw.index.has_duplicates
    assert nearest_raw_frame(raw, 1.4) == (45, 1.5)
