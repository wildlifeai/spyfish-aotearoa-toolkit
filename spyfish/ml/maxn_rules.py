"""The MaxN persistence rule: which per-frame counts are real visits.

Pure numpy, no other imports, so light callers (the UoA dashboard, whose Cloud
deploy installs no opencv) use the exact same rule as the pipeline's MaxN
(process_ml_annotations.process_maxn) and maxn_compare.py.
"""

import numpy as np


def gap_fill(counts: np.ndarray, max_gap: int) -> np.ndarray:
    """Morphological closing on a count series: zero-runs up to `max_gap` slots
    flanked by detections on BOTH sides take min(neighbours). Never invents a
    detection — an isolated blip has zeros on both sides and stays exposed."""
    if max_gap <= 0:
        return counts
    filled = counts.copy()
    n = len(filled)
    i = 0
    while i < n:
        if filled[i] != 0:
            i += 1
            continue
        j = i
        while j < n and filled[j] == 0:
            j += 1
        if 0 < i and j < n and (j - i) <= max_gap:
            filled[i:j] = min(filled[i - 1], filled[j])
        i = j
    return filled


def centered_rolling_min(counts: np.ndarray, window: int) -> np.ndarray:
    """Rolling min over `window` slots, value assigned to the window's middle
    slot. Slots beyond the series edges count as zero, so a visit shorter than
    the window is suppressed even when it touches the first or last detection."""
    if window <= 1:
        return counts
    left = window // 2
    right = window - 1 - left
    padded = np.concatenate([np.zeros(left), counts, np.zeros(right)])
    return np.lib.stride_tricks.sliding_window_view(padded, window).min(axis=1)
