"""MaxN aggregation against hand-checked ground truth.

`process_maxn` turns a raw per-frame detection CSV into one MaxN row per
species interval. The expected rows live in `conftest.EXPECTED_MAXN` and were
worked out by hand, so a change in interval bucketing, the peak-frame tiebreak
or the confidence filter fails here rather than silently shifting every
abundance number in the dashboard.
"""

import pandas as pd

from spyfish.ml.process_ml_annotations import process_maxn
from tests.conftest import DROP_NORMAL, MODEL_NAME


def test_process_maxn_matches_ground_truth(pipeline_env):
    """
    process_maxn() on DROP_NORMAL's raw CSV must produce the exact hardcoded
    EXPECTED_MAXN rows, including the tiebreak between frames 25 and 37.
    """
    env = pipeline_env
    raw_df = pd.read_csv(env.raw_csv_paths[DROP_NORMAL])
    output_csv = env.tmp_path / f"{DROP_NORMAL}_{MODEL_NAME}_maxn.csv"

    result = process_maxn(
        raw_df=raw_df,
        output_csv_path=str(output_csv),
        drop_id=DROP_NORMAL,
        interval_seconds=10,
        confidence_threshold=0.50,
        model_name=MODEL_NAME,
    )

    expected = env.expected_maxn[DROP_NORMAL]
    result = result.sort_values("TimeOfMaxAbsSeconds").reset_index(drop=True)

    assert len(result) == len(expected)
    assert list(result["MaxInterval"]) == list(expected["MaxInterval"])
    assert list(result["TimeOfMaxAbsSeconds"]) == list(expected["TimeOfMaxAbsSeconds"])
    assert list(result["ConfidenceAgreement"]) == list(expected["ConfidenceAgreement"])
    assert list(result["TimeOfMax"]) == list(expected["TimeOfMax"])


def test_process_maxn_respects_confidence_threshold(pipeline_env):
    """
    Raising confidence_threshold to 0.91 should exclude all but the single
    highest-confidence detection and produce MaxInterval=1 only for interval 10.
    """
    env = pipeline_env
    raw_df = pd.read_csv(env.raw_csv_paths[DROP_NORMAL])
    output_csv = env.tmp_path / "maxn_high_threshold.csv"

    result = process_maxn(
        raw_df=raw_df,
        output_csv_path=str(output_csv),
        drop_id=DROP_NORMAL,
        interval_seconds=10,
        confidence_threshold=0.91,
        model_name=MODEL_NAME,
    )

    assert len(result) == 1
    assert result.iloc[0]["MaxInterval"] == 1
    assert result.iloc[0]["TimeOfMaxAbsSeconds"] == 10.0
