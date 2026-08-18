"""
A re-run that detects nothing must not leave the previous run's rows behind.

`clear_annotations()` has to run whether or not there is anything to write. Put
it behind an `if annotations_to_add:` guard and a zero-detection re-run skips
it entirely, so the earlier run's annotations stay in the database for good and
the deployment reads as populated when the current model finds nothing in it.

What a zero-detection run leaves behind is one null-species row, not an empty
table: "reviewed, saw nothing" has to be distinguishable from "ML never ran",
and detection-rate denominators need this deployment counted as a real zero.
"""

import pandas as pd

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.ml.process_ml_annotations import _ingest_ml_annotations, process_maxn
from tests.conftest import DROP_NORMAL, MODEL_NAME

# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_ml_annotations(ann_db, drop_id: str) -> int:
    """Return the number of ML annotation rows for drop_id in ann_db."""
    with ann_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM annotations WHERE drop_id = ? AND annotated_by = 'ml'",
            (drop_id,),
        )
        return cursor.fetchone()[0]


def _ml_species(ann_db, drop_id: str) -> list:
    """Scientific names on this drop's ML rows. NULL_DEPLOYMENT marks "saw nothing"."""
    with ann_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT scientific_name FROM annotations "
            "WHERE drop_id = ? AND annotated_by = 'ml'",
            (drop_id,),
        )
        return [row[0] for row in cursor.fetchall()]


# ── Test data ─────────────────────────────────────────────────────────────────

# Five detections, one per 10-second interval, using two species.
# All above maxn_confidence_threshold (0.50) so process_maxn produces exactly 5 rows:
#   (interval=0,  class="Pagrus auratus")
#   (interval=10, class="Pagrus auratus")
#   (interval=20, class="Notolabrus fucicola")
#   (interval=30, class="Pagrus auratus")
#   (interval=40, class="Notolabrus fucicola")
_RAW_DETECTIONS_5 = pd.DataFrame(
    [
        {
            "frame": 1,
            "time_seconds": 0.5,
            "class": "Pagrus auratus",
            "confidence": 0.80,
            "x": 160,
            "y": 120,
            "h": 50,
            "w": 40,
        },
        {
            "frame": 15,
            "time_seconds": 10.5,
            "class": "Pagrus auratus",
            "confidence": 0.75,
            "x": 160,
            "y": 120,
            "h": 50,
            "w": 40,
        },
        {
            "frame": 30,
            "time_seconds": 20.5,
            "class": "Notolabrus fucicola",
            "confidence": 0.85,
            "x": 160,
            "y": 120,
            "h": 50,
            "w": 40,
        },
        {
            "frame": 45,
            "time_seconds": 30.5,
            "class": "Pagrus auratus",
            "confidence": 0.90,
            "x": 160,
            "y": 120,
            "h": 50,
            "w": 40,
        },
        {
            "frame": 60,
            "time_seconds": 40.5,
            "class": "Notolabrus fucicola",
            "confidence": 0.95,
            "x": 160,
            "y": 120,
            "h": 50,
            "w": 40,
        },
    ]
)

# Same frames and timestamps, confidence dropped to 0.30 — below the 0.50 threshold.
# process_maxn() will return an empty DataFrame, simulating a re-run on a video
# where the model finds nothing above the MaxN confidence cutoff.
_RAW_DETECTIONS_ZERO = _RAW_DETECTIONS_5.assign(confidence=0.30)


# ── Test ──────────────────────────────────────────────────────────────────────


def test_stale_annotations_cleared_on_zero_detection_rerun(pipeline_env):
    """
    After a first run that writes 5 annotations, a re-run that detects nothing
    must leave exactly one null-species row — the "reviewed, saw nothing"
    marker — and none of the 5 stale rows.
    """
    ann_db = pipeline_env.ann_db
    drop_id = DROP_NORMAL

    # ── Run 1: 5 detections above threshold ───────────────────────────────────
    maxn_csv_run1 = pipeline_env.tmp_path / f"{drop_id}_run1_maxn.csv"
    maxn_df_run1 = process_maxn(
        raw_df=_RAW_DETECTIONS_5,
        output_csv_path=str(maxn_csv_run1),
        drop_id=drop_id,
        interval_seconds=10,
        confidence_threshold=0.50,
        model_name=MODEL_NAME,
    )

    assert len(maxn_df_run1) == 5, (
        f"Test setup error: expected 5 MaxN rows from 5 distinct intervals, "
        f"got {len(maxn_df_run1)}"
    )

    _ingest_ml_annotations(ann_db, drop_id, maxn_df_run1, MODEL_NAME)

    count_run1 = _count_ml_annotations(ann_db, drop_id)
    assert (
        count_run1 == 5
    ), f"Expected 5 annotations after first inference run, got {count_run1}"

    # ── Run 2: same frames, all confidence below threshold → empty MaxN ───────
    maxn_csv_run2 = pipeline_env.tmp_path / f"{drop_id}_run2_maxn.csv"
    maxn_df_run2 = process_maxn(
        raw_df=_RAW_DETECTIONS_ZERO,
        output_csv_path=str(maxn_csv_run2),
        drop_id=drop_id,
        interval_seconds=10,
        confidence_threshold=0.50,
        model_name=MODEL_NAME,
    )

    assert maxn_df_run2.empty, (
        f"Test setup error: expected empty MaxN when all confidence < 0.50, "
        f"got {len(maxn_df_run2)} rows"
    )

    _ingest_ml_annotations(ann_db, drop_id, maxn_df_run2, MODEL_NAME)

    # ── Regression assertions ─────────────────────────────────────────────────
    # Species, not just the count: a bare count of 1 would also pass if one of
    # run 1's five rows had survived and the null row had never been written,
    # which is the exact failure this test exists to catch.
    species_run2 = _ml_species(ann_db, drop_id)
    assert species_run2 == [NULL_DEPLOYMENT], (
        f"Expected exactly one {NULL_DEPLOYMENT!r} row after a zero-detection "
        f"re-run, got {species_run2}. Either run 1's annotations were not "
        f"cleared, or the 'reviewed, saw nothing' row was not written."
    )
