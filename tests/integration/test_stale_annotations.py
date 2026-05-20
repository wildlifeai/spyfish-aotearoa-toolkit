"""
Regression test for the stale annotations bug.

Bug location: process_ml_annotations.py, _ingest_ml_annotations(), lines 148–154.

    if annotations_to_add:          # ← guard controls clear_annotations too
        ann_db.clear_annotations(drop_id, "ml")
        ann_db.add_annotations(annotations_to_add)

When a re-run produces zero detections above the MaxN confidence threshold,
`process_maxn()` returns an empty DataFrame. The `for ... in maxn_df.iterrows()`
loop runs zero times, `annotations_to_add` stays `[]`, and the `if` guard is False.
`clear_annotations()` is never called. ML annotations from the previous run remain
in the database permanently.

Fix: move `ann_db.clear_annotations(drop_id, "ml")` outside (above) the `if` guard
so it always runs, regardless of whether there are new annotations to write.

This test FAILS against the unfixed code and PASSES after the fix is applied.
"""

import pandas as pd

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
    After a first run that writes N annotations, a subsequent run that produces
    zero detections above threshold must leave the database with 0 annotations —
    not the stale N from the first run.

    Sequence:
        1. process_maxn() + _ingest_ml_annotations() with 5 detections → assert 5 in DB
        2. process_maxn() + _ingest_ml_annotations() with 0 detections → assert 0 in DB  ← FAILS
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

    # ── Regression assertion ───────────────────────────────────────────────────
    # CURRENT BEHAVIOUR (bug): _ingest_ml_annotations builds annotations_to_add = []
    # from an empty maxn_df. The guard `if annotations_to_add:` is False.
    # clear_annotations() is never called. The 5 rows from run 1 survive.
    #
    # EXPECTED BEHAVIOUR (after fix): clear_annotations() runs unconditionally,
    # then add_annotations() is skipped because there is nothing to add.
    count_run2 = _count_ml_annotations(ann_db, drop_id)
    assert count_run2 == 0, (
        f"Expected 0 ML annotations after zero-detection re-run, got {count_run2}. "
        f"Stale annotations from run 1 were not cleared.\n"
        f"Fix: in _ingest_ml_annotations() (process_ml_annotations.py), move "
        f"ann_db.clear_annotations(drop_id, 'ml') above the `if annotations_to_add:` guard."
    )
