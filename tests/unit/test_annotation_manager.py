"""
Tests for AnnotationDatabaseManager — annotation CRUD and aggregation.

Covers:
  - clear_synced_annotations: only deletes rows WITH external_id
  - get_annotations_for_drop: with and without annotated_by filter
  - get_maxn_summary: MAX(max_interval) per drop × species × source
"""

import pytest

from spyfish.database.annotation_manager import AnnotationDatabaseManager

DROP = "KSF_20240124_BUV_KSF_085_01"


@pytest.fixture
def ann_db(tmp_path):
    db = AnnotationDatabaseManager(db_path=str(tmp_path / "test_annotations.db"))
    return db


def _seed(ann_db, rows):
    ann_db.add_annotations(rows)


def _ann(
    drop_id=DROP,
    species="Pagrus auratus",
    max_interval=1,
    annotated_by="ml",
    external_id="yolov8n",
    time_of_max="00:00:05",
):
    return {
        "drop_id": drop_id,
        "scientific_name": species,
        "time_of_max": time_of_max,
        "max_interval": max_interval,
        "annotated_by": annotated_by,
        "interval_annotation": "",
        "confidence_agreement": 0.9,
        "external_id": external_id,
    }


# ── clear_synced_annotations ────────────────────────────────────────────────


def test_clear_synced_only_deletes_with_external_id(ann_db):
    """Rows with external_id=None (manual entry) should survive the clear."""
    _seed(
        ann_db,
        [
            _ann(annotated_by="expert", external_id="biigle_42"),
            _ann(annotated_by="expert", external_id=None, species="Manual Entry"),
        ],
    )
    ann_db.clear_synced_annotations(DROP, "expert")
    remaining = ann_db.get_annotations_for_drop(DROP, "expert")
    assert len(remaining) == 1
    assert remaining[0]["scientific_name"] == "Manual Entry"


def test_clear_synced_does_not_touch_other_sources(ann_db):
    _seed(
        ann_db,
        [
            _ann(annotated_by="expert", external_id="biigle_1"),
            _ann(annotated_by="ml", external_id="yolov8n"),
        ],
    )
    ann_db.clear_synced_annotations(DROP, "expert")
    ml = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(ml) == 1


# ── get_annotations_for_drop ────────────────────────────────────────────────


def test_get_annotations_for_drop_all(ann_db):
    _seed(
        ann_db,
        [
            _ann(annotated_by="ml"),
            _ann(annotated_by="expert", external_id="biigle_1"),
            _ann(annotated_by="citsci", external_id=None),
        ],
    )
    all_anns = ann_db.get_annotations_for_drop(DROP)
    assert len(all_anns) == 3


def test_get_annotations_for_drop_filtered(ann_db):
    _seed(
        ann_db,
        [
            _ann(annotated_by="ml"),
            _ann(annotated_by="expert", external_id="biigle_1"),
        ],
    )
    expert_only = ann_db.get_annotations_for_drop(DROP, annotated_by="expert")
    assert len(expert_only) == 1
    assert expert_only[0]["annotated_by"] == "expert"


def test_get_annotations_for_drop_empty(ann_db):
    assert ann_db.get_annotations_for_drop("NONEXISTENT") == []


# ── get_maxn_summary ────────────────────────────────────────────────────────


def test_get_maxn_summary_returns_peak_per_species(ann_db):
    """MAX(max_interval) across time intervals, not SUM."""
    _seed(
        ann_db,
        [
            _ann(max_interval=3, time_of_max="00:00:05"),
            _ann(max_interval=5, time_of_max="00:00:15"),
            _ann(max_interval=2, time_of_max="00:00:25"),
        ],
    )
    df = ann_db.get_maxn_summary(drop_id=DROP, annotated_by="ml")
    assert len(df) == 1
    assert df.iloc[0]["maxn"] == 5  # peak, not sum


def test_get_maxn_summary_separates_species(ann_db):
    _seed(
        ann_db,
        [
            _ann(species="Pagrus auratus", max_interval=3),
            _ann(species="Notolabrus fucicola", max_interval=7, time_of_max="00:00:15"),
        ],
    )
    df = ann_db.get_maxn_summary(drop_id=DROP)
    assert len(df) == 2
    species_maxn = dict(zip(df["scientific_name"], df["maxn"]))
    assert species_maxn["Pagrus auratus"] == 3
    assert species_maxn["Notolabrus fucicola"] == 7


def test_get_maxn_summary_empty(ann_db):
    df = ann_db.get_maxn_summary(drop_id="NONEXISTENT")
    assert df.empty


# ── null_deployment_row ─────────────────────────────────────────────────────


def test_null_deployment_row_round_trip(ann_db):
    """The absence record survives storage and the MaxN summary.

    get_maxn_summary once dropped absence records via a NULL = NULL subquery;
    the sentinel plus the null-safe IS comparison keeps them visible.
    """
    from spyfish.config.base import NULL_DEPLOYMENT
    from spyfish.database.annotation_manager import null_deployment_row

    row = null_deployment_row(DROP, "citsci")
    assert row["scientific_name"] == NULL_DEPLOYMENT
    assert row["max_interval"] == 0
    assert row["time_of_max_seconds"] is None

    _seed(ann_db, [row])
    stored = ann_db.get_annotations_for_drop(DROP, "citsci")
    assert len(stored) == 1
    assert stored[0]["scientific_name"] == NULL_DEPLOYMENT

    summary = ann_db.get_maxn_summary(drop_id=DROP)
    assert len(summary) == 1
    assert summary.iloc[0]["maxn"] == 0


def test_null_deployment_row_biigle_replaceable(ann_db):
    """A volume-scoped external_id makes the row clear_synced-replaceable."""
    from spyfish.database.annotation_manager import null_deployment_row

    _seed(ann_db, [null_deployment_row(DROP, "expert", external_id="biigle_volume_7")])
    ann_db.clear_synced_annotations(DROP, "expert")
    assert ann_db.get_annotations_for_drop(DROP, "expert") == []
