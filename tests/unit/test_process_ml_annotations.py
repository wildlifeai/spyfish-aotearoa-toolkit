"""
Tests for ML annotation ingestion — the null-row convention.

A deployment where inference completed with zero detections must still leave
a record in the annotations DB: one null-species row with max_interval=0,
matching the convention expert reviews use. "ML ran and found nothing" has to
be distinguishable from "ML never ran".
"""

import pandas as pd
import pytest

from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.ml.process_ml_annotations import _ingest_ml_annotations

DROP = "KSF_20240124_BUV_KSF_085_01"
MODEL = "yolov8n_test"


@pytest.fixture
def ann_db(tmp_path):
    return AnnotationDatabaseManager(db_path=str(tmp_path / "test_annotations.db"))


def _maxn_df(rows):
    return pd.DataFrame(
        rows,
        columns=[
            config.drop_id_column,
            config.csv_scientific_name_column,
            config.csv_maxn_time_column,
            config.csv_max_interval_column,
            config.csv_annotated_by_column,
            config.csv_interval_annotation_column,
            config.csv_confidence_agreement_column,
            config.csv_maxn_time_seconds_column,
        ],
    )


def _detection_row(species="Pagrus auratus", count=3):
    return [DROP, species, "00:00:05", count, MODEL, 30, 0.9, 5.0]


def test_zero_detections_writes_null_row(ann_db):
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(rows) == 1
    assert rows[0]["scientific_name"] is None
    assert rows[0]["max_interval"] == 0
    assert rows[0]["external_id"] == MODEL


def test_detections_write_real_rows_not_null(ann_db):
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([_detection_row()]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(rows) == 1
    assert rows[0]["scientific_name"] == "Pagrus auratus"
    assert rows[0]["max_interval"] == 3


def test_rerun_with_detections_replaces_null_row(ann_db):
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([]), MODEL)
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([_detection_row()]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(rows) == 1
    assert rows[0]["scientific_name"] == "Pagrus auratus"


def test_rerun_with_zero_detections_replaces_real_rows(ann_db):
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([_detection_row()]), MODEL)
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(rows) == 1
    assert rows[0]["scientific_name"] is None
    assert rows[0]["max_interval"] == 0


def test_null_row_scoped_to_model(ann_db):
    """A zero-detection run of one model must not clear another model's rows."""
    other_model = "species_v2"
    df_other = _maxn_df([_detection_row()])
    df_other[config.csv_annotated_by_column] = other_model
    _ingest_ml_annotations(ann_db, DROP, df_other, other_model)

    _ingest_ml_annotations(ann_db, DROP, _maxn_df([]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    by_model = {r["external_id"]: r for r in rows}
    assert by_model[other_model]["scientific_name"] == "Pagrus auratus"
    assert by_model[MODEL]["scientific_name"] is None
