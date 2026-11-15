"""
Tests for ML annotation ingestion — the null-row convention.

A deployment where inference completed with zero detections must still leave
a record in the annotations DB: one NULL_DEPLOYMENT sentinel row with
max_interval=0,
matching the convention expert reviews use. "ML ran and found nothing" has to
be distinguishable from "ML never ran".
"""

import pandas as pd
import pytest

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.ml.process_ml_annotations import _ingest_ml_annotations

DROP = "KSF_20240124_BUV_KSF_085_01"
MODEL = "yolov8n_test_20260101_000000"


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


def _detection_row(species="Chrysophrys auratus", count=3):
    return [DROP, species, "00:00:05", count, MODEL, 30, 0.9, 5.0]


def test_zero_detections_writes_null_row(ann_db):
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(rows) == 1
    assert rows[0]["scientific_name"] == NULL_DEPLOYMENT
    assert rows[0]["max_interval"] == 0
    assert rows[0]["external_id"] == MODEL


def test_detections_write_real_rows_not_null(ann_db):
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([_detection_row()]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(rows) == 1
    assert rows[0]["scientific_name"] == "Chrysophrys auratus"
    assert rows[0]["max_interval"] == 3


def test_rerun_with_detections_replaces_null_row(ann_db):
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([]), MODEL)
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([_detection_row()]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(rows) == 1
    assert rows[0]["scientific_name"] == "Chrysophrys auratus"


def test_rerun_with_zero_detections_replaces_real_rows(ann_db):
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([_detection_row()]), MODEL)
    _ingest_ml_annotations(ann_db, DROP, _maxn_df([]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(rows) == 1
    assert rows[0]["scientific_name"] == NULL_DEPLOYMENT
    assert rows[0]["max_interval"] == 0


def test_null_row_scoped_to_model(ann_db):
    """A zero-detection run of one model must not clear another model's rows."""
    other_model = "species_v2_20260101_000000"
    df_other = _maxn_df([_detection_row()])
    _ingest_ml_annotations(ann_db, DROP, df_other, other_model)

    _ingest_ml_annotations(ann_db, DROP, _maxn_df([]), MODEL)

    rows = ann_db.get_annotations_for_drop(DROP, "ml")
    by_model = {r["external_id"]: r for r in rows}
    assert by_model[other_model]["scientific_name"] == "Chrysophrys auratus"
    assert by_model[MODEL]["scientific_name"] == NULL_DEPLOYMENT


# ── One raw CSV read per drop ────────────────────────────────────────────────


def test_process_one_drop_reads_the_raw_csv_once(tmp_path, monkeypatch):
    """MaxN, the timeline plot and the QA frames all work from the one
    DataFrame `process_one_drop` loads; drawing used to read the file again."""
    import cv2
    import numpy as np

    from spyfish.ml import draw_frames
    from spyfish.ml import process_ml_annotations as pm

    video = tmp_path / f"{DROP}.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 3, (64, 48))
    for i in range(12):
        writer.write(np.full((48, 64, 3), (i * 10) % 255, dtype=np.uint8))
    writer.release()

    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "frame": range(12),
            "time_seconds": [i / 3 for i in range(12)],
            "class": "fish",
            "confidence": 0.9,
            "x": 32,
            "y": 24,
            "w": 20,
            "h": 16,
        }
    ).to_csv(raw_csv, index=False)

    cfg = type(pm.config)
    monkeypatch.setattr(cfg, "get_drop_annotations_dir", lambda s, d: tmp_path)
    monkeypatch.setattr(cfg, "get_raw_csv_path", lambda s, d, m: raw_csv)
    monkeypatch.setattr(cfg, "get_maxn_csv_path", lambda s, d, m: tmp_path / "maxn.csv")
    monkeypatch.setattr(cfg, "get_drop_dir", lambda s, d: tmp_path)
    monkeypatch.setattr(cfg, "get_frames_dir", lambda s, d, kind=None: tmp_path / "qa")
    monkeypatch.setattr(pm, "plot_maxn_timeline", lambda **kwargs: None)
    monkeypatch.setattr(pm, "load_selection_frames", lambda drop_id: {})
    monkeypatch.setattr(pm, "_ingest_ml_annotations", lambda *args: None)
    monkeypatch.setattr(pm, "write_legacy_review_frames", lambda **kwargs: [])
    monkeypatch.setattr(
        pm,
        "DatabaseManager",
        lambda: type("Db", (), {"get_deployment": lambda s, d: {}})(),
    )
    monkeypatch.setattr(draw_frames, "common_name_labeller", lambda: str)

    reads = []
    real_read = pm.read_raw_detections

    def counting_read(path):
        reads.append(str(path))
        return real_read(path)

    monkeypatch.setattr(pm, "read_raw_detections", counting_read)
    monkeypatch.setattr(draw_frames, "read_raw_detections", counting_read)

    pm.process_one_drop(
        drop_id=DROP,
        video_dir=tmp_path,
        ann_db=None,
        model_name=MODEL,
        interval=30,
        base_conf=0.25,
        maxn_conf=0.4,
    )

    assert reads == [str(raw_csv)]
    assert (tmp_path / "maxn.csv").exists()
    assert list((tmp_path / "qa").glob("*.jpg")), "QA frames must still be drawn"
