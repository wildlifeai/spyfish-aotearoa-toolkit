"""
Smoke tests that verify the pipeline_env fixture is correctly wired.

These are not pipeline logic tests — they confirm that:
  - The test config.yaml is written to disk and loaded into the singleton
  - All databases are created and seeded with the expected deployments
  - Video files exist and are readable by cv2
  - Raw ML CSVs are written with the expected schema
  - The stuck (PROCESSING_ML) deployment has no ML CSV on disk
  - The ML_COMPLETE deployment has a MaxN CSV and annotations pre-seeded

Import these constants in integration tests that need the canonical drop IDs:

    from tests.conftest import DROP_NORMAL, DROP_STUCK, DROP_ML_COMPLETE, MODEL_NAME
"""

import cv2
import pandas as pd
import pytest

from spyfish.config.base import PipelineStatus
from spyfish.config.wrapper import config
from spyfish.ml.process_ml_annotations import process_maxn
from tests.conftest import (
    DROP_ML_COMPLETE,
    DROP_NORMAL,
    DROP_STUCK,
    MODEL_NAME,
    RAW_ML_DETECTIONS,
)

# ── Config wiring ─────────────────────────────────────────────────────────────


def test_config_yaml_written_to_disk(pipeline_env):
    """The test config.yaml must exist in tmp_path and parse cleanly."""
    import yaml

    assert pipeline_env.config_path.exists()
    loaded = yaml.safe_load(pipeline_env.config_path.read_text())
    assert loaded["orchestrator"]["is_test_run"] is True
    assert loaded["paths"]["bucket_name"] == "marine-buv-test"


def test_config_singleton_uses_test_config(pipeline_env):
    """config.* properties must reflect the test config, not the production one."""
    # bucket_name is "marine-buv-test" in the test config, "marine-buv-kalindi" in prod
    assert config.s3_bucket == "marine-buv-test"


def test_all_paths_resolve_under_tmp(pipeline_env):
    """Every config path helper must resolve inside tmp_path."""
    env = pipeline_env
    assert str(env.db.db_path).startswith(str(env.tmp_path))
    assert str(env.ann_db.db_path).startswith(str(env.tmp_path))
    for video_path in env.video_paths.values():
        assert str(video_path).startswith(str(env.tmp_path))
    for csv_path in env.raw_csv_paths.values():
        assert str(csv_path).startswith(str(env.tmp_path))


# ── Database seeding ──────────────────────────────────────────────────────────


def test_deployments_seeded_with_correct_statuses(pipeline_env):
    db = pipeline_env.db

    normal = db.get_deployment(DROP_NORMAL)
    assert normal is not None
    assert normal["status"] == PipelineStatus.READY_FOR_ML
    assert normal["sampling_start"] == 1
    assert normal["sampling_end"] == 5

    stuck = db.get_deployment(DROP_STUCK)
    assert stuck is not None
    assert stuck["status"] == PipelineStatus.PROCESSING_ML

    ml_done = db.get_deployment(DROP_ML_COMPLETE)
    assert ml_done is not None
    assert ml_done["status"] == PipelineStatus.ML_COMPLETE
    assert ml_done["ml_annotations"] == 3


def test_get_deployments_by_status(pipeline_env):
    db = pipeline_env.db
    ready = db.get_deployments_by_status(PipelineStatus.READY_FOR_ML)
    stuck = db.get_deployments_by_status(PipelineStatus.PROCESSING_ML)
    done = db.get_deployments_by_status(PipelineStatus.ML_COMPLETE)

    assert len(ready) == 1 and ready[0]["drop_id"] == DROP_NORMAL
    assert len(stuck) == 1 and stuck[0]["drop_id"] == DROP_STUCK
    assert len(done) == 1 and done[0]["drop_id"] == DROP_ML_COMPLETE


# ── Videos ────────────────────────────────────────────────────────────────────


def test_videos_are_readable_by_cv2(pipeline_env):
    """cv2 must open each video and report the correct 320×240 dimensions."""
    for drop_id, video_path in pipeline_env.video_paths.items():
        assert video_path.exists(), f"Video missing: {drop_id}"
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"cv2 could not open video: {drop_id}"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert w == 320, f"Wrong width for {drop_id}: {w}"
        assert h == 240, f"Wrong height for {drop_id}: {h}"


# ── ML CSV files ──────────────────────────────────────────────────────────────


def test_raw_csv_written_for_drop_normal_only(pipeline_env):
    """Only DROP_NORMAL should have a raw ML CSV — DROP_STUCK crashed before inference."""
    assert DROP_NORMAL in pipeline_env.raw_csv_paths
    assert pipeline_env.raw_csv_paths[DROP_NORMAL].exists()

    # DROP_STUCK: no raw CSV should exist (simulates mid-run crash)
    stuck_csv = config.get_raw_csv_path(DROP_STUCK, MODEL_NAME)
    assert not stuck_csv.exists()


def test_raw_csv_schema(pipeline_env):
    """Raw CSV must have the 8 columns expected by run_inference.py."""
    expected = {"frame", "time_seconds", "class", "confidence", "x", "y", "h", "w"}
    df = pd.read_csv(pipeline_env.raw_csv_paths[DROP_NORMAL])
    assert expected.issubset(df.columns)
    assert len(df) == len(RAW_ML_DETECTIONS[DROP_NORMAL])


def test_maxn_csv_pre_written_for_ml_complete(pipeline_env):
    """DROP_ML_COMPLETE must have a MaxN CSV already on disk from a prior run."""
    maxn_path = config.get_maxn_csv_path(DROP_ML_COMPLETE, MODEL_NAME)
    assert maxn_path.exists()
    df = pd.read_csv(maxn_path)
    assert len(df) == 1
    assert df.iloc[0]["ScientificName"] == "Notolabrus fucicola"


def test_annotations_pre_seeded_for_ml_complete(pipeline_env):
    """ann_db must have the pre-seeded ML annotations for DROP_ML_COMPLETE."""
    ann_db = pipeline_env.ann_db
    with ann_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM annotations WHERE drop_id = ? AND annotated_by = 'ml'",
            (DROP_ML_COMPLETE,),
        )
        rows = cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["scientific_name"] == "Notolabrus fucicola"


# ── Ground truth ──────────────────────────────────────────────────────────────


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
    result = result.sort_values("time_of_maxn_ms").reset_index(drop=True)

    assert len(result) == len(expected)
    assert list(result["MaxInterval"]) == list(expected["MaxInterval"])
    assert list(result["time_of_maxn_ms"]) == list(expected["time_of_maxn_ms"])
    assert list(result["ConfidenceAgreement"]) == list(expected["ConfidenceAgreement"])
    assert list(result["TimeOfMax"]) == list(expected["TimeOfMax"])


def test_process_maxn_respects_confidence_threshold(pipeline_env):
    """
    Raising confidence_threshold to 0.91 should exclude all but the single
    highest-confidence detection and produce MaxInterval=1 only for interval 10.
    Frame 10 had confidences 0.85 and 0.90 — both below 0.91 — so interval 0
    disappears. Frame 25 had 0.95 — the only detection above 0.91.
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
    assert result.iloc[0]["time_of_maxn_ms"] == 10.0
