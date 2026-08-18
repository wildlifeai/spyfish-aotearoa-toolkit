"""
Shared fixtures for unit and integration tests.

Integration tests use the `pipeline_env` fixture, which builds an isolated
environment under pytest's tmp_path:

    - A test config.yaml written to disk (monkeypatched into the config singleton)
    - Pipeline and annotation SQLite databases, pre-seeded with 3 deployments
    - Tiny real videos (5 s, 320×240, 25 fps) for each deployment
    - Raw ML CSV and pre-written MaxN CSV for the appropriate drops
    - Known ground truth for MaxN calculations

Canonical deployment identifiers:

    DROP_NORMAL       KSF_20240124_BUV_KSF_085_01  ml_status=ready    — happy-path drop
    DROP_STUCK        KSF_20240124_BUV_KSF_085_02  ml_status=running  — simulates a mid-run crash
    DROP_ML_COMPLETE  KSF_20240124_BUV_KSF_085_03  ml_status=complete — for post-ML step tests

Usage in tests:

    from tests.conftest import DROP_NORMAL, DROP_STUCK, DROP_ML_COMPLETE, MODEL_NAME

    def test_something(pipeline_env):
        env = pipeline_env
        assert env.db.get_deployment(DROP_NORMAL)["ml_status"] == MlStatus.READY
"""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from spyfish.config.base import MlStatus
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager

# ── Canonical identifiers ─────────────────────────────────────────────────────

MODEL_NAME = "yolov8n"

# All three drops come from the same survey so cross-drop survey-level tests work.
DROP_NORMAL = "KSF_20240124_BUV_KSF_085_01"  # READY_FOR_ML  — inference not yet run
DROP_STUCK = "KSF_20240124_BUV_KSF_085_02"  # PROCESSING_ML — crash before CSV written
DROP_ML_COMPLETE = (
    "KSF_20240124_BUV_KSF_085_03"  # ML_COMPLETE   — MaxN CSV already on disk
)

# ── Test config.yaml ──────────────────────────────────────────────────────────
# Written to tmp_path/config.yaml and loaded into the singleton.
# All path keys are relative; project_root is monkeypatched to tmp_path so every
# config.get_*_path() resolves under the temp directory.
TEST_CONFIG_YAML = """\
csv_mapping:
  drop_id_column: "DropID"
  pipeline_status_column: "PipelineStatus"
  survey_id_column: "SurveyID"
  site_id_column: "SiteID"
  replicate_column: "ReplicateWithinSite"
  file_name_column: "FileName"
  link_to_marine_reserve_column: "LinkToMarineReserve"
  site_name_column: "SiteName"
  is_bad_deployment_column: "IsBadDeployment"
  video_file_link_column: "LinkToVideoFile"
  sampling_start_column: "SamplingStart"
  sampling_end_column: "SamplingEnd"
  clip_start_absolute_column: "ClipStartAbsoluteSeconds"
  clip_end_absolute_column: "ClipEndAbsoluteSeconds"
  clip_max_time_column: "TimeOfMaxAbsSeconds"
  maxn_time_seconds_column: "TimeOfMaxAbsSeconds"
  confidence_agreement_column: "ConfidenceAgreement"
  confusion_score_column: "ConfusionScore"
  scientific_name_column: "ScientificName"
  maxn_time_column: "TimeOfMax"
  max_interval_column: "MaxInterval"
  annotated_by_column: "AnnotatedBy"
  interval_annotation_column: "IntervalAnnotation"
  time_seconds_column: "TimeAbsSeconds"

paths:
  base_dir: "process_files"
  deployment_targets_csv: process_files/deployment_targets.csv
  movie_extensions: ["avi", "mov", "mp4", "mpg", "wmv"]
  missing_files_filename: "missing_files_in_aws.txt"
  extra_files_filename: "extra_files_in_aws.txt"
  legacy:
    zooniverse: "process_files/zooniverse/legacy_classifications"
  metadata:
    root: "spyfish_metadata"
    sharepoint_dir: "sharepoint_lists"
    status_dir: "status"
    files:
      deployment_csv: "BUV Deployment.csv"
      survey_csv: "BUV Survey Metadata.csv"
      site_csv: "BUV Survey Sites.csv"
      species_csv: "BUV Species.csv"
      reserves_csv: "Marine Reserves.csv"
      legacy_experts_csv: "BUV Annotations Legacy Experts.csv"
  sub_dirs:
    media: "media"
    annotations: "annotations"
    deployment_data: "deployment_data"
    db: "db"
    logs: "logs"
    training: "training"
    models: "models"
    base_model: "base_model"
    pipeline_model: "pipeline_model"
    archived_models: "archived_models"

ml_inference:
  limit_processing: 1
  log_interval_frames: 10
  ml_fps: 3
  imgsz: 640
  confidence_threshold: 0.25
  maxn_confidence_threshold: 0.50
  remote_host: "mahuika"
  remote_user: "testuser"
  extraction:
    interval_seconds: 10
    annotated_by_prefix: "ml"

extraction:
  clip_length: 10.0
  clip_cap: 50
  video_start_threshold_seconds: 120
  force_binary_strategy: false
  sample_all_clips: false
  binary_strategy:
    maxn_export: 10
    confusing_export: 20
    empty_export: 5
    start_export: 2
    temporal_spacing_seconds: 10
  multiclass_strategy:
    per_species_maxn_export: 5
    per_species_confusing_export: 10
    per_video_empty_export: 3
    per_video_start_export: 2
    temporal_spacing_seconds: 10

zooniverse:
  project_id: 99999
  size_limit_mb: 12.0
  min_clips_per_video: 6
  min_agreement_pct: 25.0
  max_frames_per_run: 3

orchestrator:
  is_test_run: true
  log_output: "console"

ffmpeg:
  crf: 22
  preset: "fast"
  codec: "libx264"

biigle:
  project_id: 1
  disk_id: 1
  annotation_report_type_video: 8
  annotation_report_type_images: 3
  done_labels: ["Done Volume", "Done QA Review"]
  default_fish_label_id: 1
  default_label_tree_id: 1
  label_mapping: {}
  volume_finalize_max_retries: 3
  volume_finalize_retry_interval_secs: 0.1
  report_download_max_retries: 3
  report_download_retry_interval_secs: 0.1

training:
  image_extensions: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]
  epochs: 100
  patience: 25
  imgsz: 640
  train_pct: 0.85
  val_pct: 0.15
  test_pct: 0.0
  class_floor_pct: 0.02
  local_training_dir: "process_files/training"
  retrain_min_improvement_pct: 2.0

validation_patterns:
  drop_id:   "^[A-Z]{3}_\\\\d{8}_BUV_[A-Z]{3}_\\\\d{3}_\\\\d{2}$"
  survey_id: "^[A-Z]{3}_\\\\d{8}_BUV$"
  site_id:   "^[A-Z]{3}_\\\\d+$"

validation_rules:
  deployments:
    file_name: "spyfish_metadata/sharepoint_lists/BUV Deployment.csv"
    required: ["DropID", "SurveyID", "SiteID", "SamplingStart", "SamplingEnd"]
    unique: ["DropID"]
    info_columns: ["SurveyID", "SiteID"]
    foreign_keys:
      surveys: "SurveyID"
      sites: "SiteID"
    relationships:
      - column: "DropID"
        rule: "equals"
        template: "{SurveyID}_{SiteID}_{ReplicateWithinSite:02}"
      - column: "FileName"
        rule: "equals"
        template: "{DropID}.mp4"
        allowed_values: ["NO VIDEO BAD DEPLOYMENT"]
        allow_null: true
      - column: "LinkToVideoFile"
        rule: "equals"
        template: "media/{SurveyID}/{DropID}/{DropID}.mp4"
        allowed_values: ["NO VIDEO BAD DEPLOYMENT"]
        allow_null: true
    values:
      - column: "Latitude"
        rule: "value_range"
        range: [-46, -36]
        allowed_values: [0]
      - column: "Longitude"
        rule: "value_range"
        range: [170, 178.5]
        allowed_values: [0]
  surveys:
    file_name: "spyfish_metadata/sharepoint_lists/BUV Survey Metadata.csv"
    required: ["SurveyID"]
    unique: ["SurveyID"]
    info_columns: ["SurveyName"]
    formats: ["SurveyID"]
    foreign_keys: {}
    relationships: []
  sites:
    file_name: "spyfish_metadata/sharepoint_lists/BUV Survey Sites.csv"
    required: ["SiteID", "LinkToMarineReserve"]
    unique: ["SiteID"]
    info_columns: ["SiteName", "LinkToMarineReserve"]
    formats: ["SiteID"]
    foreign_keys: {}
    relationships: []
  species:
    file_name: "spyfish_metadata/sharepoint_lists/BUV Species.csv"
    required: ["AphiaID", "CommonName", "ScientificName"]
    unique: ["AphiaID", "ScientificName", "CommonName"]
    info_columns: ["AphiaID", "CommonName", "ScientificName"]
    foreign_keys: {}
    relationships: []
  reserves:
    file_name: "spyfish_metadata/sharepoint_lists/Marine Reserves.csv"
    required: []
    unique: []
    info_columns: []
    foreign_keys: {}
    relationships: []
"""

# ── Raw ML detections ─────────────────────────────────────────────────────────
# Columns match the hardcoded schema in run_inference.py:
#   frame, time_seconds, class, confidence, x, y, h, w
#
# DROP_NORMAL layout (interval_seconds=10, maxn_confidence_threshold=0.50):
#
#   interval 0–10 s:
#     frame 10 has 2 detections above threshold → MaxN = 2 at t = 4.0 s
#     mean_confidence = (0.85 + 0.90) / 2 = 0.875
#
#   interval 10–20 s:
#     frame 25 (conf 0.95) and frame 37 (conf 0.72) both have MaxN = 1 — a tie.
#     Tiebreak: highest mean confidence across all boxes in that frame.
#     Frame 25 wins: 0.95 > 0.72 → best_second = 10.0 s
#
# DROP_STUCK has no raw CSV — the process crashed before inference wrote output.
#
# DROP_ML_COMPLETE has no raw CSV — inference already ran in a prior pipeline run.
RAW_ML_DETECTIONS: dict[str, pd.DataFrame] = {
    DROP_NORMAL: pd.DataFrame(
        [
            {
                "frame": 10,
                "time_seconds": 4.0,
                "class": "Pagrus auratus",
                "confidence": 0.85,
                "x": 160,
                "y": 120,
                "h": 50,
                "w": 40,
            },
            {
                "frame": 10,
                "time_seconds": 4.0,
                "class": "Pagrus auratus",
                "confidence": 0.90,
                "x": 200,
                "y": 120,
                "h": 50,
                "w": 40,
            },
            {
                "frame": 25,
                "time_seconds": 10.0,
                "class": "Pagrus auratus",
                "confidence": 0.95,
                "x": 160,
                "y": 120,
                "h": 50,
                "w": 40,
            },
            {
                "frame": 37,
                "time_seconds": 14.8,
                "class": "Pagrus auratus",
                "confidence": 0.72,
                "x": 160,
                "y": 120,
                "h": 50,
                "w": 40,
            },
        ]
    ),
}

# ── Ground truth MaxN output ──────────────────────────────────────────────────
# Exact rows that process_maxn() must produce from RAW_ML_DETECTIONS[DROP_NORMAL].
# Column names match csv_mapping in config.yaml — accessed via config.*_column properties.
# Values are hardcoded (no pipeline code) so any regression in MaxN logic breaks these tests.
EXPECTED_MAXN: dict[str, pd.DataFrame] = {
    DROP_NORMAL: pd.DataFrame(
        [
            {
                "DropID": DROP_NORMAL,
                "ScientificName": "Pagrus auratus",
                "TimeOfMax": "00:00:04.000",  # seconds_to_time(4.0)
                "MaxInterval": 2,
                "AnnotatedBy": MODEL_NAME,
                "IntervalAnnotation": 10,
                "ConfidenceAgreement": 0.875,  # (0.85 + 0.90) / 2
                "TimeOfMaxAbsSeconds": 4.0,
            },
            {
                "DropID": DROP_NORMAL,
                "ScientificName": "Pagrus auratus",
                "TimeOfMax": "00:00:10.000",  # seconds_to_time(10.0)
                "MaxInterval": 1,
                "AnnotatedBy": MODEL_NAME,
                "IntervalAnnotation": 10,
                "ConfidenceAgreement": 0.95,  # frame 25 wins tiebreak
                "TimeOfMaxAbsSeconds": 10.0,
            },
        ]
    )
    .sort_values("TimeOfMaxAbsSeconds")
    .reset_index(drop=True),
}

# Pre-written MaxN data for DROP_ML_COMPLETE — stored in the fixture as if a prior
# pipeline run already completed inference and wrote this output. Three rows so
# `sync_annotation_counts` yields ml_annotations = 3.
_ML_COMPLETE_MAXN_ROWS = [
    {
        "DropID": DROP_ML_COMPLETE,
        "ScientificName": "Notolabrus fucicola",
        "TimeOfMax": "00:00:05.000",
        "MaxInterval": 3,
        "AnnotatedBy": MODEL_NAME,
        "IntervalAnnotation": 10,
        "ConfidenceAgreement": 0.80,
        "TimeOfMaxAbsSeconds": 5.0,
    },
    {
        "DropID": DROP_ML_COMPLETE,
        "ScientificName": "Pagrus auratus",
        "TimeOfMax": "00:00:15.000",
        "MaxInterval": 2,
        "AnnotatedBy": MODEL_NAME,
        "IntervalAnnotation": 10,
        "ConfidenceAgreement": 0.75,
        "TimeOfMaxAbsSeconds": 15.0,
    },
    {
        "DropID": DROP_ML_COMPLETE,
        "ScientificName": "Parika scaber",
        "TimeOfMax": "00:00:25.000",
        "MaxInterval": 1,
        "AnnotatedBy": MODEL_NAME,
        "IntervalAnnotation": 10,
        "ConfidenceAgreement": 0.90,
        "TimeOfMaxAbsSeconds": 25.0,
    },
]


# ── PipelineEnv dataclass ─────────────────────────────────────────────────────


@dataclass
class PipelineEnv:
    """
    Fully isolated test environment created by the pipeline_env fixture.

    Attributes:
        tmp_path:          pytest tmp_path — all test files live here.
        config_path:       Path to the test config.yaml written on disk.
        db:                Pipeline DatabaseManager, DB at tmp_path/process_files/db/.
        ann_db:            Annotation DatabaseManager, DB at tmp_path/process_files/db/.
        raw_csv_paths:     {drop_id: Path} — raw ML CSVs for drops that have them.
        expected_maxn:     {drop_id: DataFrame} — ground truth for process_maxn().
        model_name:        Model name string used in CSV/DB path construction.
    """

    tmp_path: Path
    config_path: Path
    db: DatabaseManager
    ann_db: AnnotationDatabaseManager
    raw_csv_paths: dict
    expected_maxn: dict
    model_name: str = MODEL_NAME


# ── Integration fixture ───────────────────────────────────────────────────────


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    """
    Creates a fully isolated pipeline environment. Tears down automatically
    when the test completes because everything lives under pytest's tmp_path.

    Folder structure created under tmp_path:
        config.yaml
        media/                                 (empty — no test needs a video)
        process_files/
            db/
                spyfish_pipeline.db
                spyfish_annotations.db
            data_quality/
                KSF_20240124_BUV_KSF_085_01/annotations/
                    KSF_20240124_BUV_KSF_085_01_yolov8n_raw.csv
                KSF_20240124_BUV_KSF_085_02/annotations/   (empty — crash before inference)
                KSF_20240124_BUV_KSF_085_03/annotations/
                    KSF_20240124_BUV_KSF_085_03_ml_yolov8n_maxn.csv

    No video files: the fixture used to shell out to ffmpeg for three synthetic
    MP4s, which gated every test that used it on a binary only one test needed.
    That test is gone; a test that genuinely needs a decodable video should
    generate one in its own fixture rather than putting the dependency back
    here.
    """
    # ── 1. Write test config.yaml to disk ─────────────────────────────────
    config_path = tmp_path / "config.yaml"
    config_path.write_text(TEST_CONFIG_YAML)
    test_yaml = yaml.safe_load(TEST_CONFIG_YAML)

    # ── 2. Patch the config singleton ─────────────────────────────────────
    # _project_root controls where all path properties resolve to.
    # _yaml_config is replaced with the test config so the singleton reads
    # the on-disk test file rather than the production config.yaml.
    monkeypatch.setattr(config, "_project_root", tmp_path)
    monkeypatch.setattr(config, "_yaml_config", test_yaml)
    monkeypatch.setenv("S3_BUCKET", "marine-buv-test")

    # ── 3. Create the expected folder tree ────────────────────────────────
    (tmp_path / "media").mkdir()
    (tmp_path / "process_files" / "db").mkdir(parents=True)
    for drop_id in [DROP_NORMAL, DROP_STUCK, DROP_ML_COMPLETE]:
        config.get_drop_annotations_dir(drop_id).mkdir(parents=True)

    # ── 4. Create databases ───────────────────────────────────────────────
    # Paths now resolve inside tmp_path thanks to the monkeypatch above.
    db = DatabaseManager()
    ann_db = AnnotationDatabaseManager()

    # ── 5. Seed the three deployments ─────────────────────────────────────
    db.add_or_update_deployment(
        drop_id=DROP_NORMAL,
        ml_status=MlStatus.READY,
        sampling_start=1,
        sampling_end=5,
    )
    # DROP_STUCK: video was present when the crash happened; ml_status was set to
    # 'running' before the crash so it is now permanently stuck there.
    db.add_or_update_deployment(
        drop_id=DROP_STUCK,
        ml_status=MlStatus.RUNNING,
        sampling_start=1,
        sampling_end=5,
    )
    # DROP_ML_COMPLETE: a prior pipeline run finished inference successfully.
    # ml_annotations count is derived from the annotations DB via
    # sync_annotation_counts() in step 8 below — not set directly here.
    db.add_or_update_deployment(
        drop_id=DROP_ML_COMPLETE,
        ml_status=MlStatus.COMPLETE,
        sampling_start=1,
        sampling_end=5,
    )

    # ── 6. Write raw ML CSV for DROP_NORMAL ───────────────────────────────
    # DROP_STUCK has no CSV (the crash happened before inference wrote output).
    # DROP_ML_COMPLETE has no raw CSV (a prior run already post-processed it).
    raw_csv_paths = {}
    raw_csv_path = config.get_raw_csv_path(DROP_NORMAL, MODEL_NAME)
    RAW_ML_DETECTIONS[DROP_NORMAL].to_csv(raw_csv_path, index=False)
    raw_csv_paths[DROP_NORMAL] = raw_csv_path

    # ── 7. Pre-write MaxN CSV and seed annotations for DROP_ML_COMPLETE ───
    maxn_df = pd.DataFrame(_ML_COMPLETE_MAXN_ROWS)
    maxn_csv_path = config.get_maxn_csv_path(DROP_ML_COMPLETE, MODEL_NAME)
    maxn_df.to_csv(maxn_csv_path, index=False)

    ann_db.add_annotations(
        [
            {
                "drop_id": row["DropID"],
                "scientific_name": row["ScientificName"],
                "time_of_max": row["TimeOfMax"],
                "max_interval": row["MaxInterval"],
                "annotated_by": "ml",
                "interval_annotation": "",
                "confidence_agreement": row["ConfidenceAgreement"],
                "external_id": MODEL_NAME,
            }
            for row in _ML_COMPLETE_MAXN_ROWS
        ]
    )

    # Propagate counts from annotations DB → deployments table. This matches
    # how the real pipeline works: sync_annotation_counts() is the only writer
    # to the ml_annotations / citsci_annotations / expert_annotations columns.
    db.sync_annotation_counts([DROP_ML_COMPLETE])

    return PipelineEnv(
        tmp_path=tmp_path,
        config_path=config_path,
        db=db,
        ann_db=ann_db,
        raw_csv_paths=raw_csv_paths,
        expected_maxn=EXPECTED_MAXN,
    )


# ── Unit test fixtures (unchanged) ────────────────────────────────────────────


# NOTE: there is deliberately no ":memory:" fixture. DatabaseManager resolves
# db_path via Path(...).absolute(), which turns the string ":memory:" into a
# literal file named `:memory:` in the repo root — and get_connection() opens a
# fresh connection per call, so an in-memory DB would be empty each time anyway.
# Use temp_db.
@pytest.fixture
def temp_db(tmp_path):
    """Provides a DatabaseManager connected to a temporary file database."""
    db_file = tmp_path / "test_pipeline.db"
    db = DatabaseManager(db_path=str(db_file))
    yield db


@pytest.fixture
def mock_s3_handler():
    """Mocks the S3Handler to prevent real AWS calls during testing."""
    with patch("spyfish.storage.s3_handler.S3Handler") as MockS3:
        mock_instance = MockS3.return_value
        mock_instance.download_object_from_s3.return_value = True
        yield mock_instance
