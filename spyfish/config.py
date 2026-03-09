import os
import yaml
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

def str_to_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() == "true"

def get_required(config_dict, key: str, section: str = ""):
    """Robust getter that works with both Python dicts and ConfigWrapper objects"""
    if isinstance(config_dict, dict):
        if key not in config_dict:
            section_msg = f" in section '{section}'" if section else ""
            raise ValueError(f"Missing required config key '{key}'{section_msg}")
        return config_dict[key]
    else:
        # Handle object access (e.g. ConfigWrapper)
        if not hasattr(config_dict, key) or getattr(config_dict, key) is None:
            section_msg = f" in section '{section}'" if section else ""
            raise ValueError(f"Missing required config key '{key}'{section_msg}")
        return getattr(config_dict, key)

def load_env_wrapper() -> None:
    """
    Wrapper for loading environment variables from a .env file.

    Guard clause added to prevent loading environment variables when running in
    a GitHub Actions environment.
    """
    if os.getenv("GITHUB_ACTIONS") == "true":
        return

    env_path = find_dotenv()
    # Check if the file exists before trying to load it
    if env_path:
        logging.info(f"Loading .env file from: {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        logging.warning(
            f".env file not found at '{env_path}'. Environment variables might not be loaded."
        )

load_env_wrapper()

def load_config() -> dict:
    """Loads the main pipeline configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ConfigWrapper:
    def __init__(self):
        self._yaml_config = load_config()
        # Absolute path to the project root (where config.yaml lives)
        self._project_root = Path(__file__).parent.parent.resolve()

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root directory."""
        return self._project_root

    @property
    def base_dir(self) -> str:
        return self.paths.get("base_dir", "process_files")

    def db_rel_path(self, filename: str) -> str:
        """The relative path (key) for a database file, starting from base_dir."""
        db_dir = self.sub_dirs.get("db", "db")
        return f"{self.base_dir}/{db_dir}/{filename}"

    def get_db_path(self, filename: str) -> Path:
        """LOCAL path to a SQLite database by name."""
        path = self.project_root / self.db_rel_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_s3_db_key(self, filename: str) -> str:
        """S3 key for a database by name."""
        return self.db_rel_path(filename)

    @property
    def db_path(self) -> Path:
        """LOCAL path to the main pipeline database."""
        return self.get_db_path("spyfish_pipeline.db")

    @property
    def csv_mapping(self):
        return self._yaml_config.get("csv_mapping", {})

    @property
    def s3_bucket(self):
        return self.paths.get("bucket_name")

    @property
    def sub_dirs(self):
        return self.paths.get("sub_dirs", {})

    @property
    def metadata(self):
        return self.paths.get("metadata", {})

    @property
    def aws_access_key_id(self):
        return os.getenv("AWS_ACCESS_KEY_ID")

    @property
    def aws_secret_access_key(self):
        return os.getenv("AWS_SECRET_ACCESS_KEY")

    @property
    def aws_region(self):
        return os.getenv("AWS_REGION", "eu-central-1")

    @property
    def zooniverse_user(self):
        return os.getenv("ZOONIVERSE_USER") or os.getenv("PANOPTES_USERNAME")

    @property
    def zooniverse_password(self):
        return os.getenv("ZOONIVERSE_PASSWORD") or os.getenv("PANOPTES_PASSWORD")

    @property
    def zooniverse_project_id(self) -> int | None:
        return self._yaml_config.get("zooniverse_extraction", {}).get("project_id")

    @property
    def biigle_email(self) -> str | None:
        return os.getenv("BIIGLE_API_EMAIL")

    @property
    def biigle_token(self) -> str | None:
        return os.getenv("BIIGLE_API_TOKEN")

    @property
    def biigle_project_id(self) -> int:
        val = os.getenv("BIIGLE_PROJECT_ID")
        if val:
            return int(val)
        return self._yaml_config.get("biigle", {}).get("project_id", 3711)

    @property
    def biigle_disk_id(self) -> int:
        val = os.getenv("BIIGLE_DISK_ID")
        if val:
            return int(val)
        return self._yaml_config.get("biigle", {}).get("disk_id", 134)

    @property
    def biigle_annotation_report_type_video(self) -> int:
        return self._yaml_config.get("biigle", {}).get("annotation_report_type_video", 8)

    @property
    def biigle_annotation_report_type_images(self) -> int:
        return self._yaml_config.get("biigle", {}).get("annotation_report_type_images", 3)

    @property
    def biigle_done_labels(self) -> list:
        """Whole-file labels that indicate a volume is ready to be ingested (e.g. 'Done Volume', 'Done QA Review')."""
        return self._yaml_config.get("biigle", {}).get("done_labels", ["Done Volume", "Done QA Review"])

    @property
    def paths(self):
        return self._yaml_config.get("paths", {})

    @property
    def sub_dirs(self):
        return self.paths.get("sub_dirs", {})

    # S3 Prefixes / Local Dirs are now unified in sub_dirs

    @property
    def biigle_s3_images_prefix(self) -> str:
        """S3 prefix for Biigle JPEG frames."""
        return f"{self.base_dir}/{self.sub_dirs.get('biigle_images', 'media/biigle_images')}"

    @property
    def export_local(self):
        return str_to_bool(os.getenv("EXPORT_LOCAL"))

    @property
    def local_data_folder_path(self):
        return os.getenv("LOCAL_DATA_FOLDER_PATH", str(Path.cwd() / "data"))

    @property
    def metadata_root(self) -> str:
        """The root directory for metadata in S3."""
        return self.metadata.get("root", "spyfish_metadata")

    @property
    def sharepoint_root(self) -> str:
        """The full S3 path to sharepoint lists."""
        return f"{self.metadata_root}/{self.metadata.get('sharepoint_dir', 'sharepoint_lists')}"

    @property
    def status_root(self) -> str:
        """The full S3 path to status reports."""
        return f"{self.metadata_root}/{self.metadata.get('status_dir', 'status')}"

    @property
    def metadata_files(self) -> dict:
        """Dictionary of metadata filenames."""
        return self.metadata.get("files", {})

    @property
    def s3_sharepoint_deployment_csv(self):
        return f"{self.sharepoint_root}/{self.metadata_files.get('deployment_csv', 'BUV Deployment.csv')}"

    @property
    def test_deployment_metadata_csv(self) -> Path:
        """The path to the test deployment metadata CSV."""
        return self.project_root / self.metadata_files.get("test_deployment_csv", "manual_testing/test_deployment_metadata.csv")

    @property
    def s3_sharepoint_survey_csv(self):
        return f"{self.sharepoint_root}/{self.metadata_files.get('survey_csv', 'BUV Survey Metadata.csv')}"

    @property
    def s3_sharepoint_site_csv(self):
        return f"{self.sharepoint_root}/{self.metadata_files.get('site_csv', 'BUV Survey Sites.csv')}"

    @property
    def s3_sharepoint_species_csv(self):
        return f"{self.sharepoint_root}/{self.metadata_files.get('species_csv', 'BUV Species.csv')}"

    @property
    def s3_sharepoint_reserves_csv(self):
        return f"{self.sharepoint_root}/{self.metadata_files.get('reserves_csv', 'Marine Reserves.csv')}"

    @property
    def s3_sharepoint_annotations_legacy_experts_csv(self):
        return f"{self.sharepoint_root}/{self.metadata_files.get('legacy_experts_csv', 'BUV Annotations Legacy Experts.csv')}"

    @property
    def s3_sharepoint_path(self):
        return self.sharepoint_root

    @property
    def drop_id_column(self):
        return "DropID"

    @property
    def survey_id_column(self):
        return "SurveyID"

    @property
    def site_id_column(self):
        return "SiteID"

    @property
    def replicate_column(self):
        return "ReplicateWithinSite"

    @property
    def file_name_column(self):
        return "FileName"

    @property
    def link_to_marine_reserve_column(self):
        return "LinkToMarineReserve"

    @property
    def site_name_column(self):
        return "SiteName"

    @property
    def s3_status_path(self):
        return self.status_root

    @property
    def missing_files_filename(self):
        return "missing_files_in_aws.txt"

    @property
    def extra_files_filename(self):
        return "extra_files_in_aws.txt"

    @property
    def s3_missing_files(self):
        return f"{self.s3_status_path}/{self.missing_files_filename}"

    @property
    def s3_extra_files(self):
        return f"{self.s3_status_path}/{self.extra_files_filename}"

    @property
    def s3_deployment_status_csv(self):
        return f"{self.s3_status_path}/deployments_status.csv"

    @property
    def s3_survey_status_csv(self):
        return os.path.join(self.s3_status_path, "surveys_status.csv")


    @property
    def biigle_annotation_report_type(self):
        return 8

    @property
    def biigle_annotation_report_type_images(self):
        return 3

    @property
    def biigle_volume_report_type(self):
        return 10

    @property
    def movie_extensions(self):
        return ["avi", "mov", "mp4", "mpg", "wmv"]

    @property
    def file_presence_rules(self):
        return {
            "file_presence": {
                "bucket": self.s3_bucket,
                "s3_sharepoint_path": self.s3_sharepoint_path,
                "csv_filename": "BUV Deployment.csv",
                "csv_column_to_extract": "LinkToVideoFile",
                "column_filter": None,
                "column_value": None,
                "valid_extensions": self.movie_extensions,
                "path_prefix": "media",
            }
        }

    @property
    def validation_patterns(self):
        return {
            self.drop_id_column: r"^[A-Z]{3}_\d{8}_BUV_[A-Z]{3}_\d{3}_\d{2}$",
            self.survey_id_column: r"^[A-Z]{3}_\d{8}_BUV$",
            self.site_id_column: r"^[A-Z]{3}_\d+$",
        }

    @property
    def validation_rules(self):
        # Driven entirely by config.yaml now
        return self._yaml_config.get("validation_rules", {})

    @property
    def zooniverse_extraction(self):
        return self._yaml_config.get("zooniverse_extraction", {})

    @property
    def zooniverse_clip_length(self) -> int:
        return int(self.zooniverse_extraction.get("clip_length_seconds", 10))

    @property
    def zooniverse_health_check_count(self) -> int:
        return int(self.zooniverse_extraction.get("health_check_count", 6))

    @property
    def zooniverse_video_start_threshold(self) -> int:
        return int(self.zooniverse_extraction.get("video_start_threshold_seconds", 60))

    @property
    def zooniverse_clip_cap(self) -> int:
        return int(self.zooniverse_extraction.get("clip_cap", 50))

    @property
    def zooniverse_force_binary_strategy(self) -> bool:
        return bool(self.zooniverse_extraction.get("force_binary_strategy", False))

    @property
    def zooniverse_temporal_spacing(self) -> int:
        return int(self.zooniverse_extraction.get("temporal_spacing_seconds", 0))

    @property
    def zooniverse_binary_strategy(self) -> dict:
        return self.zooniverse_extraction.get("binary_strategy", {
            "maxn_clips": 10,
            "confusing_clips": 20,
            "empty_clips": 5,
            "start_clips": 2,
            "temporal_spacing_seconds": 10
        })

    @property
    def zooniverse_multiclass_strategy(self) -> dict:
        return self.zooniverse_extraction.get("multiclass_strategy", {
            "per_species_maxn_clips": 5,
            "per_species_confusing_clips": 10,
            "per_video_empty_clips": 3,
            "per_video_start_clips": 2,
            "temporal_spacing_seconds": 10
        })

    @property
    def ml_inference(self):
        return self._yaml_config.get("ml_inference", {})

    @property
    def limit_processing(self):
        return self.ml_inference.get("limit_processing")

    @property
    def test_drops(self):
        """Returns list of (drop_id, video_path, sampling_start, sampling_end) tuples."""
        try:
            import pandas as pd
            csv_path = self.project_root / self.test_deployment_metadata_csv
            if not csv_path.exists():
                return []
            df = pd.read_csv(csv_path)
            drops = []
            for _, row in df.iterrows():
                drops.append((
                    str(row['drop_id']),
                    str(row['video_path']),
                    int(row['sampling_start']),
                    int(row['sampling_end'])
                ))
            return drops
        except Exception as e:
            logging.error(f"Failed to load test drops: {e}")
            return []
    @property
    def training_config(self):
        return self._yaml_config.get("training", {})

    @property
    def models_root_dir(self) -> Path:
        """Local root for all models."""
        return self.project_root / self.base_dir / self.sub_dirs.get("models", "models")

    @property
    def base_model_dir(self) -> Path:
        """Directory for base models."""
        return self.models_root_dir / self.sub_dirs.get("base_model", "base_model")

    @property
    def pipeline_model_dir(self) -> Path:
        """Directory for the active pipeline model."""
        return self.models_root_dir / self.sub_dirs.get("pipeline_model", "pipeline_model")

    @property
    def trained_model_dir(self) -> Path:
        """Directory for newly trained models."""
        return self.models_root_dir / self.sub_dirs.get("trained", "trained")

    @property
    def pipeline_model_path(self) -> Path:
        """Path to the active model file. Defaults to first .pt file in the pipeline_model_dir."""
        # Check if there's a specific .pt file in the directory
        if self.pipeline_model_dir.exists():
            pt_files = list(self.pipeline_model_dir.glob("*.pt"))
            if pt_files:
                # TODO Return the most recent or first one? Usually there should only be one main one.
                return pt_files[0]

        # Fallback to a default filename
        else:
            raise FileNotFoundError(f"No model file found in {self.pipeline_model_dir}")

    @property
    def s3_model_key(self):
        """S3 key for the active ML model, mimicking local structure."""
        return f"{self.base_dir}/models/pipeline_model/{self.pipeline_model_path.name}"

    @property
    def s3_training_base_model_key(self):
        """S3 key for the starting base model."""
        # Find the base model name
        if self.base_model_dir.exists():
            pt_files = list(self.base_model_dir.glob("*.pt"))
            if pt_files:
                return f"{self.base_dir}/models/base_model/{pt_files[0].name}"
        return f"{self.base_dir}/models/base_model/model.pt"

    @property
    def s3_training_output_prefix(self):
        """S3 prefix for artifacts produced during training."""
        return f"{self.base_dir}/{self.sub_dirs.get('training', 'training')}/"

    @property
    def frame_skip(self):
        return get_required(self.ml_inference, "frame_skip", "ml_inference")

    @property
    def ml_log_interval_frames(self) -> int:
        return int(self.ml_inference.get("log_interval_frames", 10))

    @property
    def imgsz(self):
        return self.ml_inference.get("imgsz", 640)

    @property
    def confidence_threshold(self):
        return get_required(self.ml_inference, "confidence_threshold", "ml_inference")

    @property
    def maxn_confidence_threshold(self):
        return get_required(self.ml_inference, "maxn_confidence_threshold", "ml_inference")

    @property
    def interval_seconds(self):
        return self.ml_inference.get("extraction", {}).get("interval_seconds", 10)

    @property
    def orchestrator(self):
        return self._yaml_config.get("orchestrator", {})

    @property
    def media_dir(self) -> Path:
        return self.project_root / self.sub_dirs.get("media", "media")

    @property
    def is_test_run(self):
        return bool(self.orchestrator.get("is_test_run", False))

    @property
    def data_quality_dir(self) -> Path:
        return self.project_root / self.base_dir / self.sub_dirs.get("data_quality", "data_quality")

    @property
    def logs_dir(self) -> Path:
        return self.project_root / self.base_dir / self.sub_dirs.get("logs", "logs")

    @property
    def s3_data_quality_dir(self):
        return f"{self.base_dir}/{self.sub_dirs.get('data_quality', 'data_quality')}"

    @property
    def s3_annotations_dir(self):
        return f"{self.base_dir}/{self.sub_dirs.get('annotations', 'annotations')}"

    @property
    def biigle_default_fish_label_id(self) -> int:
        return int(get_required(self._yaml_config.get("biigle", {}), "default_fish_label_id"))

    @property
    def biigle_label_mapping(self) -> dict:
        mapping = self._yaml_config.get("biigle", {}).get("label_mapping")
        return mapping if mapping is not None else {}

    @property
    def biigle_default_label_tree_id(self) -> int:
        return int(get_required(self._yaml_config.get("biigle", {}), "default_label_tree_id"))

    @property
    def volume_finalize_max_retries(self) -> int:
        return int(self._yaml_config.get("biigle", {}).get("volume_finalize_max_retries", 10))

    @property
    def volume_finalize_retry_interval_secs(self) -> float:
        return float(self._yaml_config.get("biigle", {}).get("volume_finalize_retry_interval_secs", 3.0))

    @property
    def report_download_max_retries(self) -> int:
        return int(self._yaml_config.get("biigle", {}).get("report_download_max_retries", 60))

    @property
    def report_download_retry_interval_secs(self) -> float:
        return float(self._yaml_config.get("biigle", {}).get("report_download_retry_interval_secs", 2.0))


    # TODO not sure we need this,  there is structural validation performed by the DataValidator
    # (in spyfish/validation/data_validator.py) using the regex patterns defined in config.py
    # This method is now called by every path-generating helper for security
    def validate_drop_id(self, drop_id: str) -> str:
        """
        Validates that a drop_id matches the expected pattern and contains no path traversal.
        Raises ValueError if invalid.
        """
        import re
        pattern = self.validation_patterns.get(self.drop_id_column)
        if not pattern:
            # Fallback to a safe default if pattern is missing
            pattern = r"^[A-Z0-9_\-]+$"

        if not re.match(pattern, drop_id):
            raise ValueError(f"Invalid DropID format: '{drop_id}'. Must match {pattern}")

        # Explicit check for directory traversal characters
        if ".." in drop_id or "/" in drop_id or "\\" in drop_id:
            raise ValueError(f"Security Alert: Malicious DropID detected (potential path traversal): '{drop_id}'")

        return drop_id

    def get_drop_annotations_dir(self, drop_id: str) -> Path:
        """Helper to consistently get the annotations directory for a drop."""
        val_drop_id = self.validate_drop_id(drop_id)
        return self.data_quality_dir / val_drop_id / "annotations"

    def get_video_path(self, drop_id: str) -> Path:
        """Helper to get the correct video path depending on the environment."""
        val_drop_id = self.validate_drop_id(drop_id)
        return self.media_dir / f"{val_drop_id}.mp4"

    def get_maxn_csv_path(self, drop_id: str, model_name: str) -> Path:
        val_drop_id = self.validate_drop_id(drop_id)
        return self.get_drop_annotations_dir(val_drop_id) / f"{val_drop_id}_{model_name}_maxn.csv"

    def get_selections_csv_path(self, drop_id: str) -> Path:
        val_drop_id = self.validate_drop_id(drop_id)
        return self.get_drop_annotations_dir(val_drop_id) / f"{val_drop_id}_frames_selection.csv"

    def get_raw_csv_path(self, drop_id: str, model_name: str) -> Path:
        val_drop_id = self.validate_drop_id(drop_id)
        return self.get_drop_annotations_dir(val_drop_id) / f"{val_drop_id}_{model_name}_raw.csv"

    # TODO I do not think we need the target argument here
    def get_clips_dir(self, drop_id: str, target: str = "") -> Path:
        """Get the localized clips directory for a drop."""
        val_drop_id = self.validate_drop_id(drop_id)
        sub_path = f"{target}_clips" if target else "clips"
        return self.data_quality_dir / val_drop_id / sub_path

    def get_frames_dir(self, drop_id: str, target: str = "") -> Path:
        """Get the localized frames directory for a drop."""
        val_drop_id = self.validate_drop_id(drop_id)
        sub_path = f"{target}_frames" if target else "frames"
        return self.data_quality_dir / val_drop_id / sub_path

    @property
    def csv_video_file_link_column(self) -> str:
        return self.csv_mapping.get("video_file_link_column", "LinkToVideoFile")

    @property
    def csv_sampling_start_column(self) -> str:
        return self.csv_mapping.get("sampling_start_column", "SamplingStart")

    @property
    def csv_sampling_end_column(self) -> str:
        return self.csv_mapping.get("sampling_end_column", "SamplingEnd")

    @property
    def csv_clip_start_column(self) -> str:
        return self.csv_mapping.get("clip_start_column", "ClipStartRelative")

    @property
    def csv_clip_end_column(self) -> str:
        return self.csv_mapping.get("clip_end_column", "ClipEndRelative")

    @property
    def csv_clip_max_time_column(self) -> str:
        return self.csv_mapping.get("clip_max_time_column", "TimeOfMaxnMs")

    @property
    def csv_confidence_agreement_column(self) -> str:
        return self.csv_mapping.get("confidence_agreement_column", "ConfidenceAgreement")

    @property
    def csv_confusion_score_column(self) -> str:
        return self.csv_mapping.get("confusion_score_column", "ConfusionScore")

    @property
    def csv_scientific_name_column(self) -> str:
        return self.csv_mapping.get("scientific_name_column", "ScientificName")

    @property
    def csv_maxn_time_column(self) -> str:
        return self.csv_mapping.get("maxn_time_column", "TimeOfMax")

    @property
    def csv_maxn_time_ms_column(self) -> str:
        return self.csv_mapping.get("maxn_time_ms_column", "TimeOfMaxnMs")

    @property
    def csv_max_interval_column(self) -> str:
        return self.csv_mapping.get("max_interval_column", "MaxInterval")

    @property
    def csv_annotated_by_column(self) -> str:
        return self.csv_mapping.get("annotated_by_column", "AnnotatedBy")

    @property
    def csv_interval_annotation_column(self) -> str:
        return self.csv_mapping.get("interval_annotation_column", "IntervalAnnotation")

    @property
    def csv_time_seconds_column(self) -> str:
        return self.csv_mapping.get("time_seconds_column", "TimeSeconds")

    # TODO we shoudn't need the extra db, the path should mimic local, the only difference is that it will have the s3key
    @property
    def s3_db_key(self) -> str:
        """S3 key for the main pipeline database."""
        return self.get_s3_db_key("spyfish_pipeline.db")

    @property
    def annotations_db_path(self) -> Path:
        """LOCAL path to the annotations database."""
        return self.get_db_path("spyfish_annotations.db")

    @property
    def s3_annotations_db_key(self) -> str:
        """S3 key for the annotations database."""
        return self.get_s3_db_key("spyfish_annotations.db")

    @property
    def ffmpeg_config(self) -> dict:
        return self._yaml_config.get("ffmpeg", {})

    @property
    def ffmpeg_crf(self) -> str:
        return str(self.ffmpeg_config.get("crf", "22"))

    @property
    def ffmpeg_preset(self) -> str:
        return str(self.ffmpeg_config.get("preset", "fast"))

    @property
    def ffmpeg_codec(self) -> str:
        return str(self.ffmpeg_config.get("codec", "libx264"))

config = ConfigWrapper()

class PipelineStatus:
    """
    Constant string stages of the Spyfish pipeline.
    """
    # Holds and Errors
    ON_HOLD = "ON_HOLD"
    EXCLUDED = "EXCLUDED"
    ERROR = "ERROR"
    MISSING_METADATA = "MISSING_METADATA"

    # Healthy cylce
    PENDING_ARRIVAL = "PENDING_ARRIVAL"
    READY_FOR_ML = "READY_FOR_ML"
    PROCESSING_ML = "PROCESSING_ML"
    ML_COMPLETE = "ML_COMPLETE"
    AWAITING_CITSCI_CLIPS = "AWAITING_CITSCI_CLIPS"
    CITSCI_CLIPS_COMPLETE = "CITSCI_CLIPS_COMPLETE"
    AWAITING_CITSCI_FRAMES = "AWAITING_CITSCI_FRAMES"
    CITSCI_COMPLETE = "CITSCI_COMPLETE"
    AWAITING_EXPERT_REVIEW = "AWAITING_EXPERT_REVIEW"
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"

    VIDEO_PRESENT_STATUSES = [
        READY_FOR_ML, PROCESSING_ML, ML_COMPLETE,
        CITSCI_CLIPS_COMPLETE, CITSCI_COMPLETE, AWAITING_EXPERT_REVIEW, PIPELINE_COMPLETE
    ]

    STAGE_ORDER = [
        ("PENDING_ARRIVAL",        "⏳ Pending Arrival",        "Waiting for video to arrive in S3"),
        ("READY_FOR_ML",           "🤖 Ready for ML",           "Video present, queued for ML inference"),
        ("PROCESSING_ML",          "⚙️ Processing ML",          "ML inference actively running"),
        ("ML_COMPLETE",            "✅ ML Complete",            "ML done, awaiting next steps"),
        ("AWAITING_CITSCI_CLIPS",  "⏳ Awaiting CitSci Clips",  "Queued for Zooniverse clip selection"),
        ("CITSCI_CLIPS_COMPLETE",  "✅ CitSci Clips Complete",  "CitSci clips extracted, awaiting CitSci annotations"),
        ("AWAITING_CITSCI_FRAMES", "⏳ Awaiting CitSci Frames", "Zooniverse frames extracted, awaiting annotations"),
        ("CITSCI_COMPLETE",        "✅ CitSci Complete",        "CitSci fully done"),
        ("AWAITING_EXPERT_REVIEW", "🔬 Awaiting Expert",        "Volume created in Biigle, awaiting expert annotation"),
        ("PIPELINE_COMPLETE",      "🎉 Pipeline Complete",      "Fully processed and synced from Biigle"),
        ("ON_HOLD",                "⏸️ On Hold",                "Paused for investigation"),
        ("EXCLUDED",               "🚫 Excluded",               "Bad deployment, not processing"),
        ("ERROR",                  "❌ Error",                  "Failed a pipeline step"),
        ("MISSING_METADATA",       "⚠️ Missing Metadata",       "Required metadata absent"),
    ]
