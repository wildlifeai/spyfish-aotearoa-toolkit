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
    def db_path(self) -> Path:
        return self._project_root / "spyfish_pipeline.db"

    @property
    def csv_mapping(self):
        return self._yaml_config.get("csv_mapping", {})

    @property
    def storage(self):
        return self._yaml_config.get("storage", {})

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
        val = os.getenv("ZOONIVERSE_PROJECT_ID")
        return int(val) if val else None

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
    def biigle_volume_report_type(self) -> int:
        return self._yaml_config.get("biigle", {}).get("volume_report_type", 10)

    @property
    def biigle_s3_images_prefix(self) -> str:
        """S3 prefix for Biigle JPEG frames — append /{survey_id}/{drop_id}/ at runtime."""
        return self._yaml_config.get("biigle", {}).get("s3_images_prefix", "process_files/media/biigle_images")

    @property
    def biigle_s3_clips_prefix(self) -> str:
        """S3 prefix for Biigle video clips — append /{survey_id}/{drop_id}/ at runtime."""
        return self._yaml_config.get("biigle", {}).get("s3_clips_prefix", "process_files/media/biigle_clips")

    @property
    def export_local(self):
        return str_to_bool(os.getenv("EXPORT_LOCAL"))

    @property
    def local_data_folder_path(self):
        return os.getenv("LOCAL_DATA_FOLDER_PATH", str(Path.cwd() / "data"))

    @property
    def s3_bucket(self):
        return self._yaml_config.get("storage", {}).get("bucket_name")

    @property
    def s3_spyfish_metadata(self):
        return "spyfish_metadata"

    @property
    def s3_sharepoint_path(self):
        return os.path.join(self.s3_spyfish_metadata, "sharepoint_lists")

    @property
    def s3_sharepoint_deployment_csv(self):
        return os.path.join(self.s3_sharepoint_path, "BUV Deployment.csv")

    @property
    def s3_sharepoint_survey_csv(self):
        return os.path.join(self.s3_sharepoint_path, "BUV Survey Metadata.csv")

    @property
    def s3_sharepoint_site_csv(self):
        return os.path.join(self.s3_sharepoint_path, "BUV Survey Sites.csv")

    @property
    def s3_sharepoint_species_csv(self):
        return os.path.join(self.s3_sharepoint_path, "BUV Species.csv")

    @property
    def s3_sharepoint_reserves_csv(self):
        return os.path.join(self.s3_sharepoint_path, "Marine Reserves.csv")

    @property
    def s3_sharepoint_annotations_legacy_experts_csv(self):
        return os.path.join(self.s3_sharepoint_path, "BUV Annotations Legacy Experts.csv")

    @property
    def s3_sharepoint_test_csv(self):
        return os.path.join(self.s3_sharepoint_path, "Test.csv")

    @property
    def s3_kso_path(self):
        return os.path.join(self.s3_spyfish_metadata, "kso_csvs")

    @property
    def s3_kso_annotations_csv(self):
        return os.path.join(self.s3_kso_path, "annotations_buv_doc.csv")

    @property
    def s3_kso_movie_csv(self):
        return os.path.join(self.s3_kso_path, "movies_buv_doc.csv")

    @property
    def s3_kso_site_csv(self):
        return os.path.join(self.s3_kso_path, "sites_buv_doc.csv")

    @property
    def s3_kso_species_csv(self):
        return os.path.join(self.s3_kso_path, "species_buv_doc.csv")

    @property
    def s3_kso_survey_csv(self):
        return os.path.join(self.s3_kso_path, "surveys_buv_doc.csv")

    @property
    def s3_kso_test_csv(self):
        return os.path.join(self.s3_kso_path, "test_buv_doc.csv")

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
        return os.path.join(self.s3_spyfish_metadata, "status")

    @property
    def errors_filename(self):
        return "data_errors.csv"

    @property
    def missing_files_filename(self):
        return "missing_files_in_aws.txt"

    @property
    def extra_files_filename(self):
        return "extra_files_in_aws.txt"

    @property
    def s3_errors_csv(self):
        return os.path.join(self.s3_status_path, self.errors_filename)

    @property
    def s3_missing_files(self):
        return os.path.join(self.s3_status_path, self.missing_files_filename)

    @property
    def s3_extra_files(self):
        return os.path.join(self.s3_status_path, self.extra_files_filename)

    @property
    def s3_deployment_status_csv(self):
        return os.path.join(self.s3_status_path, "deployments_status.csv")

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
    def ml_inference(self):
        return self._yaml_config.get("ml_inference", {})

    @property
    def limit_processing(self):
        return self.ml_inference.get("limit_processing")

    @property
    def test_drops(self):
        """Returns list of (drop_id, drop_key, sampling_start, sampling_end) tuples."""
        try:
            from spyfish.test_setup import TEST_DROPS
            return TEST_DROPS
        except ImportError:
            return []

    @property
    def model_path(self):
        return self.ml_inference.get("model_path")

    @property
    def mock_model_path(self):
        return self.ml_inference.get("mock_model_path")

    @property
    def frame_skip(self):
        return get_required(self.ml_inference, "frame_skip", "ml_inference")

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
    def mock_video_dir(self):
        return self.orchestrator.get("mock_video_dir", "mock_media")

    @property
    def is_test_run(self):
        return bool(self.orchestrator.get("is_test_run", False))

    @property
    def is_local(self):
        return bool(self.orchestrator.get("is_local", False))

    @property
    def local_manifest_dir_path(self):
        return self.orchestrator.get("local_manifest_dir_path", "process_files/annotations/")

    @property
    def local_data_quality_dir(self):
        return self.orchestrator.get("local_data_quality_dir", "process_files/data_quality/")

    @property
    def local_manifest_name(self):
        return self.orchestrator.get("local_manifest_name", "videos_to_ml.csv")

    @property
    def local_logs_dir(self):
        return self.orchestrator.get("local_logs_dir", "process_files/logs")

    @property
    def s3_annotations_dir(self):
        return self.orchestrator.get("s3_annotations_dir", "process_files/annotations")

    @property
    def nesi_video_dir(self):
        return self.orchestrator.get("nesi_video_dir")

    @property
    def s3_db_key(self) -> str:
        return self.orchestrator.get("s3_db_key", "process_files/spyfish_pipeline.db")

    @property
    def annotations_db_path(self) -> Path:
        return self.project_root / "process_files" / "spyfish_annotations.db"

    @property
    def s3_annotations_db_key(self) -> str:
        return "process_files/spyfish_annotations.db"

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
    READY_FOR_CITSCI = "READY_FOR_CITSCI"
    PROCESSING_CITSCI = "PROCESSING_CITSCI"
    CITSCI_COMPLETE = "CITSCI_COMPLETE"
    READY_FOR_EXPERT = "READY_FOR_EXPERT"
    PROCESSING_EXPERT = "PROCESSING_EXPERT"
    EXPERT_COMPLETE = "EXPERT_COMPLETE"
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"

    VIDEO_PRESENT_STATUSES = [
        READY_FOR_ML, PROCESSING_ML, ML_COMPLETE,
        READY_FOR_CITSCI, PROCESSING_CITSCI, CITSCI_COMPLETE,
        READY_FOR_EXPERT, PROCESSING_EXPERT, EXPERT_COMPLETE,
        PIPELINE_COMPLETE
    ]

    STAGE_ORDER = [
        ("PENDING_ARRIVAL",      "⏳ Pending Arrival",       "Waiting for video to arrive in S3"),
        ("READY_FOR_ML",         "🤖 Ready for ML",          "Video present, queued for ML inference"),
        ("PROCESSING_ML",        "⚙️ Processing ML",         "ML inference actively running"),
        ("ML_COMPLETE",          "✅ ML Complete",            "ML done, awaiting Zooniverse/Biigle"),
        ("READY_FOR_CITSCI",     "🌍 Ready for CitSci",      "Queued for citizen science upload"),
        ("PROCESSING_CITSCI",    "⚙️ Processing CitSci",    "CitSci workflow running"),
        ("CITSCI_COMPLETE",      "✅ CitSci Complete",       "CitSci done, awaiting expert review"),
        ("READY_FOR_EXPERT",     "🔬 Ready for Expert",      "Queued for expert annotation in Biigle"),
        ("PROCESSING_EXPERT",    "⚙️ Processing Expert",    "Expert annotation in progress"),
        ("EXPERT_COMPLETE",      "✅ Expert Complete",       "Expert done, syncing annotations"),
        ("PIPELINE_COMPLETE",    "🎉 Pipeline Complete",     "Fully processed"),
        ("ON_HOLD",              "⏸️ On Hold",               "Paused for investigation"),
        ("EXCLUDED",             "🚫 Excluded",              "Bad deployment, not processing"),
        ("ERROR",                "❌ Error",                 "Failed a pipeline step"),
        ("MISSING_METADATA",     "⚠️ Missing Metadata",     "Required metadata absent"),
    ]
