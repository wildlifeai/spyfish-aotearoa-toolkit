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
    env_path = find_dotenv()
    if env_path:
        logging.info(f"Loading .env file from: {env_path}")
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        logging.warning(".env file not found. Environment variables might not be loaded.")

load_env_wrapper()

def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Singleton loaded config
_YAML_CONFIG = load_config()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

class BaseConfig:
    def __init__(self):
        self._yaml_config = _YAML_CONFIG
        self._project_root = _PROJECT_ROOT

    def get_section(self, section_name: str, default: dict = None) -> dict:
        if default is None:
            default = {}
        return self._yaml_config.get(section_name, default)

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def is_test_run(self):
        orchestrator = self.get_section("orchestrator")
        return bool(get_required(orchestrator, "is_test_run", "orchestrator"))

class PipelineStatus:
    """Constant string stages of the Spyfish pipeline."""
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
