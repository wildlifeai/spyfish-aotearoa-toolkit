import logging
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv


def get_required(config_dict, key: str, section: str = ""):
    """Fetch a required config value, raising if missing or None.

    Works with both Python dicts and objects (e.g. `ConfigWrapper`). A value
    of `None` is treated as missing — config.yaml keys that resolve to None
    are almost always a sign of a bad edit or placeholder. Use a falsy sentinel
    like `""` or `0` instead if you genuinely want an absent-but-set value.

    Raises `KeyError` with a config.yaml-style path for easy debugging, e.g.
    `config.yaml [zooniverse.min_votes]`.
    """
    location = f"config.yaml [{section}.{key}]" if section else f"config.yaml [{key}]"

    if isinstance(config_dict, dict):
        if key not in config_dict or config_dict[key] is None:
            raise KeyError(f"Required config key missing: {location}")
        return config_dict[key]

    # Object access (ConfigWrapper, dataclass, etc.)
    if not hasattr(config_dict, key) or getattr(config_dict, key) is None:
        raise KeyError(f"Required config key missing: {location}")
    return getattr(config_dict, key)


def load_env_wrapper() -> None:
    env_path = find_dotenv()
    if env_path:
        logging.info(f"Loading .env file from: {env_path}")
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        logging.warning(
            ".env file not found. Environment variables might not be loaded."
        )


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


class InvalidTransitionError(Exception):
    """Raised when a pipeline status transition is not permitted."""


class VideoPresence:
    """Tracks whether a deployment's video is present in S3.

    Stored in the video_presence column. Updated on every ingest run to reflect
    current S3 state — if the video disappears from S3, it reverts to ABSENT.
    """

    PRESENT = "present"  # Video in S3 with a directly-downloadable storage class
    ARCHIVED = "archived"  # Video in S3 but in DEEP_ARCHIVE — needs restore
    ABSENT = "absent"  # Video not found in S3
    NO_VIDEO_BAD_DEP = "no_video_bad_dep"  # Bad deployment — no video expected


class IngestStatus:
    """Data-quality gate for a deployment.

    Set only at ingestion — never advanced. Not in the `SECTIONS` registry
    because it has no `advance_*_status` method. All `get_deployments_eligible()`
    queries filter `ingest_status = 'ok'`, so anything else is invisible to stages.
    """

    COLUMN = "ingest_status"
    OK = "ok"
    EXCLUDED = "excluded"
    METADATA_ERROR = "metadata_error"
    VALIDATION_ERROR = "validation_error"
    REMOVED = "removed"


class MlStatus:
    COLUMN = "ml_status"

    PENDING = "ml_pending"
    READY = "ml_ready"
    RUNNING = "ml_running"
    COMPLETE = "ml_complete"
    ERROR = "ml_error"
    SKIPPED = "ml_skipped"

    VALID_TRANSITIONS: dict = {
        PENDING: {READY, SKIPPED},
        READY: {RUNNING, SKIPPED},
        RUNNING: {COMPLETE, ERROR},
        ERROR: {READY},
    }


class CitSciStatus:
    COLUMN = "citsci_status"

    PENDING = "citsci_pending"
    CLIPS_UPLOADED = "citsci_clips_uploaded"
    CLIPS_DONE = "citsci_clips_done"
    FRAMES_UPLOADED = "citsci_frames_uploaded"
    FRAMES_DONE = "citsci_frames_done"
    COMPLETE = "citsci_complete"
    SKIPPED = "citsci_skipped"
    ERROR = "citsci_error"

    # TODO — CLIPS_DONE / FRAMES_DONE retirement gates not yet wired.
    #
    # Current state:
    #   - zooniverse-clips     : pending         → clips_uploaded
    #   - zooniverse-images    : clips_uploaded  → frames_uploaded     (bypass gate)
    #   - zooniverse-sync      : frames_uploaded → complete             (bypass gate)
    # The "bypass" transitions are kept in VALID_TRANSITIONS so the pipeline runs
    # today. sync_zooniverse_drop() only reads the bundled {drop_id}_zooniverse_maxn.csv
    # written by spyfish.zooniverse.live_extract and ingests it
    # as citsci annotations — it does not distinguish clip vs frame subject retirement.
    #
    # When the Caesar retirement check is wired up, advance through the gates:
    #   clips_uploaded  → clips_done    once clip subjects are retired + parsed
    #   clips_done      → frames_uploaded (or → complete for skip-frames path)
    #   frames_uploaded → frames_done   once frame subjects are retired + parsed
    #   frames_done     → complete
    # Needed to make this work:
    #   1. Split subject_completion_from_csv/api() by subject_type in
    #      spyfish/zooniverse/parse_classifications.py (the "clip"/"frame" field is
    #      already parsed per row — see parse_classifications ~line 314).
    #   2. Either emit two MaxN CSVs (clip phase, frame phase) or parameterise
    #      ingest_zooniverse_annotations() by subject_type.
    #   3. Replace the single sync stage with two stages: one advancing clips_uploaded→
    #      clips_done, one frames_uploaded → frames_done.
    # At that point, DELETE the bypass edges marked below so the pipeline is forced
    # through the gates.
    VALID_TRANSITIONS: dict = {
        PENDING: {CLIPS_UPLOADED, SKIPPED},
        CLIPS_UPLOADED: {
            CLIPS_DONE,  # retirement-gate path (future)
            FRAMES_UPLOADED,  # bypass: remove once retirement gate wired
            ERROR,
        },
        CLIPS_DONE: {FRAMES_UPLOADED, COMPLETE, ERROR},
        FRAMES_UPLOADED: {
            FRAMES_DONE,  # retirement-gate path (future)
            COMPLETE,  # bypass: remove once retirement gate wired
            ERROR,
        },
        FRAMES_DONE: {COMPLETE, ERROR},
        ERROR: {CLIPS_UPLOADED, FRAMES_UPLOADED},  # retries
    }


class BiigleStatus:
    COLUMN = "biigle_status"

    PENDING = "expert_pending"
    UPLOADED = "expert_uploaded"
    COMPLETE = "expert_complete"
    SKIPPED = "expert_skipped"
    ERROR = "expert_error"

    VALID_TRANSITIONS: dict = {
        PENDING: {UPLOADED, SKIPPED},
        UPLOADED: {COMPLETE, ERROR},
        ERROR: {PENDING},
    }


class ReportingStatus:
    COLUMN = "reporting_status"

    PENDING = "reporting_pending"
    COMPLETE = "reporting_complete"
    ERROR = "reporting_error"

    VALID_TRANSITIONS: dict = {
        PENDING: {COMPLETE},
        ERROR: {PENDING},
    }


# ── Section registry ──────────────────────────────────────────────────────
#
# Single source of truth for everything that needs to iterate over sections
# or look one up by its DB column name. To add a new section, define a new
# status class with COLUMN, ERROR, and VALID_TRANSITIONS, then add it to
# SECTION_STATUSES — no other lookup tables need updating.
#
# `ERROR` is used in two places: the section status column value, AND the
# `validation_errors.ErrorType` discriminator. Same string, one concept.
#
# Consumers:
#   - DatabaseManager.advance_status()   — uses VALID_TRANSITIONS for checks
#                                          and ERROR for clearing validation_errors
#   - StageRunner._run_drop_stage()      — sets section to ERROR on exception

SECTION_STATUSES: tuple = (MlStatus, CitSciStatus, BiigleStatus, ReportingStatus)

SECTIONS: dict[str, type] = {s.COLUMN: s for s in SECTION_STATUSES}
