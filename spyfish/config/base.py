import logging
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv


def get_required(config_dict, key: str, section: str = ""):
    """Fetch a required config value, raising if missing or None.

    Works with both Python dicts and objects (e.g. `ConfigWrapper`). A value
    of `None` is treated as missing, config.yaml keys that resolve to None
    are almost always a sign of a bad edit or placeholder. Use a falsy sentinel
    like `""` or `0` instead if you genuinely want an absent-but-set value.

    Raises `KeyError` with a config.yaml-style path for easy debugging, e.g.
    `config.yaml [zooniverse.min_agreement_pct]`.
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
    current S3 state, if the video disappears from S3, it reverts to ABSENT.
    """

    PRESENT = "present"  # Video in S3 with a directly-downloadable storage class
    ARCHIVED = "archived"  # Video in S3 but in DEEP_ARCHIVE, needs restore
    ABSENT = "absent"  # Video not found in S3
    NO_VIDEO_BAD_DEP = "no_video_bad_dep"  # Bad deployment, no video expected


class IngestStatus:
    """Data-quality gate for a deployment.

    Set only at ingestion, never advanced. Not in the `SECTIONS` registry
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
    COMPLETE = "citsci_complete"
    SKIPPED = "citsci_skipped"
    ERROR = "citsci_error"

    # Happy path: pending → clips_uploaded → complete (zooniverse-sync checks retirement).
    # Frame subjects intentionally not modelled, see git history if reintroducing.
    VALID_TRANSITIONS: dict = {
        PENDING: {CLIPS_UPLOADED, SKIPPED},
        CLIPS_UPLOADED: {COMPLETE, ERROR},
        ERROR: {CLIPS_UPLOADED},
    }


class ExpertStatus:
    """Expert-review state for a deployment.

    Source-agnostic: a drop is `expert_complete` once it has expert annotations
    from any path (BIIGLE round-trip, legacy CSV ingest, future direct review).
    Provenance lives on each annotation row's `external_id` field, not in this
    status. `PENDING → COMPLETE` is allowed for non-BIIGLE direct paths;
    `PENDING → UPLOADED → COMPLETE` is the BIIGLE pipeline path.
    """

    COLUMN = "expert_status"

    PENDING = "expert_pending"
    UPLOADED = "expert_uploaded"
    COMPLETE = "expert_complete"
    SKIPPED = "expert_skipped"
    ERROR = "expert_error"

    VALID_TRANSITIONS: dict = {
        PENDING: {UPLOADED, SKIPPED, COMPLETE},
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
# SECTION_STATUSES, no other lookup tables need updating.
#
# `ERROR` is used in two places: the section status column value, AND the
# `validation_errors.ErrorType` discriminator. Same string, one concept.
#
# Consumers:
#   - DatabaseManager.advance_status()  , uses VALID_TRANSITIONS for checks
#                                          and ERROR for clearing validation_errors
#   - StageRunner._run_drop_stage()     , sets section to ERROR on exception

SECTION_STATUSES: tuple = (MlStatus, CitSciStatus, ExpertStatus, ReportingStatus)

SECTIONS: dict[str, type] = {s.COLUMN: s for s in SECTION_STATUSES}

# Every status column, including ingest_status, which is absent from SECTIONS
# because it is never advanced, but still needs its values policed at the edges.
STATUS_CLASSES: dict[str, type] = {IngestStatus.COLUMN: IngestStatus, **SECTIONS}

# column name → the values that column is allowed to hold. Any path that accepts
# a status from outside the code (the --set-targets CSV, the set_status CLI) must
# check against this: advance_status() validates transitions, not vocabulary, and
# update_section_status() deliberately validates neither.
SECTION_VALUES: dict[str, list] = {
    col: sorted(
        value
        for name, value in vars(cls).items()
        if not name.startswith("_") and name != "COLUMN" and isinstance(value, str)
    )
    for col, cls in STATUS_CLASSES.items()
}

# `scientific_name` on a row that records "this deployment was reviewed and
# nothing was seen". Every annotation source writes one, so a null deployment is
# an explicit statement rather than an absence of rows, which is indistinguishable
# from "this source never looked".
#
# A named value rather than SQL NULL: a null vanishes from a GROUP BY, does not
# match an `IN` filter and reads as missing data on a dashboard, so the one row
# that says "we checked" is exactly the row most likely to be dropped in
# aggregation. In code, not config, for the same reason as the status values,
# readers match on the exact string.
NULL_DEPLOYMENT = "NULL DEPLOYMENT"
