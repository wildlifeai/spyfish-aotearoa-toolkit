import logging
import os
import time
from pathlib import Path

from spyfish.config.wrapper import config
from spyfish.storage.s3_handler import S3Handler


def _download_db_if_newer(s3_key: str, local_path: Path, label: str) -> bool:
    """Download a SQLite DB from S3 to `local_path` if S3 has a newer version.

    Returns True if the local file is up-to-date (downloaded, already newer,
    or S3 has nothing yet). Returns False on an unexpected error.

    `label` is used only in log messages (e.g. "pipeline database").
    """
    s3 = S3Handler()
    try:
        last_modified = s3.get_object_last_modified(s3_key)
        if last_modified is None:
            logging.info(f"{label.capitalize()} not found on S3. Starting fresh.")
            return True

        s3_mtime = last_modified.timestamp()

        if local_path.exists() and local_path.stat().st_mtime >= s3_mtime:
            logging.info(f"Local {label} is up-to-date with S3. Skipping download.")
            return True
    except Exception as e:
        logging.error(f"Error checking {label} on S3: {e}")
        return False

    logging.info(f"Downloading {label} to {local_path} (S3 is newer)...")
    try:
        s3.download_object_from_s3(s3_key, str(local_path))
        # Match local mtime to S3 so the next run's timestamp check skips correctly.
        os.utime(local_path, (time.time(), s3_mtime))
        logging.info(f"{label.capitalize()} downloaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to download {label}: {e}")
        return False


def _upload_db(s3_key: str, local_path: Path, label: str) -> bool:
    """Upload a SQLite DB from `local_path` to S3.

    Returns True on success, False if the local file is missing or upload fails.
    """
    if not local_path.exists():
        logging.warning(
            f"{label.capitalize()} file {local_path} does not exist. Skipping upload."
        )
        return False

    logging.info(f"Uploading {label} to s3://{config.s3_bucket}/{s3_key}...")
    try:
        s3 = S3Handler()
        s3.upload_file_to_s3(str(local_path), s3_key)
        logging.info(f"{label.capitalize()} uploaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to upload {label}: {e}")
        return False


# ── Public wrappers ──
# One-liners over the generic helpers. Kept as distinct functions so callers
# can name the specific DB they mean — changing the label or wiring up a third
# DB is a single-line edit.


def download_db() -> bool:
    return _download_db_if_newer(config.s3_db_key, config.db_path, "pipeline database")


def upload_db() -> bool:
    return _upload_db(config.s3_db_key, config.db_path, "pipeline database")


def download_annotations_db() -> bool:
    return _download_db_if_newer(
        config.s3_annotations_db_key, config.annotations_db_path, "annotations database"
    )


def upload_annotations_db() -> bool:
    return _upload_db(
        config.s3_annotations_db_key, config.annotations_db_path, "annotations database"
    )


def sync_annotations() -> bool:
    """
    Synchronizes the local nested annotations and images to S3.
    Excludes .mp4 files — Zooniverse clips are uploaded directly to Zooniverse,
    and raw BUV footage already lives in media/ on S3.
    """
    s3 = S3Handler()
    local_dq_dir = config.deployment_data_dir
    s3_prefix = config.s3_deployment_data_dir

    # Start with global exclude
    filters = ["--exclude", "*"]

    # Include metadata and annotations
    filters += ["--include", "*/annotations/*.csv"]
    filters += ["--include", "*/annotations/*.json"]
    filters += ["--include", "*/clips/*.csv"]

    # Include images (standardize on .jpg/.jpeg/.png)
    image_patterns = ["*/qa_frames/*", "*/frames/*"]
    for pattern in image_patterns:
        filters += ["--include", f"{pattern}.jpg"]
        filters += ["--include", f"{pattern}.jpeg"]
        filters += ["--include", f"{pattern}.png"]

    # NOTE: training/ outputs are intentionally NOT synced here.
    # The directory contains a mix of useful artifacts (model weights, metrics,
    # data.yaml) and noise (YOLO debug-image dumps, tensorboard event files,
    # symlinked dataset trees that re-resolve to deployment_data/ frames,
    # intermediate label-staging dirs, spot-check audits). A proper home for
    # training artifacts in S3 — and the right include/exclude rules — is a
    # design decision for later. Promoted models still flow to S3 via the
    # models/ prefix below.
    # See claude_docs/todo.md "ML pipeline" entry.

    # Include promoted models
    models_prefix = "models"
    filters += ["--include", f"{models_prefix}/**"]

    # Exclude external Biigle volumes that live under extra_no_survey_id/.
    # Those are pre-curated bulk imports already hosted on Biigle's storage
    # (disk-134) — re-uploading them to the project bucket duplicates GBs of
    # frames + annotations for no benefit. Drop annotations + class_map + frames
    # were synced once at download time; if a volume needs to be re-shared,
    # re-pull from Biigle rather than relying on this bucket as backup.
    # Filter ordering matters: this `--exclude` must come AFTER the includes
    # above so it overrides them for paths under extra_no_survey_id/.
    filters += ["--exclude", "extra_no_survey_id/*"]

    # This sync is additive only (no --delete flag is passed to aws s3 sync inside sync_local_to_s3)
    return s3.sync_local_to_s3(str(local_dq_dir), s3_prefix, filters=filters)


def sync_pipeline_results() -> bool:
    """
    Comprehensive sync of all pipeline outputs to S3:
    1. Uploads both databases (pipeline and annotations)
    2. Syncs the data_quality directory (CSVs, images, models — no .mp4s)
    """
    logging.info("Starting consolidated pipeline sync to S3...")
    success = True

    # 1. Databases
    if not upload_db():
        logging.error("Failed to upload pipeline database.")
        success = False

    if not upload_annotations_db():
        logging.error("Failed to upload annotations database.")
        success = False

    # 2. Annotations & Selections
    if not sync_annotations():
        logging.error("Failed to sync annotations directory.")
        success = False

    if success:
        logging.info("Consolidated S3 sync completed successfully.")
    else:
        logging.warning("Consolidated S3 sync completed with errors.")

    return success
