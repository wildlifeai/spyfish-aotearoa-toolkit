import logging
import os

from spyfish.config.wrapper import config
from spyfish.storage.s3_handler import S3Handler


def download_db() -> bool:
    """
    Downloads the pipeline database from S3 if it exists.
    Returns True if successfully downloaded or if it doesn't exist on S3 yet.
    """
    s3 = S3Handler()
    local_path = config.db_path
    s3_key = config.s3_db_key
    bucket = config.s3_bucket

    # Check if object exists and get metadata
    try:
        last_modified = s3.get_object_last_modified(s3_key)
        if last_modified is None:
            logging.info("Database not found on S3. Starting fresh.")
            return True

        s3_mtime = last_modified.timestamp()

        # If local file exists, check if it's already newer or same as S3
        if local_path.exists():
            local_mtime = local_path.stat().st_mtime
            if local_mtime >= s3_mtime:
                logging.info("Local database is up-to-date with S3. Skipping download.")
                return True

    except Exception as e:
        logging.error(f"Error checking database on S3: {e}")
        return False

    logging.info(f"Downloading database to {local_path} (S3 is newer)...")
    try:
        s3.download_object_from_s3(s3_key, str(local_path))
        # Ensure local mtime matches S3 so we don't re-download next time
        import time

        os.utime(local_path, (time.time(), s3_mtime))
        logging.info("Database downloaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to download database: {e}")
        return False


def upload_db() -> bool:
    """
    Uploads the pipeline database to S3.
    """
    if config.is_test_run:
        logging.info("Skipping database upload in test run.")
        return True
    s3 = S3Handler()
    local_path = config.db_path
    s3_key = config.s3_db_key

    if not local_path.exists():
        logging.warning(f"Database file {local_path} does not exist. Skipping upload.")
        return False

    logging.info(f"Uploading database to s3://{config.s3_bucket}/{s3_key}...")
    try:
        s3.upload_file_to_s3(str(local_path), s3_key)
        logging.info("Database uploaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to upload database: {e}")
        return False


def download_annotations_db() -> bool:
    """
    Downloads the annotations database from S3 if it doesn't exist locally or S3 is newer.
    Returns True if successful or not needed.
    """
    s3_key = config.s3_annotations_db_key
    local_path = config.annotations_db_path

    try:
        s3 = S3Handler()
        last_modified = s3.get_object_last_modified(s3_key)
        if last_modified is None:
            logging.info("Annotations database not found on S3 yet. Starting fresh.")
            return True

        s3_mtime = last_modified.timestamp()

        if local_path.exists():
            local_mtime = local_path.stat().st_mtime
            if local_mtime >= s3_mtime:
                logging.info(
                    "Local annotations database is up-to-date with S3. Skipping download."
                )
                return True

        logging.info("Downloading annotations database...")
        s3.download_object_from_s3(s3_key, str(local_path))
        import time

        os.utime(local_path, (time.time(), s3_mtime))
        logging.info("Annotations database downloaded successfully.")
        return True

    except Exception as e:
        logging.warning(f"Could not download annotations database: {e}")
        return False


def upload_annotations_db() -> bool:
    """
    Uploads the annotations database to S3.
    """
    s3_key = config.s3_annotations_db_key
    local_path = config.annotations_db_path

    if not local_path.exists():
        logging.warning(
            f"Annotations database {local_path} does not exist. Skipping upload."
        )
        return False

    logging.info(
        f"Uploading annotations database to s3://{config.s3_bucket}/{s3_key}..."
    )
    try:
        s3 = S3Handler()
        s3.upload_file_to_s3(str(local_path), s3_key)
        logging.info("Annotations database uploaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to upload annotations database: {e}")
        return False


def sync_annotations() -> bool:
    """
    Synchronizes the local nested annotations and images to S3.
    Excludes .mp4 files — Zooniverse clips are uploaded directly to Zooniverse,
    and raw BUV footage already lives in media/ on S3.
    """
    s3 = S3Handler()
    local_dq_dir = config.data_quality_dir
    s3_prefix = config.s3_data_quality_dir

    # Start with global exclude
    filters = ["--exclude", "*"]

    # Include metadata and annotations
    filters += ["--include", "*/annotations/*.csv"]
    filters += ["--include", "*/annotations/*.json"]
    filters += ["--include", "*/clips/*.csv"]

    # Include images (standardize on .jpg/.jpeg/.png)
    image_patterns = ["*/qa_frames/*", "*/biigle_cache/*", "*/frames/*"]
    for pattern in image_patterns:
        filters += ["--include", f"{pattern}.jpg"]
        filters += ["--include", f"{pattern}.jpeg"]
        filters += ["--include", f"{pattern}.png"]

    # Include training results (models, metrics, curves)
    training_prefix = "training"
    filters += ["--include", f"{training_prefix}/**"]

    # Include promoted models
    models_prefix = "models"
    filters += ["--include", f"{models_prefix}/**"]

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
    if not config.is_test_run:
        if not sync_annotations():
            logging.error("Failed to sync annotations directory.")
            success = False

    if success:
        logging.info("Consolidated S3 sync completed successfully.")
    else:
        logging.warning("Consolidated S3 sync completed with errors.")

    return success
