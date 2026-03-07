import logging
import os
from pathlib import Path

from spyfish.config import config
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
        from botocore.exceptions import ClientError
        response = s3.s3.head_object(Bucket=bucket, Key=s3_key)
        s3_mtime = response['LastModified'].timestamp()

        # If local file exists, check if it's already newer or same as S3
        if local_path.exists():
            local_mtime = local_path.stat().st_mtime
            if local_mtime >= s3_mtime:
                logging.info("Local database is up-to-date with S3. Skipping download.")
                return True

    except ClientError as e:
        if e.response['Error']['Code'] in ["404", "403"]:
            logging.info("Database not found on S3. Starting fresh.")
            return True
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
        from botocore.exceptions import ClientError
        s3 = S3Handler()
        response = s3.s3.head_object(Bucket=config.s3_bucket, Key=s3_key)
        s3_mtime = response['LastModified'].timestamp()

        if local_path.exists():
            local_mtime = local_path.stat().st_mtime
            if local_mtime >= s3_mtime:
                logging.info("Local annotations database is up-to-date with S3. Skipping download.")
                return True

        logging.info(f"Downloading annotations database...")
        s3.download_object_from_s3(s3_key, str(local_path))
        import time
        os.utime(local_path, (time.time(), s3_mtime))
        logging.info("Annotations database downloaded successfully.")
        return True

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ["404", "403"]:
            logging.info("Annotations database not found on S3 yet. Starting fresh.")
            return True
        logging.error(f"Error downloading annotations database: {e}")
        return False
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
        logging.warning(f"Annotations database {local_path} does not exist. Skipping upload.")
        return False

    logging.info(f"Uploading annotations database to s3://{config.s3_bucket}/{s3_key}...")
    try:
        s3 = S3Handler()
        s3.upload_file_to_s3(str(local_path), s3_key)
        logging.info("Annotations database uploaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to upload annotations database: {e}")
        return False

def sync_annotations(upload_videos: bool = False) -> bool:
    """
    Synchronizes the local nested annotations and images to S3.
    By default, excludes large video files to save bandwidth and storage.

    Args:
        upload_videos: If True, also uploads .mp4 files. Default False.
    """
    s3 = S3Handler()
    local_dq_dir = config.data_quality_dir
    s3_prefix = config.s3_data_quality_dir

    # Start with global exclude
    filters = ["--exclude", "*"]

    # Include metadata and annotations
    filters += ["--include", "*/annotations/*.csv"]
    filters += ["--include", "*/annotations/*.json"]
    filters += ["--include", "*/zooniverse_clips/*.csv"]

    # Include images (standardize on .jpg/.jpeg/.png)
    image_patterns = ["*/qa_frames/*", "*/biigle_cache/*", "*/zooniverse_images/*"]
    for pattern in image_patterns:
        filters += ["--include", f"{pattern}.jpg"]
        filters += ["--include", f"{pattern}.jpeg"]
        filters += ["--include", f"{pattern}.png"]

    # Optionally include videos
    if upload_videos:
        filters += ["--include", "*.mp4"]

    return s3.sync_local_to_s3(str(local_dq_dir), s3_prefix, filters=filters)

def sync_pipeline_results(upload_videos: bool = False) -> bool:
    """
    Comprehensive sync of all pipeline outputs to S3:
    1. Uploads both databases (pipeline and annotations)
    2. Syncs the data_quality directory (filtered to exclude large binaries)
    """
    logging.info(f"Starting consolidated pipeline sync to S3 (upload_videos={upload_videos})...")
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
        if not sync_annotations(upload_videos=upload_videos):
            logging.error("Failed to sync annotations directory.")
            success = False

    if success:
        logging.info("Consolidated S3 sync completed successfully.")
    else:
        logging.warning("Consolidated S3 sync completed with errors.")

    return success
