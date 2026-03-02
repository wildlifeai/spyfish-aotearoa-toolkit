import logging
from spyfish.config import config, PipelineStatus
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler
from spyfish.storage.local import LocalStorageHandler

mode = config.storage.get("mode", "local")
if mode == "aws":
    storage = S3Handler(bucket=config.storage.get("bucket_name"))
else:
    video_folder = config.storage.get("local_video_dir", ".")
    storage = LocalStorageHandler(video_folder=video_folder)

def check_pending_arrivals():
    db = DatabaseManager()
    if config.storage.get("mode") == "aws":
        from spyfish.storage.db_sync import download_db, upload_db
        download_db()

    pending = db.get_deployments_by_status(PipelineStatus.PENDING_ARRIVAL)

    if not pending:
        logging.info("No PENDING_ARRIVAL drops found.")
        return

    logging.info(f"Checking storage ({mode}) for {len(pending)} PENDING_ARRIVAL drops...")

    updated_count = 0
    known_files = set()

    # Batch extraction interface
    if mode == "aws":
        logging.info(f"Downloading master file list from S3 bucket...")
        known_files = set(storage.get_file_paths_set_from_s3(prefix="media/"))
    else:
        known_files = storage.get_all_videos()

    for drop in pending:
        drop_id = drop["drop_id"]
        video_path = drop["video_path"]

        # Only check against known files if video_path is truthy
        if video_path and video_path in known_files:
            logging.info(f"✅ Video confirmed for {drop_id}. Updating status to VIDEO_PRESENT.")
            db.update_status(drop_id, PipelineStatus.VIDEO_PRESENT, auto_sync=False)
            updated_count += 1

    if config.storage.get("mode") == "aws":
        from spyfish.storage.db_sync import upload_db
        upload_db()

    logging.info(f"Loop complete. Advanced {updated_count} drops to VIDEO_PRESENT.")

if __name__ == "__main__":
    check_pending_arrivals()
