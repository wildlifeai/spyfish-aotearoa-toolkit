import logging
from spyfish.config import config, PipelineStatus
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler

storage = S3Handler(bucket=config.storage.get("bucket_name"))

def check_pending_arrivals():
    db = DatabaseManager()
    pending = db.get_deployments_by_status(PipelineStatus.PENDING_ARRIVAL)

    if not pending:
        logging.info("No PENDING_ARRIVAL drops found.")
        return

    logging.info(f"Checking S3 for {len(pending)} PENDING_ARRIVAL drops...")

    logging.info("Downloading master file list from S3 bucket...")
    known_files = set(storage.get_file_paths_set_from_s3(prefix="media/"))

    updated_count = 0
    for drop in pending:
        drop_id = drop["drop_id"]
        video_path = drop["video_path"]

        if video_path and video_path in known_files:
            logging.info(f"✅ Video confirmed for {drop_id}. Updating status to VIDEO_PRESENT.")
            db.update_status(drop_id, PipelineStatus.VIDEO_PRESENT)
            updated_count += 1

    logging.info(f"Loop complete. Advanced {updated_count} drops to VIDEO_PRESENT.")

if __name__ == "__main__":
    check_pending_arrivals()
