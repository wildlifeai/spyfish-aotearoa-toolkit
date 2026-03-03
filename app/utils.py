import streamlit as st
from spyfish.config import config
from spyfish.storage.db_sync import download_db
from spyfish.storage.s3_handler import S3Handler
from botocore.exceptions import ClientError

import logging


@st.cache_data(ttl=300)  # Check S3 at most every 5 minutes
def sync_db_if_needed():
    """Helper to sync database from S3 if in AWS mode or missing locally."""

    # Download the main pipeline DB
    if not config.db_path.exists():
        download_db()

    # Also handle annotations DB since components use it now
    annotations_db_path = config.project_root / "process_files" / "spyfish_annotations.db"
    if not annotations_db_path.exists():
        try:
            s3 = S3Handler()
            s3.download_file("process_files/spyfish_annotations.db", str(annotations_db_path))

        except ClientError as e:
            # It's okay if the file doesn't exist yet (404), but other S3 errors should be logged.
            if e.response.get("Error", {}).get("Code") != "404":
                logging.warning(f"Could not download annotations DB due to an S3 error: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            # Not an error if it doesn't exist yet. Log for visibility in case of other issues.
            logging.warning(f"Could not download annotations DB: {e}")
    return True
