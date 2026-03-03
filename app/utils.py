import streamlit as st
from spyfish.config import config
from spyfish.storage.db_sync import download_db

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
            from spyfish.storage.s3_handler import S3Handler
            s3 = S3Handler()
            s3.download_file("process_files/spyfish_annotations.db", str(annotations_db_path))
        except Exception as e:
            # Not an error if it doesn't exist yet, we just gracefully continue
            pass

    return True
