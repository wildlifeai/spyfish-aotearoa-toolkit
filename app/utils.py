import hmac
import streamlit as st
from spyfish.config import config
from spyfish.storage.db_sync import download_db, download_annotations_db

import logging


@st.cache_data(ttl=300)  # Check S3 at most every 5 minutes
def sync_db_if_needed():
    """Helper to sync database from S3 if in AWS mode or missing locally."""

    # Download the main pipeline DB
    download_db()

    # Also handle annotations DB
    download_annotations_db()

    return True


# --- Password protection ---
def check_password():
    """Returns True if the user entered the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            app_password = st.secrets.get("APP_PASSWORD")
            if app_password is not None and hmac.compare_digest(password, app_password):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        return False
    return True
