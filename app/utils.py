import hmac

import streamlit as st

from spyfish.storage.db_sync import download_annotations_db, download_db


@st.cache_data(ttl=None)  # Only sync once per session unless 'Refresh Cache' is clicked
def sync_db_if_needed():
    """Helper to sync database from S3 if in AWS mode or missing locally."""

    # Download the main pipeline DB
    download_db()

    # Also handle annotations DB
    download_annotations_db()

    return True


def render_sidebar_refresh():
    """Renders a common sidebar button to clear cache and refresh data."""
    st.sidebar.header("Controls")
    if st.sidebar.button(
        "🔄 Refresh Cache", help="Clears local cache and re-syncs from S3 if needed"
    ):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
        st.rerun()


def render_contact_note():
    """Sidebar fallback contact note, shown on every page.

    Single source of truth for the "something broke, tell Kalindi" message — keep
    the wording here so it only changes in one place. Best-effort: it renders only
    if the page got far enough to call it. The guaranteed fallback when a page
    fails to load at all is the dependency-free
    ``pages/0_🆘_Error_-_Inform_Kalindi.py`` page.
    """
    st.sidebar.error(
        "⚠️ **If this tool doesn't work**, contact Kalindi immediately on "
        "Slack, or email [kalindi@wildlife.ai](mailto:kalindi@wildlife.ai)."
    )


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
