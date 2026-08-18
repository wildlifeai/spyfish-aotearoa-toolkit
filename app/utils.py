import hmac

import streamlit as st

from spyfish.storage.db_sync import download_annotations_db, download_db

# One TTL for every cached data read in the app. 300s keeps widget
# interactions snappy (reruns hit the cache) while bounding staleness; the
# sidebar Refresh Cache button is the instant path when the data has changed.
# Per-cache TTLs drifted into a zoo (1s, 300s, 600s, None), a 1-second TTL in
# particular silently disables caching, re-running the full query on every
# widget click.
CACHE_TTL_SECONDS = 300


# st.cache_data is process-global, not per-session: the first visitor triggers
# the sync and everyone shares the result until the TTL lapses or someone
# clicks Refresh Cache. An hour bounds how stale a long-running server can get.
@st.cache_data(ttl=3600)
def sync_db_if_needed() -> bool:
    """Sync both databases from S3 if it holds newer copies.

    Returns False when either download failed (S3 unreachable, bad
    credentials), db_sync logs the detail. Callers that get False should call
    ``sync_db_if_needed.clear()`` before rerunning, so the failure is retried
    on the next run instead of being cached as success for an hour.
    """
    ok_pipeline = download_db()
    ok_annotations = download_annotations_db()
    return bool(ok_pipeline and ok_annotations)


# The sidebar refresh button lives in `cache_controls`, which imports nothing
# but Streamlit, so the entrypoint can render it on every page without putting
# this module's S3 and config imports on the load path of the whole app.
# Re-exported because callers already import it from here.
from cache_controls import render_sidebar_refresh  # noqa: E402,F401


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
def check_password(secret_key: str = "APP_PASSWORD", *, label: str = "Password"):
    """Returns True once the user has entered the password held in `secret_key`.

    Each secret gets its own session flag and its own widget keys, so gates are
    independent: unlocking the map must not also unlock anything guarded by
    ``APP_PASSWORD``, and two gates on one page cannot collide on widget IDs.

    Secrets are read from ``.streamlit/secrets.toml`` **relative to the working
    directory Streamlit was launched from**, run the app from ``app/`` or
    ``st.secrets`` will be empty and every gate silently stays locked.
    """
    state_key = f"_pw_ok_{secret_key}"
    if not st.session_state.get(state_key, False):
        expected = st.secrets.get(secret_key)
        if expected is None:
            st.error(
                f"`{secret_key}` is not set in `.streamlit/secrets.toml`, or the app "
                "was launched from the wrong directory, run it from `app/`."
            )
            return False
        password = st.text_input(label, type="password", key=f"_pw_in_{secret_key}")
        if st.button("Unlock", key=f"_pw_btn_{secret_key}"):
            # compare_digest keeps the check constant-time; both sides must be
            # ASCII str or it raises, which is why the stored value is alphanumeric.
            if hmac.compare_digest(password, str(expected)):
                st.session_state[state_key] = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        return False
    return True
