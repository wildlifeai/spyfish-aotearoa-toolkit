"""Cache controls for the sidebar.

Its own module, and deliberately importing nothing but Streamlit.

The obvious home is `utils`, next to `sync_db_if_needed` and the download
helpers, and that is where this lived. The problem is who has to call it: the
button belongs on every page, so the entrypoint renders it once, and the
entrypoint runs on every page load. Importing `utils` there would put
`spyfish.storage.db_sync` — config, credentials, S3 — on the load path of the
whole app, including the support page that exists to survive exactly that kind
of breakage.

Clearing a cache needs none of that, so the button moved out to where it can be
imported safely. `utils.render_sidebar_refresh` still resolves, re-exported, for
anything already importing it from there.
"""

import streamlit as st


def render_sidebar_refresh() -> None:
    """Sidebar button to clear the cached frames and re-sync from S3.

    `st.cache_data.clear()` drops every cached frame, so the next run reloads
    the databases and re-reads them. `sync_db_if_needed` is cached too, which is
    what makes this a re-sync rather than just a re-read.

    Called once by the entrypoint, so every page gets the button. Do not call
    it from a page.
    """
    st.sidebar.header("Controls")
    if st.sidebar.button(
        "🔄 Refresh Cache", help="Clears local cache and re-syncs from S3 if needed"
    ):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
        st.rerun()
