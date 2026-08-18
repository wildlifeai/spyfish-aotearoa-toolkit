"""Support chrome. Deliberately imports only streamlit.

`utils.py` pulls in `spyfish.storage.db_sync`, which reaches config and S3. The
entrypoint runs on every page load under `st.navigation`, so anything it imports
becomes a single point of failure for the whole app, including the support page
that is supposed to survive breakage. Keeping this module dependency-free is
what preserves that.

Do not add imports here.
"""

import streamlit as st

SUPPORT_PAGE_PATH = "support"


# The contact note renders only in the sidebar (below), not as a banner above
# every page, a banner would cost a row above the fold on every view just to
# repeat the sidebar note. Known trade-off: a page that throws before the
# sidebar renders leaves no visible route to /support, the URL still works.


def render_contact_note() -> None:
    """Sidebar contact note. Called once from the entrypoint, after `nav.run()`.

    Single source of truth for the "something broke, tell Kalindi" message, so
    the wording changes in one place.

    Called after `nav.run()` so it sits at the bottom of the sidebar, under the
    page list. `st.navigation` runs the entrypoint on every page load, so one
    call covers every page.
    """
    # Streamlit routes only pages that appear in the nav, so the support page
    # has to be registered. Hiding its one link keeps it out of the list while
    # leaving /support reachable from the message below.
    #
    # Anchored on the link's own href rather than on a wrapper element:
    # `section[data-testid="stSidebarNav"]` stopped existing in Streamlit 1.50
    # (the list is now `ul[data-testid="stSidebarNavItems"]`), and the rule
    # failed silently, so the link came back. The `li:has(...)` form also takes
    # the list item with it, leaving no gap where it was.
    st.markdown(
        f"""
        <style>
          a[href$="/{SUPPORT_PAGE_PATH}"][data-testid="stSidebarNavLink"],
          li:has(> div > a[href$="/{SUPPORT_PAGE_PATH}"][data-testid="stSidebarNavLink"]) {{
            display: none;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.error(
        "⚠️ **If this tool doesn't work**, contact Kalindi immediately on "
        "Slack, or email [kalindi@wildlife.ai](mailto:kalindi@wildlife.ai).  \n"
        f"Read more about it here: [Error page](/{SUPPORT_PAGE_PATH})"
    )
