"""UoA mussel-farm dashboard. A separate app from Spyfish Data Tools.

The University of Auckland dataset (Underwood & Jeffs, Hauraki Gulf mussel
farms, Jan 2022) is a collaborator's data, not DOC's. It used to be a
password-gated "Dashboard test" entry in the DOC app; it is its own app so the
DOC deployment carries none of its data files or secrets
(`TEST_DASHBOARD_PASSWORD`, `[UOA_SITE_COORDS]`). Everything UoA-only lives in
this folder; see README.md.

Run from `app/`, so `.streamlit/secrets.toml` resolves (see
`utils.check_password`):

    cd app && streamlit run ../in_progress/uoa/app.py

Same shape as the DOC entrypoint: `st.navigation` with imports kept to
streamlit and `support`, so a broken page cannot take the app down with it.
"""

import sys
from pathlib import Path

# Streamlit puts only this file's folder (in_progress/uoa/) on sys.path. The
# shared modules (`support`, `theme`, `utils`, `ecology_data`) live in app/ and
# this package (`uoa.data`, ...) in in_progress/, so add both once, here. Pages
# run in this same process, so they need no path setup of their own.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _dir in (_REPO_ROOT / "app", _REPO_ROOT / "in_progress"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import streamlit as st  # noqa: E402
from support import SUPPORT_PAGE_PATH, render_contact_note  # noqa: E402

st.set_page_config(
    page_title="Mussel Insights",
    page_icon="🐚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The page gates itself on TEST_DASHBOARD_PASSWORD and holds every view,
# including the ML vs Expert comparison, so it is
# the only page this app needs.
mussel_page = st.Page(
    "mussel_insights.py",
    title="Mussel Insights",
    icon="🐚",
    url_path="mussel-insights",
    default=True,
)

# Registered because the sidebar contact note links to /support and Streamlit
# routes only pages in the nav; `render_contact_note` hides its link. The page
# is shared with the DOC app, so it stays in app/pages/. st.Page resolves a
# relative path against this file's folder and accepts one outside it.
support_page = st.Page(
    "../../app/pages/0_🆘_Support.py",
    title="Error page",
    icon="🆘",
    url_path=SUPPORT_PAGE_PATH,
)

nav = st.navigation([mussel_page, support_page])
nav.run()
render_contact_note()
