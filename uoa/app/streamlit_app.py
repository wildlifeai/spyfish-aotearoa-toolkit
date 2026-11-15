"""UoA mussel-farm dashboard. A separate app from Spyfish Data Tools.

The University of Auckland dataset (Underwood & Jeffs, Hauraki Gulf mussel
farms, Jan 2022) is a collaborator's data, not DOC's. It used to be a
password-gated "Dashboard test" entry in the DOC app; it is its own app so the
DOC deployment carries none of its data files or secrets
(`[UOA_AWS]`, `[UOA_SITE_COORDS]`). Everything UoA-only lives in
this folder; see README.md.

Run from this folder, which holds the app's own `.streamlit/secrets.toml`
(Streamlit reads secrets from the folder it is launched in):

    cd uoa/app && streamlit run streamlit_app.py

Same shape as the DOC entrypoint: `st.navigation` with imports kept to
streamlit, so a broken page cannot take the app down with it. No contact note
or support page: the app is shown to people outside the team, so it names no
one.
"""

import sys
from pathlib import Path

# Streamlit puts only this file's folder (uoa/app/) on sys.path. This package
# (`uoa.app.data`, ...) and `spyfish` (S3Handler, not pip-installed on Streamlit
# Cloud) both resolve from the repo root, and the shared modules (`theme`,
# `utils`) live in app/: add both once, here. Pages
# run in this same process, so they need no path setup of their own.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _dir in (_REPO_ROOT, _REPO_ROOT / "app"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Mussel Insights",
    page_icon="🐚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The page holds every view, including the ML vs Expert comparison, so it is
# the only page this app needs.
mussel_page = st.Page(
    "mussel_insights.py",
    title="Mussel Insights",
    icon="🐚",
    url_path="mussel-insights",
    default=True,
)

nav = st.navigation([mussel_page])
nav.run()
