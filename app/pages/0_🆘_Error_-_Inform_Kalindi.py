import streamlit as st

# Intentionally dependency-light: this page imports ONLY streamlit — no config, no
# S3Handler, no DB. That is the whole point. Streamlit builds the sidebar nav from
# the pages/ directory regardless of whether other pages run, so this page stays
# reachable and renders even when another tool is completely broken (bad config,
# S3 outage, import error). Do not add heavy imports here.

st.title("🆘 Error? Inform Kalindi")

st.error(
    "**If any tool on this site doesn't work, contact Kalindi immediately.**\n\n"
    "- **Slack:** message Kalindi\n"
    "- **Email:** [kalindi@wildlife.ai](mailto:kalindi@wildlife.ai)\n\n"
    "Please include what you were doing, which tool/page, the **DropID** if "
    "there is one, and a **screenshot** if you can. This makes things much "
    "faster to fix."
)

st.caption(
    "This page has no dependencies, so it loads even when the other pages are "
    "broken. If you can read this, the site itself is up."
)
