import streamlit as st

# Imports ONLY streamlit: no config, no S3Handler, no DB. That is the point. It
# stays readable when another tool is completely broken (bad config, S3 outage,
# import error).
#
# The app entrypoint uses st.navigation, which runs on every page load, so this
# page is only as reachable as the entrypoint. That is why the entrypoint imports
# streamlit and app/support.py and nothing else. Do not add heavy imports to
# either file.

st.title("🆘 Error page")

st.error(
    "**If any tool on this site doesn't work, contact Kalindi immediately.**\n\n"
    "- **Slack:** message Kalindi\n"
    "- **Email:** [kalindi@wildlife.ai](mailto:kalindi@wildlife.ai)\n\n"
    "Please include what you were doing, which tool/page, the **DropID** if "
    "there is one, and a **screenshot** if you can. This makes things much "
    "faster to fix."
)

st.caption(
    "This page loads even when the other pages are broken. If you can read "
    "this, the site itself is up."
)
