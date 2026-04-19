"""
Biigle Annotation Fetcher — download and preview annotation reports from Biigle.

Retrieves MaxN, 30-second interval counts, and size annotations from a Biigle volume or project.
"""

import pandas as pd
import streamlit as st
from utils import render_sidebar_refresh

from spyfish.biigle.biigle_parser import BiigleParser
from spyfish.config.wrapper import config

st.set_page_config(
    page_title="Biigle Annotation Fetcher", page_icon="📥", layout="wide"
)
st.title("📥 Biigle Annotation Fetcher")
render_sidebar_refresh()
st.markdown(
    "Retrieve and parse annotation reports from Biigle clip volumes.  \n"
    "Exports MaxN (whole video), MaxN (every 30s), and size annotations.  \n"
    "Find the Volume ID in the Biigle URL, e.g. `https://biigle.de/volumes/25173` → `25173`."
)

# ── Scope selection ──────────────────────────────────────────────────────────

whole_project = st.checkbox(
    "Download report for the whole Spyfish Aotearoa project?",
    value=False,
)

volume_placeholder = st.empty()

if whole_project:
    resource = "projects"
    volume_id_str = str(config.biigle_project_id or "")
    volume_placeholder.empty()
else:
    with volume_placeholder:
        resource = "volumes"
        volume_id_str = st.text_input(
            "Volume ID",
            placeholder="Enter volume ID (number)",
            help="ID of the Biigle volume you want to download. Found in the URL on Biigle.",
            key="volume_id",
        ).strip()

# ── Credentials form ─────────────────────────────────────────────────────────

with st.form("biigle_form"):
    email = st.text_input(
        "Biigle Email",
        value=config.email or "",
        placeholder="you@example.com",
        help="The email you use to sign in to Biigle.",
    ).strip()
    token = st.text_input(
        "Biigle API Token",
        type="password",
        placeholder="Paste your token…",
        help=(
            "Find your token at https://biigle.de/settings/tokens.  \n"
            "Treat this like a password. You can also set BIIGLE_API_TOKEN in your .env file."
        ),
    ).strip()

    submitted = st.form_submit_button("Fetch annotations", type="primary")

# ── Fetch & display ──────────────────────────────────────────────────────────

if submitted:
    if not (email and token and volume_id_str.isdigit()):
        st.error("Please provide a valid Email, Token, and numeric Volume ID.")
        st.stop()

    try:
        with st.spinner("Creating report and downloading ZIP from Biigle…"):
            parser = BiigleParser(email=email, token=token)
            processed = parser.process_video_annotations(
                volume_id=int(volume_id_str),
                resource=resource,
            )

        if not processed:
            st.warning(f"No annotations found for {resource} {volume_id_str}.")
            st.stop()

        max_n_df = processed.get("max_n_df")
        max_n_30s_df = processed.get("max_n_30s_df")
        sizes_df = processed.get("sizes_df")

        st.success(f"Annotations loaded from {resource} {volume_id_str}.")

        def _section(df: pd.DataFrame | None, label: str, fname: str):
            st.subheader(label)
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.caption(f"{len(df)} rows")
                st.dataframe(df, width="stretch")
                st.download_button(
                    label=f"⬇️ Download {label} (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=fname,
                    mime="text/csv",
                    width="stretch",
                )
            else:
                st.info(f"No data for **{label}**.")

        tab1, tab2, tab3 = st.tabs(
            ["Max N (whole video)", "Max N (every 30s)", "Size annotations"]
        )
        with tab1:
            _section(max_n_df, "Max N — whole video", f"maxn_{volume_id_str}.csv")
        with tab2:
            _section(max_n_30s_df, "Max N — every 30s", f"maxn_30s_{volume_id_str}.csv")
        with tab3:
            _section(sizes_df, "Sizes", f"sizes_{volume_id_str}.csv")

    except Exception as e:
        st.error(f"Error fetching annotations: {e}")
        st.stop()
