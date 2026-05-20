import pandas as pd
import streamlit as st

from sftk.biigle_parser import BiigleParser
from sftk.common import BIIGLE_PROJECT_ID

# TODO add logging if necessary


st.set_page_config(
    page_title="BIIGLE Annotation Fetcher", page_icon="🐟", layout="wide"
)
st.title("🐟  BIIGLE Annotation Fetcher")
st.markdown(
    "This app is used to retrieve and parse the annotation report from each of the Biigle clip groups.  \nIt allows you to export the MaxN total video, MaxN every 30 seconds, and size annotations.  \nFor more info on where to get the form values, check the help icons next to each entry.  \n If there are any issues, or if it breaks please write an email to Kalindi, or open an issue on: https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/issues"
)


whole_project = st.checkbox(
    "Download report for the whole Spyfish Aotearoa Project?", value=False
)

# Placeholder for the volume_id input
volume_placeholder = st.empty()

if whole_project:
    resource = "projects"
    volume_id_str = str(BIIGLE_PROJECT_ID)
    volume_placeholder.empty()  # hides the volume ID input immediately
else:
    with volume_placeholder:
        resource = "volumes"
        volume_id_str = st.text_input(
            "Volume ID",
            placeholder="Enter volume ID (number)",
            help="ID of the BIIGLE volume you want to download",
            key="volume_id",
        ).strip()


with st.form("biigle_form"):
    help_msg_email = "The email you sign in to BIIGLE with and the one associated with the volume you want to download."
    help_msg_token = "A token is a special password-like number used for the BIIGLE API.  \nFind yours here: https://biigle.de/settings/tokens. Treat this token as you would a password."
    help_msg_volume_id = "The ID of the video volume you want to export annotations from.  \nYou can find it in the URL on BIIGLE when you are on the page with all the clips.  \nFor example: https://biigle.de/volumes/25173, 25173 is the volume ID."

    email = st.text_input(
        "Email", placeholder="you@example.com", help=help_msg_email
    ).strip()
    token = st.text_input(
        "BIIGLE Token",
        type="password",
        placeholder="Paste your token…",
        help=help_msg_token,
    ).strip()

    submitted = st.form_submit_button("Fetch", type="primary")

if submitted:
    # Validate before constructing anything
    if not (email and token and volume_id_str.isdigit()):
        st.error("Please provide Email, Token, and a numeric Volume ID.")

    else:
        try:
            with st.spinner("Creating report and downloading ZIP…"):
                biigle_parser = BiigleParser(email=email, token=token)
                processed = biigle_parser.process_video_annotations(
                    volume_id=int(volume_id_str),
                    resource=resource,
                )

            if not processed:
                st.warning(
                    f"No annotations found for volume {volume_id_str}. The volume might be empty."
                )
                st.stop()

            # Extract dataframes
            max_n_30s_df = processed.get("max_n_30s_df")
            max_n_df = processed.get("max_n_df")
            sizes_df = processed.get("sizes_df")

            def _render_df_section(df: pd.DataFrame | None, label: str, fname: str):
                st.subheader(label)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    st.caption(f"{len(df)} rows")
                    st.dataframe(df, width="stretch")
                    st.download_button(
                        label=f"Download {label} (CSV)",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=fname,
                        mime="text/csv",
                        width="stretch",
                    )
                else:
                    st.info(f"No data available for **{label}**.")

            st.success(f"Loaded annotations.")

            # TODO add some graphs here
            tab1, tab2, tab3 = st.tabs(
                ["Max N (whole video)", "Max N (every 30s)", "Size annotations"]
            )
            with tab1:
                _render_df_section(
                    max_n_df, "Max N of whole video", f"annotations_{volume_id_str}_max_n.csv"
                )
            with tab2:
                _render_df_section(
                    max_n_30s_df,
                    "Max N every 30 seconds",
                    f"annotations_{volume_id_str}_max_n_30s.csv",
                )
            with tab3:
                _render_df_section(
                    sizes_df, "Sizes (if annotated)", f"annotations_{volume_id_str}_sizes.csv"
                )
        except Exception as e:
            st.error(f"An error occurred while fetching annotations: {e}")
            st.stop()
