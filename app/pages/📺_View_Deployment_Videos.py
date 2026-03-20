import os
import subprocess
import tempfile

import streamlit as st
from utils import check_password

from spyfish.config.wrapper import config
from spyfish.storage.s3_handler import S3Handler


# --- Helper to generate a presigned URL ---
@st.cache_resource
def get_s3_handler():
    return S3Handler()


def get_presigned_url(key: str, expires_in: int = 3600) -> str | None:
    handler = get_s3_handler()
    return handler.generate_presigned_url(key, expiration=expires_in)


def extract_clip_bytes(video_url: str, start_s: float, end_s: float) -> bytes:
    duration = end_s - start_s
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-i", video_url,
            "-t", str(duration),
            "-c:v", "copy",   # stream copy — no re-encode, near-instant
            "-an",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- MAIN APP ---
if not check_password():
    st.stop()
else:
    st.write("You are logged in! 🎉")

    # --- Streamlit UI ---
    st.title("Deployment Video player")

    # --- Checkbox: Use direct S3 path OR DropID ---
    use_direct_path = st.checkbox("Provide full S3 path instead of DropID")
    if use_direct_path:
        s3_key = st.text_input("Enter full S3 object path (key)")
        drop_id = s3_key  # fallback label for clip filename
    else:
        drop_id = st.text_input("Provide DropID").strip()
        if drop_id:
            try:
                config.validate_drop_id(drop_id)
                survey_id = config.get_survey_id_from_drop(drop_id)
                s3_key = f"media/{survey_id}/{drop_id}/{drop_id}.mp4"
            except ValueError as e:
                s3_key = ""
                st.error(str(e))
        else:
            s3_key = ""

    if st.button("Generate URL and play video"):
        if not s3_key:
            st.error("Please provide an S3 path or a DropID.")
        else:
            try:
                ps_url = get_presigned_url(s3_key)
                if ps_url is None:
                    st.error(f"Video not found at path: {s3_key}")
            except Exception as e:
                st.error(f"AWS Error: {e}")
                ps_url = None

            if ps_url:
                st.session_state["presigned_url"] = ps_url
                st.session_state["presigned_drop_id"] = drop_id
                # Clear any previous clip when a new video is loaded
                st.session_state.pop("clip_bytes", None)
                st.session_state.pop("clip_filename", None)

    presigned_url = st.session_state.get("presigned_url")
    if presigned_url:
        st.subheader("Video preview.")
        st.write(
            "Does the path look ok? (In the future this will check automatically.)"
        )
        st.code(st.session_state.get("presigned_drop_id", ""), language="text")
        st.write(
            "The video box will show even when there are issues, so check above/try again later, or raise an issue."
        )
        st.video(presigned_url)
        st.caption("Generated Presigned URL:")
        st.code(presigned_url, language="text")

        # --- Clip extractor ---
        st.divider()
        st.subheader("✂️ Extract Clip")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Start")
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                start_min = st.number_input("Minutes", min_value=0, step=1, value=0, key="start_min")
            with s_col2:
                start_sec = st.number_input("Seconds", min_value=0, max_value=59, step=1, value=0, key="start_sec")
            clip_start = start_min * 60 + start_sec
        with col2:
            st.caption("End")
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                end_min = st.number_input("Minutes", min_value=0, step=1, value=0, key="end_min")
            with e_col2:
                end_sec = st.number_input("Seconds", min_value=0, max_value=59, step=1, value=30, key="end_sec")
            clip_end = end_min * 60 + end_sec

        if st.button("Extract Clip", disabled=(clip_end <= clip_start)):
            if clip_end <= clip_start:
                st.error("End must be after start.")
            else:
                with st.spinner("Extracting clip..."):
                    try:
                        clip_bytes = extract_clip_bytes(presigned_url, clip_start, clip_end)
                        st.session_state["clip_bytes"] = clip_bytes
                        current_drop_id = st.session_state.get("presigned_drop_id") or "clip"
                        start_label = f"{start_min}m{start_sec:02d}s"
                        end_label = f"{end_min}m{end_sec:02d}s"
                        st.session_state["clip_filename"] = (
                            f"{current_drop_id}_clip_{start_label}_{end_label}.mp4"
                        )
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
                        st.session_state.pop("clip_bytes", None)

        if "clip_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download Clip",
                data=st.session_state["clip_bytes"],
                file_name=st.session_state["clip_filename"],
                mime="video/mp4",
                type="primary",
            )
