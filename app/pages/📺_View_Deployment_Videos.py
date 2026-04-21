import io

import av
import streamlit as st

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
    buf = io.BytesIO()
    with av.open(video_url) as in_container:
        video_stream = in_container.streams.video[0]
        # Seek to near start_s using HTTP range requests — doesn't download the whole file
        in_container.seek(int(start_s * 1_000_000))
        with av.open(buf, "w", format="mp4") as out_container:
            out_stream = out_container.add_stream(template=video_stream)
            for packet in in_container.demux(video_stream):
                if packet.pts is None:
                    continue
                pts_s = float(packet.pts * video_stream.time_base)
                if pts_s < start_s:
                    continue
                if pts_s > end_s:
                    break
                packet.stream = out_stream
                out_container.mux(packet)
    return buf.getvalue()


# --- MAIN APP ---
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
    st.write("Does the path look ok? (In the future this will check automatically.)")
    st.code(st.session_state.get("presigned_drop_id", ""), language="text")
    st.write(
        "The video box will show even when there are issues, so check above/try again later, or raise an issue."
    )
    st.video(presigned_url)
    st.caption("Generated Presigned URL:")
    st.code(presigned_url, language="text")

    # --- Clip extractor ---
    # TODO: hide this section behind a dev flag (e.g. SPYFISH_DEV env var) and
    # move `import av` into extract_clip_bytes so non-dev installs don't need PyAV.
    st.divider()
    st.subheader("✂️ Extract Clip")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Start")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            start_min = st.number_input(
                "Minutes", min_value=0, step=1, value=0, key="start_min"
            )
        with s_col2:
            start_sec = st.number_input(
                "Seconds",
                min_value=0,
                max_value=59,
                step=1,
                value=0,
                key="start_sec",
            )
        clip_start = start_min * 60 + start_sec
    with col2:
        st.caption("End")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            end_min = st.number_input(
                "Minutes", min_value=0, step=1, value=0, key="end_min"
            )
        with e_col2:
            end_sec = st.number_input(
                "Seconds",
                min_value=0,
                max_value=59,
                step=1,
                value=30,
                key="end_sec",
            )
        clip_end = end_min * 60 + end_sec

    if st.button("Extract Clip", disabled=(clip_end <= clip_start)):
        if clip_end <= clip_start:
            st.error("End must be after start.")
        else:
            with st.spinner("Extracting clip..."):
                try:
                    clip_bytes = extract_clip_bytes(presigned_url, clip_start, clip_end)
                    st.session_state["clip_bytes"] = clip_bytes
                    current_drop_id = (
                        st.session_state.get("presigned_drop_id") or "clip"
                    )
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
