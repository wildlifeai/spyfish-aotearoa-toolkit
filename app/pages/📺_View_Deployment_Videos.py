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
    # Lazy-import: PyAV pulls in ffmpeg native libs — keep page cold-start light
    # for visitors who only watch videos and never click "Extract Clip".
    import logging
    import os
    import tempfile
    import time

    import av

    t_start = time.monotonic()
    # Write to a NamedTemporaryFile rather than BytesIO. `movflags=faststart`
    # makes the mov muxer do a post-write moov-atom shift; some FFmpeg builds
    # error with "[Errno 2] No such file or directory: '<none>'" when that
    # shift is requested against a file-like object — the muxer needs a real
    # path. Disk round-trip is cheap (~50 MB local SSD).
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        with av.open(video_url) as in_container:
            video_stream = in_container.streams.video[0]
            logging.info(f"clip extract: opened in {time.monotonic() - t_start:.2f}s")
            # Seek to near start_s via HTTP range requests — doesn't download
            # the whole file. PyAV lands on the keyframe immediately preceding
            # start_s.
            t_seek = time.monotonic()
            in_container.seek(int(start_s * 1_000_000))
            logging.info(f"clip extract: seek in {time.monotonic() - t_seek:.2f}s")
            n_packets = 0
            n_bytes = 0
            with av.open(
                tmp_path, "w", format="mp4", options={"movflags": "faststart"}
            ) as out_container:
                out_stream = out_container.add_stream_from_template(video_stream)
                # QuickTime rejects HEVC streams with the `hev1` codec tag (its
                # default in many FFmpeg builds + most GoPro sources) and
                # accepts `hvc1`. The bitstream itself is identical — only the
                # 4-byte container tag differs. Browsers/VLC don't care.
                src_codec = video_stream.codec_context.name
                if src_codec == "hevc":
                    out_stream.codec_tag = "hvc1"
                logging.info(
                    f"clip extract: source codec is {src_codec!r}"
                    f"{' — tagged hvc1 for QuickTime' if src_codec == 'hevc' else ''}"
                )
                # Don't filter packets with pts_s < start_s: seek lands us on
                # the keyframe immediately preceding start_s, and dropping it
                # leaves the clip starting on a P-frame (decodable in browsers,
                # refused by QuickTime). The clip will be ≤ ~one GOP longer at
                # the front (typically 1–2 s), worth it for player compatibility.
                #
                # Rebase PTS/DTS so the clip starts at 0. Packets carry the
                # source video's absolute timestamps; without rebasing, players
                # see a clip with `duration = end_s` and `first frame at start_s`,
                # so they pad the front with invisible/black content. We use
                # the first packet's DTS as the offset (DTS ≤ PTS for B-frames,
                # so subtracting DTS keeps all DTS values non-negative).
                pts_offset = None
                for packet in in_container.demux(video_stream):
                    if packet.pts is None:
                        continue
                    pts_s = float(packet.pts * video_stream.time_base)
                    if pts_s > end_s:
                        break
                    if pts_offset is None:
                        pts_offset = (
                            packet.dts if packet.dts is not None else packet.pts
                        )
                    packet.pts -= pts_offset
                    if packet.dts is not None:
                        packet.dts -= pts_offset
                    packet.stream = out_stream
                    out_container.mux(packet)
                    n_packets += 1
                    n_bytes += packet.size
        with open(tmp_path, "rb") as fh:
            data = fh.read()
        logging.info(
            f"clip extract: {n_packets} packets, {n_bytes / 1e6:.1f} MB in "
            f"{time.monotonic() - t_start:.2f}s total"
        )
        return data
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# --- MAIN APP ---
# --- Streamlit UI ---
st.title("Deployment Video player")


def _presign_into_session_state(key: str, drop_label: str) -> bool:
    """Run the presign and populate session_state. Returns True on success."""
    try:
        ps_url = get_presigned_url(key)
    except Exception as e:
        st.error(f"AWS Error: {e}")
        return False
    if ps_url is None:
        st.error(f"Video not found at path: {key}")
        return False
    st.session_state["presigned_url"] = ps_url
    st.session_state["presigned_drop_id"] = drop_label
    return True


# Shareable links: ?drop_id=ABC123 pre-fills the input and auto-presigns on first
# load, so a recipient lands on a playing video rather than an empty form.
url_drop_id = st.query_params.get("drop_id", "").strip()

# --- Checkbox: Use direct S3 path OR DropID ---
use_direct_path = st.checkbox("Provide full S3 path instead of DropID")
if use_direct_path:
    s3_key = st.text_input("Enter full S3 object path (key)")
    drop_id = s3_key  # fallback label for clip filename
else:
    drop_id = st.text_input("Provide DropID", value=url_drop_id).strip()
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

# Auto-presign on first arrival via a shared link. Guarded on session_state so we
# don't re-fire on subsequent reruns; user can still click Generate to refresh
# the presign if it expires.
if (
    not use_direct_path
    and url_drop_id
    and drop_id == url_drop_id
    and s3_key
    and "presigned_url" not in st.session_state
):
    _presign_into_session_state(s3_key, drop_id)

if st.button("Generate URL and play video"):
    if not s3_key:
        st.error("Please provide an S3 path or a DropID.")
    elif _presign_into_session_state(s3_key, drop_id) and not use_direct_path:
        # Reflect the active drop in the URL so the address bar is the share link.
        st.query_params["drop_id"] = drop_id

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
    # TODO: hide this section behind a dev flag (e.g. SPYFISH_DEV env var) so
    # non-dev installs can avoid PyAV entirely.
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
            import time as _t

            duration = clip_end - clip_start
            t0 = _t.monotonic()
            with st.status(
                f"Extracting {duration:.0f}s clip, wait time depends on your "
                "connection (S3 download + browser upload). Often under a "
                "minute, can be several minutes on slow networks…",
                expanded=True,
            ) as status:
                try:
                    clip_bytes = extract_clip_bytes(presigned_url, clip_start, clip_end)
                except Exception as e:
                    status.update(label=f"Extraction failed: {e}", state="error")
                else:
                    elapsed = _t.monotonic() - t0
                    status.update(
                        label=f"Extracted {len(clip_bytes) / 1e6:.1f} MB in "
                        f"{elapsed:.1f}s, ready to download",
                        state="complete",
                    )
                    current_drop_id = (
                        st.session_state.get("presigned_drop_id") or "clip"
                    )
                    start_label = f"{start_min}m{start_sec:02d}s"
                    end_label = f"{end_min}m{end_sec:02d}s"
                    # Render the download button in the same run; bytes don't
                    # need to live in st.session_state because Streamlit ships
                    # the data to the browser at render time.
                    st.download_button(
                        label="⬇️ Download Clip",
                        data=clip_bytes,
                        file_name=f"{current_drop_id}_clip_{start_label}_{end_label}.mp4",
                        mime="video/mp4",
                        type="primary",
                    )
