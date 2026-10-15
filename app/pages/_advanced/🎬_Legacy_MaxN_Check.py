"""Watch a legacy-annotated deployment at the moment its MaxN was recorded.

A checking tool, not a reporting view. Legacy annotations came out of Excel
workbooks recorded years before this pipeline existed, and nothing has ever
put them back in front of the footage they describe. This does: pick a
deployment that has BOTH legacy expert annotations and video still on S3,
pick a species, and the player opens at that species' recorded time of MaxN.

Three things it is good for:

* **Spot-checking the legacy import.** If the workbook says eight snapper at
  12:40 and the footage at 12:40 shows sand, something went wrong between the
  spreadsheet and the database (a misparsed time, a mismatched DropID, a
  transposed column).
* **The IsBadDeployment audit** (`claude_docs/todo.md`): 41 flagged
  deployments carry real annotations, and the only way to judge whether the
  flag or the annotation is wrong is to look.
* **Time-base sanity.** Legacy sheets record time several ways (minutes into
  the drop, interval numbers, clock times); this shows whether the stored
  seconds land where they should.

Only 112 deployments qualify today (HOR, TUH, BNP, AHE), because most legacy
surveys predate the footage that survives on S3.
"""

import sqlite3
import sys
from pathlib import Path

# The shared helpers live in app/, which is not on sys.path when Streamlit
# runs a page from a subfolder. parents[2] is app/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from ecology_data import load_common_names  # noqa: E402
from utils import (  # noqa: E402
    CACHE_TTL_SECONDS,
    ensure_databases,
    stop_missing_database,
)

from spyfish.config.base import NULL_DEPLOYMENT, VideoPresence  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.database.connection import connect_readonly  # noqa: E402
from spyfish.orchestrator.legacy_extract import legacy_time_reading  # noqa: E402
from spyfish.storage.s3_handler import S3Handler  # noqa: E402

st.title("🎬 Legacy MaxN check")
st.badge("In development", color="red")
st.caption(
    "Play a legacy-annotated deployment at the second its MaxN was recorded, "
    "to check the imported numbers against the footage they describe."
)


@st.cache_resource
def _s3() -> S3Handler:
    return S3Handler()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_checkable() -> pd.DataFrame:
    """Deployments with legacy annotations AND video still on S3.

    `video_presence` is the pipeline's own record of what S3 holds, so this
    never offers a drop whose footage was archived to glacier or lost.
    """
    with connect_readonly(config.annotations_db_path) as conn:
        legacy = pd.read_sql(
            "SELECT DISTINCT drop_id FROM annotations WHERE external_id = 'legacy'",
            conn,
        )
    with connect_readonly(config.db_path) as conn:
        present = pd.read_sql(
            "SELECT drop_id, is_bad_deployment FROM deployments "
            "WHERE video_presence = ?",
            conn,
            params=(VideoPresence.PRESENT,),
        )
    return present.merge(legacy, on="drop_id", how="inner").sort_values("drop_id")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_rows(drop_id: str) -> pd.DataFrame:
    """One row per species: its MaxN and the moment it was recorded.

    The annotations table holds a row per time interval, so a legacy review
    of one deployment can be fifty rows. MaxN is the PEAK across them (the
    rule `data.species_maxn` enforces for the whole report), so the peak row
    per species is what a check should open at; ties go to the earliest,
    which is the first time the count was reached.
    """
    with connect_readonly(config.annotations_db_path) as conn:
        rows = pd.read_sql(
            "SELECT scientific_name, max_interval, time_of_max, "
            "time_of_max_seconds FROM annotations "
            "WHERE drop_id = ? AND external_id = 'legacy' "
            "AND scientific_name IS NOT NULL AND scientific_name != ?",
            conn,
            params=(drop_id, NULL_DEPLOYMENT),
        )
    if rows.empty:
        return rows
    rows["_intervals"] = 1
    counts = rows.groupby("scientific_name", dropna=False)["_intervals"].sum()
    peaks = (
        rows.sort_values(
            ["max_interval", "time_of_max_seconds"],
            ascending=[False, True],
            na_position="last",
        )
        .drop_duplicates("scientific_name", keep="first")
        .drop(columns="_intervals")
    )
    peaks["intervals"] = peaks["scientific_name"].map(counts)
    peaks = peaks.sort_values("max_interval", ascending=False).reset_index(drop=True)

    # The ONE reader of legacy time text (`tests/unit/test_one_time_parser`
    # fails if a second one appears). It matters here beyond consistency:
    # some workbook times reached the database as Excel time cells that had
    # swallowed MM:SS as HH:MM, so "23:45" meaning 23m45s became 23:45:00.
    # A naive read would send the player to hour 23 of a half-hour video.
    # `repaired` marks the rows where that rule fired, and those are the
    # timestamps most worth checking against the footage.
    reading = legacy_time_reading(
        pd.Series([drop_id] * len(peaks)), peaks["time_of_max"]
    )
    # Object dtype with real Nones, not NaN floats: everything downstream
    # asks `is None`, and NaN silently fails that test.
    seconds = reading["seconds"]
    # Cast to object FIRST: `where` fills in the Series' own dtype, so on a
    # float Series the None becomes NaN before `astype(object)` ever runs.
    peaks["start"] = seconds.astype(object).where(seconds.notna(), None).values
    peaks["repaired"] = reading["repaired"].values
    peaks["as_written"] = reading["as_written"].values
    return peaks


# How wide a check is, in seconds. Narrow on purpose: every extra second is
# ~5.7 MB off S3 at these bitrates (4K GoPro, ~340 MB per minute).
FRAME_OFFSETS = (-2.0, 0.0, 2.0)
CLIP_BEFORE, CLIP_AFTER = 5.0, 10.0


def _video_url(drop_id: str) -> str:
    """A short-lived presigned URL. Signing is local, so it costs nothing."""
    return _s3().generate_presigned_url(
        config.get_video_s3_key(drop_id), expiration=3600
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def grab_frames(drop_id: str, seconds: float) -> list:
    """JPEG stills around one moment: [(second, bytes), ...].

    ONE decode pass, not one per frame. ffmpeg seeks to the keyframe before
    the window and decodes forward, so three stills two seconds apart share
    a single seek and a single contiguous read, which halves both the wait
    (35 s against 55 s measured on a 12 GB drop) and the bytes pulled.

    Reading over HTTP rather than downloading the file: ffmpeg issues range
    requests, so the transfer is the index (1.5 to 7 MB, and these files
    keep it at the END, which is what makes opening slow) plus a few seconds
    of video, not the 12 GB the deployment weighs.
    """
    import io

    import av

    wanted = sorted(max(seconds + off, 0.0) for off in FRAME_OFFSETS)
    out: list = []
    with av.open(_video_url(drop_id), timeout=60) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(
            int(wanted[0] / stream.time_base),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        idx = 0
        for frame in container.decode(stream):
            at = float(frame.pts * stream.time_base)
            while idx < len(wanted) and at >= wanted[idx]:
                image = frame.to_image()
                # 960px is plenty to count fish and keeps the page light;
                # the source is 1920x1080.
                image.thumbnail((960, 960))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80)
                out.append((at, buffer.getvalue()))
                idx += 1
            if idx >= len(wanted):
                break
    return out


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def grab_spread(drop_id: str) -> list:
    """Three stills spread across the whole drop: [(second, bytes), ...].

    For deployments whose only legacy record is an absence ("reviewed,
    nothing seen"): there is no MaxN moment to open at, but the claim is
    worth testing, and three samples tell an empty reef from footage too
    murky to see through, which is the difference between a real zero and a
    bad deployment.

    Separate seeks here, unlike `grab_frames`: the samples are minutes
    apart, so there is no contiguous read to share.
    """
    import io

    import av

    out: list = []
    with av.open(_video_url(drop_id), timeout=60) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration
            else (float(container.duration) / 1_000_000 if container.duration else 0.0)
        )
        if duration <= 0:
            return []
        for fraction in (0.25, 0.5, 0.75):
            target = duration * fraction
            container.seek(
                int(target / stream.time_base),
                stream=stream,
                backward=True,
                any_frame=False,
            )
            for frame in container.decode(stream):
                at = float(frame.pts * stream.time_base)
                if at >= target - 0.5:
                    image = frame.to_image()
                    image.thumbnail((960, 960))
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=80)
                    out.append((at, buffer.getvalue()))
                    break
    return out


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def grab_clip(drop_id: str, seconds: float) -> bytes | None:
    """A short clip around one moment, as MP4 bytes.

    Remuxed, not re-encoded: packets are copied, so this costs a read and
    no CPU to speak of. Timestamps are rebased onto the clip, or players
    show a file that claims to start 29 minutes in and refuse to scrub it.

    Returning BYTES rather than a URL is the cost control: the browser gets
    a 15-second file and physically cannot stream on into the other 35
    minutes.
    """
    import io

    import av

    start = max(seconds - CLIP_BEFORE, 0.0)
    end = start + CLIP_BEFORE + CLIP_AFTER
    buffer = io.BytesIO()
    with av.open(_video_url(drop_id), timeout=60) as src:
        in_stream = src.streams.video[0]
        src.seek(
            int(start / in_stream.time_base),
            stream=in_stream,
            backward=True,
            any_frame=False,
        )
        # Fragmented MP4, so the output is playable without a seekable
        # file handle to write the index back into.
        with av.open(
            buffer,
            mode="w",
            format="mp4",
            options={"movflags": "frag_keyframe+empty_moov"},
        ) as dst:
            out_stream = dst.add_stream_from_template(in_stream)
            first_pts = None
            for packet in src.demux(in_stream):
                if packet.pts is None:
                    continue
                at = float(packet.pts * in_stream.time_base)
                if at > end:
                    break
                if first_pts is None:
                    first_pts = packet.pts
                packet.pts -= first_pts
                packet.dts = (
                    packet.pts if packet.dts is None else packet.dts - first_pts
                )
                packet.stream = out_stream
                dst.mux(packet)
    data = buffer.getvalue()
    return data or None


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def reviewed_empty(drop_id: str) -> bool:
    """True when the only legacy content is "reviewed, saw nothing".

    `NULL_DEPLOYMENT` is the sentinel the pipeline writes for exactly that
    (see `spyfish.config.base`). The report's own loaders map it to NaN on
    the way in; this page reads the database directly, so it has to know
    about the sentinel itself, or it lists "NULL DEPLOYMENT" as a species
    with no timestamp, which is what first crashed this page.
    """
    with connect_readonly(config.annotations_db_path) as conn:
        total, named = conn.execute(
            "SELECT COUNT(*), COUNT(CASE WHEN scientific_name IS NOT NULL "
            "AND scientific_name != ? THEN 1 END) FROM annotations "
            "WHERE drop_id = ? AND external_id = 'legacy'",
            (NULL_DEPLOYMENT, drop_id),
        ).fetchone()
    return bool(total) and not named


def _num(value) -> int:
    """An int for display, treating NaN and None as zero."""
    return 0 if value is None or value != value else int(value)


def _clock(seconds) -> str:
    # NaN, not just None: the canonical time parser returns NaN for text it
    # cannot read, and `NaN is None` is False.
    if seconds is None or seconds != seconds:
        return "not recorded"
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


# Not rendered through the report shell, so it syncs itself: opened straight
# at this URL on a fresh deploy, there is otherwise no database yet.
ensure_databases()
try:
    checkable = load_checkable()
except sqlite3.OperationalError:
    stop_missing_database()
if checkable.empty:
    st.info(
        "No deployment has both legacy annotations and video on S3 in this "
        "database. The reporting databases sync from S3 on first load, so try "
        "the Refresh Cache button in the sidebar."
    )
    st.stop()

flagged = checkable[checkable["is_bad_deployment"].fillna(0).astype(bool)]
st.caption(
    f"**{len(checkable)} deployments** carry legacy annotations and still "
    f"have footage on S3. {len(flagged)} of them are flagged "
    "`IsBadDeployment`, which is exactly the set the bad-deployment audit "
    "needs eyes on."
)

# Cost is why this page never hands the browser the whole video. These are
# 4K GoPro files: ~12 GB for a 35-minute drop, ~340 MB per MINUTE streamed,
# out of S3 in ap-southeast-2 at US$0.114/GB. A full-video player would let
# one careless tab stream US$1.40 of footage. Everything below is extracted
# server-side over a bounded window instead, so a check costs cents however
# long the page stays open.
st.caption(
    "**Bounded on purpose.** The whole video is never loaded: a check pulls "
    "three frames around the recorded MaxN (about 25 MB, under a cent), and "
    "a 15-second clip is one click away if stills are not enough (about "
    "85 MB, roughly US$0.01). Measured on a 12 GB drop: frames take about "
    "35 seconds, the clip about 80, and both are cached afterwards."
)

only_flagged = st.checkbox(
    f"Only the {len(flagged)} flagged as bad deployments",
    value=False,
    help="The audit list: deployments marked unusable that nonetheless carry "
    "a full expert review. Watching one answers whether the flag or the "
    "annotation is wrong.",
)
options = (flagged if only_flagged else checkable)["drop_id"].tolist()
if not options:
    st.warning("No deployments match that tick box.")
    st.stop()

drop_id = st.selectbox("Deployment", options, key="legacy_check_drop")

rows = load_rows(drop_id)
if rows.empty:
    if reviewed_empty(drop_id):
        st.info(
            "**Reviewed, nothing seen.** The legacy workbook records this "
            "deployment as watched with no animals identified, so there is "
            "no MaxN moment to jump to. That claim is still checkable: the "
            "button below samples three frames spread across the drop, "
            "which is enough to tell an empty reef from footage nobody "
            "could see through."
        )
        if st.button("Sample three frames across the drop", key="legacy_check_spread"):
            with st.spinner("Sampling (about 40 seconds)"):
                try:
                    shots = grab_spread(drop_id)
                except Exception as exc:  # noqa: BLE001
                    shots = []
                    st.error(f"Could not read frames: {exc}")
            if shots:
                cols = st.columns(len(shots))
                for col, (at, jpeg) in zip(cols, shots):
                    with col:
                        st.image(jpeg, caption=_clock(at), use_container_width=True)
    else:
        st.warning("No legacy rows for this deployment.")
    st.stop()

common = load_common_names()
rows = rows.assign(
    Species=rows["scientific_name"].map(lambda s: common.get(s, s) if s else s),
)

left, right = st.columns([2, 3])

with left:
    st.markdown("**What the workbook recorded**")
    st.dataframe(
        rows.assign(At=rows["start"].map(_clock))[
            ["Species", "max_interval", "At", "intervals", "repaired"]
        ].rename(
            columns={
                "max_interval": "MaxN",
                "intervals": "Rows",
                "repaired": "Time repaired",
            }
        ),
        hide_index=True,
        width="stretch",
        height=300,
        column_config={
            "Species": st.column_config.TextColumn("Species", pinned=True),
            "MaxN": st.column_config.NumberColumn("MaxN", width="small"),
            "At": st.column_config.TextColumn("Time of MaxN", width="small"),
            "Rows": st.column_config.NumberColumn(
                "Intervals",
                width="small",
                help="How many interval rows the workbook recorded for this "
                "species; the MaxN shown is the peak across them.",
            ),
            "Time repaired": st.column_config.CheckboxColumn(
                "Repaired",
                width="small",
                help="The stored time differs from the text as written, "
                "because the Excel MM:SS coercion rule fired. These are the "
                "rows most worth checking against the footage.",
            ),
        },
    )

    # The picker is the species list, so choosing a row and jumping the
    # player are one action rather than two.
    labels = {
        f"{r.Species} — MaxN {_num(r.max_interval)} at {_clock(r.start)}": i
        for i, r in rows.iterrows()
    }
    chosen = st.radio(
        "Jump to",
        list(labels),
        key="legacy_check_row",
        help="The player reloads at this row's recorded time of MaxN.",
    )
    row = rows.loc[labels[chosen]]
    start = row["start"]

with right:
    st.markdown("**The footage at that moment**")
    if start is None:
        st.info(
            "This row has no usable time, so there is nothing to seek to. A "
            "legacy row can carry an interval number rather than a timestamp."
        )
    else:
        st.caption(
            f"Recorded: **{_num(row['max_interval'])} "
            f"{row['Species']}** at **{_clock(start)}**"
            + (
                "  ·  this time was repaired from the workbook text "
                f"(`{row['time_of_max']}`), so it is worth a close look"
                if bool(row.get("repaired"))
                else ""
            )
        )
        with st.spinner("Pulling frames from S3 (about 35 seconds, then cached)"):
            try:
                shots = grab_frames(drop_id, float(start))
            except Exception as exc:  # noqa: BLE001, a decode failure is not fatal
                shots = []
                st.error(f"Could not read frames: {exc}")
        if shots:
            cols = st.columns(len(shots))
            for col, (at, jpeg) in zip(cols, shots):
                with col:
                    st.image(jpeg, caption=_clock(at), use_container_width=True)
            st.caption(
                "Two seconds before, at, and two seconds after the recorded "
                "MaxN. Fish move: a count that looks wrong on a still can be "
                "right on the clip."
            )

        if st.button("Pull a 15-second clip (~85 MB)", key="legacy_check_clip"):
            with st.spinner("Extracting the clip (about 80 seconds)"):
                try:
                    clip = grab_clip(drop_id, float(start))
                except Exception as exc:  # noqa: BLE001
                    clip = None
                    st.error(f"Could not extract the clip: {exc}")
            if clip:
                st.video(clip)
                st.caption(
                    "Five seconds before the recorded MaxN to ten after. "
                    "Served from extracted bytes, so it cannot stream on "
                    "past the window."
                )

st.divider()
st.markdown(
    "**Reading it.** Agreement is the common case and needs no note. Worth "
    "recording when the frames show nothing at the recorded time (a time "
    "base or import problem), when they show clearly more or fewer animals "
    "than the workbook says (a counting or transcription problem), or when a "
    "deployment flagged bad looks perfectly usable (a flag problem). The "
    "audit note lives in `claude_docs/todo.md`."
)
