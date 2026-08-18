"""ML vs Expert MaxN comparison.

Block-aligned comparison: each video file covers ~11.8 min of its parent
deployment, which the expert CSV slices into 6-min interval rows. For each
ML video, we find the 6-min blocks fully contained in the video, compute
ML MaxN inside that block's time window, and compare to the matching
expert CSV row for that block.
"""

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide", page_title="ML vs Expert MaxN")

# ── Constants ─────────────────────────────────────────────────────────────────

CSV_PATH = Path("/Users/kalindi/Downloads/Annotations_CLEANED (1).csv")
SMOKETEST_DIR = Path("/Users/kalindi/code/spyfish-merged/media/smoketest_output")

VIDEO_DURATION_SEC = 707.7
VIDEO_DURATION_MIN = VIDEO_DURATION_SEC / 60  # 11.795
INTERVAL_MIN = 6
BAIT_SOAK_MIN = 6  # first 6-min block of deployment = bait soak, excluded
SRC_FPS = 60

MODEL_CLASSES = {
    0: "Jasus edwardsii",
    1: "Notolabrus celidotus",
    2: "Pagrus auratus",
    3: "Parapercis colias",
    4: "Pseudolabrus miles",
    5: "bait",
    6: "fish",
}

# Expert CSV column → model class index (direct species mappings only)
EXPERT_TO_MODEL_CLASS = {
    "snapper_maxn": 2,  # Pagrus auratus
    "spotty_maxn": 1,  # Notolabrus celidotus
}

# Smoke-test runs (video stem → run folder + stride).
# The three picked candidates (busy / less-busy / empty), each with expert
# annotations in the CSV for cross-source comparison.
RUNS = {
    "ESK_C_S_20220118_PM_v001_videoonly": {
        "species_run": "run5",
        "community_run": "run5_community",
        "stride": 100,
    },  # 🕳️ EMPTY
    "ESK_X_S_20220120_MM_v007_videoonly": {
        "species_run": "run6",
        "community_run": "run6_community",
        "stride": 100,
    },  # 🐟 LESS-BUSY
    "ESK_X_S_20220120_PM_v003_videoonly": {
        "species_run": "run7",
        "community_run": "run7_community",
        "stride": 100,
    },  # 🔥 BUSY
    "WHA_X_S_20220118_PM_v004_videoonly": {
        "species_run": "run8",
        "community_run": "_none_",
        "stride": 100,
    },  # 🐠 NON-BLENNY BUSY (CFD not run; also wrong session — actually Evening)
    "ESK_X_S_20220120_PM_v005_videoonly": {
        "species_run": "run9",
        "community_run": "_none_",
        "stride": 100,
    },  # 🔥 BUSY chapter 5 (covers interval 7-8, post-peak)
}


# ── Helpers ───────────────────────────────────────────────────────────────────


@st.cache_data
def load_expert() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def parse_video_stem(stem: str):
    """ESK_X_S_20220118_MD_v005_videoonly → ('ESK_X_S_20220118_MD', 5)."""
    m = re.match(r"^(.+?)_v(\d+)_videoonly$", stem)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def blocks_for_video(video_num: int):
    """6-min blocks that overlap the video.

    Block N in the legacy CSV corresponds to recording minute 6N to 6(N+1) —
    the first 6 minutes of recording are bait-soak and not annotated, so the
    CSV's first row already starts at minute 6.
    """
    v_start = (video_num - 1) * VIDEO_DURATION_MIN
    v_end = video_num * VIDEO_DURATION_MIN
    blocks = []
    n = 1
    while n * INTERVAL_MIN < v_end:
        b_start = n * INTERVAL_MIN  # block N starts at min 6N
        b_end = (n + 1) * INTERVAL_MIN  # block N ends at min 6(N+1)
        overlap_start = max(b_start, v_start)
        overlap_end = min(b_end, v_end)
        if overlap_end > overlap_start:
            fully = (b_start >= v_start) and (b_end <= v_end)
            blocks.append(
                {
                    "block": n,
                    "dep_start_min": b_start,
                    "dep_end_min": b_end,
                    "video_sec_start": max(0.0, (b_start - v_start) * 60),
                    "video_sec_end": min(VIDEO_DURATION_SEC, (b_end - v_start) * 60),
                    "fully_covered": fully,
                }
            )
        n += 1
    return blocks


def ml_in_window(
    labels_dir: Path,
    video_stem: str,
    start_sec: float,
    end_sec: float,
    stride: int,
    conf_threshold: float = 0.25,
):
    """
    Within the time window, returns:
      - per_class_peak: {cls: max single-frame count}
      - fish_peak_total: max total detections (any class) in any single frame
    Detections below conf_threshold are skipped.
    """
    start_frame = int(start_sec * SRC_FPS)
    end_frame = int(end_sec * SRC_FPS)
    min_n = max(1, (start_frame // stride) + 1)
    max_n = (end_frame // stride) + 1

    per_class_peak = defaultdict(int)
    fish_peak_total = 0
    for n in range(min_n, max_n + 1):
        p = labels_dir / f"{video_stem}_{n}.txt"
        if not p.exists():
            continue
        frame_counts = defaultdict(int)
        frame_total = 0
        for line in p.read_text().splitlines():
            parts = line.split()
            if not parts or len(parts) < 6:
                continue
            conf = float(parts[5])
            if conf < conf_threshold:
                continue
            cls = int(parts[0])
            frame_counts[cls] += 1
            frame_total += 1
        for cls, c in frame_counts.items():
            if c > per_class_peak[cls]:
                per_class_peak[cls] = c
        if frame_total > fish_peak_total:
            fish_peak_total = frame_total
    return dict(per_class_peak), fish_peak_total


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🆚 ML vs Expert MaxN")
st.caption(
    f"Block-aligned comparison. Each video = **{VIDEO_DURATION_MIN:.2f} min**. "
    f"Expert CSV = **{INTERVAL_MIN}-min** blocks per deployment, starting at "
    f"recording minute {BAIT_SOAK_MIN} (the first {BAIT_SOAK_MIN} min are bait "
    f"soak and excluded from annotation). Block N = recording min "
    f"{BAIT_SOAK_MIN}N to {BAIT_SOAK_MIN}(N+1). For each video, we map which "
    f"deployment block(s) it fully covers, then compare expert MaxN (from "
    f"that block's CSV row) to ML MaxN (computed from frames in the same time "
    f"window inside the video)."
)

expert_df = load_expert()

# Confidence threshold slider — affects every ML count on this page
conf_threshold = st.slider(
    "ML confidence threshold",
    min_value=0.25,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help=(
        "Detections below this confidence are excluded from ML counts. YOLO's "
        "default at inference time was 0.25; the label files include everything "
        "≥ 0.25, so any slider value ≥ 0.25 just re-filters them on the fly."
    ),
)

MIN_COVERAGE_PCT = (
    25  # rows with less than this % of the 6-min block covered are hidden
)

# ── Section 1: Block-aligned comparison table ─────────────────────────────────

st.divider()
st.header("1. Comparison table — block-aligned (rows ≥25% covered)")
st.caption(
    f"One row per (video, 6-min block) with at least **{MIN_COVERAGE_PCT}%** of "
    "the block covered by the video. Expert MaxN for a partial block is still "
    "the full interval's CSV value — so ML may look low when coverage is partial."
)

species_cols = [c for c in expert_df.columns if c.endswith("_maxn")]

comparison_rows = []
for video_stem, run_info in RUNS.items():
    deployment_id, video_num = parse_video_stem(video_stem)
    species_labels_dir = SMOKETEST_DIR / run_info["species_run"] / "labels"
    community_labels_dir = SMOKETEST_DIR / run_info["community_run"] / "labels"
    for b in blocks_for_video(video_num):
        # Compute coverage fraction (full = 1.0, partial < 1.0)
        block_min = b["dep_end_min"] - b["dep_start_min"]
        covered_min = (b["video_sec_end"] - b["video_sec_start"]) / 60
        coverage_pct = (covered_min / block_min) * 100

        # Hide rows where the video covers less than MIN_COVERAGE_PCT of the block
        if coverage_pct < MIN_COVERAGE_PCT:
            continue

        # Species model in this block's window
        ml_per_class, ml_fish_peak = ml_in_window(
            species_labels_dir,
            video_stem,
            b["video_sec_start"],
            b["video_sec_end"],
            run_info["stride"],
            conf_threshold=conf_threshold,
        )

        # Expert row N for deployment_id = the Nth interval (1-indexed)
        expert_match = expert_df[expert_df["deployment_id"] == deployment_id]
        expert_row = (
            expert_match.iloc[b["block"] - 1]
            if not expert_match.empty and b["block"] <= len(expert_match)
            else None
        )

        def exp(col):
            if expert_row is None or pd.isna(expert_row[col]):
                return None
            return int(expert_row[col])

        expert_fish_max = None
        if expert_row is not None:
            vals = [int(expert_row[c]) for c in species_cols if pd.notna(expert_row[c])]
            expert_fish_max = int(max(vals)) if vals else 0

        comparison_rows.append(
            {
                "deployment_id": deployment_id,
                "block": b["block"],
                "block_dep_min": f"{b['dep_start_min']}–{b['dep_end_min']}",
                "video_sec": f"{b['video_sec_start']:.0f}–{b['video_sec_end']:.0f}",
                "coverage": (
                    "✅ FULL" if b["fully_covered"] else f"⚠️ {coverage_pct:.0f}%"
                ),
                "expert_match": "✅" if expert_row is not None else "—",
                "fish_expert": expert_fish_max,
                "fish_ml": ml_fish_peak,
                "snapper_expert": exp("snapper_maxn"),
                "snapper_ml": ml_per_class.get(2, 0),
                "spotty_expert": exp("spotty_maxn"),
                "spotty_ml": ml_per_class.get(1, 0),
            }
        )

if not comparison_rows:
    st.warning(
        f"No blocks with ≥{MIN_COVERAGE_PCT}% coverage. Run ML on a video that "
        "spans more of a 6-min interval to see a comparison."
    )
    st.stop()

df = pd.DataFrame(comparison_rows)

# Multi-level columns for the display
display_df = df.copy()
display_df.columns = pd.MultiIndex.from_tuples(
    [
        ("Deployment", ""),
        ("Block", "#"),
        ("Block", "Dep min"),
        ("Block", "Video sec"),
        ("Coverage", ""),
        ("Expert match", ""),
        ("Fish (any)", "Expert (max species)"),
        ("Fish (any)", "ML (peak/frame)"),
        ("Snapper", "Expert"),
        ("Snapper", "ML"),
        ("Spotty", "Expert"),
        ("Spotty", "ML"),
    ]
)
st.dataframe(display_df, hide_index=True, use_container_width=True)

# ── Section 2: Comparison chart ───────────────────────────────────────────────

st.divider()
st.header("2. Fish (any) MaxN — Expert vs ML per block")

chart_df = df[["deployment_id", "block", "fish_expert", "fish_ml"]].copy()
chart_df["row_label"] = (
    chart_df["deployment_id"] + " · blk " + chart_df["block"].astype(str)
)
chart_long = chart_df.melt(
    id_vars=["row_label"],
    value_vars=["fish_expert", "fish_ml"],
    var_name="Source",
    value_name="Fish MaxN",
)
chart_long["Source"] = chart_long["Source"].map(
    {
        "fish_expert": "Expert (max species)",
        "fish_ml": "ML (peak/frame)",
    }
)

# Skip rows with no value at all
chart_long = chart_long.dropna(subset=["Fish MaxN"])

if chart_long.empty:
    st.info("No values to plot — both Expert and ML are missing for all blocks.")
else:
    fig = px.bar(
        chart_long,
        x="row_label",
        y="Fish MaxN",
        color="Source",
        barmode="group",
        title="Fish MaxN per (deployment, block) — Expert (sum of species) vs ML (peak detections / frame)",
        labels={"row_label": "(deployment · block)"},
    )
    fig.update_layout(height=380, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

# ── Section 3: Video → block mapping (reference) ──────────────────────────────

st.divider()
st.header("3. Video ↔ block mapping (reference)")
st.caption("Per-video block coverage detail — useful for verifying the time math.")

for video_stem in sorted(RUNS.keys()):
    deployment_id, video_num = parse_video_stem(video_stem)
    v_start = (video_num - 1) * VIDEO_DURATION_MIN
    v_end = video_num * VIDEO_DURATION_MIN
    blocks = blocks_for_video(video_num)
    full_blocks = [b for b in blocks if b["fully_covered"]]

    with st.expander(
        f"**`{video_stem}`** — v{video_num:03d} of `{deployment_id}`", expanded=False
    ):
        st.markdown(
            f"- Video position in deployment: "
            f"**({video_num}-1) × {VIDEO_DURATION_MIN:.2f} = {v_start:.2f} min** "
            f"to **{video_num} × {VIDEO_DURATION_MIN:.2f} = {v_end:.2f} min**"
        )
        st.markdown(
            f"- Fully-covered blocks: "
            f"**{[b['block'] for b in full_blocks] if full_blocks else 'none'}**"
        )

        block_rows = [
            {
                "Block #": b["block"],
                "Dep min": f"{b['dep_start_min']}–{b['dep_end_min']}",
                "Video sec": f"{b['video_sec_start']:.0f}–{b['video_sec_end']:.0f}",
                "Video min": f"{b['video_sec_start']/60:.2f}–{b['video_sec_end']/60:.2f}",
                "Coverage": ("✅ FULL" if b["fully_covered"] else "⚠️ partial"),
            }
            for b in blocks
        ]
        st.dataframe(
            pd.DataFrame(block_rows), hide_index=True, use_container_width=True
        )

# ── Methodology ───────────────────────────────────────────────────────────────

with st.expander("Methodology & caveats", expanded=False):
    st.markdown(
        f"""
**Time mapping:**
- Each video is **{VIDEO_DURATION_MIN:.2f} min** ({VIDEO_DURATION_SEC} sec). Video `vNNN` starts at deployment minute **(NNN-1) × {VIDEO_DURATION_MIN:.2f}**.
- Expert CSV slices the deployment into **{INTERVAL_MIN}-min blocks** — row N of the CSV (for that deployment_id) corresponds to block N (1-indexed).
- Block 1 (0–{BAIT_SOAK_MIN} min) excluded as bait soak.
- A block is "fully covered" by a video only if both its start and end fall within the video's range — partial overlaps are skipped (can't compute a fair MaxN from a partial interval).

**Expert MaxN:**
- Per-species: taken directly from the matching block's CSV row.
- "Fish (any)" Expert column: sum of all `*_maxn` columns for that row. May over-count if different species peaks happen at the same wall-clock time (but for MaxN that's expected behaviour — each species' peak is tracked independently).

**ML MaxN (this page):**
- For each fully-covered block, we compute MaxN using **only the frames inside that block's time window** within the video.
- Per-species: max single-frame detection count for that class within the window.
- "Fish (any)" ML column: peak total detections across ALL classes (species + class-6 fish) in any single frame within the window.

**Why neither test video has expert match right now:**
- Test videos are from deployment time-blocks `_EM` and `_MD` (early morning, midday). The CSV for the same site/date contains only `_AM`, `_PM` — different deployments. Comparison rows show ML values only; Expert columns are blank.

**Species mappings (expert column → model class):**
- `snapper_maxn` → class 2 (*Pagrus auratus*)
- `spotty_maxn` → class 1 (*Notolabrus celidotus*)
- 17 other expert species columns roll up into "Fish (any) Expert" only — no direct model class.
"""
    )
