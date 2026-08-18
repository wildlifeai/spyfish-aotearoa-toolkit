"""Shared data layer for the UoA (Underwood & Jeffs) mussel-farm BUV dataset.

One home for the constants, loaders and comparison metrics that the Mussel
Insights dashboard and the ML-vs-Expert comparison both need. They each grew
their own copies of the paths, the raw-CSV filename regex, the chapter length
and the label maps, and copies of facts about an external dataset drift.

Dataset shape (see ``claude_docs/uoa_design.md``):

* **Expert annotations**, ``hold/Annotations_CLEANED_v2.csv``: one row per
  6-minute interval per deployment, with per-species ``<Species> MaxN``
  columns. Rows flagged ``unannotated=True`` are placeholders where no expert
  ever looked (kept distinct from "looked and saw none").
* **ML detections**, per GoPro-chapter raw CSVs under ``hold/`` named
  ``{deployment}_v{chapter}_{model}_raw.csv``, holding every detection at
  confidence ≥ 0.15 (the inference floor).
* **deployment_id** = ``SITE_HAB_YYYYMMDD_BUCKET_TREAT``
  (e.g. ``ESK_B_20220118_EV_C``).
"""

import hashlib
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Locations ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERT_CSV = REPO_ROOT / "hold" / "Annotations_CLEANED_v2.csv"
# Any *_raw.csv under hold/ is discovered (e.g. hold/20220118_raw/, future days).
RAW_ROOT = REPO_ROOT / "hold"
# Per-deployment MaxN frames + manifest, rendered on NeSI by
# render_deployment_maxn_frames.py and scp'd here.
FRAMES_ROOT = REPO_ROOT / "hold" / "uoa_frames"

RAW_FILENAME_RE = re.compile(r"^(?P<dep>.+)_v(?P<chap>\d+)_(?P<model>.+)_raw\.csv$")

# ── Dataset constants ────────────────────────────────────────────────────────

CHAPTER_SEC = 707.7  # nominal GoPro chapter length (FAT32 4-GB split @ 60fps)
INTERVAL_SEC = 360  # 6-min expert annotation interval
BAIT_SOAK_SEC = 360  # first 6 min of the recording (chapter 1), excluded, as
# in the paper; expert interval 1 already starts here
# Default re-filter threshold; raw CSVs hold everything ≥ 0.15.
DEFAULT_CONF = 0.40

SITE_NAMES = {
    "ESK": "Esk Point",
    "MOT": "Motukopake",
    "RAT": "Rat Island",
    "WHA": "Whanganui",
}
TREAT_LABEL = {"M": "Mussel farm", "C": "Control"}
DEPTH_LABEL = {"B": "Benthic", "S": "Surface"}

# Expert "<Species> MaxN" column → ML class name, and the reverse. Only species
# a loaded model actually emits can be compared per-species; everything else
# rolls into "fish (any)". An unmapped class is skipped rather than silently
# compared against the wrong column.
EXPERT_TO_ML_CLASS = {
    "Snapper MaxN": "Pagrus auratus",
    "Spotty MaxN": "Notolabrus celidotus",
}
ML_CLASS_TO_EXPERT = {
    "Pagrus auratus": "Snapper MaxN",
    "snapper": "Snapper MaxN",
    "Notolabrus celidotus": "Spotty MaxN",
    "spotty": "Spotty MaxN",
    "Girella tricuspidata": "Parore MaxN",
    "parore": "Parore MaxN",
    "Scorpis lineolata": "Sweep MaxN",
    "sweep": "Sweep MaxN",
    "Parika scaber": "Leatherjacket MaxN",
    "leatherjacket": "Leatherjacket MaxN",
}


def _load_site_coords() -> dict[tuple[str, str], tuple[float, float]]:
    """Real UoA survey positions, from secrets rather than from this file.

    Kept out of git: the positions belong to a collaborator's dataset, and a
    repository is the wrong place to make that call on their behalf.

    TOML has no tuple keys, so `.streamlit/secrets.toml` uses "SITE|Treatment":

        [UOA_SITE_COORDS]
        "ESK|Mussel farm" = [-36.012345, 175.012345]
        "ESK|Control"     = [-36.012345, 175.012345]

    Returns {} when the section is missing, so the map says it has no positions
    instead of drawing an empty ocean. A malformed entry is skipped rather than
    raising: one bad row must not take down a page that works without any map
    at all.
    """
    try:
        raw = st.secrets["UOA_SITE_COORDS"]
    except (KeyError, FileNotFoundError):
        return {}

    coords = {}
    for key, value in raw.items():
        site, _, treatment = str(key).partition("|")
        if not treatment or len(value) != 2:
            continue
        coords[(site, treatment)] = (float(value[0]), float(value[1]))
    return coords


SITE_COORDS = _load_site_coords()

# ── Fake coordinates for the comparison map ──────────────────────────────────
# The annotations CSV carries NO per-deployment GPS. These are arbitrary anchor
# points in the Hauraki Gulf region, one per site, used purely to give the
# ML-vs-Expert map something to plot. They are NOT the real survey locations.
SITE_FAKE_COORDS = {
    "ESK": (-36.80, 174.95),
    "MOT": (-36.88, 175.06),
    "RAT": (-36.72, 174.86),
    "WHA": (-36.97, 175.16),
}
FALLBACK_COORD = (-36.85, 175.00)  # any site not in the table above
CONTROL_LON_OFFSET = 0.020  # nudge controls east of the farm cluster


def fake_latlon(deployment_id: str, site: str, treatment: str) -> tuple[float, float]:
    """Deterministic fake (lat, lon) for one deployment.

    Clusters every deployment around its site anchor, shifts controls east of
    the farm cluster (so the farm/control split is visible), and adds a small
    per-deployment jitter derived from a hash of the deployment_id, stable
    across reruns (unlike random), and unique per deployment so points don't
    stack. ~0.006° ≈ 600 m of scatter.
    """
    base_lat, base_lon = SITE_FAKE_COORDS.get(site, FALLBACK_COORD)
    if treatment == "Control":
        base_lon += CONTROL_LON_OFFSET
    h = hashlib.md5(deployment_id.encode()).digest()
    jit_lat = ((h[0] / 255) * 2 - 1) * 0.006  # [-0.006, +0.006)
    jit_lon = ((h[1] / 255) * 2 - 1) * 0.006
    return round(base_lat + jit_lat, 5), round(base_lon + jit_lon, 5)


# ── Loaders ──────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Reading expert annotations…")
def load_expert_rows() -> tuple[pd.DataFrame, list[str]]:
    """Every expert annotation row, with derived columns and numeric MaxN.

    Returns (df, species_cols). All rows are kept, including the
    ``unannotated=True`` placeholders, callers decide whether "nobody looked"
    rows belong in their question. Derived columns: ``site``, ``depth``,
    ``treatment``, ``date`` (YYYY-MM-DD string), all parsed from
    ``deployment_id``. Some MaxN cells hold stray strings that ``notna``
    passes but ``float()`` rejects, so the block is coerced once here.
    """
    df = pd.read_csv(EXPERT_CSV)
    species_cols = [c for c in df.columns if re.search(r"max ?n", c, re.I)]
    df[species_cols] = df[species_cols].apply(pd.to_numeric, errors="coerce")

    parts = df["deployment_id"].str.split("_")
    df["site"] = parts.str[0]
    df["depth"] = parts.str[1].map(DEPTH_LABEL)
    df["treatment"] = parts.str[-1].map(TREAT_LABEL)
    df["date"] = pd.to_datetime(
        parts.str[2], format="%Y%m%d", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return df, species_cols


@st.cache_data(show_spinner="Reading machine detections…")
def load_ml_raw() -> tuple[dict, dict]:
    """Discover every *_raw.csv under hold/, keyed for window lookups.

    Returns:
      raw_by_key:  {(deployment_id, chapter_int): DataFrame}
      model_by_dep: {deployment_id: model_name}
    """
    raw_by_key: dict = {}
    model_by_dep: dict = {}
    for path in sorted(RAW_ROOT.rglob("*_raw.csv")):
        m = RAW_FILENAME_RE.match(path.name)
        if not m:
            continue
        raw_by_key[(m.group("dep"), int(m.group("chap")))] = pd.read_csv(path)
        model_by_dep[m.group("dep")] = m.group("model")
    return raw_by_key, model_by_dep


# ── Window / peak computations ───────────────────────────────────────────────


def maxn_in_window(
    raw_by_key: dict,
    deployment_id: str,
    chapter: int,
    start_sec: float,
    conf: float,
    class_name: str | None = None,
) -> int:
    """Peak per-frame detection count in [start_sec, start_sec+INTERVAL_SEC).

    Spills into the next chapter when the window overruns CHAPTER_SEC. If
    class_name is given, counts only that class; otherwise counts all classes
    (fish-any). Returns 0 if the chapter(s) have no raw CSV.
    """
    end_sec = start_sec + INTERVAL_SEC
    segments = []
    if end_sec <= CHAPTER_SEC:
        segments.append((chapter, start_sec, end_sec))
    else:
        segments.append((chapter, start_sec, CHAPTER_SEC))
        segments.append((chapter + 1, 0.0, end_sec - CHAPTER_SEC))

    peak = 0
    for chap, w_start, w_end in segments:
        df = raw_by_key.get((deployment_id, chap))
        if df is None or df.empty:
            continue
        d = df[
            (df["confidence"] >= conf)
            & (df["time_seconds"] >= w_start)
            & (df["time_seconds"] < w_end)
        ]
        if class_name is not None:
            d = d[d["class"] == class_name]
        if d.empty:
            continue
        # MaxN = max over frames of (detections in that frame)
        peak = max(peak, int(d.groupby("frame").size().max()))
    return peak


def deployment_peak(
    raw_by_key: dict, deployment_id: str, conf: float, class_name: str | None = None
) -> int:
    """Peak per-frame count across ALL chapters of a deployment, excluding the
    first 6 min of chapter 1 (bait soak / boat footage), matching the paper."""
    peak = 0
    for (dep, chap), df in raw_by_key.items():
        if dep != deployment_id or df is None or df.empty:
            continue
        d = df[df["confidence"] >= conf]
        if chap == 1:
            d = d[d["time_seconds"] >= BAIT_SOAK_SEC]
        if class_name is not None:
            d = d[d["class"] == class_name]
        if d.empty:
            continue
        peak = max(peak, int(d.groupby("frame").size().max()))
    return peak


# ── Disagreement metrics ─────────────────────────────────────────────────────


def severity(expert, ml):
    """A single 'level of mistake' score = magnitude × completeness.

        severity = |expert − ml|² / max(expert, ml, 1)
                 = (number missed) × (fraction missed)

    Plain % error saturates at 100% whenever ML predicts 0, so missing 1 fish
    and missing 20 fish look identical. This instead multiplies the count error
    by how completely ML failed: a *total* miss of a busy scene (called empty)
    scores its full count, while a *partial* miss (got the abundance roughly
    right) is discounted. None when unannotated.

    Examples: (20,0)→20 [called busy empty], (100,80)→4 [undercount on busy],
    (1,0)→1, (0,20)→20 [symmetric false alarm], (5,5)→0.
    """
    if expert is None:
        return None
    d = abs(expert - ml)
    return round(d * d / max(expert, ml, 1), 1)


def presence_flip(expert, ml):
    """Flag a presence/absence disagreement, one side says fish, the other says
    none (0 ↔ non-zero). The "is there *something* here?" rows worth a look,
    regardless of count. None when unannotated.
    """
    if expert is None:
        return None
    return "⚠️" if (expert == 0) != (ml == 0) else ""


def ml_call(expert, ml):
    """Direction of the ML error vs expert truth, as a symbol. None = unannotated.

    −  ML under-counted (predicted too few / missed fish)
    +  ML over-counted (predicted too many / false positives)
    =  exact match
    """
    if expert is None:
        return None
    if ml < expert:
        return "−"
    if ml > expert:
        return "+"
    return "="


# ── Small helpers ────────────────────────────────────────────────────────────


def chapter_from_symlink(symlink: str) -> int | None:
    """'ESK_B_20220118_EV_C/v_02.mp4' → 2."""
    if not isinstance(symlink, str):
        return None
    m = re.search(r"v_?(\d+)\.mp4$", symlink)
    return int(m.group(1)) if m else None


def date_key(dep: str):
    """Sort key: by date token first, so deployments order chronologically
    rather than alphabetically by reserve code."""
    parts = dep.split("_")
    return (parts[2] if len(parts) > 2 else "", dep)
