"""Shared data layer for the UoA (Underwood & Jeffs) mussel-farm BUV dataset.

One home for the constants, loaders and comparison metrics that the Mussel
Insights dashboard and the ML-vs-Expert comparison both need. They each grew
their own copies of the paths, the raw-CSV filename regex, the chapter length
and the label maps, and copies of facts about an external dataset drift.

Dataset shape (see ``uoa/docs/uoa_design.md``):

* **Expert annotations**, ``Annotations_CLEANED_v2.csv``: one row per
  6-minute interval per deployment, with per-species ``<Species> MaxN``
  columns. Rows flagged ``unannotated=True`` are placeholders where no expert
  ever looked (kept distinct from "looked and saw none").
* **ML detections**, per GoPro-chapter raw CSVs under ``*_raw/`` named
  ``{deployment}_v{chapter}_{model}_raw.csv``, holding every detection at
  confidence ≥ 0.15 (the inference floor).
* **deployment_id** = ``SITE_HAB_YYYYMMDD_BUCKET_TREAT``
  (e.g. ``ESK_B_20220118_EV_C``).

All of it lives on S3 under ``uoa/kutai/`` and is mirrored into the same path
locally by ``ensure_local_data``; see Locations below.
"""

import hashlib
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from spyfish.config.wrapper import config
from spyfish.ml.maxn_rules import centered_rolling_min
from spyfish.storage.s3_handler import S3Handler

# ── Locations ────────────────────────────────────────────────────────────────
#
# S3 is the source of truth: `uoa/kutai/` in the bucket named by the
# `[UOA_AWS]` secret, laid out like the Spyfish side (`metadata/` for the
# study's own records, `process_files/deployment_data/<deployment>/` for what the
# pipeline produced). Locally the same tree sits at `uoa/kutai/` (gitignored),
# so a path reads the same on the Mac, on NeSI and in the bucket.
#
# `ensure_local_data` downloads only what this app reads (`app_reads`), not the
# whole prefix: the deployment folders also hold BIIGLE frames the app never shows.

UOA_PREFIX = "uoa/kutai/"

DATA_ROOT = config.project_root / "uoa" / "kutai"
EXPERT_CSV = DATA_ROOT / "metadata" / "Annotations_CLEANED_v2.csv"
DEPLOYMENT_DATA = DATA_ROOT / "process_files" / "deployment_data"
# Every *_raw.csv under the deployment folders is discovered (<deployment>/annotations/).
RAW_ROOT = DEPLOYMENT_DATA
# Per-deployment MaxN frames + manifest, `<deployment>/maxn_frames/`, rendered
# on NeSI by uoa/scripts/render_deployment_maxn_frames.py. Loaders search for
# the manifests, so FRAMES_ROOT is the deployment folders' parent.
FRAMES_ROOT = DEPLOYMENT_DATA
# The frozen rules of the run that made the raw CSVs: a copy of
# process_files/wip_two_stage/uoa/frozen_rules.yaml (freeze_rules.py, NeSI),
# uploaded with the CSVs. See `default_conf`.
FROZEN_RULES_YAML = DATA_ROOT / "process_files" / "frozen_rules.yaml"


def app_reads(rel: str) -> bool:
    """Whether a key (relative to UOA_PREFIX) is one this app loads."""
    return (
        rel == "metadata/Annotations_CLEANED_v2.csv"
        or rel == FROZEN_RULES_YAML.relative_to(DATA_ROOT).as_posix()
        or (
            rel.startswith("process_files/deployment_data/")
            and (rel.endswith("_raw.csv") or "/maxn_frames/" in rel)
        )
    )


@st.cache_resource(show_spinner="Fetching the UoA data from S3…")
def ensure_local_data() -> str:
    """Mirror `s3://{S3_BUCKET}/{UOA_PREFIX}` into DATA_ROOT; returns a status line.

    Downloads only files that are missing locally or differ in size, so a warm
    cache costs one listing request. Credentials come from the `[UOA_AWS]`
    secret (`S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
    `AWS_REGION`), a key for this bucket alone. In a section, not at the top
    level: Streamlit exports top-level secrets as environment variables, and
    `AWS_*` / `S3_BUCKET` there would replace whatever the process got from
    `.env`, which `spyfish` config reads too. Without the secret, the cache is
    used as it is. Cached for the life of the server.
    """
    try:
        creds = st.secrets.get("UOA_AWS")
    except FileNotFoundError:  # no secrets.toml at all (StreamlitSecretNotFoundError)
        creds = None
    if not creds:
        return "S3 not configured (no [UOA_AWS] secret), using the local cache."
    bucket = creds["S3_BUCKET"]
    # S3Handler takes an injected client: it otherwise builds one from the
    # pipeline's env credentials, which belong to the DOC bucket.
    s3 = S3Handler(
        s3_client=boto3.client(
            "s3",
            aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
            region_name=creds["AWS_REGION"],
        ),
        bucket=bucket,
    )
    objects = s3.get_objects_from_s3(prefix=UOA_PREFIX)
    root = DATA_ROOT.resolve()
    todo = []
    wanted = 0
    for obj in objects:
        rel = obj["Key"][len(UOA_PREFIX) :]
        if not rel or rel.endswith("/") or not app_reads(rel):
            continue
        wanted += 1
        local = (DATA_ROOT / rel).resolve()
        # A key is a path we did not write: never let one escape the data root.
        if not local.is_relative_to(root):
            raise ValueError(f"S3 key escapes the data root: {obj['Key']}")
        if not local.exists() or local.stat().st_size != obj["Size"]:
            todo.append((obj["Key"], local))

    def fetch(item):
        key, local = item
        local.parent.mkdir(parents=True, exist_ok=True)
        if not s3.download_object_from_s3(key, str(local), callback=lambda _: None):
            raise RuntimeError(f"Download failed: s3://{bucket}/{key}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(fetch, todo))
    logging.info(f"UoA data: {wanted} app files on S3, {len(todo)} downloaded")
    return f"{wanted} files from s3://{bucket}/{UOA_PREFIX}, {len(todo)} downloaded."


RAW_FILENAME_RE = re.compile(r"^(?P<dep>.+)_v(?P<chap>\d+)_(?P<model>.+)_raw\.csv$")

# ── Diversity indices ────────────────────────────────────────────────────────
#
# Pure functions over a vector of per-species counts (MaxN). Shannon and
# Pielou's evenness are the two indices the Underwood & Jeffs paper reports
# (see uoa/docs/uoa_methods_draft.md); Simpson is shown alongside. Moved
# here from app/ecology_data.py on 2026-11-10 once the DOC app stopped using
# them, so this app no longer imports the DOC data layer.
#
# All three take counts in any order; zeros and NaNs are dropped. None of them
# know where the counts came from, so they work equally on a deployment, a site
# or a whole reserve, the caller decides what to aggregate first.
#
# What each one is for:
#   shannon , rewards having many species AND having them in balance. Its
#              ceiling is ln(S), so it is NOT comparable between datasets with
#              different species pools.
#   simpson , probability two individuals drawn at random are different
#              species. Less sensitive to rare species than Shannon.
#   pielou  . Shannon divided by its own maximum, so bounded 0-1 and therefore
#              comparable across datasets. 1 = perfectly even, near 0 = one
#              species dominates.


def _clean_counts(counts) -> list[float]:
    """Positive counts only, as plain floats."""
    return [float(c) for c in counts if pd.notna(c) and float(c) > 0]


def shannon(counts) -> float:
    """Shannon diversity H' = -sum(p * ln p)."""
    vals = _clean_counts(counts)
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log(c / total) for c in vals))


def simpson(counts) -> float:
    """Simpson's index in its 1-D form: higher = more diverse."""
    vals = _clean_counts(counts)
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float(1 - sum((c / total) ** 2 for c in vals))


def pielou(counts) -> float:
    """Pielou's evenness J' = H' / ln(S). Bounded 0-1.

    Zero when only one species is present: with nothing to be uneven against,
    evenness is undefined, and 0 is the conventional reported value.
    """
    vals = _clean_counts(counts)
    if len(vals) < 2:
        return 0.0
    return float(shannon(vals) / math.log(len(vals)))


# ── Dataset constants ────────────────────────────────────────────────────────

CHAPTER_SEC = 707.7  # nominal GoPro chapter length (FAT32 4-GB split @ 60fps)
INTERVAL_SEC = 360  # 6-min expert annotation interval
BAIT_SOAK_SEC = 360  # first 6 min of the recording (chapter 1), excluded, as
# in the paper; expert interval 1 already starts here
# The observers' "unidentified fish" column. A fish, so it counts toward total
# abundance; not a species, so it is left out of every species count (richness,
# Shannon, evenness, the species pool, dominant species). Decided 2026-11-10.
UNIDENTIFIED_COL = "UnID MaxN"
# Inference confidence of runs made without frozen rules (the June 2026 runs):
# their raw CSVs hold every box at or above it.
INFERENCE_CONF = 0.15
# Default for runs without frozen rules (decided 2026-11-12). On the June Esk
# Point runs against the BIIGLE expert boxes: F1 0.71 (0.72 at 0.15, 0.68 at
# 0.40), precision 0.88 vs 0.76 at 0.15, ML count / true count 0.67.
UNFROZEN_DEFAULT_CONF = 0.30


def _frozen_conf() -> float | None:
    """`detector.conf` of the run's frozen rules, or None without the file.

    Read on every call, never at import: on a fresh deploy the file arrives
    with `ensure_local_data`, after the modules are imported.
    """
    if not FROZEN_RULES_YAML.exists():
        return None
    return float(yaml.safe_load(FROZEN_RULES_YAML.read_text())["detector"]["conf"])


def default_conf() -> float:
    """The detection threshold every ML panel starts at.

    With frozen rules, the run's own cut: it kept only boxes at or above its
    frozen detector confidence (`freeze_rules.py`: best F1 on the val
    deployments), and a higher default drops boxes the run kept and, for
    counting, deepens the undercount (June Esk Point runs: ML count / true
    count 0.91 at 0.15, 0.67 at 0.30, 0.59 at 0.40). Without them,
    UNFROZEN_DEFAULT_CONF.
    """
    frozen = _frozen_conf()
    return UNFROZEN_DEFAULT_CONF if frozen is None else frozen


def conf_floor() -> float:
    """The lowest threshold that means anything: the CSVs hold nothing below it."""
    frozen = _frozen_conf()
    return INFERENCE_CONF if frozen is None else frozen


# A fish counts only if it stays this long: the frozen UoA rule
# (wip_two_stage/uoa/frozen_rules.yaml maxn.persistence_seconds, 2026-11-11,
# gap fill 0), the one maxn_compare.py applies. Without it, single-frame boxes
# on the seabed made ~30% of empty intervals look like sweep. 0 = every frame.
DEFAULT_PERSISTENCE_SEC = 2.0
# The frozen sampling step (ml_fps 1). Caps the step `persistent_peak` reads off
# the detection times: isolated boxes 5 s apart must not pass for a 5-s grid.
ML_SAMPLE_SEC = 1.0

# Days the UoA detector and classifier trained on (expert boxes from BIIGLE
# 4510). ML on any other day is on footage the models never saw: 20 Jan is the
# held-out day (hand counts, never trains), 21 Jan was never annotated at all.
TRAINING_DAYS = {"20220118", "20220119"}

SITE_NAMES = {
    "ESK": "Esk Point",
    "MOT": "Motukopake",
    "RAT": "Rat Island",
    "WHA": "Whanganui",
}
TREAT_LABEL = {"M": "Mussel farm", "C": "Control"}
DEPTH_LABEL = {"B": "Benthic", "S": "Surface"}

# Lucy's `Unique ID` in Camera data_working.xlsx: {SiteTreat}-{Depth}-{Window},
# e.g. EM-B-A = Esk mussel, benthic, afternoon. No date, so not a key; carried
# so joins to her analysis sheet are one-liners (see uoa_design.md).
LUCY_SITE = {"ESK": "E", "MOT": "M", "RAT": "R", "WHA": "W"}
LUCY_WINDOW = {"EM": "EM", "MM": "MM", "PM": "A", "EV": "E"}


def lucy_unique_id(deployment_id: str) -> str | None:
    parts = str(deployment_id).split("_")
    if len(parts) != 5 or parts[0] not in LUCY_SITE or parts[3] not in LUCY_WINDOW:
        return None
    return f"{LUCY_SITE[parts[0]]}{parts[4]}-{parts[1]}-{LUCY_WINDOW[parts[3]]}"


# ML class name → expert "<Species> MaxN" column. Class names are the
# scientific_name from the BIIGLE volume class_map.json (what
# build_volume_dataset.py --keep-species writes into data.yaml), with the
# common-name spellings older runs used. Only species a loaded model actually
# emits are compared per species (`active_species_map`); every expert column
# without a class of its own is what the model's catch-all should be counting
# (`other_expert_columns`). An unmapped class is skipped rather than silently
# compared against the wrong column.
ML_CLASS_TO_EXPERT = {
    "Chrysophrys auratus": "Snapper MaxN",
    "Pagrus auratus": "Snapper MaxN",  # class name in checkpoints trained before 2026-11-04
    "snapper": "Snapper MaxN",
    "Notolabrus celidotus": "Spotty MaxN",
    "spotty": "Spotty MaxN",
    "Girella tricuspidata": "Parore MaxN",
    "parore": "Parore MaxN",
    "Scorpis lineolata": "Sweep MaxN",
    "sweep": "Sweep MaxN",
    # The observers' "mackerel spp" column; the model class is jack mackerel.
    "Trachurus novaezelandiae": "mackerel spp MaxN",
    "jack mackerel": "mackerel spp MaxN",
    "Tripterygiidae": "Triplefin MaxN",
    "triplefin": "Triplefin MaxN",
    "Parika scaber": "Leatherjacket MaxN",
    "leatherjacket": "Leatherjacket MaxN",
}
# Kept for callers that still import it: every column that has a class.
EXPERT_TO_ML_CLASS = {col: cls for cls, col in ML_CLASS_TO_EXPERT.items()}
OTHER_ML_CLASS = "fish"  # species model's catch-all; the binary model's only class
OTHER_LABEL = "other"
FISH_ANY_LABEL = "fish_any"


def active_species_map(ml_class_names) -> dict[str, str]:
    """{expert column -> ML class} for the classes a loaded model emits.

    Walks ML_CLASS_TO_EXPERT so either spelling of a class lands on its
    column; the first spelling present wins per column.
    """
    out: dict[str, str] = {}
    for cls in sorted(ml_class_names):
        col = ML_CLASS_TO_EXPERT.get(cls)
        if col and col not in out:
            out[col] = cls
    return out


def other_expert_columns(maxn_cols, active: dict[str, str]) -> list[str]:
    """Expert columns with no class of their own."""
    return [c for c in maxn_cols if c not in active]


def has_species_catchall(ml_class_names) -> bool:
    """True when `fish` sits beside named classes (species model). For the
    binary model `fish` is everything, so there is no `other` to compare."""
    names = set(ml_class_names)
    return OTHER_ML_CLASS in names and len(names) > 1


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


def window_segments(chapter: int, start_sec: float) -> list[tuple[int, float, float]]:
    """The (chapter, start, end) pieces of one 6-min window.

    One piece, or two when the window overruns CHAPTER_SEC and spills into the
    next chapter. Shared by `maxn_in_window` and `window_coverage` so the count
    and its coverage always describe the same seconds.
    """
    end_sec = start_sec + INTERVAL_SEC
    if end_sec <= CHAPTER_SEC:
        return [(chapter, start_sec, end_sec)]
    return [
        (chapter, start_sec, CHAPTER_SEC),
        (chapter + 1, 0.0, end_sec - CHAPTER_SEC),
    ]


def window_coverage(
    raw_by_key: dict, deployment_id: str, chapter: int, start_sec: float
) -> float:
    """Share of the 6-min window (0-1) whose chapter has an ML raw CSV.

    `maxn_in_window` reads a missing chapter as zero detections, so without
    this an interval ML never ran on looks exactly like one where ML saw no
    fish. Typical cause: the spill into a next chapter that was not
    processed, or a recording that ended mid-window.
    """
    covered = sum(
        w_end - w_start
        for chap, w_start, w_end in window_segments(chapter, start_sec)
        if (deployment_id, chap) in raw_by_key
    )
    return covered / INTERVAL_SEC


def persistent_peak(d: pd.DataFrame, persistence_sec: float) -> int:
    """MaxN of one chapter's (or window's) detections: the peak per-frame count
    once every count is held to what lasted `persistence_sec`.

    Counts sit on the sampling grid (step = the smallest gap between detection
    times, at most ML_SAMPLE_SEC), unsampled slots are zero, and each slot takes the minimum over a
    window of persistence_sec centred on it: a box seen in one frame only is
    dropped, and a school of 8 seen for one frame among frames of 3 counts 3.
    The rule itself is spyfish/ml/maxn_rules.py, shared with the pipeline
    and maxn_compare.py.
    """
    if d.empty:
        return 0
    per_t = d.groupby("time_seconds").size()
    if persistence_sec <= 0:
        return int(per_t.max())
    t = per_t.index.to_numpy(float)
    gaps = np.diff(t)
    dt = min(float(gaps.min()), ML_SAMPLE_SEC) if len(gaps) else ML_SAMPLE_SEC
    window = max(1, round(persistence_sec / dt))
    slots = np.round((t - t[0]) / dt).astype(int)
    counts = np.zeros(slots[-1] + 1)
    np.maximum.at(counts, slots, per_t.to_numpy(float))
    return int(centered_rolling_min(counts, window).max())


def maxn_in_window(
    raw_by_key: dict,
    deployment_id: str,
    chapter: int,
    start_sec: float,
    conf: float,
    class_name: str | None = None,
    persistence_sec: float = DEFAULT_PERSISTENCE_SEC,
) -> int:
    """Peak per-frame detection count in [start_sec, start_sec+INTERVAL_SEC),
    after the persistence rule (`persistent_peak`).

    Spills into the next chapter when the window overruns CHAPTER_SEC. If
    class_name is given, counts only that class; otherwise counts all classes
    (fish-any). Returns 0 if the chapter(s) have no raw CSV: see
    `window_coverage` for telling that apart from a real zero.
    """
    peak = 0
    for chap, w_start, w_end in window_segments(chapter, start_sec):
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
        peak = max(peak, persistent_peak(d, persistence_sec))
    return peak


def deployment_peak(
    raw_by_key: dict,
    deployment_id: str,
    conf: float,
    class_name: str | None = None,
    persistence_sec: float = DEFAULT_PERSISTENCE_SEC,
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
        peak = max(peak, persistent_peak(d, persistence_sec))
    return peak


def species_deployment_table(
    expert_ml: pd.DataFrame,
    raw_by_key: dict,
    maxn_cols: list[str],
    active: dict[str, str],
    conf: float,
    catchall: bool,
    persistence_sec: float = DEFAULT_PERSISTENCE_SEC,
) -> pd.DataFrame:
    """One row per deployment x category: expert peak vs ML peak.

    The peak is the paper's unit, the highest interval value of the
    deployment. Categories: each mapped species; `other` (unmapped expert
    columns summed per interval, against the model's catch-all) when the model
    has one; and `fish_any` (all expert columns summed per interval, against
    every ML box). An unannotated deployment gets expert None. Site, depth,
    date, time bucket and treatment come from the deployment id, so the frame
    is ready for the Site / Deployment Time / Date tests.
    """
    other_cols = other_expert_columns(maxn_cols, active)
    rows = []
    for dep, sub in expert_ml.groupby("deployment_id", sort=False):
        unann = sub["unannotated"].astype(str).str.strip().str.lower().eq("true").all()
        num = sub[maxn_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        parts = dep.split("_")
        base = {
            "deployment_id": dep,
            "unique_id": lucy_unique_id(dep),
            "site": parts[0],
            "depth": DEPTH_LABEL.get(parts[1], parts[1]),
            "date": parts[2],
            "time_bucket": parts[3],
            "treatment": TREAT_LABEL.get(parts[-1], parts[-1]),
        }
        cats = [
            (
                col.replace(" MaxN", ""),
                num[col].max(),
                deployment_peak(raw_by_key, dep, conf, cls, persistence_sec),
            )
            for col, cls in active.items()
        ]
        if catchall:
            ex_other = num[other_cols].sum(axis=1).max() if other_cols else 0
            cats.append(
                (
                    OTHER_LABEL,
                    ex_other,
                    deployment_peak(
                        raw_by_key, dep, conf, OTHER_ML_CLASS, persistence_sec
                    ),
                )
            )
        cats.append(
            (
                FISH_ANY_LABEL,
                num.sum(axis=1).max(),
                deployment_peak(raw_by_key, dep, conf, None, persistence_sec),
            )
        )
        for name, ex, ml in cats:
            rows.append(
                {
                    **base,
                    "category": name,
                    "expert_peak": None if unann else int(ex),
                    "ml_peak": int(ml),
                }
            )
    return pd.DataFrame(rows)


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
