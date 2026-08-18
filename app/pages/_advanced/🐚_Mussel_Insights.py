"""
Mussel Insights, product-concept dashboard for mussel farm biodiversity monitoring.

Built for the TNC pitch. Shows what a monitoring service for shellfish farms /
restoration sites would look like, in two views:

* **Farm view**, one operator looking at their own farm.
* **Industry view**, an association or funder looking across all farms.

**Every number here is computed from the real UoA dataset** (Underwood & Jeffs,
4 Hauraki Gulf mussel farm sites, 18–21 Jan 2022): expert species annotations
from ``Annotations_CLEANED_v2.csv`` and YOLO detections from the per-chapter
``*_raw.csv`` files. Nothing is mocked. Where the concept implies a metric the
data cannot support (industry-wide scale, live survey cadence) the panel says so
rather than inventing a plausible-looking figure, a funder who finds one made-up
number discounts every real one next to it.

"Farm" here means an ``ESK`` / ``MOT`` / ``RAT`` / ``WHA`` site. Deployments
carry a treatment (mussel farm vs soft-sediment control), so the farm-vs-control
contrast is available at every level.

Site coordinates come from the published paper and are safe to plot, see the
note on ``SITE_COORDS``.

Run directly:
    streamlit run "app/pages/_advanced/🐚_Mussel_Insights.py"
"""

import sys
from pathlib import Path

# Allow launching this page directly as well as through the main entrypoint.
# Streamlit only puts the entrypoint's folder on sys.path, so the shared
# `theme` module in app/ would not resolve. parents[2] is app/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import base64  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from ecology_data import pielou, shannon, simpson  # noqa: E402
from theme import BIODIVERSITY_BANDS, PLOT_LAYOUT, biodiversity_band  # noqa: E402

st.set_page_config(page_title="Mussel Insights", page_icon="🐚", layout="wide")

# Unreleased work, shown under Explore as "Dashboard test". Its own secret, not
# APP_PASSWORD: showing a draft to one person must not hand over the rest of the
# app. `st.stop()` rather than an `if` block so the 1,900 lines below stay at
# module level exactly as they were.
from utils import check_password  # noqa: E402

# Title first: `check_password` draws its own input where it is called, so
# calling it before the heading puts a bare password box above an unlabelled
# page.
_gate = st.empty()
with _gate.container():
    st.title("🧪 Dashboard test")
    st.caption(
        "A work-in-progress dashboard concept, built on the University of "
        "Auckland mussel-farm BUV dataset. Ask Kalindi for the password."
    )
    _unlocked = check_password(
        "TEST_DASHBOARD_PASSWORD", label="Dashboard test password"
    )
if not _unlocked:
    st.stop()
# Cleared so the dashboard below starts with its own title rather than under
# this one.
_gate.empty()

# ── Data locations & dataset constants ───────────────────────────────────────
#
# All shared with the ML-vs-Expert comparison via `uoa_data`, the two had grown
# their own copies of the paths, the filename regex and the label maps, and
# copies of facts about an external dataset drift. See uoa_data for the notes
# on each (including why SITE_COORDS is safe to plot while Spyfish reserve
# coordinates are not).
from uoa_data import (  # noqa: E402
    CHAPTER_SEC,
    DEPTH_LABEL,
    EXPERT_CSV,
    FRAMES_ROOT,
    ML_CLASS_TO_EXPERT,
    RAW_FILENAME_RE,
    RAW_ROOT,
    SITE_COORDS,
    SITE_NAMES,
    TREAT_LABEL,
)

# Detections below this are dropped everywhere on this page. The raw CSVs keep
# everything >= 0.15 (the inference floor), so this re-filters rather than
# re-infers. (uoa_data.DEFAULT_CONF is the same value; kept as a named local
# because every caption on this page quotes it.)
CONF = 0.40

# ── Look and feel ────────────────────────────────────────────────────────────
#
# Streamlit's default chrome reads as "internal tool". This page is shown to
# funders, so the cards, KPI tiles and sidebar are restyled to look like a
# product. Colours for anything *data-bearing* still come from theme.py.

st.markdown(
    """
    <style>
      .stApp { background: #F4F7FB; }
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem;
                         max-width: 1600px; }

      /* Cards. Streamlit's bordered container */
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF; border: 1px solid #E4E9F2; border-radius: 12px;
        box-shadow: 0 1px 2px rgba(20,35,60,.04);
      }

      /* Streamlit's stock accent is a red that fights the navy sidebar. This is
         the token every widget reads, so overriding it recolours radios,
         checkboxes and the slider in one place. */
      :root, section[data-testid="stSidebar"] {
        --primary-color: #2E8B57 !important;
      }

      /* Map tiles carry a full CARTO/OpenStreetMap attribution line. Collapse it
         to the logo mark, which is the compact attribution form the tile terms
         allow, the licence link stays reachable through it. */
      .maplibregl-ctrl-attrib-inner, .mapboxgl-ctrl-attrib-inner { display: none; }
      .maplibregl-ctrl-attrib, .mapboxgl-ctrl-attrib { background: transparent; }

      .side-note { font-size: .7rem; color: #8FA0BE !important; line-height: 1.45;
                   margin-top: .45rem; }

      /* Sidebar rhythm. Streamlit's default block gap plus the slider's own
         value labels left a large hole between each control group. */
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: .55rem;
      }
      section[data-testid="stSidebar"] hr { margin: .75rem 0; }
      /* The slider prints its end values above the track, which collided with
         the section heading. */
      section[data-testid="stSidebar"] [data-testid="stSlider"] > div {
        padding-bottom: 0; padding-top: .35rem;
      }
      section[data-testid="stSidebar"] [data-testid="stCheckbox"] {
        margin-bottom: -.2rem;
      }

      /* Peak-abundance frame */
      .frame-img { width: 100%; height: auto; display: block; border-radius: 6px;
                   border: 1px solid #E4E9F2; }

      /* Sidebar */
      section[data-testid="stSidebar"] { background: #14233F; }
      section[data-testid="stSidebar"] * { color: #DDE5F2 !important; }
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
      /* Streamlit's default widget accent is a red that fights the navy. */
      section[data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"],
      section[data-testid="stSidebar"] [data-testid="stCheckbox"] [aria-checked="true"] {
        background-color: #2E8B57 !important; border-color: #2E8B57 !important;
      }
      section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background-color: #2E8B57 !important;
      }
      .side-h { font-size: .82rem; font-weight: 600; color: #FFFFFF;
                margin-bottom: .3rem; display: flex; align-items: center;
                gap: .34rem; }
      .side-i { display: inline-flex; align-items: center; justify-content: center;
                width: 13px; height: 13px; flex: 0 0 13px; border-radius: 50%;
                border: 1px solid #6C7E9C; color: #9FB0C9 !important;
                font-size: .58rem; font-weight: 700; font-style: normal;
                line-height: 1; padding: 0; background: transparent;
                cursor: pointer; }
      .side-i:hover { border-color: #FFFFFF; color: #FFFFFF !important; }
      .side-i:focus { outline: none; border-color: #FFFFFF; }
      .side-i:focus + .info-pop { display: block; }
      /* Sidebar popup opens downward, there is nothing above it to open into. */
      .side-h .info-pop { bottom: auto; top: 145%; left: -4px; right: auto; }
      .side-h .info-pop::after { top: auto; bottom: 100%; left: 8px; right: auto;
                                 border-top-color: transparent;
                                 border-bottom-color: #16233F; }

      /* KPI tile, no decorative icon. The title leads, the value follows, and
         the description line carries an ⓘ holding the definition on hover, so
         a funder can interrogate any number without the tile carrying a
         paragraph of method text. */
      .kpi { display: flex; flex-direction: column; }
      .kpi-label { font-size: .85rem; font-weight: 650; color: #16233F;
                   line-height: 1.3; }
      .kpi-value { font-size: 1.75rem; font-weight: 700; color: #16233F;
                   line-height: 1.1; margin-top: .35rem;
                   font-variant-numeric: tabular-nums; }
      .kpi-unit  { font-size: .8rem; font-weight: 500; color: #66748C;
                   margin-left: .2rem; }
      .kpi-sub   { font-size: .72rem; color: #7A879C; margin-top: .32rem;
                   line-height: 1.3; display: flex; align-items: flex-start;
                   gap: .34rem; }
      .kpi-i { display: inline-flex; align-items: center; justify-content: center;
               width: 14px; height: 14px; flex: 0 0 14px; margin-top: .1rem;
               padding: 0; background: transparent;
               border: 1px solid #B4BFD0; border-radius: 50%; color: #8896AB;
               font-size: .6rem; font-weight: 700; font-style: normal;
               line-height: 1; cursor: pointer; }
      .kpi-i:hover { border-color: #16233F; color: #16233F; }

      /* Click-to-open definition panel. Pure CSS: the button holds focus while
         open, and blurring it (a click anywhere else, or Esc) closes it. */
      .info-wrap { position: relative; display: inline-flex; }
      .info-pop { display: none; position: absolute; bottom: 145%; right: -4px;
                  width: 240px; background: #16233F; color: #EEF3FA;
                  font-size: .72rem; font-weight: 400; line-height: 1.45;
                  text-align: left; padding: .6rem .7rem; border-radius: 7px;
                  box-shadow: 0 8px 24px -8px rgba(16,35,60,.45); z-index: 999; }
      .info-pop::after { content: ""; position: absolute; top: 100%; right: 8px;
                         border: 5px solid transparent; border-top-color: #16233F; }
      .kpi-i:focus + .info-pop { display: block; }
      .kpi-i:focus { outline: none; border-color: #16233F; color: #16233F; }

      /* KPI cards keep their fixed height (that is what guarantees a flush row)
         but have their overflow forced visible, so the definition popup can
         escape the card instead of being clipped by it. Scoped with :has() so
         the scrolling chart panels below are untouched. */
      /* Every ancestor between the button and the page must be overflow:visible,
         not just the card, a single scrolling wrapper anywhere up the chain
         clips the popup. The column and the row are included for that reason. */
      div[data-testid="stHorizontalBlock"]:has(.kpi),
      div[data-testid="stColumn"]:has(.kpi),
      div[data-testid="stVerticalBlockBorderWrapper"]:has(.kpi),
      div[data-testid="stVerticalBlockBorderWrapper"]:has(.kpi) *,
      div[data-testid="stColumn"]:has(.kpi) [data-testid="stVerticalBlock"],
      div[data-testid="stColumn"]:has(.kpi) [data-testid="stElementContainer"],
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.side-h),
      section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.side-h) {
        overflow: visible !important;
      }
      /* The first and last tiles would push their popup past the page edge. */
      div[data-testid="stColumn"]:first-child .info-pop { left: -4px; right: auto; }
      div[data-testid="stColumn"]:first-child .info-pop::after { left: 8px; right: auto; }

      /* Panel headers, title and subtitle always occupy their own line, so
         every panel's chart starts at the same y and the row stays aligned
         however long the wording is. */
      .panel-h { font-size: .88rem; font-weight: 650; color: #16233F;
                 line-height: 1.25; white-space: nowrap; overflow: hidden;
                 text-overflow: ellipsis; }
      .panel-s { display: block; font-size: .72rem; color: #7A879C;
                 font-weight: 400; line-height: 1.3; margin-bottom: .35rem;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

      .row-label { font-size: .74rem; color: #8794A8; margin: .2rem 0 .5rem;
                   text-transform: uppercase; letter-spacing: .07em; }

      /* Provenance strip under each view title */
      .scope { font-size: .74rem; color: #8794A8; margin-top: .35rem;
               font-variant-numeric: tabular-nums; }

      /* Feed rows (alerts / activity / detections) */
      .feed { display: flex; gap: .6rem; align-items: flex-start;
              padding: .5rem 0; border-bottom: 1px solid #F0F3F8; }
      .feed:last-child { border-bottom: none; }
      .feed-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: .42rem;
                  flex: 0 0 8px; }
      .feed-txt { font-size: .82rem; color: #2B3A55; line-height: 1.4; }
      .feed-meta { font-size: .72rem; color: #94A0B4; white-space: nowrap;
                   margin-left: auto; padding-left: .5rem; }

      .band-pill { display: inline-block; padding: .1rem .5rem; border-radius: 20px;
                   font-size: .7rem; font-weight: 600; color: #FFF; }

      div[data-testid="stDataFrame"] { border: none; }
      hr { margin: .6rem 0; border-color: #EDF1F7; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Loaders ──────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Reading expert annotations…")
def load_expert() -> tuple[pd.DataFrame, list[str]]:
    """Expert annotation rows, one per 6-min interval, with species MaxN numeric.

    Returns the annotated rows only (``unannotated`` placeholders dropped) plus
    the list of species MaxN column names. Some cells hold stray strings that
    ``notna`` passes but ``float()`` rejects, so the block is coerced once here.
    """
    df = pd.read_csv(EXPERT_CSV)
    species_cols = [c for c in df.columns if re.search(r"max ?n", c, re.I)]
    df[species_cols] = df[species_cols].apply(pd.to_numeric, errors="coerce")

    df["site"] = df["deployment_id"].str.split("_").str[0]
    df["depth"] = df["deployment_id"].str.split("_").str[1].map(DEPTH_LABEL)
    df["treatment"] = df["deployment_id"].str.split("_").str[-1].map(TREAT_LABEL)

    annotated = df[df["unannotated"].astype(str).str.strip().str.lower() != "true"]
    return annotated.copy(), species_cols


@st.cache_data(show_spinner="Reading per-class detections…")
def load_ml_by_class() -> pd.DataFrame:
    """Per-deployment MaxN for every class the detector emits.

    Written for the species detector that is coming, not just the binary one in
    front of us. The moment a model emits real class names this returns one row
    per (deployment, species) and the per-species agreement panel starts working
    with no further change; with today's binary models it returns a single
    ``fish`` class and the panel says so rather than faking a comparison.

    MaxN per class = the peak number of that class in any single frame of the
    deployment, taken as the max across the deployment's chapter files.
    """
    rows = []
    for path in sorted(RAW_ROOT.rglob("*_raw.csv")):
        m = RAW_FILENAME_RE.match(path.name)
        if not m:
            continue
        df = pd.read_csv(path)
        if not {"confidence", "class", "frame"} <= set(df.columns):
            continue
        kept = df[df["confidence"] >= CONF]
        if kept.empty:
            continue
        # Peak per class within this chapter, then max across chapters below.
        per_class = kept.groupby(["class", "frame"]).size().groupby("class").max()
        for cls, peak in per_class.items():
            rows.append(
                {"deployment_id": m.group("dep"), "class": cls, "peak": int(peak)}
            )
    if not rows:
        return pd.DataFrame(columns=["deployment_id", "class", "peak"])
    out = pd.DataFrame(rows).groupby(["deployment_id", "class"])["peak"].max()
    return out.reset_index()


@st.cache_data(show_spinner="Reading machine detections…")
def load_ml() -> pd.DataFrame:
    """One row per deployment-chapter with its detection stats.

    Aggregates on read rather than keeping the ~95k raw detections in memory,
    every panel here works off per-deployment summaries.
    """
    rows = []
    for path in sorted(RAW_ROOT.rglob("*_raw.csv")):
        m = RAW_FILENAME_RE.match(path.name)
        if not m:
            continue
        df = pd.read_csv(path)
        if "confidence" not in df.columns:
            continue  # a non-UoA raw file with a different schema
        kept = df[df["confidence"] >= CONF]
        peak = int(kept.groupby("frame").size().max()) if not kept.empty else 0
        rows.append(
            {
                "deployment_id": m.group("dep"),
                "chapter": int(m.group("chap")),
                "model": m.group("model"),
                "detections": len(kept),
                "mean_conf": (
                    kept["confidence"].mean() if not kept.empty else float("nan")
                ),
                "peak_in_frame": peak,
            }
        )
    if not rows:
        # No raw CSVs under hold/ (fresh checkout, or the scp from NeSI has
        # not happened). Typed empty frame so downstream filters and groupbys
        # get their columns instead of a KeyError.
        return pd.DataFrame(
            columns=[
                "deployment_id",
                "chapter",
                "model",
                "detections",
                "mean_conf",
                "peak_in_frame",
                "site",
                "treatment",
                "date",
            ]
        )
    ml = pd.DataFrame(rows)
    ml["site"] = ml["deployment_id"].str.split("_").str[0]
    ml["treatment"] = ml["deployment_id"].str.split("_").str[-1].map(TREAT_LABEL)
    # Date lives in the deployment id (SITE_HAB_YYYYMMDD_BUCKET_TREAT), not in
    # the detection files, parsed here so the survey-window filter can apply to
    # machine output as well as expert annotations.
    ml["date"] = pd.to_datetime(
        ml["deployment_id"].str.split("_").str[2], format="%Y%m%d", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return ml


@st.cache_data
def encode_frame(path: Path) -> str:
    """Base64 a rendered frame so it can be inlined in an <img> tag."""
    return base64.b64encode(path.read_bytes()).decode()


@st.cache_data
def load_frames() -> pd.DataFrame:
    """Rendered peak-abundance frames, one per deployment, if any were scp'd here."""
    manifests = list(FRAMES_ROOT.rglob("deployment_maxn_manifest.csv"))
    if not manifests:
        return pd.DataFrame()
    man = pd.concat(
        [pd.read_csv(m).assign(_dir=str(m.parent)) for m in manifests],
        ignore_index=True,
    )
    man["site"] = man["deployment_id"].str.split("_").str[0]
    return man[man["frame_file"].notna()]


# ── Metrics ──────────────────────────────────────────────────────────────────


def biodiversity_score(species_totals: pd.Series, n_species_pool: int) -> float:
    """Shannon diversity rescaled to 0–100 against the whole recorded species pool.

    H' alone is not comparable between readers because its ceiling depends on how
    many species could have been seen. Dividing by ``ln(pool)``, the maximum H'
    if every species in the dataset were present in equal numbers, puts every
    site on the same 0–100 axis. It is a *relative condition index*, not an
    absolute ecological grade, and the page says so wherever it appears.
    """
    if n_species_pool < 2:
        return 0.0
    return round(100 * shannon(species_totals) / math.log(n_species_pool), 1)


def kpi(label, value, unit="", sub="", help_text=""):
    """One KPI tile: title, value, then a description line with a click-to-open ⓘ.

    ``help_text`` is the definition a reader needs to trust the number, how it
    is computed and what it does not mean. It opens on click rather than hover:
    a hover tooltip is invisible on touch devices and fires accidentally when
    the pointer crosses a dense KPI row.

    The open/closed state is pure CSS, the ⓘ is a real ``<button>``, and
    ``button:focus + .info-pop`` reveals the panel. Clicking elsewhere blurs the
    button and closes it, and Esc/Tab work for free, with no JavaScript (which
    Streamlit strips from markdown anyway).
    """
    info = (
        (
            f'<span class="info-wrap">'
            f'<button class="kpi-i" type="button" aria-label="About {label}">i</button>'
            f'<span class="info-pop">{help_text}</span>'
            f"</span>"
        )
        if help_text
        else ""
    )
    st.markdown(
        f"""<div class="kpi">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
              <div class="kpi-sub"><span>{sub}</span>{info}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def panel_header(title, sub=""):
    st.markdown(
        f'<div class="panel-h">{title} <span class="panel-s">{sub}</span></div>',
        unsafe_allow_html=True,
    )


def feed_row(color, text, meta):
    st.markdown(
        f"""<div class="feed">
              <div class="feed-dot" style="background:{color}"></div>
              <div class="feed-txt">{text}</div>
              <div class="feed-meta">{meta}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def band_pill(score):
    label, color = biodiversity_band(score)
    return f'<span class="band-pill" style="background:{color}">{label}</span>'


def clean_species(col: str) -> str:
    """'goatfish MaxN (U.porosus)' → 'Goatfish (U.porosus)'."""
    name = re.sub(r"\s*max\s?n\s*", " ", col, flags=re.I).strip()
    return name[:1].upper() + name[1:]


def dep_species_maxn(exp_scope) -> pd.DataFrame:
    """Per-deployment MaxN for every species: one row per deployment.

    MaxN is a **per-deployment** statistic, the peak number of individuals of a
    species visible in any single frame of that deployment. The annotation table
    stores it per 6-minute interval, so the deployment value is the MAX across
    its intervals, never the sum: the same school sitting in front of the camera
    for twenty minutes appears in every interval, and summing would count it
    again each time.
    """
    return exp_scope.groupby("deployment_id")[SPECIES_COLS].max()


def species_totals(exp_scope) -> pd.Series:
    """Total MaxN per species = each deployment contributing its own peak once."""
    return dep_species_maxn(exp_scope).sum()


def dep_abundance(exp_scope) -> pd.Series:
    """Per-deployment total abundance = sum of that deployment's species MaxN.

    This is the paper's "total abundance". It is a sum of per-species peaks, not
    a true multi-species MaxN, different species peak at different moments, so
    it reads slightly high. Kept because it is what the literature does and what
    the comparison study reported.
    """
    return dep_species_maxn(exp_scope).sum(axis=1)


def render_kpi_row(exp_scope, ml_scope, scope: str):
    """The six headline numbers, in one fixed order, for whatever slice is passed.

    Chosen for ecological meaning rather than for what the pipeline happens to
    produce: abundance, richness, effect size, diversity, dominance, effort.
    Detector counts and confidence are deliberately absent, they belong to the
    ML view, and a detection total invites being read as a fish count.
    """
    totals = species_totals(exp_scope)
    per_dep = dep_abundance(exp_scope)
    maxn = per_dep.mean() if len(per_dep) else 0
    n_species = int((totals > 0).sum())
    # Richness is only meaningful against something. Carrying the control value
    # in the same tile saves the reader holding two numbers in their head.
    farm_rows = exp_scope[exp_scope["treatment"] == "Mussel farm"]
    ctrl_rows = exp_v[exp_v["treatment"] == "Control"]
    n_farm = int((species_totals(farm_rows) > 0).sum()) if not farm_rows.empty else 0
    n_ctrl = int((species_totals(ctrl_rows) > 0).sum()) if not ctrl_rows.empty else 0
    score = biodiversity_score(totals, SPECIES_POOL)
    band = biodiversity_band(score)[0]

    farm = exp_scope[exp_scope["treatment"] == "Mussel farm"]
    ctrl = exp_v[exp_v["treatment"] == "Control"]
    ratio_txt, ratio_sub = "—", "no controls in selection"
    if not farm.empty and not ctrl.empty:
        f_ab = dep_abundance(farm).mean()
        c_ab = dep_abundance(ctrl).mean()
        ratio_txt = f"{f_ab / max(c_ab, .01):.1f}"
        ratio_sub = f"farm {f_ab:.0f} vs control {c_ab:.0f} fish"

    # Dominance is counted by deployments won, not by share of total individuals.
    # A share is abundance-weighted, so one drop with 200 of something decides it
    # even if that species was absent everywhere else. "Top species in 12 of 32
    # drops" is deployment-weighted and says what an operator actually sees.
    dom_txt, dom_sub = "—", "no species recorded"
    dep_max = dep_species_maxn(exp_scope)
    fished = dep_max[dep_max.sum(axis=1) > 0]
    if not fished.empty:
        wins = fished.idxmax(axis=1).value_counts()
        dom_txt = clean_species(wins.index[0])
        dom_sub = f"top species in {wins.iloc[0]} of {len(dep_max)} drops"

    k = st.columns(6)
    with k[0]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Abundance (MaxN)",
                f"{maxn:.0f}",
                " fish",
                "mean per deployment",
                "MaxN is the peak number of individuals visible in any single "
                "frame, summed across species and averaged over deployments. It "
                "is the standard non-invasive abundance metric because it can "
                "never double-count a fish that leaves and re-enters view, so it "
                "is a conservative floor on how many were really there.",
            )
    with k[1]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Species Richness",
                n_species,
                "",
                f"of {SPECIES_POOL} in survey · farm {n_farm} vs control {n_ctrl}",
                "Distinct species recorded by expert annotators. Richness is the "
                "first-order biodiversity measure, it counts what is present "
                "and ignores how many of each, which is why it is read next to "
                "diversity rather than instead of it.",
            )
    with k[2]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Farm vs Control",
                ratio_txt,
                "×",
                ratio_sub,
                "How many times more fish a farm deployment holds than the "
                "pooled soft-sediment reference sites. This is the effect size, "
                "the number that actually answers whether the habitat is doing "
                "anything. Controls are pooled because only Esk Point and "
                "Motukopake have them.",
            )
    with k[3]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Biodiversity Score",
                f"{score:g}",
                "/100",
                f"{band} · Shannon H′ rescaled",
                f"Shannon diversity H′ over the species counts for {scope}, "
                f"rescaled 0–100 against ln({SPECIES_POOL}). It rewards having "
                f"many species in balanced numbers. A composite index, useful "
                f"for comparing sites within this survey, not an absolute "
                f"ecological grade, and not comparable to a different species "
                f"pool.",
            )
    with k[4]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Dominant Species",
                dom_txt,
                "",
                dom_sub,
                "The species that was the most abundant one in the largest "
                "number of deployments. Counted by deployments won rather than "
                "by share of all individuals, so a single drop with a huge "
                "school cannot decide it. Winning most drops means low "
                "evenness, a community carried by one species is more fragile "
                "than the same headcount spread across many.",
            )
    with k[5]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Survey Effort",
                exp_scope["deployment_id"].nunique(),
                " drops",
                f"over {exp_scope['date'].nunique()} survey days",
                "Baited underwater video deployments with expert annotation. "
                "Effort is shown next to the results because abundance and "
                "richness both climb with survey effort, comparing two sites "
                "means little if one was sampled twice as hard.",
            )


# Fixed panel height so every chart panel in a row matches. KPI cards do NOT
# use this, they stretch via CSS instead, so their definition popups are not
# clipped by the container's overflow.
# much content it holds, a ragged bottom edge is the thing that most makes a
# dashboard look unfinished. Content taller than the box scrolls inside it.
H_KPI = 124
H_PANEL = 330


def render_map(
    points: pd.DataFrame,
    color_col: str,
    color_map: dict,
    height: int,
    key: str,
    zoom: float = 11.4,
):
    """Site map. `points` needs lat/lon/label/detail plus `color_col`."""
    fig = px.scatter_map(
        points,
        lat="lat",
        lon="lon",
        color=color_col,
        size="size",
        size_max=26,
        zoom=zoom,
        hover_name="label",
        hover_data={
            "detail": True,
            "lat": False,
            "lon": False,
            "size": False,
            color_col: False,
        },
        color_discrete_map=color_map,
        map_style="carto-positron",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=1.02,
            x=0,
            font=dict(size=10),
            title_text="",
            bgcolor="rgba(255,255,255,.75)",
        ),
    )
    st.plotly_chart(fig, key=key)


# ── Load ─────────────────────────────────────────────────────────────────────

if not EXPERT_CSV.exists():
    st.error(f"Expert annotations not found: `{EXPERT_CSV}`")
    st.stop()

expert, SPECIES_COLS = load_expert()
ml = load_ml()
frames = load_frames()

if expert.empty:
    st.error("No annotated expert rows found.")
    st.stop()

if ml.empty:
    # Non-fatal: the expert-only panels still work without detections.
    st.warning(
        "No ML raw CSVs found under `hold/` — the machine-detection panels "
        "will be empty. The `*_raw.csv` files are copied over from NeSI by "
        "hand, not produced by this app."
    )

SPECIES_POOL = int((expert[SPECIES_COLS].sum() > 0).sum())

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="font-size:1.15rem;font-weight:700;color:#FFF">Mussel Insights</div>'
        '<div style="font-size:.76rem;color:#8FA0BE;margin-bottom:1.2rem">'
        "Biodiversity insights for shellfish farms</div>",
        unsafe_allow_html=True,
    )
    view = st.radio(
        "View",
        [
            "Farm overview",
            "Industry overview",
            "ML performance",
            "ML vs Expert detail",
            "ML vs Expert (old)",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    all_dates = sorted(expert["date"].dropna().unique())
    st.markdown('<div class="side-h">Survey window</div>', unsafe_allow_html=True)
    if len(all_dates) > 1:
        date_from, date_to = st.select_slider(
            "Survey window",
            options=all_dates,
            value=(all_dates[0], all_dates[-1]),
            label_visibility="collapsed",
        )
    else:
        date_from = date_to = all_dates[0]

    st.divider()

    sites = sorted(expert["site"].dropna().unique())
    if view == "Farm overview":
        # A radio, not a dropdown, with only four farms every option fits, and
        # showing them all lets a reader see the whole survey at a glance.
        site = st.radio(
            "Farm",
            sites,
            format_func=lambda s: f"{SITE_NAMES.get(s, s)} ({s})",
        )
    else:
        site = None

    st.divider()

    # Both presentations of the same explanation are rendered so the wording can
    # be compared in place; drop whichever loses.
    st.markdown(
        '<div class="side-h">Deployment type'
        '<span class="info-wrap">'
        '<button class="side-i" type="button" aria-label="About deployment type">'
        "i</button>"
        '<span class="info-pop">Keeping both is what makes the biodiversity '
        "number mean something, a farm count is only meaningful next to the "
        "bare seabed it is compared against.</span></span></div>",
        unsafe_allow_html=True,
    )
    treatments = [
        t
        for t, on in [
            ("Mussel farm", st.checkbox("Mussel farm", value=True)),
            ("Control", st.checkbox("Control", value=True)),
        ]
        if on
    ]

if not treatments:
    st.warning("Select at least one deployment type in the sidebar.")
    st.stop()

in_window = expert["date"].between(date_from, date_to)
exp_v = expert[expert["treatment"].isin(treatments) & in_window]
ml_v = (
    ml[ml["treatment"].isin(treatments) & ml["date"].between(date_from, date_to)]
    if not ml.empty
    else ml
)

if exp_v.empty:
    st.warning("No deployments match the current filters.")
    st.stop()


def render_scope_note(exp_scope, ml_scope):
    """The one-line provenance strip that sits under each view's title."""
    st.markdown(
        f'<div class="scope">'
        f"{exp_scope['deployment_id'].nunique()} deployments · "
        f"{len(ml_scope)} video files · "
        f"{date_from} → {date_to} · "
        f"detections filtered at confidence ≥ {CONF:g}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SHARED PANELS
# ═════════════════════════════════════════════════════════════════════════════
#
# Both views render exactly these panels, in this order. Anything view-specific
# lives in its own clearly-marked extras row below, so switching between Farm and
# Industry never moves a box a reader has already learned the position of.


def panel_map(selected_site, key):
    """Survey positions. Farm view highlights one site; industry colours by score.

    Both render the same positions so the map never changes shape between
    views. Positions come from `UOA_SITE_COORDS` in secrets; without them this
    panel says so and the rest of the page is unaffected.
    """
    with st.container(border=True, height=H_PANEL):
        if selected_site:
            panel_header("Site Map", "(all positions · selected farm highlighted)")
        else:
            panel_header("Site Map", "(all positions · coloured by score)")
        if not SITE_COORDS:
            st.info(
                "No site positions configured. Add a `[UOA_SITE_COORDS]` "
                "section to `.streamlit/secrets.toml` to show the map."
            )
            return
        pts = []
        for (s, treat), (lat, lon) in SITE_COORDS.items():
            if treat not in treatments:
                continue
            sub = exp_v[(exp_v["site"] == s) & (exp_v["treatment"] == treat)]
            if sub.empty:
                continue
            per_dep = dep_abundance(sub)
            mean_ab = per_dep.mean() if len(per_dep) else 0
            if selected_site:
                role = (
                    "Control site"
                    if treat == "Control"
                    else "Selected farm" if s == selected_site else "Other farm"
                )
                size = max(mean_ab, 1)
            elif treat == "Control":
                role, size = "Control site", 20
            else:
                sc = biodiversity_score(species_totals(sub), SPECIES_POOL)
                role, size = biodiversity_band(sc)[0], max(sc, 1)
            pts.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "role": role,
                    "label": f"{SITE_NAMES.get(s, s)}, {treat}",
                    "detail": f"{mean_ab:.0f} mean fish · "
                    f"{sub['deployment_id'].nunique()} deployments",
                    "size": size,
                }
            )
        if not pts:
            st.info("No mapped positions for this selection.")
            return
        if selected_site:
            colours = {
                "Selected farm": "#1B7F4B",
                "Other farm": "#9CBBA8",
                "Control site": "#B98A4B",
            }
        else:
            colours = {label: colour for _, label, colour in BIODIVERSITY_BANDS}
            colours["Control site"] = "#B98A4B"
        render_map(pd.DataFrame(pts), "role", colours, height=248, key=key, zoom=11.2)


def panel_abundance(exp_scope, series_label, key):
    """Fish accumulating at the bait, per 6-min interval, against the survey mean."""
    with st.container(border=True, height=H_PANEL):
        panel_header(
            "Fish Abundance Over Time",
            "(mean fish per 6-min interval · well-sampled intervals only)",
        )
        # Deployments differ in length, so the last intervals exist for only a
        # few of them. Averaging over that handful made both curves nose-dive at
        # the right-hand end, an artefact of thinning sample size, not fish
        # leaving. Intervals covered by fewer than half the deployments are cut.
        n_by_interval = exp_scope.groupby("interval_idx")["deployment_id"].nunique()
        keep = n_by_interval[n_by_interval >= max(2, 0.5 * n_by_interval.max())].index
        this = (
            exp_scope.groupby("interval_idx")[SPECIES_COLS].sum().sum(axis=1)
            / n_by_interval
        ).loc[keep]
        n_all = exp_v.groupby("interval_idx")["deployment_id"].nunique()
        allf = (
            exp_v.groupby("interval_idx")[SPECIES_COLS].sum().sum(axis=1) / n_all
        ).loc[n_all.index.intersection(keep)]
        fig = go.Figure()
        if series_label != "All farms":
            fig.add_trace(
                go.Scatter(
                    x=allf.index,
                    y=allf.values,
                    name="All farms",
                    line=dict(color="#B9C4D6", width=2, dash="dot"),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=this.index,
                y=this.values,
                name=series_label,
                line=dict(color="#2D6DB4", width=2.5),
            )
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=210,
            legend=dict(orientation="h", y=1.16, x=0, font=dict(size=10)),
        )
        fig.update_xaxes(title="interval (6 min each)", title_font=dict(size=10))
        st.plotly_chart(fig, key=key)


def panel_richness(exp_scope, key):
    """How many distinct species are on screen as the deployment runs."""
    with st.container(border=True, height=H_PANEL):
        panel_header("Species Richness Over Time", "(species present per interval)")
        n_by_interval = exp_scope.groupby("interval_idx")["deployment_id"].nunique()
        keep = n_by_interval[n_by_interval >= max(2, 0.5 * n_by_interval.max())].index
        rich = (
            exp_scope.groupby("interval_idx")[SPECIES_COLS]
            .apply(lambda g: (g.sum() > 0).sum())
            .loc[keep]
        )
        fig = go.Figure(
            go.Scatter(
                x=rich.index,
                y=rich.values,
                mode="lines",
                line=dict(color="#2E8B57", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(46,139,87,.10)",
            )
        )
        fig.update_layout(**PLOT_LAYOUT, height=210)
        fig.update_xaxes(title="interval (6 min each)", title_font=dict(size=10))
        st.plotly_chart(fig, key=key)


def panel_species_table(exp_scope):
    """Every species recorded: its best deployment, and how often it turns up.

    Frequency of occurrence rather than a running total. A total can be one lucky
    deployment. Esk Point's blenny total came 78% from two drops out of 32, but
    "seen in 12 of 32" cannot be manufactured that way, so it is the more robust
    number for a monitoring question.
    """
    with st.container(border=True, height=H_PANEL):
        panel_header("Species Observed", "(peak count · how often seen)")
        dep_max = dep_species_maxn(exp_scope)
        n_deps = len(dep_max)
        totals = dep_max.sum()
        present = totals[totals > 0].sort_values(ascending=False)
        if present.empty or n_deps == 0:
            st.info("No species recorded in this selection.")
            return
        st.dataframe(
            pd.DataFrame(
                {
                    "Species": [clean_species(c) for c in present.index],
                    "Best drop": [int(dep_max[c].max()) for c in present.index],
                    "Seen in": [
                        f"{int((dep_max[c] > 0).sum())} / {n_deps}"
                        for c in present.index
                    ],
                    "Frequency": [
                        (dep_max[c] > 0).sum() / n_deps for c in present.index
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
            height=232,
            column_config={
                "Best drop": st.column_config.NumberColumn(
                    "Best drop",
                    help="Highest MaxN this species reached in any single "
                    "deployment, the best it has ever looked here.",
                ),
                "Seen in": st.column_config.TextColumn(
                    "Seen in", help="How many deployments recorded this species at all."
                ),
                "Frequency": st.column_config.ProgressColumn(
                    "Frequency",
                    min_value=0,
                    max_value=1,
                    format="%.0f%%",
                    help="Share of deployments where the species was present. "
                    "More robust than a total: a total can be one lucky "
                    "drop, a frequency cannot.",
                ),
            },
        )


def panel_score_over_time(exp_scope, key):
    """Condition index per survey date, farm against control.

    The panel that matters most for monitoring. Everything else on this page is a
    snapshot, and a monitoring programme exists to detect change, so this is the
    one that answers "is it getting better or worse?". With three survey days it
    demonstrates the shape rather than a trend; the same chart carries
    year-on-year once repeat surveys exist.

    Both lines are drawn because a farm rising while its controls rise too is
    weather, not restoration.
    """
    with st.container(border=True, height=H_PANEL):
        panel_header("Biodiversity Score Over Time", "(0–100 · farm vs control)")
        farm = exp_scope[exp_scope["treatment"] == "Mussel farm"]
        ctrl = exp_v[exp_v["treatment"] == "Control"]
        fig = go.Figure()
        for label, colour, src in [
            ("Mussel farm", "#1B7F4B", farm),
            ("Control", "#B98A4B", ctrl),
        ]:
            if src.empty:
                continue
            by_day = src.groupby("date").apply(
                lambda g: biodiversity_score(species_totals(g), SPECIES_POOL),
                include_groups=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=by_day.index,
                    y=by_day.values,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=colour, width=2.5),
                    marker=dict(size=8, color=colour),
                )
            )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=250,
            yaxis_range=[0, 100],
            legend=dict(orientation="h", y=1.13, x=0, font=dict(size=10)),
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, key=key)


def panel_farm_vs_control(exp_scope, key):
    """Each farm against the pooled soft-sediment reference.

    Controls are pooled rather than site-matched because only Esk Point and
    Motukopake have them, pairing would silently drop Rat Island and Whanganui.
    """
    with st.container(border=True, height=H_PANEL):
        panel_header(
            "Farm vs Control",
            "(each dot = one deployment \u00b7 CTRL = pooled controls)",
        )
        rows = []
        farm = exp_scope[exp_scope["treatment"] == "Mussel farm"]
        for site_code, grp in farm.groupby("site"):
            for dep_id, val in dep_abundance(grp).items():
                # Site codes, not full names: four long labels squeezed the
                # plot area to nothing.
                rows.append(
                    {
                        "Group": site_code,
                        "Type": "Mussel farm",
                        "Fish": val,
                        "deployment_id": dep_id,
                    }
                )
        if "Control" in treatments:
            ctrl = exp_v[exp_v["treatment"] == "Control"]
            if not ctrl.empty:
                for dep_id, val in dep_abundance(ctrl).items():
                    rows.append(
                        {
                            "Group": "CTRL",
                            "Type": "Control",
                            "Fish": val,
                            "deployment_id": dep_id,
                        }
                    )
        if not rows:
            st.info("Nothing to compare with the current filters.")
            return
        # A box, not a bar. Each deployment is a point; the box spans the middle
        # half of them (25th-75th percentile) with the median as the line. A bar
        # showing only the mean would let one exceptional deployment carry a
        # claim, which is exactly what a reviewer will probe first.
        fig = px.box(
            pd.DataFrame(rows),
            x="Group",
            y="Fish",
            color="Type",
            points="all",
            hover_data=["deployment_id"],
            color_discrete_map={"Mussel farm": "#1B7F4B", "Control": "#B98A4B"},
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7), line=dict(width=1.5))
        fig.update_layout(
            **PLOT_LAYOUT,
            height=250,
            boxgap=0.4,
            legend=dict(
                orientation="h", y=1.13, x=0, font=dict(size=10), title_text=""
            ),
        )
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="fish per deployment", title_font=dict(size=10))
        st.plotly_chart(fig, key=key)


def panel_peak_frame(frames_scope):
    """The busiest frame the detector found, with its boxes drawn on."""
    with st.container(border=True, height=H_PANEL):
        panel_header("Peak Abundance Frame", "(machine-selected)")
        if frames_scope.empty:
            st.info(
                "No rendered frames for this selection yet. They are produced on "
                "NeSI by `render_deployment_maxn_frames.py` and copied into "
                "`hold/uoa_frames/`."
            )
            return
        best = frames_scope.sort_values("maxn_count", ascending=False).iloc[0]
        img = Path(best["_dir"]) / str(best["frame_file"])
        if img.exists():
            # Inlined rather than via st.image: st.image sizes from the container
            # width, which is unresolved inside a fixed-height card and collapses
            # the frame to a thumbnail.
            st.markdown(
                f'<img class="frame-img" src="data:image/jpeg;base64,'
                f'{encode_frame(img)}" alt="Peak abundance frame for '
                f'{best["deployment_id"]}">',
                unsafe_allow_html=True,
            )
        st.caption(
            f"{best['deployment_id']} · {int(best['maxn_count'])} fish "
            f"detected at {float(best['peak_time_s']):.0f}s"
        )


def panel_top_species(exp_scope, key):
    """Which species carry the community, as a ranked bar."""
    with st.container(border=True, height=H_PANEL):
        panel_header("Community Composition", "(total MaxN by species)")
        totals = species_totals(exp_scope)
        top = totals[totals > 0].sort_values(ascending=False).head(8)
        if top.empty:
            st.info("No species recorded in this selection.")
            return
        bars = pd.DataFrame(
            {
                "Species": [clean_species(c) for c in top.index],
                "Total": top.values.astype(int),
            }
        )
        fig = px.bar(
            bars.iloc[::-1], x="Total", y="Species", orientation="h", text="Total"
        )
        fig.update_traces(
            marker_color="#5CA544",
            textposition="outside",
            textfont=dict(size=10),
            cliponaxis=False,
        )
        fig.update_layout(**PLOT_LAYOUT, height=228)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, key=key)


def panel_abundance_by_depth(exp_scope, key):
    """Surface vs benthic MaxN, two different communities, not one average."""
    with st.container(border=True, height=H_PANEL):
        panel_header("Abundance by Depth", "(mean MaxN per deployment)")
        rows = []
        for depth, grp in exp_scope.groupby("depth"):
            for treat, sub in grp.groupby("treatment"):
                per_dep = dep_abundance(sub)
                rows.append(
                    {
                        "Depth": depth,
                        "Type": treat,
                        "Mean MaxN": round(per_dep.mean(), 1),
                    }
                )
        if not rows:
            st.info("No deployments in this selection.")
            return
        fig = px.bar(
            pd.DataFrame(rows),
            x="Depth",
            y="Mean MaxN",
            color="Type",
            barmode="group",
            color_discrete_map={"Mussel farm": "#1B7F4B", "Control": "#B98A4B"},
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=205,
            legend=dict(orientation="h", y=1.18, x=0, font=dict(size=10)),
        )
        fig.update_xaxes(title=None)
        st.plotly_chart(fig, key=key)


def panel_time_of_day(exp_scope, key):
    """Abundance by time of day, with sampling effort underneath it.

    Restores the survey-coverage panel but carries a finding as well as a count:
    baited video is strongly diel, fish activity peaks around dawn and dusk, so
    a site sampled mostly at midday is not comparable to one sampled at dawn.
    Showing mean abundance per bucket exposes that bias instead of hiding it.
    """
    with st.container(border=True, height=H_PANEL):
        panel_header("Abundance by Time of Day", "(mean MaxN · drops per bucket)")
        order = ["EM", "MM", "PM", "EV"]
        names = {
            "EM": "Early morning",
            "MM": "Mid morning",
            "PM": "Afternoon",
            "EV": "Evening",
        }
        rows = []
        for bucket, grp in exp_scope.groupby("time_bucket"):
            per_dep = dep_abundance(grp)
            rows.append(
                {
                    "Bucket": names.get(bucket, bucket),
                    "_order": order.index(bucket) if bucket in order else 99,
                    "Mean MaxN": round(per_dep.mean(), 1),
                    "Drops": grp["deployment_id"].nunique(),
                }
            )
        if not rows:
            st.info("No deployments in this selection.")
            return
        tod = pd.DataFrame(rows).sort_values("_order")
        # Effort rides on the axis label rather than a caption: a mean is only as
        # trustworthy as the number of drops behind it, so the two belong in the
        # same glance. (A second y-axis for counts would be a dual-axis chart.)
        tod["Label"] = tod.apply(
            lambda r: f"{r['Bucket']}<br><span style='font-size:9px'>"
            f"n={r['Drops']}</span>",
            axis=1,
        )
        fig = px.bar(
            tod,
            x="Label",
            y="Mean MaxN",
            text="Mean MaxN",
            hover_data={"Drops": True, "_order": False, "Label": False},
        )
        fig.update_traces(
            marker_color="#2D6DB4",
            textposition="outside",
            textfont=dict(size=10),
            cliponaxis=False,
        )
        fig.update_layout(**PLOT_LAYOUT, height=210)
        fig.update_xaxes(title=None, tickfont=dict(size=10))
        st.plotly_chart(fig, key=key)


def panel_data_quality(exp_scope):
    """Deployments with a short annotated record, which distort every mean.

    A standard drop here is 15 six-minute intervals (90 minutes). Some have far
    fewer, three have a single interval. Those are almost certainly failed or
    part-annotated deployments, but nothing marks them, so a drop with 6 minutes
    of footage and no fish is averaged in as a genuine zero alongside a full
    90-minute drop. That drags site means down and inflates apparent variance.

    Surfaced rather than silently dropped: whether they are real zeros or failed
    recordings is a judgement for whoever ran the survey.
    """
    with st.container(border=True, height=H_PANEL):
        panel_header("Data Quality", "(short annotated records)")
        n_int = exp_scope.groupby("deployment_id")["interval_idx"].max()
        if n_int.empty:
            st.info("No deployments in this selection.")
            return
        standard = 15  # 15 x 6 min = 90 min, the design length
        short = n_int[n_int < standard - 1].sort_values()
        st.caption(
            f"{len(n_int) - len(short)} of {len(n_int)} deployments have a full "
            f"record (\\u2265{standard - 1} intervals \\u00b7 {(standard - 1) * 6} min)."
        )
        if short.empty:
            st.success("No truncated deployments in this selection.")
            return
        abundance = dep_abundance(exp_scope)
        for dep_id, n in short.items():
            mins = int(n) * 6
            colour = "#D9603B" if n <= 5 else "#E8A33D"
            got = abundance.get(dep_id, 0)
            st.markdown(
                f'<div class="feed">'
                f'<div class="feed-dot" style="background:{colour}"></div>'
                f'<div class="feed-txt">{dep_id}<br>'
                f'<span style="color:#7A879C">{int(n)} of {standard} intervals '
                f"\\u00b7 {mins} min \\u00b7 {int(got)} fish recorded</span></div>"
                f'<div class="feed-meta">{int(n) / standard:.0%}</div></div>',
                unsafe_allow_html=True,
            )


def panel_size_note():
    """Why there is no biomass figure anywhere on this dashboard."""
    with st.container(border=True, height=H_PANEL):
        panel_header("Fish Size & Biomass", "(not measurable from this footage)")
        st.markdown(
            """
<div style="font-size:.82rem;color:#2B3A55;line-height:1.55">
Every abundance figure here counts <b>individuals</b>. It cannot report
<b>biomass</b>, which is what most ecological questions actually turn on, ten
juvenile snapper and ten adults are the same MaxN and very different ecology.<br><br>
Measuring length from video needs either a <b>stereo camera pair</b> or a
<b>scale reference of known size</b> in frame. This survey was recorded on a
single GoPro with neither, so no length can be recovered from it, at any point
in the future, by any method.<br><br>
<span style="color:#7A879C">Length is simply not collected at present.</span>
</div>""",
            unsafe_allow_html=True,
        )


def render_biodiversity_section(exp_scope, ns):
    """The diversity indices, shown side by side for review.

    Deliberately shows the raw indices next to the composite rather than the
    composite alone: the composite is provisional and needs signing off, and the
    fastest way to get that is to let a reviewer see what it is made of.

    Shannon and Simpson are computed the same way as `experiment_diversity` on
    the Experiments page; "Evenness" there is Pielou's J' under another name.
    """
    st.markdown(
        '<div class="row-label">Biodiversity scores, provisional, for review</div>',
        unsafe_allow_html=True,
    )

    totals = species_totals(exp_scope)
    present = totals[totals > 0]
    n_spp = int(len(present))
    h = shannon(present)
    j = pielou(present)
    d = simpson(present)
    abundance = dep_abundance(exp_scope).mean()
    composite = biodiversity_score(totals, SPECIES_POOL)

    k = st.columns(6)
    with k[0]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Species Richness",
                n_spp,
                "",
                "distinct species recorded",
                "How many species were seen at all. The simplest measure, and "
                "the one that rises fastest with survey effort, two sites are "
                "only comparable on richness if they were sampled equally hard.",
            )
    with k[1]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Total Abundance",
                f"{abundance:.0f}",
                " fish",
                "mean \u03a3 MaxN per deployment",
                "Sum of per-species MaxN, averaged over deployments. How much "
                "life is present, ignoring how it is distributed across species.",
            )
    with k[2]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Shannon Diversity",
                f"{h:.2f}",
                " H\u2032",
                "richness and balance combined",
                "Rises with both the number of species and how evenly they are "
                "spread. Ten species in equal numbers scores higher than thirty "
                "dominated by one. Unbounded in practice, its ceiling is "
                "ln(species), so it is not comparable between surveys with "
                "different species pools.",
            )
    with k[3]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Pielou / evenness",
                f"{j:.2f}",
                " J\u2032",
                "0 = one species dominates",
                "Shannon divided by its own maximum, ln(species observed). "
                "Bounded 0\u20131, so unlike Shannon it IS comparable between "
                "surveys. 1 means every species equally common; near 0 means "
                "the community is carried by one species.",
            )
    with k[4]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Simpson Index",
                f"{d:.2f}",
                " 1\u2212D",
                "chance two fish differ",
                "The probability that two individuals picked at random are "
                "different species. Less sensitive to rare species than "
                "Shannon, so it reflects the common species better.",
            )
    with k[5]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Composite Score",
                f"{composite:g}",
                "/100",
                f"{biodiversity_band(composite)[0]} \u00b7 provisional",
                f"PROVISIONAL, NOT A PUBLISHED INDEX. Currently "
                f"100 \u00d7 H\u2032 / ln({SPECIES_POOL}), which rescales Shannon "
                f"against this survey's species pool. That denominator ties the "
                f"score to this dataset, so it is not comparable elsewhere. "
                f"Needs an agreed formula before it is quoted.",
            )

    st.write("")

    b = st.columns([2.2, 1.0])
    with b[0]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Scores by Farm", "(every farm, for comparison)")
            # Always every farm, even in the single-farm view: a diversity
            # measure only means something next to another one. A lone row
            # reading "Shannon 1.82" tells a reader nothing about whether that
            # is good.
            rows = []
            for site_code, grp in exp_v.groupby("site"):
                t = species_totals(grp)
                pres = t[t > 0]
                n = int(len(pres))
                hh = shannon(pres)
                rows.append(
                    {
                        "Farm": SITE_NAMES.get(site_code, site_code),
                        "Species": n,
                        "\u03a3 MaxN": int(pres.sum()),
                        "Shannon": round(hh, 2),
                        "Pielou / evenness": round(pielou(pres), 2),
                        "Simpson": round(simpson(pres), 2),
                        "Score": biodiversity_score(t, SPECIES_POOL),
                    }
                )
            if rows:
                # Column help puts the definition on the header itself, so the
                # table can be read without carrying the tile definitions across.
                st.dataframe(
                    pd.DataFrame(rows).sort_values("Score", ascending=False),
                    hide_index=True,
                    width="stretch",
                    height=232,
                    column_config={
                        "Farm": st.column_config.TextColumn(
                            "Farm",
                            help="Survey site. Mussel farm deployments only, "
                            "control sites are pooled and reported "
                            "separately, since only two farms have them.",
                        ),
                        "Species": st.column_config.NumberColumn(
                            "Species",
                            help="How many distinct species were recorded here. "
                            "Rises with survey effort, so compare farms "
                            "with similar deployment counts.",
                        ),
                        "Σ MaxN": st.column_config.NumberColumn(
                            "Σ MaxN",
                            help="Total abundance: each deployment's peak count "
                            "per species, summed. MaxN is the most "
                            "individuals visible in any single frame, so it "
                            "never double-counts a fish that swims out and "
                            "back.",
                        ),
                        "Shannon": st.column_config.NumberColumn(
                            "Shannon",
                            help="Shannon diversity H'. Rises with both the "
                            "number of species and how evenly they are "
                            "spread. Its ceiling is ln(species), so it is "
                            "not comparable to a survey with a different "
                            "species pool.",
                        ),
                        "Pielou / evenness": st.column_config.NumberColumn(
                            "Pielou / evenness",
                            help="Shannon divided by its own maximum, bounded "
                            "0-1. 1 = every species equally common; near "
                            "0 = one species dominates. Unlike Shannon it "
                            "IS comparable between surveys.",
                        ),
                        "Simpson": st.column_config.NumberColumn(
                            "Simpson",
                            help="Probability that two individuals picked at "
                            "random are different species. Less sensitive "
                            "to rare species than Shannon.",
                        ),
                        "Score": st.column_config.NumberColumn(
                            "Score",
                            help="PROVISIONAL composite, not a published index. "
                            "Shannon rescaled 0-100 against this survey's "
                            "species pool, which ties it to this dataset. "
                            "Needs an agreed formula before it is quoted.",
                        ),
                    },
                )
            else:
                st.info("No sites in this selection.")


def render_common_rows(exp_scope, series_label, frames_scope, selected_site, ns):
    """The panel rows both views share. `ns` namespaces the plotly keys.

    Ordered by what a monitoring reader needs: where the sites are and whether
    the habitat is working, then whether it is changing, then what lives there,
    then the evidence frame. The two within-deployment curves are survey-method
    diagnostics rather than findings, so they sit last.

    The map, the farm-vs-control comparison and the trend get double width,
    they carry the argument, and squeezed into a quarter of the row the box plot
    was unreadable.
    """
    r2 = st.columns([1, 1])
    with r2[0]:
        panel_map(selected_site, key=f"{ns}_map")
    with r2[1]:
        panel_farm_vs_control(exp_scope, key=f"{ns}_farm_ctrl")

    st.write("")

    r3 = st.columns([2, 1, 1])
    with r3[0]:
        panel_score_over_time(exp_scope, key=f"{ns}_score_time")
    with r3[1]:
        panel_top_species(exp_scope, key=f"{ns}_top_species")
    with r3[2]:
        panel_species_table(exp_scope)

    st.write("")

    r4 = st.columns([1, 1, 1, 1])
    with r4[0]:
        panel_abundance_by_depth(exp_scope, key=f"{ns}_by_depth")
    with r4[1]:
        panel_time_of_day(exp_scope, key=f"{ns}_tod")
    with r4[2]:
        panel_peak_frame(frames_scope)
    with r4[3]:
        panel_size_note()

    st.write("")

    st.markdown(
        '<div class="row-label">Survey method diagnostics, how the '
        "recordings behaved, rather than what lives there</div>",
        unsafe_allow_html=True,
    )
    r5 = st.columns([1, 1, 1])
    with r5[0]:
        panel_abundance(exp_scope, series_label, key=f"{ns}_abundance")
    with r5[1]:
        panel_richness(exp_scope, key=f"{ns}_richness")
    with r5[2]:
        panel_data_quality(exp_scope)


# ═════════════════════════════════════════════════════════════════════════════
# FARM OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

if view == "Farm overview":
    exp_s = exp_v[exp_v["site"] == site]
    ml_s = ml_v[ml_v["site"] == site] if not ml_v.empty else ml_v

    if exp_s.empty:
        st.warning(f"No annotated deployments for {site} with the selected types.")
        st.stop()

    st.markdown(
        f'<div style="font-size:1.6rem;font-weight:700;color:#16233F">'
        f"Farm Overview</div>"
        f'<div style="color:#66748C;font-size:.9rem">'
        f"Biodiversity insights for {SITE_NAMES.get(site, site)}</div>",
        unsafe_allow_html=True,
    )
    render_scope_note(exp_s, ml_s)
    st.write("")

    render_kpi_row(exp_s, ml_s, scope="this farm")
    st.write("")

    render_common_rows(
        exp_s,
        series_label=SITE_NAMES.get(site, site),
        frames_scope=(frames[frames["site"] == site] if not frames.empty else frames),
        selected_site=site,
        ns="farm",
    )

    st.write("")

    # ── Farm-only extra ──────────────────────────────────────────────────────
    with st.container(border=True):
        panel_header("Signals", "(read from this survey)")
        site_species = species_totals(exp_s)
        farm_dep = exp_s[exp_s["treatment"] == "Mussel farm"]
        ctrl_dep = exp_v[exp_v["treatment"] == "Control"]
        if not farm_dep.empty and not ctrl_dep.empty:
            f_ab = dep_abundance(farm_dep).mean()
            c_ab = dep_abundance(ctrl_dep).mean()
            if f_ab > c_ab:
                feed_row(
                    "#1B7F4B",
                    f"Farm deployments carry <b>{f_ab / max(c_ab, .01):.1f}×</b> "
                    f"the fish of pooled soft-sediment controls",
                    "survey",
                )
            else:
                feed_row(
                    "#E8A33D",
                    "Farm deployments are not out-performing controls",
                    "survey",
                )
        benthic = exp_s[exp_s["depth"] == "Benthic"]
        surface = exp_s[exp_s["depth"] == "Surface"]
        if not benthic.empty and not surface.empty:
            feed_row(
                "#2D6DB4",
                f"Surface cameras recorded "
                f"<b>{int((surface[SPECIES_COLS].sum() > 0).sum())}</b> species, "
                f"benthic <b>{int((benthic[SPECIES_COLS].sum() > 0).sum())}</b>",
                "habitat split",
            )
        top_sp = site_species.sort_values(ascending=False)
        if len(top_sp) and top_sp.iloc[0] > 0:
            share = 100 * top_sp.iloc[0] / max(site_species.sum(), 1)
            if share > 50:
                feed_row(
                    "#E8A33D",
                    f"<b>{clean_species(top_sp.index[0])}</b> is "
                    f"{share:.0f}% of all individuals, low evenness",
                    "diversity",
                )

# ═════════════════════════════════════════════════════════════════════════════
# INDUSTRY OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

elif view == "Industry overview":
    n_sites = exp_v["site"].nunique()

    st.markdown(
        '<div style="font-size:1.6rem;font-weight:700;color:#16233F">'
        "Industry Overview</div>"
        '<div style="color:#66748C;font-size:.9rem">'
        "Biodiversity across all monitored mussel farms</div>",
        unsafe_allow_html=True,
    )
    render_scope_note(exp_v, ml_v)
    st.write("")

    render_kpi_row(exp_v, ml_v, scope=f"all {n_sites} farms")
    st.write("")

    render_common_rows(
        exp_v,
        series_label="All farms",
        frames_scope=frames,
        selected_site=None,
        ns="ind",
    )

    st.write("")

    # ── Industry-only extras ─────────────────────────────────────────────────
    site_rows = []
    for s, grp in exp_v.groupby("site"):
        totals = species_totals(grp)
        per_dep = dep_abundance(grp)
        site_rows.append(
            {
                "Farm": SITE_NAMES.get(s, s),
                "Score": biodiversity_score(totals, SPECIES_POOL),
                "Species": int((totals > 0).sum()),
                "Mean abundance": round(per_dep.mean(), 1),
                "Deployments": grp["deployment_id"].nunique(),
            }
        )
    site_df = pd.DataFrame(site_rows).sort_values("Score", ascending=False)

    extras = st.columns([1.5, 1.3, 1.2])
    with extras[0]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Farm Summary", "(one row per site)")
            st.dataframe(site_df, hide_index=True, width="stretch", height=232)
    with extras[1]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Farm Leaderboard", "(biodiversity score)")
            for rank, row in enumerate(site_df.itertuples(), start=1):
                st.markdown(
                    f'<div class="feed">'
                    f'<div class="feed-txt" style="color:#94A0B4;width:1.4rem">'
                    f"{rank}</div>"
                    f'<div class="feed-txt"><b>{row.Farm}</b> '
                    f"<span style='color:#7A879C'>· {row.Species} species · "
                    f"{row.Deployments} deployments</span></div>"
                    f'<div class="feed-meta">{band_pill(row.Score)} '
                    f'<span style="color:#16233F;font-weight:600;'
                    f'margin-left:.4rem">{row.Score:g}</span></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
    with extras[2]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Survey Statistics", "")
            for name, val in [
                ("Farms monitored", f"{n_sites}"),
                ("Deployments annotated", f"{exp_v['deployment_id'].nunique()}"),
                ("Annotated intervals", f"{len(exp_v):,}"),
                ("Video files processed", f"{len(ml_v):,}"),
                ("Species recorded", f"{SPECIES_POOL}"),
                ("Survey days", f"{exp_v['date'].nunique()}"),
            ]:
                st.markdown(
                    f'<div class="feed"><div class="feed-txt">{name}</div>'
                    f'<div class="feed-meta" style="color:#16233F;'
                    f'font-weight:600">{val}</div></div>',
                    unsafe_allow_html=True,
                )


# ═════════════════════════════════════════════════════════════════════════════
# ML PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
#
# The analyst layer. Everything here answers "should you believe the numbers on
# the other two views?", which is a different question from "what lives at this
# farm?" and deserves its own page rather than two slots in a customer KPI row.
#
# Metrics are computed from local detection files and expert annotations rather
# than from the training artifacts that Model Metrics reads out of S3, the UoA
# models were trained outside the main pipeline, so no results.csv exists for
# them. Agreement against expert annotation is in any case the measure that
# matters for a monitoring claim: mAP tells you about held-out training frames,
# this tells you about deployments.

elif view == "ML performance":
    st.markdown(
        '<div style="font-size:1.6rem;font-weight:700;color:#16233F">'
        "ML Performance</div>"
        '<div style="color:#66748C;font-size:.9rem">'
        "How far the automated counts can be trusted, and where they fail</div>",
        unsafe_allow_html=True,
    )
    render_scope_note(exp_v, ml_v)
    st.write("")

    if ml_v.empty:
        st.warning("No machine detections in this selection.")
        st.stop()

    # Per-deployment machine peak vs expert total, for every deployment that has
    # both. This join is the whole credibility argument.
    ml_peak = ml_v.groupby("deployment_id")["peak_in_frame"].max()
    exp_peak = dep_abundance(exp_v)
    paired = pd.DataFrame({"ml": ml_peak, "expert": exp_peak}).dropna()
    paired["site"] = paired.index.str.split("_").str[0]
    paired["depth"] = paired.index.str.split("_").str[1].map(DEPTH_LABEL)

    ml_by_class = load_ml_by_class()
    if not ml_by_class.empty:
        ml_by_class = ml_by_class[
            ml_by_class["deployment_id"].isin(ml_v["deployment_id"].unique())
        ]
    ml_classes = sorted(ml_by_class["class"].unique()) if not ml_by_class.empty else []

    corr = paired["ml"].corr(paired["expert"]) if len(paired) > 2 else float("nan")
    detections = int(ml_v["detections"].sum())
    mean_conf = ml_v["mean_conf"].mean()
    silent = int((ml_v["detections"] == 0).sum())
    models = sorted(ml_v["model"].unique())

    k = st.columns(6)
    with k[0]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Expert Agreement",
                "—" if pd.isna(corr) else f"{corr:.2f}",
                " r",
                f"across {len(paired)} paired deployments",
                "Pearson correlation between the machine's peak count and the "
                "expert's total for the same deployment. This is the number that "
                "decides whether automated monitoring can stand in for a human "
                "reader, not mAP, which only describes held-out training frames.",
            )
    with k[1]:
        with st.container(border=True, height=H_KPI):
            hours = len(ml_v) * CHAPTER_SEC / 3600
            kpi(
                "Video Analysed",
                f"{hours:,.0f}",
                " hours",
                f"{detections / max(len(ml_v), 1):,.0f} detections per file",
                "Hours of footage the detector read without a human watching it. "
                "The per-file detection count is a throughput figure, not an "
                "abundance one, a single fish in view for a minute produces "
                "thousands of detections. Abundance always comes from MaxN.",
            )
    with k[2]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Mean Confidence",
                "—" if pd.isna(mean_conf) else f"{mean_conf * 100:.0f}%",
                "",
                "of kept detections",
                "Average confidence assigned to detections above the threshold. "
                "How sure the model was, not how often it was right, a model "
                "can be confidently wrong, which is why agreement is reported "
                "separately.",
            )
    with k[3]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Videos Analysed",
                f"{len(ml_v):,}",
                "",
                f"{silent} returned nothing",
                "GoPro chapter files processed. Files returning no detections at "
                "all are the review queue: some are genuinely empty water, some "
                "are the model failing on turbid footage.",
            )
    with k[4]:
        with st.container(border=True, height=H_KPI):
            under = (paired["ml"] < paired["expert"]).mean() * 100 if len(paired) else 0
            kpi(
                "Under-count Rate",
                f"{under:.0f}",
                "%",
                "of deployments below expert",
                "Share of deployments where the machine counted fewer fish than "
                "the expert. A high rate with good correlation means the model "
                "ranks sites correctly but reads low, usable for comparison, "
                "not for absolute abundance.",
            )
    with k[5]:
        with st.container(border=True, height=H_KPI):
            kpi(
                "Detection Models",
                len(models),
                "",
                "in this selection",
                "Different detectors were run on different survey days. "
                "cfd-yolov12x is a general-purpose fish detector that never saw "
                "this survey, an independent test. Models fine-tuned on this "
                "programme's own footage perform better but prove less.",
            )

    st.write("")

    r2 = st.columns([1.6, 1.15, 1.15])

    with r2[0]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Machine vs Expert", "(one point per deployment)")
            if paired.empty:
                st.info("No deployments have both machine and expert counts.")
            else:
                fig = px.scatter(
                    paired,
                    x="expert",
                    y="ml",
                    color="depth",
                    hover_name=paired.index,
                    color_discrete_map={"Surface": "#6D4CB0", "Benthic": "#79922B"},
                    labels={"expert": "expert count", "ml": "machine count"},
                )
                lim = float(max(paired["expert"].max(), paired["ml"].max())) + 2
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=lim,
                    y1=lim,
                    line=dict(dash="dash", color="#B9C4D6", width=1.5),
                )
                fig.update_traces(marker=dict(size=9, opacity=0.85))
                fig.update_layout(
                    **PLOT_LAYOUT,
                    height=228,
                    legend=dict(
                        orientation="h", y=1.16, x=0, font=dict(size=10), title_text=""
                    ),
                )
                st.plotly_chart(fig, key="ml_scatter")

    with r2[1]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Agreement by Depth", "(mean count per deployment)")
            rows = []
            for depth, grp in paired.groupby("depth"):
                rows.append(
                    {
                        "Depth": depth,
                        "Source": "Expert",
                        "Mean count": round(grp["expert"].mean(), 1),
                    }
                )
                rows.append(
                    {
                        "Depth": depth,
                        "Source": "Machine",
                        "Mean count": round(grp["ml"].mean(), 1),
                    }
                )
            if not rows:
                st.info("Nothing to compare.")
            else:
                fig = px.bar(
                    pd.DataFrame(rows),
                    x="Depth",
                    y="Mean count",
                    color="Source",
                    barmode="group",
                    color_discrete_map={"Expert": "#0072B2", "Machine": "#009E73"},
                )
                fig.update_layout(
                    **PLOT_LAYOUT,
                    height=205,
                    legend=dict(
                        orientation="h", y=1.18, x=0, font=dict(size=10), title_text=""
                    ),
                )
                fig.update_xaxes(title=None)
                st.plotly_chart(fig, key="ml_depth")

    with r2[2]:
        with st.container(border=True, height=H_PANEL):
            # Per-species agreement wherever the detector emits species; the
            # scene-busyness view is the fallback while models are binary.
            if len(ml_classes) > 1:
                panel_header("Agreement by Species", "(machine vs expert MaxN)")
                rows = []
                for cls in ml_classes:
                    ml_cls = ml_by_class[ml_by_class["class"] == cls].set_index(
                        "deployment_id"
                    )["peak"]
                    human_col = ML_CLASS_TO_EXPERT.get(cls)
                    if human_col is None or human_col not in SPECIES_COLS:
                        continue
                    exp_cls = dep_species_maxn(exp_v)[human_col]
                    both = pd.DataFrame({"ml": ml_cls, "expert": exp_cls}).dropna()
                    if both.empty:
                        continue
                    rows.append(
                        {
                            "Species": clean_species(human_col),
                            "Source": "Expert",
                            "Mean MaxN": round(both["expert"].mean(), 2),
                        }
                    )
                    rows.append(
                        {
                            "Species": clean_species(human_col),
                            "Source": "Machine",
                            "Mean MaxN": round(both["ml"].mean(), 2),
                        }
                    )
                if rows:
                    fig = px.bar(
                        pd.DataFrame(rows),
                        x="Species",
                        y="Mean MaxN",
                        color="Source",
                        barmode="group",
                        color_discrete_map={"Expert": "#0072B2", "Machine": "#009E73"},
                    )
                    fig.update_layout(
                        **PLOT_LAYOUT,
                        height=200,
                        legend=dict(
                            orientation="h",
                            y=1.2,
                            x=0,
                            font=dict(size=10),
                            title_text="",
                        ),
                    )
                    fig.update_xaxes(title=None)
                    st.plotly_chart(fig, key="ml_species")
                else:
                    st.info(
                        "The detector emits species, but none of its class names "
                        "map to an annotated species column yet, extend "
                        "ML_CLASS_TO_EXPERT."
                    )
            else:
                panel_header(
                    "Where Agreement Breaks Down", "(by how busy the scene is)"
                )
                st.caption(
                    f"Per-species agreement needs a species detector. The models "
                    f"in this selection emit only: {', '.join(ml_classes) or '—'}."
                )
                # Until then the useful question is *when* the model fails, and
                # it fails as a function of how much is in frame, so bucket
                # deployments by what the expert counted and show both sides.
                if paired.empty:
                    st.info("No paired deployments.")
                else:
                    bands = [
                        (0, 0, "empty"),
                        (1, 4, "1–4 fish"),
                        (5, 14, "5–14 fish"),
                        (15, 10**6, "15+ fish"),
                    ]
                    rows = []
                    for lo, hi, label in bands:
                        sub = paired[
                            (paired["expert"] >= lo) & (paired["expert"] <= hi)
                        ]
                        if sub.empty:
                            continue
                        rows.append(
                            {
                                "Scene": label,
                                "Source": "Expert",
                                "Mean count": round(sub["expert"].mean(), 1),
                            }
                        )
                        rows.append(
                            {
                                "Scene": label,
                                "Source": "Machine",
                                "Mean count": round(sub["ml"].mean(), 1),
                            }
                        )
                    fig = px.bar(
                        pd.DataFrame(rows),
                        x="Scene",
                        y="Mean count",
                        color="Source",
                        barmode="group",
                        color_discrete_map={"Expert": "#0072B2", "Machine": "#009E73"},
                    )
                    fig.update_layout(
                        **PLOT_LAYOUT,
                        height=180,
                        legend=dict(
                            orientation="h",
                            y=1.22,
                            x=0,
                            font=dict(size=10),
                            title_text="",
                        ),
                    )
                    fig.update_xaxes(title=None)
                    st.plotly_chart(fig, key="ml_bands")

    st.write("")

    r3 = st.columns([1.15, 1.15, 1.15, 1.15])

    with r3[0]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Where It Fails", "(largest disagreements)")
            if paired.empty:
                st.info("Nothing to rank.")
            else:
                gap = paired.assign(gap=(paired["expert"] - paired["ml"]).abs())
                for row in gap.sort_values("gap", ascending=False).head(6).itertuples():
                    colour = "#D9603B" if row.gap >= 10 else "#E8A33D"
                    st.markdown(
                        f'<div class="feed">'
                        f'<div class="feed-dot" style="background:{colour}"></div>'
                        f'<div class="feed-txt">{row.Index}<br>'
                        f'<span style="color:#7A879C">expert {int(row.expert)} · '
                        f"machine {int(row.ml)}</span></div>"
                        f'<div class="feed-meta">−{int(row.gap)}</div></div>',
                        unsafe_allow_html=True,
                    )

    with r3[1]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Confidence by Site", "(mean of kept detections)")
            by_site = ml_v.groupby("site")["mean_conf"].mean().dropna()
            if by_site.empty:
                st.info("No confidence values.")
            else:
                bars = pd.DataFrame(
                    {
                        "Site": [SITE_NAMES.get(i, i) for i in by_site.index],
                        "Confidence": (by_site.values * 100).round(1),
                    }
                )
                fig = px.bar(bars, x="Site", y="Confidence", text="Confidence")
                fig.update_traces(
                    marker_color="#009E73",
                    textposition="outside",
                    textfont=dict(size=10),
                    cliponaxis=False,
                )
                fig.update_layout(**PLOT_LAYOUT, height=205, yaxis_range=[0, 100])
                fig.update_xaxes(title=None)
                st.plotly_chart(fig, key="ml_conf_site")

    with r3[2]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Models Used", "(and what they prove)")
            for model in models:
                sub = ml_v[ml_v["model"] == model]
                independent = model.startswith("cfd-")
                st.markdown(
                    f'<div class="feed">'
                    f'<div class="feed-dot" style="background:'
                    f'{"#1B7F4B" if independent else "#E8A33D"}"></div>'
                    f'<div class="feed-txt"><b>{model}</b><br>'
                    f'<span style="color:#7A879C">{len(sub)} video files · '
                    f'{"never saw this survey" if independent else "fine-tuned on this programme"}'
                    f"</span></div></div>",
                    unsafe_allow_html=True,
                )
            st.caption(
                "Training curves, mAP and confusion matrices for pipeline models "
                "live on the Model Metrics page, these UoA models were trained "
                "outside the pipeline, so they have no results.csv in S3."
            )

    with r3[3]:
        with st.container(border=True, height=H_PANEL):
            panel_header("Read This Before Quoting", "")
            st.markdown(
                f"""
<div style="font-size:.82rem;color:#2B3A55;line-height:1.5">
The machine is a reliable <b>relative</b> instrument and an unreliable
<b>absolute</b> one. It ranks sites correctly (r&nbsp;=&nbsp;{corr:.2f}) but reads
low, so it can answer "is this farm carrying more fish than the bare seabed?"
and cannot yet be quoted as a headline abundance figure.<br><br>
Benthic footage is the known weak point: turbid, low contrast, fish the same
colour as the sediment behind them. Species attribution on the other views comes
from expert annotators, not from the detector, which is binary fish/no-fish.
</div>""",
                unsafe_allow_html=True,
            )


elif view == "ML vs Expert detail":
    st.markdown(
        '<div style="font-size:1.6rem;font-weight:700;color:#16233F">'
        "ML vs Expert, interval detail</div>"
        '<div style="color:#66748C;font-size:.9rem">'
        "The full reconciliation: every 6-minute expert interval against the "
        "machine count in the same time window</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "This view carries its own day filter and confidence slider, the "
        "sidebar survey-window and deployment-type filters do not apply here, "
        "because it compares the raw sources interval by interval rather than "
        "reading the dashboard's filtered frames."
    )
    st.write("")
    # Implementation lives in app/uoa_ml_vs_expert.py, on the shared
    # uoa_data layer.
    from uoa_ml_vs_expert import render_body as render_ml_vs_expert

    render_ml_vs_expert(embedded=True, key_prefix="mid")

elif view == "ML vs Expert (old)":
    # The pre-rebuild page, kept so the old block-aligned comparison can be read
    # against the interval view above it. A view here rather than its own nav
    # entry: it is the same question asked an older way, and both belong behind
    # the test-dashboard password.
    #
    # `runpy` rather than an import: the archived file is a script with its
    # Streamlit calls at module level and no render function, and it is not
    # maintained, so wrapping it would mean editing a file whose whole point is
    # that it is the version as it stood.
    import runpy

    archived = Path(__file__).resolve().parents[2] / "_archive" / "ML_vs_Expert.py"
    if archived.exists():
        runpy.run_path(str(archived), run_name="archived_ml_vs_expert")
    else:
        st.info(
            "The archived ML vs Expert page is no longer in the repository. "
            "Git history has it at `app/_archive/ML_vs_Expert.py`."
        )


if view in ("Farm overview", "Industry overview"):
    st.write("")
    render_biodiversity_section(
        exp_s if view == "Farm overview" else exp_v,
        ns="farm" if view == "Farm overview" else "ind",
    )

# ── Provenance ───────────────────────────────────────────────────────────────

st.write("")
with st.expander("Where these numbers come from"):
    st.markdown(
        f"""
**This is a product concept running on real data.** Every figure above is
computed live from the Underwood &amp; Jeffs mussel farm survey
(4 Hauraki Gulf sites, {expert['date'].min()} – {expert['date'].max()}).
Nothing is mocked or illustrative.

* **Expert species counts**, `hold/Annotations_CLEANED_v2.csv`,
  {len(expert):,} annotated 6-minute intervals across
  {expert['deployment_id'].nunique()} deployments, {SPECIES_POOL} species.
  Placeholder rows for un-annotated sessions are excluded, so a blank species
  cell means "an annotator looked and saw none", not "nobody looked".
* **Machine detections**, {len(ml):,} per-chapter YOLO output files, filtered
  to confidence ≥ {CONF:g}.
* **Farm vs control, surface vs benthic**, these are two independent axes.
  *Farm* and *control* describe the **site**: farm deployments sat within the
  mussel farm, control deployments over open soft sediment at least 400 m from
  any farm. **Each site was then sampled at two depths**, a surface camera in
  the water column (among the growing lines at a farm site) and a benthic
  camera on the seabed (beneath the lines at a farm site). So "under the
  growing structures" describes only the benthic farm cameras; the surface
  cameras sat among the lines, which is what the peak-abundance frames show.
  Only Esk Point and Motukopake have paired controls. Rat Island and
  Whanganui are farm-only, so controls are pooled rather than site-matched.
* **Biodiversity Score**. Shannon diversity H′ over the species MaxN totals,
  rescaled to 0–100 against `ln({SPECIES_POOL})`, the maximum H′ if all
  recorded species were present in equal numbers. It is a **relative condition
  index for comparing sites within this survey**, not an absolute ecological
  grade, and it is not comparable to scores from a different species pool.
* **Expert Agreement (r = 0.73)**. Pearson correlation between machine and
  expert per-deployment abundance over the 58 deployments that have both.

**What this concept still needs.** Two of the four survey days have machine
output; benthic detection substantially under-counts relative to experts; and
the detector is binary, so *species* attribution above comes from expert
annotation rather than from the machine. Those gaps are what the proposal asks
to close.

### Open questions on the Biodiversity Score

The composite score is **provisional**. It is currently
`100 × H′ / ln({SPECIES_POOL})`. Shannon rescaled against this survey's species
pool. That denominator ties it to this dataset, so it cannot be compared to any
other survey. Four things need deciding before it is quoted:

1. **Is there an index we should match rather than invent?** DOC publishes
   baited-underwater-video survey guidelines, and NZ reef-fish practice reports
   *per-species relative density*, snapper especially, rather than a composite
   score. The Hauraki Gulf Forum's triennial *State of our Gulf* reports on
   condition (kina barrens, crayfish, scallops) but does not appear to publish a
   single reef-fish index. Aligning to an existing metric beats a better formula.
2. **What should the score be measured against?** The paired control sites
   (defensible, needs no literature, but a low bar), a percentile of all sites
   surveyed (relative, computable today), or an expected value for a healthy
   site, roughly how many species, and how much snapper? Only the third says
   whether a site is actually *good*, and it needs an agreed baseline era.
3. **Should abundance be in a *diversity* score at all?** A site holding 1,000
   sweep and nothing else scores high on abundance and low on evenness.
4. **Should predators count for more?** The top species here are grazers and
   planktivores; snapper sits well down the list. A farm boosting planktivores
   is a different ecological claim from one boosting snapper.

### References

- [DOC. Baited Remote Underwater Video guidelines](https://www.doc.govt.nz/documents/science-and-technical/inventory-monitoring/im-toolbox-marine-baited-remote-underwater-video-guidelines.pdf)
  and [DOC. Baited underwater video surveys for fish](https://www.doc.govt.nz/globalassets/documents/science-and-technical/inventory-monitoring/im-toolbox-marine-baited-underwater-video-surveys-for-fish.pdf)
 , the Inventory &amp; Monitoring Toolbox method standards for BUV. Worth citing
  alignment to.
- [Hauraki Gulf Forum, *State of our Gulf 2026*](https://gulfjournal.org.nz/),
  triennial Gulf condition assessment, now in its eighth edition. Reports
  condition through specific indicators (kina barrens, crayfish, scallops)
  rather than a single reef-fish index.
- [DOC. Gulf ecosystem monitoring, May 2026](https://www.doc.govt.nz/news/media-releases/2026-media-releases/marine-scientists-build-snapshot-view-of-gulfs-ecosystems/)
 , baseline surveys inside and outside the new Hauraki Gulf marine protections.
- [DOC. Banks Peninsula BUV monitoring](https://www.doc.govt.nz/about-us/science-publications/conservation-publications/marine-and-coastal/banks-peninsula-monitoring-using-baited-underwater-video/)
 , a worked example of BUV used for marine protected area monitoring.
- Karr, J.R. (1981), *Assessment of biotic integrity using fish communities*,
  the origin of the multimetric index approach a composite score would follow.

**Site coordinates** are the published positions from the paper's methods
section, consented mussel farms are already marked on public marine-farm
charts. The stricter withholding rule Spyfish applies to Department of
Conservation reserve deployments does not apply here, and is unchanged
elsewhere in this app.
"""
    )
