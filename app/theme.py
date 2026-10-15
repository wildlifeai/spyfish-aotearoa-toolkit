"""Shared chart colours for the Streamlit app.

Single source of truth for every palette. Define a colour here, never in a
page: a category that changes hex between two views reads as two different
things.

Three kinds of colour live here, and they are not interchangeable:

* **Categorical** (`SOURCE_COLORS`), identity, no order. Validated for
  colour-vision deficiency; see the note on that dict before changing a value.
* **Status** (`INGEST_STATUS_COLORS`, `VIDEO_PRESENCE_COLORS`), reserved for
  good/warning/bad state. Never reuse these for a data series, or a neutral
  category inherits "this is broken".
* **Ordered** (`PROTECTION_COLORS`), protection strength is ordinal, so it gets
  a graded scale rather than arbitrary hues.

Below those sit the per-chart role palettes (one dict per chart's categories),
the unencoded chart furniture (reference, guide and fit lines, grids) and the
interface chrome written into inline HTML/CSS. Name every new colour by its
role, never by its hue.

Streamlit here runs in its default light theme (there is no `[theme]` block in
`.streamlit/config.toml`), so the light values are what actually ship. The dark
variants are kept validated alongside them so that switching the app theme later
is a one-line change rather than a re-derivation.
"""

# stdlib only, deliberately. This module is imported by anything that wants a
# hex value, so it stays free of the config loader and the plotting stack —
# `apply_plotly_defaults` imports plotly lazily for the same reason.
import logging

# ── Annotation sources (categorical) ─────────────────────────────────────────
#
# Okabe-Ito derived. Both sets pass the six palette checks (lightness band,
# chroma floor, CVD separation, normal-vision separation, contrast) against
# their surface. Adjacent-pair separation is ΔE 11.4 protan on light and 8.1 on
# dark, the dark pair sits close to the floor, so charts using it must keep a
# legend or direct labels rather than relying on hue alone.
#
# Do not add a fourth source colour by picking something that "looks different".
# Re-run the palette validator over the whole set.
SOURCE_COLORS = {
    "expert": "#0072B2",
    "citsci": "#E69F00",
    "ml": "#009E73",
}

# ── Species (categorical) ────────────────────────────────────────────────────
#
# Okabe-Ito, assigned in fixed order and never cycled. The first three are the
# same steps as SOURCE_COLORS and carry the same validation (worst adjacent-pair
# CVD separation ΔE 11.4 protan on the light surface). Past eight series, fold
# the tail into "Other" rather than generating a ninth hue.
#
# Species colour is identity, not magnitude, so it must stay attached to the
# species regardless of rank: `species_color_map` assigns by sorted name so a
# filter that removes one species never repaints the survivors.
CATEGORICAL = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#8C6D31",  # bronze
    "#5D3A9B",  # violet
]


def species_color_map(names) -> dict:
    """Name → colour, assigned in sorted order.

    PASS THE FULL SET OF SPECIES AVAILABLE, NOT THE CURRENT SELECTION. Colours
    are assigned by position in the sorted list, so a map built from a filtered
    subset shifts every colour up whenever a species is removed, the survivors
    get repainted and two views of the same data disagree about what colour a
    species is. Building from the complete list makes the mapping independent of
    what happens to be selected; Plotly ignores keys for absent series, so the
    extra entries cost nothing.

    Sorted rather than order-of-appearance for the same reason: arrival order
    depends on the data, sorted order does not.
    """
    unique = sorted({n for n in names if n})
    return {n: CATEGORICAL[i % len(CATEGORICAL)] for i, n in enumerate(unique)}


# ── Pipeline state (status, reserved) ───────────────────────────────────────
NEUTRAL = "#9E9E9E"

INGEST_STATUS_COLORS = {
    "ok": "#26A69A",
    "excluded": "#FF9800",
    "metadata_error": "#EF5350",
    "validation_error": "#EF5350",
    "removed": NEUTRAL,
    # `ecology_data.NO_SHAREPOINT_RECORD`: legacy-only drops synthesised from
    # annotations. Muted violet, so it reads as "a different kind of row",
    # not as another shade of failure.
    "no SharePoint record": "#9575CD",
}

VIDEO_PRESENCE_COLORS = {
    "present": "#26A69A",
    "archived": "#FF9800",
    "absent": "#EF5350",
    "no_video_bad_dep": NEUTRAL,
}

# ── Unencoded marks ──────────────────────────────────────────────────────────
#
# A bar chart with a single series has no category to encode, and a threshold
# line is a reference, not data. One colour each, so they match across views.
SINGLE_SERIES = "#1E6FB4"
THRESHOLD_LINE = "#EF5350"

# Lines computed from the data rather than drawn from it, and the faint guides
# that sit behind a chart. None of them is a category, so none may borrow a
# categorical or status hue. Same-value pairs below are kept as separate names
# on purpose: they mean different things and may diverge.
#
# MODEL_LINE: a fitted or summarised line over the points (regression fit, the
# mean species-accumulation curve). ANCHOR_LINE: the dark neutral summary that
# the other marks are read against (the all-species line, the pooled estimate).
MODEL_LINE = "#1976D2"
ANCHOR_LINE = "#2B3A55"
# The 95% range of a pooled estimate, drawn as a translucent band.
INTERVAL_FILL = "#8794A8"
# Dashed comparison lines: the 1:1 line on a calibration scatter and the
# quadrant splits on the strategy chart.
REFERENCE_LINE = "#888888"
# Dotted "no difference" / target guides (x = 0, y = 95%), and the fainter
# minute ticks on the timing chart.
GUIDE_LINE = "#AAAAAA"
MINOR_GUIDE_LINE = "#DDDDDD"
# The expert = ML diagonal on the Operations annotations scatter, and its label.
AGREEMENT_LINE = "#B9C4D6"
AGREEMENT_LABEL = "#8794A8"
# White on a filled bar segment, and the hairline round scatter markers.
TEXT_ON_FILL = "white"
MARKER_OUTLINE = "white"

# ── Protection status (ordered) ──────────────────────────────────────────────
#
# Keyed on the exact values stored in `sites.protection_status` after ingest
# normalisation, never matched on substrings: substring matching (e.g.
# "marine reserve" in status.lower()) would silently mis-colour anything
# phrased unexpectedly and give no signal that it had done so.
#
# Cool and graded = degrees of protection, darkest is strongest. Warm = none.
# Unknown stays neutral grey so it never reads as a finding.
PROTECTION_ORDER = [
    "Type I MPA (Marine Reserve)",
    "High Protection Area",
    "Type II MPA",
    "Seafloor Protection Area",
    "Fisheries Act closure areas",
    "Mataitai",
    "Taiapure",
    "No protection",
    "Other",
    "unknown",
]

PROTECTION_COLORS = {
    "Type I MPA (Marine Reserve)": "#0B3D6B",
    "High Protection Area": "#14568F",
    "Type II MPA": "#1E6FB4",
    "Seafloor Protection Area": "#3A8CCB",
    "Fisheries Act closure areas": "#66A9DB",
    "Mataitai": "#93C4E8",
    "Taiapure": "#BFDCF2",
    "No protection": "#D96C3F",
    "Other": NEUTRAL,
    "unknown": "#BDBDBD",
}


# Statuses already reported as unknown. Both lookups below run on every rerun,
# for every status on the chart, so without this one unrecognised value would
# fill the log and bury whatever else was wrong.
_UNKNOWN_PROTECTION_SEEN: set = set()


def _note_unknown_protection(status) -> None:
    """Report a protection status this module has no colour or position for.

    The fall-backs are deliberate, grey and last, so an unclassified status
    looks unclassified, but silence makes them indistinguishable from a
    correctly-styled category. The usual cause is a value added to
    `known_protection_statuses` in config.yaml without the matching entry here.
    """
    if status is None or status in _UNKNOWN_PROTECTION_SEEN:
        return
    _UNKNOWN_PROTECTION_SEEN.add(status)
    logging.warning(
        "Protection status %r has no entry in theme.PROTECTION_COLORS / "
        "PROTECTION_ORDER, so it renders neutral grey and sorts last. Add it "
        "to both if it is a real category.",
        status,
    )


def protection_color_map(statuses) -> dict:
    """Plotly `color_discrete_map` for the protection statuses actually present.

    Anything not in `PROTECTION_COLORS` falls back to neutral grey instead of
    raising or being auto-assigned a hue, a status added upstream should look
    unclassified, not like a new category with meaning. It is logged once, so
    the fall-back is visible without being repeated on every rerun.
    """
    out = {}
    for status in statuses:
        if status in PROTECTION_COLORS:
            out[status] = PROTECTION_COLORS[status]
        else:
            _note_unknown_protection(status)
            out[status] = NEUTRAL
    return out


def protection_sort_key(status: str) -> int:
    """Sort position for a protection status, strongest first. Unknown sorts last."""
    try:
        return PROTECTION_ORDER.index(status)
    except ValueError:
        _note_unknown_protection(status)
        return len(PROTECTION_ORDER)


# ── Biodiversity condition score (ordered) ───────────────────────────────────
#
# A 0–100 condition band, worst → best. Ordinal like PROTECTION_COLORS, but it
# gets its own warm-to-cool ramp rather than reusing the protection blues,
# because a site's measured condition and its legal protection status are
# independent facts that routinely appear on the same page, sharing a ramp
# would imply one follows from the other.
#
# Never used as colour alone: every band is rendered with its numeric score and
# its label, so the ramp is reinforcement, not the encoding.
BIODIVERSITY_BANDS = [
    (80, "Excellent", "#1B7F4B"),
    (60, "Good", "#5CA544"),
    (40, "Moderate", "#E8A33D"),
    (0, "Poor", "#D9603B"),
]

BIODIVERSITY_NO_DATA = NEUTRAL


def biodiversity_band(score) -> tuple:
    """(label, hex) for a 0–100 condition score. None/NaN reads as no data."""
    if score is None or (isinstance(score, float) and score != score):
        return "No data", BIODIVERSITY_NO_DATA
    for floor, label, color in BIODIVERSITY_BANDS:
        if score >= floor:
            return label, color
    return "No data", BIODIVERSITY_NO_DATA


# ── Sequential scales (ordered) ──────────────────────────────────────────────
#
# Light-to-dark ramps for the species-by-site heatmaps. Frequency ends on the
# strongest-protection blue and abundance on the "Excellent" green by value
# only: they are magnitude scales, not protection or condition, so they are
# named for what they measure and must not be read through those palettes.
DETECTION_FREQUENCY_SCALE = ["#EAF2FB", "#0B3D6B"]
ABUNDANCE_SCALE = ["#EAF3EC", "#1B7F4B"]


# ── View palettes (one dict per chart role) ──────────────────────────────────
#
# Each dict below belongs to one chart's categories. Where a value repeats a
# status or source token AND means the same thing, it references that token;
# where the hex merely coincides (e.g. "#D9603B" is the biodiversity "Poor",
# the bad-deployment red and the large-gap dot), it stays a literal under its
# own role, so changing one never silently repaints another.

# Species detection rate, by how widely a species is seen.
RARITY_COLORS = {
    "common (≥20%)": "#2E7D32",
    "intermediate (5–20%)": "#FBC02D",
    "rare (<5%)": "#C62828",
}

# Frequency-vs-abundance quadrants. Keys are the lower-case strategy values the
# chart colours by; the quadrant labels look themselves up here too.
STRATEGY_COLORS = {
    "core": "#2E7D32",
    "transient": "#FF9800",
    "patchy": "#1976D2",
    "incidental": "#9E9E9E",
}

# Settled-direction palette for the effect charts: a species whose interval
# clears zero gets a side's colour, one whose interval spans zero stays grey
# however big its point estimate. Colour therefore means "the data settles the
# direction", not "the bar is long". "Higher inside" is the Protected group
# colour (`charting.group_colors`); "Higher outside" is deliberately its own
# red, not the Unprotected orange.
EFFECT_COLOURS = {
    "Higher inside": PROTECTION_COLORS["Type I MPA (Marine Reserve)"],
    "Higher outside": "#C1553B",
    "Not settled": "#9AA5B4",
}

# Arrival vs time-of-MaxN, the two timings on the Annotations timing chart.
ARRIVAL_PEAK_COLORS = {
    "Arrival": "#1976D2",
    "MaxN time": "#FF7043",
}

# Size of an expert/ML disagreement in the "where agreement breaks down" list.
GAP_SEVERITY_COLORS = {
    "large": "#D9603B",
    "moderate": "#E8A33D",
}

# Pipeline funnel tiers ahead of annotation; the annotation tiers take their
# source's colour.
PIPELINE_FUNNEL_COLORS = {
    "All deployments": "#90A4AE",
    "Not a bad deployment": "#D9603B",
    "Passed ingest": "#E8A33D",
    "Footage exists": "#42A5F5",
    "ML annotated": SOURCE_COLORS["ml"],
    "CitSci annotated": SOURCE_COLORS["citsci"],
    "Expert annotated": SOURCE_COLORS["expert"],
}

# Footage funnel on the Operations Deployments view. "Footage exists" is the
# video-present status colour; "#42A5F5" here marks the expected tier, which is
# not the same tier as the pipeline funnel's "Footage exists".
MEDIA_FUNNEL_COLORS = {
    "All deployments": "#90A4AE",
    "Footage expected": "#42A5F5",
    "Footage exists": VIDEO_PRESENCE_COLORS["present"],
}

# Usable vs bad deployments per year.
USABLE_SPLIT_COLORS = {"ok": "#2E8B57", "bad": "#D9603B"}

# Footage state, with archived split by whether annotations already exist.
# Keyed by the display labels the media view draws.
DETAIL_COLOURS = {
    "Present": VIDEO_PRESENCE_COLORS["present"],
    "Archived, annotated": "#7FB77E",
    "Archived, not annotated": VIDEO_PRESENCE_COLORS["archived"],
    "Absent": VIDEO_PRESENCE_COLORS["absent"],
    "No footage expected": NEUTRAL,
    "Unknown": "#BDBDBD",
}

# The four states of a deployment that has footage, furthest from done first.
# "Annotated, standard storage" is annotation-done blue that happens to share
# the expert source hex; it is not an expert-only state.
FOOTAGE_STATE_COLORS = {
    "Not annotated, archived": "#D9603B",
    "Not annotated, standard storage": "#FF9800",
    "Annotated, standard storage": "#0072B2",
    "Annotated, archived": DETAIL_COLOURS["Archived, annotated"],
}

# Annotation-depth bands per survey: source colours for work done, then the
# outstanding states. Keyed by the labels in `charts/surveys.BAND_LABELS`.
BAND_COLOURS = {
    "Expert": SOURCE_COLORS["expert"],
    "CitSci": SOURCE_COLORS["citsci"],
    "ML only": SOURCE_COLORS["ml"],
    "Not annotated yet": "#E0E0E0",
    "Bad deployment": "#8C2F1F",
    "Ingest issue": "#E08E7B",
}


# ── Plotly layout defaults ───────────────────────────────────────────────────
#
# Recessive grid and axes, no chart-junk background. Spread into a figure with
# `fig.update_layout(**PLOT_LAYOUT)` so every page gets the same furniture.
TRANSPARENT = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(128,128,128,0.2)"
# A few charts draw on white instead, with a light solid grid and zero line.
PLOT_BG_WHITE = "white"
GRID_ON_WHITE = "#EEEEEE"
ZERO_LINE_ON_WHITE = "#CCCCCC"

PLOT_LAYOUT = {
    # No font here. Chart text is set once, in `apply_plotly_defaults` below,
    # as the Plotly default template, that reaches every figure in the app,
    # including the modules that never import this file. Setting a size here
    # too would override the template and put the sizes back out of step.
    "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
    "plot_bgcolor": TRANSPARENT,
    "paper_bgcolor": TRANSPARENT,
    "xaxis": {"showgrid": False},
    "yaxis": {"gridcolor": GRID_COLOR},
}


# ── App-wide text sizing ─────────────────────────────────────────────────────

# Chart text was set per figure, and five chart modules never import this file
# at all, so raising `PLOT_LAYOUT` only moved some of them. Setting Plotly's
# DEFAULT TEMPLATE reaches every figure in the app whether or not the module
# knows about this one, it just has to run once per page load.
CHART_FONT_SIZE = 17
CHART_TICK_SIZE = 16


def apply_plotly_defaults() -> None:
    """Make every Plotly figure in the app inherit readable text.

    Imported lazily so this module stays dependency-free for anything that only
    wants the colours.
    """
    import plotly.io as pio

    base = pio.templates[pio.templates.default or "plotly"]
    template = base.to_plotly_json()
    layout = template.setdefault("layout", {})
    layout.setdefault("font", {})["size"] = CHART_FONT_SIZE
    for axis in ("xaxis", "yaxis"):
        axis_layout = layout.setdefault(axis, {})
        axis_layout.setdefault("tickfont", {})["size"] = CHART_TICK_SIZE
        axis_layout.setdefault("title", {}).setdefault("font", {})[
            "size"
        ] = CHART_TICK_SIZE
    layout.setdefault("legend", {}).setdefault("font", {})["size"] = CHART_TICK_SIZE

    pio.templates["spyfish"] = template
    pio.templates.default = "spyfish"


# Streamlit's own text is left at its defaults: the complaint was chart labels,
# not the interface, and bumping captions and metrics made the page heavy.
UI_TEXT_CSS = ""


# ── Interface chrome (inline HTML/CSS) ───────────────────────────────────────
#
# Colours written into st.markdown HTML and injected CSS rather than a chart.
# Text is a three-step neutral ramp, strongest first.
TEXT_PRIMARY = "#2B3A55"
TEXT_SECONDARY = "#7A879C"
TEXT_TERTIARY = "#94A0B4"
ROW_DIVIDER = "#F0F3F8"
# Section chips in the sticky header: translucent grey so they work on either
# Streamlit theme.
CHIP_BORDER = "rgba(128,128,128,.35)"
CHIP_BORDER_HOVER = "rgba(128,128,128,.75)"
CHIP_BG_HOVER = "rgba(128,128,128,.12)"
# Sticky header: the fallback when Streamlit's --background-color is missing,
# and the soft shadow under it.
HEADER_BG_FALLBACK = "#FFFFFF"
HEADER_SHADOW = "rgba(20, 35, 60, .5)"
