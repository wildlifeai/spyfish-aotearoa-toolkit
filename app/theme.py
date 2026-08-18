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

Streamlit here runs in its default light theme (there is no `[theme]` block in
`.streamlit/config.toml`), so the light values are what actually ship. The dark
variants are kept validated alongside them so that switching the app theme later
is a one-line change rather than a re-derivation.
"""

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

SOURCE_COLORS_DARK = {
    "expert": "#3585C0",
    "citsci": "#B37B00",
    "ml": "#009468",
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
    subset shifts every colour up whenever a species is removed — the survivors
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
}

VIDEO_PRESENCE_COLORS = {
    "present": "#26A69A",
    "archived": "#FF9800",
    "absent": "#EF5350",
    "no_video_bad_dep": NEUTRAL,
}

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


def protection_color_map(statuses) -> dict:
    """Plotly `color_discrete_map` for the protection statuses actually present.

    Anything not in `PROTECTION_COLORS` falls back to neutral grey instead of
    raising or being auto-assigned a hue, a status added upstream should look
    unclassified, not like a new category with meaning.
    """
    return {status: PROTECTION_COLORS.get(status, NEUTRAL) for status in statuses}


def protection_sort_key(status: str) -> int:
    """Sort position for a protection status, strongest first. Unknown sorts last."""
    try:
        return PROTECTION_ORDER.index(status)
    except ValueError:
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


# ── Plotly layout defaults ───────────────────────────────────────────────────
#
# Recessive grid and axes, no chart-junk background. Spread into a figure with
# `fig.update_layout(**PLOT_LAYOUT)` so every page gets the same furniture.
PLOT_LAYOUT = {
    # No font here. Chart text is set once, in `apply_plotly_defaults` below,
    # as the Plotly default template, that reaches every figure in the app,
    # including the modules that never import this file. Setting a size here
    # too would override the template and put the sizes back out of step.
    "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "xaxis": {"showgrid": False},
    "yaxis": {"gridcolor": "rgba(128,128,128,0.2)"},
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
