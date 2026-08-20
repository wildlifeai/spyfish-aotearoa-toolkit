"""Shared shell for the reporting views.

Each view is its own nav entry and its own URL, but they all need the same
filters applied the same way. This builds the context once per page load and
dispatches to the selected view.

Views are registered in `VIEWS`. A view that is not built yet maps to None and
gets a placeholder saying what it will hold, rather than an empty chart that
looks like a view reporting nothing.
"""

import sqlite3
from dataclasses import dataclass

import plotly.express as px
import streamlit as st
from theme import protection_sort_key
from utils import sync_db_if_needed

from . import (
    annotations,
)
from . import data as report_data
from . import (
    layout,
    media,
    metadata,
    mpa,
    ops_mpa,
    pipeline,
    report_annotations,
    sites,
    species,
    species_search,
    surveys,
)
from .charting import style, year_axis
from .charts.deployments import (
    render_deployments_per_year,
)

# Two sections, the same seven names in each, asking different questions.
# Reporting answers "what is out there"; Operations answers "what state is the
# data in".
#
# Each entry points at its view's render function, this dict is routing only,
# no view logic lives here. Entries mapping to None are not built yet and say so.
REPORTING_VIEWS = {
    # The section landing view. Named "… home" so the tab strip does not
    # repeat the section name with no way to tell which is which.
    "Report home": None,  # replaced below by render_home
    "MPA": mpa.render,
    "Surveys": None,
    "Sites": sites.render,
    "Deployments": None,
    "Annotations": report_annotations.render,
    "Species": species.render,
    # Reporting only: an operations question is about a deployment or a
    # stage, not about finding one animal.
    "Species search": species_search.render,
}

OPERATIONS_VIEWS = {
    "Operations home": pipeline.render,
    # Operations only, a data-entry problem is not a finding about the
    # reserves, so it has no counterpart on the Reporting side.
    "Metadata error review": metadata.render,
    "MPA": ops_mpa.render,
    "Surveys": surveys.render,
    "Sites": None,
    "Deployments": media.render,
    "Annotations": annotations.render,
    "Species": None,
}

SECTIONS = {"Reporting": REPORTING_VIEWS, "Operations": OPERATIONS_VIEWS}


def register_home() -> None:
    """Point the Reporting entry at `render_home`, which is defined further down.

    Deferred because the registry sits at the top of the module for readability
    and `render_home` cannot be referenced before it exists.
    """
    REPORTING_VIEWS["Report home"] = render_home


NOT_BUILT = {
    ("Reporting", "Surveys"): (
        "what each survey found: species recorded, abundance, notable "
        "sightings. The pipeline state of a survey is in Operations.",
        "",
    ),
    ("Reporting", "Deployments"): (
        "per-deployment ecology: abundance distribution, the best drops, and "
        "peak-abundance frames.",
        "",
    ),
    ("Operations", "Sites"): (
        "site coverage: planned versus achieved deployments, how far actual "
        "positions sit from the targeted coordinates, and sites with no usable "
        "footage.",
        "the targeted coordinates come from `BUV Survey Sites.csv`.",
    ),
    ("Operations", "Species"): (
        "species data hygiene: unidentified rates, names that do not map to "
        "the class map, and species-level disagreement between sources.",
        "",
    ),
}


# ── The shared filters, in one registry ──────────────────────────────────────
#
# One entry per filter: its session key (the value that survives navigation,
# never attached to a widget), its widget key, its query-param name, and its
# label. Reset, URL adoption and URL mirroring all loop over this, so adding a
# multiselect filter is one entry here plus its options in `build_context`.
#
# Query-param names are short and singular because they are typed and read by
# people: `?reserve=X&reserve=Y&years=2018-2024`.
#
# No shared species filter for now: the MPA populations panel carries its own
# species picker, which matters more, and two species controls fighting each
# other was worse than one. The plumbing for it (a registry entry here, its
# options in `build_context`, `ctx["species"]`, the panel passthrough) is
# ready for the day it comes back.
@dataclass(frozen=True)
class FilterSpec:
    state_key: str
    widget_key: str
    qp: str
    label: str
    # "multi" filters share all the generic machinery. "range" (the year
    # slider) and "select" (the source picker) carry their own URL encoding
    # and widget handling in the functions below.
    kind: str = "multi"


FILTERS = {
    "years": FilterSpec("report_years", "_w_years", "years", "Survey year", "range"),
    "reserves": FilterSpec("report_reserves", "_w_reserves", "reserve", "MPA"),
    "regions": FilterSpec("report_regions", "_w_regions", "region", "Region"),
    "protections": FilterSpec(
        "report_protections", "_w_protections", "protection", "Protection status"
    ),
    "source": FilterSpec("report_source", "_w_source", "source", "Source", "select"),
}

_MULTI = {name: spec for name, spec in FILTERS.items() if spec.kind == "multi"}

FILTER_KEYS = tuple(
    key for spec in FILTERS.values() for key in (spec.widget_key, spec.state_key)
)


def _reset_filters() -> None:
    """Drop every filter key so the next run seeds them from the data again."""
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)
    for spec in FILTERS.values():
        if spec.qp in st.query_params:
            del st.query_params[spec.qp]


def _adopt_url_filters() -> None:
    """Seed the filters from the URL, once per session.

    A pasted link opens with its filters already applied. Once only: after
    this the widgets own the values and `_mirror_filters_to_url` keeps the URL
    following them, so a stale URL cannot fight later clicks.

    An absent param is no opinion, not a reset. Streamlit strips the query
    string every time the nav switches page, so an empty URL must never clear
    a live selection — that is also why adoption cannot simply run every time.
    """
    if st.session_state.get("_url_filters_adopted"):
        return
    st.session_state["_url_filters_adopted"] = True
    qp = st.query_params
    for spec in _MULTI.values():
        values = qp.get_all(spec.qp)
        if values:
            # Not validated against the data here: the widget seed drops values
            # the data does not hold, and the mirror rewrites the URL from what
            # survived.
            st.session_state[spec.state_key] = values
    years = qp.get(FILTERS["years"].qp)
    if years:
        try:
            lo, hi = (int(part) for part in years.split("-", 1))
            st.session_state["report_years"] = (lo, hi)
        except ValueError:
            pass  # a hand-mangled URL is ignored, not an error
    source = qp.get(FILTERS["source"].qp) or ""
    # Case-insensitive: the choices are capitalised ("Expert") but a URL is as
    # often typed as pasted, and ?source=expert should mean the obvious thing.
    match = next(
        (c for c in report_data.SOURCE_CHOICES if c.lower() == source.lower()),
        None,
    )
    if match:
        st.session_state["report_source"] = match


def _mirror_filters_to_url(selected: dict, bounds) -> None:
    """Write the current filters into the URL, so the view can be shared.

    Re-asserted on every build because a page switch rewrites the URL to the
    new page's path and drops the query string with it. Only non-default
    values are written, so an untouched page keeps a clean URL, and writes
    are skipped when the URL already agrees, so reruns do not spam the
    browser history.
    """
    qp = st.query_params

    def _sync(key: str, wanted: list) -> None:
        if qp.get_all(key) != wanted:
            if wanted:
                qp[key] = wanted
            elif key in qp:
                del qp[key]

    for name, spec in _MULTI.items():
        _sync(spec.qp, list(selected[name]))
    year_range = selected["years"]
    non_default_years = bounds and year_range and tuple(year_range) != tuple(bounds)
    _sync(
        FILTERS["years"].qp,
        [f"{year_range[0]}-{year_range[1]}"] if non_default_years else [],
    )
    _sync(
        FILTERS["source"].qp,
        (
            [selected["source"]]
            if selected["source"] != report_data.BEST_AVAILABLE
            else []
        ),
    )


def _prune_multi(spec: FilterSpec, options: list) -> None:
    """Seed the widget from the stored values, and prune it on every run.

    Option lists follow the other filters ("hide what does not exist"), so a
    value picked earlier can drop out of the list; left in the widget key it
    would make the multiselect raise. Pruning before the widget is built is
    safe — assigning to a widget key only raises after its widget exists in
    the current run.
    """
    current = st.session_state.get(spec.widget_key, st.session_state[spec.state_key])
    st.session_state[spec.widget_key] = [v for v in current if v in options]


def _multi_filter(spec: FilterSpec, options: list, **kwargs) -> list:
    """One multiselect filter: prune, render, store. Returns the selection."""
    _prune_multi(spec, options)
    value = st.multiselect(spec.label, options, key=spec.widget_key, **kwargs)
    st.session_state[spec.state_key] = value
    return value


def build_context() -> dict:
    """Sidebar filters plus the filtered frames every view reads.

    The filters render into the sidebar, above the page nav (`_sidebar_css`
    does the reordering): MPA and survey year always visible, the rest behind
    a "More filters" expander. The same controls appear on every reporting
    view and apply to whichever view is open.

    Filters live here rather than in each view so the counts stay consistent.
    Two views filtering separately would disagree about how many deployments
    exist, with both totals on screen at once.
    """
    # Sync BEFORE any reader runs, so a fresh deploy pulls the databases from
    # S3 first. The readers open SQLite read-only, so if the databases are
    # still absent they raise rather than creating empty files, an empty
    # file's fresh mtime would make db_sync skip the S3 download forever.
    if not sync_db_if_needed():
        # Do not cache an S3 failure for the TTL, clear so the next run
        # retries, and say what happened instead of silently showing stale data.
        sync_db_if_needed.clear()
        st.warning(
            "Could not check S3 for a newer database, showing whatever local "
            "data exists. Check AWS credentials in `.env` if this persists."
        )
    try:
        deployments = report_data.load_deployments()
        annotations = report_data.load_annotations()
    except sqlite3.OperationalError:
        st.error(
            "No local database found, and it could not be downloaded from S3. "
            "Check the AWS credentials in `.env`, then click **Refresh Cache** "
            "in the sidebar, or run the pipeline once on this machine."
        )
        st.stop()

    _sidebar_css()

    # Filter values are kept under their own session keys, not the widget keys.
    # Streamlit drops widget state for widgets that were not rendered in a run,
    # and switching page means exactly that, so reading the widget key back
    # would reset every filter on every navigation. These keys are never
    # attached to a widget, so nothing clears them.
    dated = deployments["survey_year"].dropna()
    bounds = (int(dated.min()), int(dated.max())) if not dated.empty else None
    st.session_state.setdefault("report_years", bounds)
    for spec in _MULTI.values():
        st.session_state.setdefault(spec.state_key, [])
    st.session_state.setdefault("report_source", report_data.BEST_AVAILABLE)

    # After the defaults, before the widget seeds: a pasted URL overrides the
    # defaults, and the widgets then seed from what it said.
    _adopt_url_filters()

    # The slider is seeded into its own key ONCE and then left alone. Passing
    # `value=` / `default=` / `index=` on every run is what caused the slider to
    # lose the first drag: those arguments are part of a widget's identity, so
    # feeding the mirrored value back in changed the identity on the rerun the
    # drag triggered, and Streamlit rebuilt the widget at the argument's value
    # instead of the dragged one. The second drag stuck because by then the
    # mirror and the widget agreed.
    if "_w_years" not in st.session_state and bounds:
        stored = st.session_state["report_years"] or bounds
        # Clamped: a stored range from a previous session can sit outside the
        # bounds the data now has, which Streamlit rejects.
        st.session_state["_w_years"] = (
            max(bounds[0], min(stored[0], bounds[1])),
            max(bounds[0], min(stored[1], bounds[1])),
        )

    # Option lists follow the other filters, so a picker only offers values
    # that exist under the current selection ("hide what does not exist").
    # They are computed from the session keys BEFORE their widgets render:
    # any widget interaction reruns the whole script, so by the time anything
    # draws, the keys already hold the new value.
    years_now = st.session_state.get("_w_years") or st.session_state["report_years"]

    def in_years(frame):
        if not years_now:
            return frame
        # Undated rows kept, same policy as `report_data.build_context`.
        year = frame["survey_year"]
        return frame[year.between(*years_now) | year.isna()]

    dep_scope = in_years(deployments)
    reserve_options = sorted(
        report_data.split_reserves(dep_scope["link_to_marine_reserve"])
    )

    # Active-filter count for the expander label, from the session keys, so a
    # closed expander still says whether anything is narrowing the page.
    active = sum(bool(st.session_state[spec.state_key]) for spec in _MULTI.values())
    stored_years = st.session_state["report_years"]
    if bounds and stored_years and tuple(stored_years) != tuple(bounds):
        active += 1
    if st.session_state["report_source"] != report_data.BEST_AVAILABLE:
        active += 1

    with st.sidebar, st.container(key="sidebar_filters"):
        # The whole block is one expander under the nav, open by default; the
        # active count stays in the label so it still reads at a glance if
        # someone closes it.
        expander = st.expander(
            f"🔍 Filters · {active} active" if active else "🔍 Filters",
            expanded=True,
        )
    with expander:
        st.caption(
            "Filters apply to every Reporting and Operations view and stick "
            "between views."
        )
        reserves = _multi_filter(
            FILTERS["reserves"],
            reserve_options,
            help="Marine protected area, from the site's "
            "`link_to_marine_reserve`. A site between two areas is "
            "counted under both.",
            placeholder="All reserves",
        )
        if reserves:
            dep_scope = dep_scope[
                report_data.matches_reserves(
                    dep_scope["link_to_marine_reserve"], reserves
                )
            ]

        year_range = st.session_state["report_years"]
        if bounds and bounds[0] < bounds[1]:
            year_range = st.slider(
                "Survey year",
                bounds[0],
                bounds[1],
                key="_w_years",
            )
        else:
            # `st.slider` raises when min equals max, so a database holding a
            # single year (a sample database, or a first season) cannot have
            # one. Name the year rather than showing nothing: a blank reads as
            # a filter that broke, not one with nothing to choose.
            only = (
                f"{bounds[0]} · only year in the data"
                if bounds
                else ("no dated deployments")
            )
            st.caption(f"Survey year: {only}")
        st.session_state["report_years"] = year_range

        # A plain container, not a nested expander (Streamlit forbids those):
        # everything sits in the one Filters expander.
        with st.container():
            # The annotations in scope under the other filters, for the
            # source picker's coverage counts below.
            ann_scope = in_years(annotations)
            if reserves:
                ann_scope = ann_scope[
                    report_data.matches_reserves(
                        ann_scope["link_to_marine_reserve"], reserves
                    )
                ]
            stored_regions = st.session_state["report_regions"]
            stored_prot = st.session_state["report_protections"]
            if stored_regions:
                ann_scope = ann_scope[ann_scope["region"].isin(stored_regions)]
            if stored_prot:
                ann_scope = ann_scope[ann_scope["protection_status"].isin(stored_prot)]

            regions = _multi_filter(
                FILTERS["regions"],
                sorted(r for r in dep_scope["region"].dropna().unique() if r),
                placeholder="All regions",
            )
            protections = _multi_filter(
                FILTERS["protections"],
                sorted(
                    (p for p in dep_scope["protection_status"].dropna().unique() if p),
                    key=protection_sort_key,
                ),
                placeholder="All statuses",
            )

            # Deployment counts in the labels, as the Experiments page does: an
            # empty chart is otherwise a mystery, when the answer is simply
            # that the chosen source has annotated almost nothing. Sources
            # with nothing at all in scope are left out of the list entirely.
            coverage = report_data.source_coverage(ann_scope)

            def _source_label(choice: str) -> str:
                if choice == report_data.BEST_AVAILABLE:
                    return "Best available (expert > citsci > ml)"
                if choice == report_data.ALL_SOURCES:
                    return "All sources (rows can repeat)"
                return f"{choice} ({coverage.get(choice, 0):,} deps)"

            always = (report_data.BEST_AVAILABLE, report_data.ALL_SOURCES)
            source_choices = [
                c
                for c in report_data.SOURCE_CHOICES
                if c in always or coverage.get(c, 0) > 0
            ]
            if "_w_source" not in st.session_state:
                st.session_state["_w_source"] = st.session_state["report_source"]
            if st.session_state["_w_source"] not in source_choices:
                st.session_state["_w_source"] = report_data.BEST_AVAILABLE
            source = st.selectbox(
                "Source",
                source_choices,
                format_func=_source_label,
                key="_w_source",
                help="Which annotation source the species numbers come from. "
                "**Best available** applies the pipeline's own precedence per "
                "deployment, expert beats citsci beats ML, and keeps only "
                "the best source's rows for each deployment, so a deployment "
                "reviewed by both an expert and volunteers is not counted "
                "twice. **All** keeps every source's rows side by side, which "
                "is what Species search wants but will double-count anything "
                "that sums rows. Picking a single source answers 'what does "
                "this source alone say', and the count beside each is how "
                "many deployments it covers.",
            )
            st.session_state["report_source"] = source

            # Deleting the keys rather than assigning defaults to them:
            # assigning to a widget key after its widget has been created in
            # this run raises, and `build_context` re-seeds whatever is missing
            # on the next run anyway. `on_click` runs before that next run
            # builds its widgets, so this is the one safe place to clear them.
            st.button(
                "↺ Reset filters",
                key="_w_reset",
                on_click=_reset_filters,
                help="Reset every filter to its default.",
            )

    selected = {
        "years": year_range,
        "reserves": reserves,
        "regions": regions,
        "protections": protections,
        "source": source,
    }
    _mirror_filters_to_url(selected, bounds)

    # Not filtered to ingested-only. Hiding excluded deployments would flatter
    # every coverage number on the page, so they stay in and the front page
    # explains what they are.
    return report_data.build_context(
        deployments,
        annotations,
        year_range,
        reserves,
        ingested_only=False,
        source=source,
        regions=regions,
        protections=protections,
    )


def _sticky_header_css() -> None:
    """Pin the view title, filters and tab strip to the top of the page.

    The sticky rule goes on the WRAPPER around the keyed container, not on the
    container itself. A sticky element only travels inside its own containing
    block, and Streamlit wraps each block in a `stLayoutWrapper` that is exactly
    as tall as its contents, so sticking the header to itself gives it zero
    room and it scrolls away immediately. The wrapper is a child of the tall
    `stVerticalBlock`, which is the scroll height the header needs to travel
    over.

    Sticky rather than fixed: fixed would leave a hole its own height in the
    flow, needing manual padding that breaks whenever the header wraps.

    A solid background is required, since sticky elements are transparent by
    default and the charts would scroll visibly underneath.
    """
    st.markdown(
        """
        <style>
          div[data-testid="stLayoutWrapper"]:has(> .st-key-report_header) {
            position: sticky;
            /* Streamlit's own toolbar is 60px tall and sits OVER the page, so
               pinning at 0 parks the header underneath it and only a sliver
               shows. `--header-height` is the app's own value where it exists;
               3.75rem is that same 60px for older versions. */
            top: var(--header-height, 3.75rem);
            z-index: 100;
            background: var(--background-color, #FFFFFF);
            padding-top: .25rem;
            box-shadow: 0 8px 10px -10px rgba(20, 35, 60, .5);
          }
          /* Streamlit's default gap would show the page through the header.
             The band holds up to three rows, shared filters, a view's own
             filters, the chip strip, so they are pulled tight against each
             other, and the bottom padding keeps the last of them off the
             header's own edge. */
          .st-key-report_header {
            background: inherit;
            gap: .1rem;
          }
          /* Clear air under the last row of the band. On the WRAPPER, not on
             the container inside it: padding on the inner block sat inside the
             sticky box without lengthening what the page flows around, so on a
             view with no chips — Report home — the first line of text was laid
             out 12px under the header and disappeared behind it. */
          div[data-testid="stLayoutWrapper"]:has(> .st-key-report_header) {
            padding-bottom: 1.4rem;
          }
          /* The header-measuring iframe (`layout.measure_header`) is
             zero-height but still a block, so it would take a full block gap
             out of every page for nothing. Out of flow rather than hidden: a
             `display: none` iframe may never load, and the script inside it is
             what publishes the height the anchors depend on. Here rather than
             in the chip stylesheet, because every view measures, and only some
             have chips. */
          [data-testid="stElementContainer"]:has(> [data-testid="stIFrame"]) {
            position: absolute;
            height: 0;
            visibility: hidden;
          }

          /* ── Page density ────────────────────────────────────────────────
             Streamlit's defaults are set for a page of prose: 96px of padding
             above the first element and 1rem between every block. A report
             view is a dozen small charts meant to be read against each other,
             and those defaults pushed the second chart of every view below the
             fold. The header is sticky, so nothing is lost by starting the
             page closer to the top. */
          [data-testid="stMainBlockContainer"] {
            /* Exactly the sticky offset, and not a pixel less. A sticky element
               whose natural position is ABOVE its `top` is pushed down to it,
               and the push is visual only: the page below keeps flowing from
               where the header would have been. At 1.5rem of padding the header
               started at 24px, got shoved to 60px, and landed on top of the
               first line of every view. Matching the two means it never moves,
               so nothing is ever underneath it. */
            padding-top: var(--header-height, 3.75rem);
            padding-bottom: 2rem;
          }
          [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
            gap: .5rem;
          }
          /* Dividers separate sections; at Streamlit's default margins they
             cost more height than the section headings they sit above. */
          [data-testid="stMainBlockContainer"] hr {
            margin: .35rem 0;
          }
          [data-testid="stMainBlockContainer"]
            [data-testid="stHeadingWithActionElements"] {
            padding-top: .2rem;
            padding-bottom: 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_css() -> None:
    """Tighten the sidebar filter widgets a little.

    Sidebar widgets default to body-text sizing, and six of them inside one
    expander is a lot of chrome. The nav stays in its default place at the
    top; the filter expander renders below it, in Streamlit's own order.
    """
    st.markdown(
        """
        <style>
          .st-key-sidebar_filters [data-testid="stWidgetLabel"] {
            margin-bottom: .05rem;
            min-height: 0;
          }
          .st-key-sidebar_filters label p {
            font-size: .75rem !important;
            margin-bottom: 0 !important;
          }
          /* The min/max end labels under a slider double its height. */
          .st-key-sidebar_filters [data-testid="stSliderTickBar"] {
            display: none;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# (section, view name) -> the StreamlitPage object the entrypoint created.
#
# `st.page_link` accepts a file path, an external URL, or a StreamlitPage. Views
# defined as functions have no file, so only the object works and a
# "/reporting-sites" string raises. The entrypoint owns the objects, so it
# registers them here rather than this module importing the entrypoint.
#
# Used by the links on the front page. Do not add a tab strip that reads this
# as well: it duplicates the sidebar, and a remembered tab selection can
# override a sidebar click and bounce the reader back to the previous view.
PAGES: dict = {}


def register_pages(pages: dict) -> None:
    """Called once by the entrypoint with {(section, view name): StreamlitPage}."""
    PAGES.clear()
    PAGES.update(pages)


def render_view_boxes(section: str, summaries: dict) -> None:
    """Bordered link boxes for a section's views, coming-soon ones included.

    Shared by both home pages. A fresh bordered row per three views: reusing
    one row's columns would stack two views inside the same box.
    """
    items = list(summaries.items())
    for start in range(0, len(items), 3):
        cols = st.columns(3, border=True)
        for col, (view_name, blurb) in zip(cols, items[start : start + 3]):
            page = PAGES.get((section, view_name))
            with col:
                if page is not None:
                    st.page_link(page, label=view_name)
                else:
                    # Not built yet, so it has no nav entry. Named here
                    # anyway: what is coming is part of the picture, and
                    # hiding it invites "where do I find this?" questions.
                    st.markdown(f"{view_name} *(coming soon)*")
                st.caption(blurb)


def render_home(ctx: dict) -> None:
    """The Reporting landing view: the programme in one screen.

    A view like any other: `render` draws the title, filters and tabs and hands
    the context in. It must not build its own header, or two places can
    disagree about what the filters are.
    """
    st.subheader("How to use it")
    st.markdown(
        "BUV survey results from the marine reserves, for MPA reporting. "
        "The sidebar filters (year, MPA and more) apply to every view and "
        "stick between views. Each section below covers one topic; the "
        "chips at the top of a view link to its parts. Pipeline state is in "
        "the Operations section."
    )
    st.caption(
        "Any feedback, issues, or something you want to see on this page? "
        "Contact us at kalindi@wildlife.ai"
    )
    st.write("")

    # What's available, straight under the intro: the reader decides where to
    # go before scrolling through the headline numbers.
    st.subheader("Sections")
    render_view_boxes(
        "Reporting",
        {
            "MPA": "Each marine protected area: how much has been surveyed "
            "there and what has been found.",
            "Surveys": "How much surveying has happened, when, and how far "
            "each survey has got through the pipeline.",
            "Sites": "Species abundance and per-site rollups, grouped by "
            "region, MPA and protection status.",
            "Deployments": "Where the footage is: ready to process, "
            "archived, or missing.",
            "Annotations": "Annotation coverage, and how the three sources " "compare.",
            "Species": "Per-species abundance and occurrence across sites "
            "and years.",
        },
    )
    st.write("")
    st.write("")

    dep = ctx["deployments"]
    ann = ctx["annotations"]
    dated = dep[dep["survey_year"].notna()]

    reserves = report_data.split_reserves(dep["link_to_marine_reserve"])

    # Computed up front so the headline comparison can sit in the KPI band with
    # everything else, rather than in a section of its own further down.
    # `protection_group` is the shared, config-driven bucketing, so this and
    # the Species view cannot disagree about which deployments count as
    # protected.
    from ecology_data import PROTECTED, UNPROTECTED, protection_group

    protection_means = {}
    protection_counts = {}
    if not ann.empty:
        per_species = report_data.species_maxn(ann)
        # survey_year and site_id ride along for the base charts further down,
        # which reuse this frame.
        meta = ann[
            ["drop_id", "protection_status", "survey_year", "site_id"]
        ].drop_duplicates("drop_id")
        per_species = per_species.merge(meta, on="drop_id", how="left")
        per_species["Group"] = protection_group(per_species["protection_status"])
        # Richness, not summed MaxN: adding a snapper to a school of sweep
        # gives a number with no meaning (the MPA view says as much), so the
        # headline compares species per deployment instead. `real_species`
        # drops the generic and unidentified classes — a deployment carrying
        # only those says nothing about richness and leaves the mean — while
        # absence records (null species) keep theirs in at zero, since
        # `nunique` ignores NaN. Protected against unprotected only; partial
        # regimes are their own group and belong in neither mean.
        per_dep = (
            report_data.real_species(per_species)
            .loc[lambda f: f["Group"].isin([PROTECTED, UNPROTECTED])]
            .groupby(["drop_id", "Group"])["scientific_name"]
            .nunique()
            .reset_index(name="richness")
        )
        grouped = per_dep.groupby("Group")["richness"]
        protection_means = grouped.mean().round(1).to_dict()
        protection_counts = grouped.count().to_dict()

    st.subheader("Overview")
    # Scope first, then results. Widest container leads: MPAs contain sites,
    # sites hold deployments, surveys are the visits.
    kpis = st.columns(4)
    kpis[0].metric(
        "MPAs",
        f"{len(reserves):,}",
        help="Distinct marine protected areas the surveyed sites link to. "
        "A site between two areas counts under both.",
    )
    kpis[1].metric("Sites", f"{dep['site_id'].nunique():,}")
    kpis[2].metric("Surveys", f"{dep['survey_id'].nunique():,}")
    kpis[3].metric("Deployments", f"{dep['drop_id'].nunique():,}")

    kpis = st.columns(4)
    kpis[0].metric(
        "Species recorded",
        f"{ann['scientific_name'].nunique():,}",
        help="Distinct species names across every annotated deployment.",
    )
    # The funnel's own headline: how many deployments carry the annotation the
    # reporting rests on. Replaces a span of years, which the per-year charts
    # below already show and in more detail.
    expert_drops = int((dep["expert_annotations"].fillna(0) > 0).sum())
    kpis[1].metric(
        "Expert annotated",
        f"{expert_drops:,}",
        help="Deployments carrying legacy or BIIGLE expert annotations. Expert "
        "wins wherever sources disagree, so this is the number the "
        "reporting rests on.",
    )
    for slot, group in ((2, "Protected"), (3, "Unprotected")):
        value = protection_means.get(group)
        kpis[slot].metric(
            f"Species/dep {group.lower()}",
            "—" if value is None else f"{value:.1f}",
            help=f"Mean distinct species recorded per annotated deployment "
            f"{'inside' if group == 'Protected' else 'outside'} an MPA, "
            f"across {protection_counts.get(group, 0):,} deployments. "
            f"Generic detections (fish, bait) and the unidentified bucket "
            f"are not counted as species; a reviewed deployment with "
            f"nothing seen counts as zero. Worth following up, not "
            f"quoting: deployments are not evenly spread across sites, "
            f"years or effort.",
        )

    st.write("")

    # No pipeline funnel here: how far the data got through processing is an
    # Operations question, and its home page opens with exactly that funnel.
    st.subheader("Survey activity per year")
    if dated.empty:
        st.info("No deployments have a parseable date in their DropID.")
    else:
        per_year = (
            dated.groupby(dated["survey_year"].astype(int))
            .agg(Deployments=("drop_id", "nunique"), Surveys=("survey_id", "nunique"))
            .reset_index()
            .rename(columns={"survey_year": "Year"})
        )
        # Stacked one above the other on a shared x-axis rather than side by
        # side: the two counts differ by a factor of about fifty, so one
        # chart would flatten surveys into the axis. Aligned years let the
        # pair be read together, which is the point, many deployments in
        # few surveys is a different year from the reverse.
        st.markdown("**Surveys per year**")
        fig = px.bar(per_year, x="Year", y="Surveys", text="Surveys")
        fig.update_traces(
            marker_color="#1E6FB4", textposition="outside", cliponaxis=False
        )
        style(fig, height=190)
        fig.update_xaxes(title=None)
        year_axis(fig)
        st.plotly_chart(fig, key="home_surveys_per_year")

        # Same chart as the Surveys view, from the same function.
        render_deployments_per_year(dated, key="home_per_year")

    # ── The base comparisons, from the same functions as the Species view ────
    # The front page answers the first questions a reader arrives with, in
    # this order: how are species doing over the years, where has the
    # surveying happened, where is each species seen, and does protection make
    # a difference. Reusing the views' own chart functions (on the same
    # per_species frame the KPIs above rest on) means home and the full views
    # cannot disagree; the links say where the full versions live.
    species_page = PAGES.get(("Reporting", "Species"))
    mpa_page = PAGES.get(("Reporting", "MPA"))
    if not ann.empty:
        from ecology_data import load_common_names

        from .charts.mpa import render_mpa_populations
        from .charts.species import (
            render_reserve_effect,
            render_species_by_site,
            render_species_over_time,
        )
        from .site_data import filtered_site_frames, render_map_gate

        common_names = load_common_names()

        # 1. Species over the years. `with_sites=False` holds back its
        # companion occurrence heatmap so the map can sit between them.
        st.divider()
        picked, all_species_mode = render_species_over_time(
            per_species, common_names, with_sites=False
        )
        if species_page is not None:
            st.page_link(
                species_page,
                label="Full version on the Species view: adds detection rates, "
                "frequency vs abundance and co-occurrence",
            )

        # 2. The MPA populations panel, straight from the MPA view: the
        # species/diversity picker, the over-time trend and the gated site
        # map whose bubbles follow the picker. Ported whole rather than
        # rebuilt — the panel's own docstring is right that splitting the
        # picker from the map it controls separates a control from the thing
        # it controls, and a hand-rolled home map immediately drifted (nan
        # site names, a picker it ignored).
        st.divider()
        render_map_gate()
        site = filtered_site_frames(ctx)
        if not site["df_context"].empty:
            render_mpa_populations(
                site["df_context"],
                site["effort_view"],
                site["show_coords"],
                year_range=site["years"],
                reserves=site["reserves"],
                regions=site["regions"],
                protections=site["protections"],
                species=site["species"],
            )
        if mpa_page is not None:
            st.page_link(
                mpa_page,
                label="MPA view: this panel in context, with the per-reserve "
                "tables, diversity and trends around it",
            )

        # 3. Where each species is seen, and how much.
        if picked is not None:
            st.divider()
            render_species_by_site(picked, per_species, all_species_mode)
            if species_page is not None:
                st.page_link(
                    species_page,
                    label="On the Species view this pairs with the species "
                    "picker above it",
                )

        # 4. Protection, per species: the slope chart rather than the box
        # plots — each species' mean MaxN outside connected to its mean
        # inside, so the direction of every line is the effect.
        st.divider()
        render_reserve_effect(per_species, common_names)
        if species_page is not None:
            st.page_link(
                species_page,
                label="Full comparison on the Species view: distributions, "
                "summary table, and what was left out and why",
            )


def render(section: str, name: str) -> None:
    """Render one view of one section, with the shared filters and tab strip."""
    _sticky_header_css()

    # Filters first: they live in the sidebar now, so nothing of theirs sits
    # in the header band.
    ctx = build_context()

    # The sticky header is one row: title on the left, the view's chip strip
    # on the right. The chips container is created here, inside the sticky
    # block, and filled later by the view via `layout.chips` — Streamlit
    # places output by container, not by call order.
    with st.container(key="report_header"):
        title_col, chips_col = st.columns([0.4, 0.6], vertical_alignment="bottom")
        with title_col:
            # "Reporting · Report home" says the same thing twice, so the
            # landing views drop the section prefix.
            heading = name if name.endswith(" home") else f"{section} · {name}"
            st.markdown(
                f'<div style="font-size:1.5rem;font-weight:700;line-height:1.2;'
                f'padding-bottom:.2rem">📊 {heading}</div>',
                unsafe_allow_html=True,
            )
        with chips_col:
            # Keyed here rather than nested inside once the view fills it: a
            # keyed container inside a plain one gave the strip a flex parent
            # that squeezed it to 11px while its chips stood 26px tall, so
            # they hung out of the header. One container, one key, one box.
            chips_slot = st.container(key="section_chips")
        layout.measure_header()

    # A view's own filters go to the sidebar, under the shared block.
    layout.set_slots(chips_slot, st.sidebar.container(key="view_filters"))

    # The line explaining that the filters are section-wide lives in the "About
    # the reporting numbers" expander at the foot of the page, not here. As a
    # caption it sat between the sticky filter row and the section chips, adding
    # a band of chrome above every view's first chart to say something you only
    # need to read once.

    view = SECTIONS[section].get(name)
    if view is not None:
        view(ctx)
    else:
        describes, blocked_by = NOT_BUILT.get((section, name), ("", ""))
        st.info(f"**Not built yet.** This view will show {describes}")
        if blocked_by:
            st.caption(f"Blocked on: {blocked_by}")

    # The panel below says the same thing on all sixteen views: what this
    # rebuild is, what the database holds, and the open questions. Read once,
    # so it sits on the two section landing pages and nowhere else, rather than
    # ending every view with a paragraph the reader has already seen.
    if not name.endswith(" home"):
        return

    dep = ctx["all_deployments"]
    # A fresh or partly-ingested database can hold no parseable DropID dates at
    # all; int(nan) would take the whole page down for a sentence of caption.
    dated_years = dep["survey_year"].dropna()
    year_span = (
        f"{int(dated_years.min())} to {int(dated_years.max())}"
        if not dated_years.empty
        else "no parseable survey dates yet"
    )
    with st.expander("About the reporting numbers"):
        st.markdown(
            f"""
Rebuild of the DOC PowerBI report. Views are added one at a time. Anything not
built says so rather than rendering an empty chart.

**The filters in the sidebar apply to every {section} view**, and stay put as
you move between them. Survey year and MPA narrow which deployments are
counted; Region and Protection status narrow further, and Source picks which
annotation source the species numbers come from. The ↺ button returns them
all to their defaults.

**Coverage in this database**: {len(dep):,} deployments across
{dep['survey_id'].nunique()} surveys,
{year_span}. Of these,
{(dep['ingest_status'] == 'ok').sum():,} passed ingest. The rest are excluded or
hold validation errors and are never picked up by processing stages.

**Numbers will not match PowerBI exactly.** PowerBI reads SharePoint directly,
while this reads the pipeline database after ingest, so anything excluded at
ingest is absent here by design.

### Open questions

* **Should the MPA filter match on MPA name or on the DropID reserve code?**
  It currently matches on name, from `link_to_marine_reserve`, because that is
  what the Sites view has always used and the two had to agree. The code (the
  `KSF` in `KSF_20240124_BUV_...`) is more reliable, since it cannot be blank or
  spelled two ways, but it is not a name anyone recognises and it does not
  capture a site that sits between two areas. Names are also comma-joined
  upstream and appear in both orders, so the list is longer than the real number
  of areas.
* **Is "MPA" the right label** for every value in that column? Most are marine
  protected areas, but not all, so **"marine area"** may be the safer heading.
* **What counts as "complete", the status column or the annotation count?**
  These views read `ml_annotations > 0` / `citsci_annotations > 0` /
  `expert_annotations > 0`. The pipeline's own answer is the status column
  (`ml_status = 'ml_complete'`), and the two disagree: a deployment can be
  genuinely complete with **zero** annotations, because the model ran and found
  no fish. On the current database that is 2 ML and 1 expert deployment counted
  here as unfinished when they are not. **To check with Kalindi**: switch these
  views to the status columns, or keep the annotation-count definition and show
  "complete, no detections" as its own category so an empty run is visible
  rather than folded in with the backlog. The archived Programme Health page is
  the only one reading the status columns, so it is the comparison.
* **Test surveys are not marked.** `KSF` is a test MPA, and its deployments
  currently count towards every total on this page. They need a flag so they can
  be excluded from reporting while staying available for pipeline testing.
"""
        )
