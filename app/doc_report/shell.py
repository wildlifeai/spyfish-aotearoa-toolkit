"""Shared shell for the reporting views.

Each view is its own nav entry and its own URL, but they all need the same
filters applied the same way. This builds the context once per page load and
dispatches to the selected view.

Views are registered in `VIEWS`. A view that is not built yet maps to None and
gets a placeholder saying what it will hold, rather than an empty chart that
looks like a view reporting nothing.
"""

import sqlite3

import plotly.express as px
import streamlit as st
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
from .charting import style
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


FILTER_KEYS = (
    "_w_years",
    "_w_reserves",
    "_w_source",
    "report_years",
    "report_reserves",
    "report_source",
)


# Query-param names for the shareable filter URL. Short and singular because
# they are typed and read by people: `?reserve=X&reserve=Y&years=2018-2024`.
_QP_RESERVES = "reserve"
_QP_YEARS = "years"
_QP_SOURCE = "source"


def _reset_filters() -> None:
    """Drop every filter key so the next run seeds them from the data again."""
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)
    for key in (_QP_RESERVES, _QP_YEARS, _QP_SOURCE):
        if key in st.query_params:
            del st.query_params[key]


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
    reserves = qp.get_all(_QP_RESERVES)
    if reserves:
        # Not validated against the data here: the widget seed drops names the
        # data does not hold, and the mirror rewrites the URL from what
        # survived.
        st.session_state["report_reserves"] = reserves
    years = qp.get(_QP_YEARS)
    if years:
        try:
            lo, hi = (int(part) for part in years.split("-", 1))
            st.session_state["report_years"] = (lo, hi)
        except ValueError:
            pass  # a hand-mangled URL is ignored, not an error
    source = qp.get(_QP_SOURCE) or ""
    # Case-insensitive: the choices are capitalised ("Expert") but a URL is as
    # often typed as pasted, and ?source=expert should mean the obvious thing.
    match = next(
        (c for c in report_data.SOURCE_CHOICES if c.lower() == source.lower()),
        None,
    )
    if match:
        st.session_state["report_source"] = match


def _mirror_filters_to_url(year_range, bounds, reserves: list, source: str) -> None:
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

    _sync(_QP_RESERVES, list(reserves))
    non_default_years = bounds and year_range and tuple(year_range) != tuple(bounds)
    _sync(
        _QP_YEARS,
        [f"{year_range[0]}-{year_range[1]}"] if non_default_years else [],
    )
    _sync(
        _QP_SOURCE,
        [source] if source != report_data.BEST_AVAILABLE else [],
    )


def build_context(cols) -> dict:
    """Page-level filters plus the filtered frames every view reads.

    `cols` are containers the caller supplies, so the filters can sit on the
    title row rather than in a band of their own. The same three controls appear
    on every reporting view and apply to whichever view is open.

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

    _compact_filter_css()

    # Filter values are kept under their own session keys, not the widget keys.
    # Streamlit drops widget state for widgets that were not rendered in a run,
    # and switching page means exactly that, so reading the widget key back
    # would reset every filter on every navigation. These keys are never
    # attached to a widget, so nothing clears them.
    dated = deployments["survey_year"].dropna()
    bounds = (int(dated.min()), int(dated.max())) if not dated.empty else None
    st.session_state.setdefault("report_years", bounds)
    st.session_state.setdefault("report_reserves", [])
    st.session_state.setdefault("report_source", report_data.BEST_AVAILABLE)

    # After the defaults, before the widget seeds: a pasted URL overrides the
    # defaults, and the widgets then seed from what it said.
    _adopt_url_filters()

    # Each widget is seeded into its own key ONCE and then left alone. Passing
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

    with cols[0]:
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
            # one. Name the year rather than leaving the column empty: a blank
            # reads as a filter that broke, not one with nothing to choose.
            st.markdown(
                '<div style="font-size:.7rem;opacity:.7;margin-bottom:.1rem">'
                "Survey year</div>",
                unsafe_allow_html=True,
            )
            only = f"{bounds[0]}" if bounds else "no dated deployments"
            st.markdown(
                f'<div style="font-size:.8rem;opacity:.55;padding-bottom:.35rem">'
                f"{only} · only year in the data</div>",
                unsafe_allow_html=True,
            )
        st.session_state["report_years"] = year_range
    with cols[1]:
        names = set()
        for value in deployments["link_to_marine_reserve"].dropna():
            names.update(p.strip() for p in str(value).split(",") if p.strip())
        names = sorted(names)
        if "_w_reserves" not in st.session_state:
            st.session_state["_w_reserves"] = [
                r for r in st.session_state["report_reserves"] if r in names
            ]
        reserves = st.multiselect(
            "MPA",
            names,
            help="Marine protected area, from the site's "
            "`link_to_marine_reserve`. A site between two areas is "
            "counted under both.",
            placeholder="All reserves",
            key="_w_reserves",
        )
        st.session_state["report_reserves"] = reserves
    with cols[2]:
        # Deployment counts in the labels, as the Experiments page does: an
        # empty chart is otherwise a mystery, when the answer is simply that
        # the chosen source has annotated almost nothing.
        coverage = report_data.source_coverage(annotations)

        def _source_label(choice: str) -> str:
            if choice == report_data.BEST_AVAILABLE:
                return "Best available (expert > citsci > ml)"
            if choice == report_data.ALL_SOURCES:
                return "All sources (rows can repeat)"
            return f"{choice} ({coverage.get(choice, 0):,} deps)"

        if "_w_source" not in st.session_state:
            st.session_state["_w_source"] = st.session_state["report_source"]
        source = st.selectbox(
            "Source",
            report_data.SOURCE_CHOICES,
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
    with cols[3]:
        # Deleting the keys rather than assigning defaults to them: assigning to
        # a widget key after its widget has been created in this run raises, and
        # `build_context` re-seeds whatever is missing on the next run anyway.
        # `on_click` runs before that next run builds its widgets, so this is
        # the one safe place to clear them.
        st.button(
            "↺",
            key="_w_reset",
            on_click=_reset_filters,
            help="Reset the year, MPA and source filters to their defaults.",
        )
    _mirror_filters_to_url(year_range, bounds, reserves, source)

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


def _compact_filter_css() -> None:
    """Shrink the filter row so it reads as chrome, not content.

    Scoped by widget key. Streamlit puts an `st-key-<key>` class on every keyed
    widget's container, which is the only reliable hook here: a wrapping div
    written with `st.markdown` does not contain the widgets, because Streamlit
    renders each one into its own container.
    """
    st.markdown(
        """
        <style>
          .st-key-_w_years, .st-key-_w_reserves,
          .st-key-_w_source { font-size: .75rem; }
          .st-key-_w_years label p,
          .st-key-_w_reserves label p,
          .st-key-_w_source label p {
            font-size: .7rem !important;
            margin-bottom: 0 !important;
          }
          /* The min/max end labels under a slider double its height. */
          .st-key-_w_years [data-testid="stSliderTickBar"] { display: none; }
          .st-key-_w_years [data-testid="stSliderThumbValue"] {
            font-size: .68rem;
          }
          .st-key-_w_source [data-baseweb="select"] > div,
          .st-key-_w_reserves [data-baseweb="select"] > div {
            min-height: 1.9rem;
            font-size: .75rem;
          }
          .st-key-_w_years, .st-key-_w_reserves { padding-bottom: .1rem; }
          /* Each widget carries a label block and its own padding, and the
             tallest of the three sets the height of the whole sticky band,
             so the band was 92px for one row of controls. These trim the
             label gap and the select's internal padding; the controls stay
             clickable at 1.7rem, which is Streamlit's own small size. */
          .st-key-_w_years [data-testid="stWidgetLabel"],
          .st-key-_w_reserves [data-testid="stWidgetLabel"],
          .st-key-_w_source [data-testid="stWidgetLabel"] {
            margin-bottom: .05rem;
            min-height: 0;
          }
          .st-key-_w_source [data-baseweb="select"] > div,
          .st-key-_w_reserves [data-baseweb="select"] > div {
            min-height: 1.7rem;
            padding-top: 0;
            padding-bottom: 0;
          }
          .st-key-_w_years [data-testid="stSlider"] { padding-top: 0; }
          /* A view's own filters (Sites has three) render into the header via
             `layout.extra_filters`, so they get the same treatment, otherwise
             the second row of the same band would be half as tall again as the
             first. */
          .st-key-view_filters label p {
            font-size: .7rem !important;
            margin-bottom: 0 !important;
          }
          .st-key-view_filters [data-testid="stWidgetLabel"] {
            margin-bottom: .05rem;
            min-height: 0;
          }
          .st-key-view_filters [data-baseweb="select"] > div {
            min-height: 1.7rem;
            font-size: .75rem;
            padding-top: 0;
            padding-bottom: 0;
          }
          /* The reset button sits on the filter row, so it is sized to the
             filters rather than to body text. */
          .st-key-_w_reset button {
            padding: .15rem .4rem;
            min-height: 1.9rem;
            font-size: .9rem;
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


def render_home(ctx: dict) -> None:
    """The Reporting landing view: the programme in one screen.

    A view like any other: `render` draws the title, filters and tabs and hands
    the context in. It must not build its own header, or two places can
    disagree about what the filters are.
    """
    st.caption(
        "How the marine protected areas are doing. Processing and data-quality "
        "state live in the Operations section."
    )

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

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Survey activity per year")
        if dated.empty:
            st.info("No deployments have a parseable date in their DropID.")
        else:
            per_year = (
                dated.groupby(dated["survey_year"].astype(int))
                .agg(
                    Deployments=("drop_id", "nunique"), Surveys=("survey_id", "nunique")
                )
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
            st.plotly_chart(fig, key="home_surveys_per_year")

            # Same chart as the Surveys view, from the same function.
            render_deployments_per_year(dated, key="home_per_year")

    with right:
        # Same funnel as the Pipeline view, from the same function, so the two
        # can never disagree about how many deployments got where.
        from .pipeline import _stage_flags, render_funnel

        render_funnel(_stage_flags(dep), compact=True)

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

        from .charts._map import gate_notice, site_scatter, site_skeleton
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

        # 2. The site map, behind the same lock as every view of coordinates.
        # Bubble size is species richness — not summed MaxN, which adds
        # snapper to sweep; the per-species abundance bubbles live on the MPA
        # view the link points to.
        st.divider()
        st.subheader("Site map")
        show_coords = render_map_gate()
        site = filtered_site_frames(ctx)
        if gate_notice(site["effort_view"], show_coords):
            rich = (
                report_data.real_species(site["df_context"])
                .groupby("site_id")["scientific_name"]
                .nunique()
                .reset_index(name="species")
            )
            sites_geo = site_skeleton(site["effort_view"], rich)
            if sites_geo.empty:
                st.info("No sites in this selection carry coordinates.")
            else:
                # A floor, so a surveyed site where nothing was identified
                # stays a visible dot instead of vanishing at size zero.
                sites_geo["size"] = sites_geo["species"].clip(lower=0.4)
                fig = site_scatter(
                    sites_geo,
                    size="size",
                    hover_name="site_name",
                    hover_data={
                        "site_id": True,
                        "species": True,
                        "region": True,
                        "size": False,
                        "latitude": False,
                        "longitude": False,
                    },
                )
                st.plotly_chart(fig, use_container_width=True, key="home_site_map")
                st.caption(
                    f"{len(sites_geo):,} surveyed sites in the current "
                    "selection, bubble size = species richness (distinct real "
                    "species recorded). A small dot is a site surveyed with "
                    "nothing identified yet."
                )
        if mpa_page is not None:
            st.page_link(
                mpa_page,
                label="MPA view: the same map with per-species abundance "
                "bubbles and diversity metrics",
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

    st.divider()
    st.subheader("Views")
    summaries = {
        "MPA": "Each marine protected area: how much has been surveyed there "
        "and what has been found.",
        "Surveys": "How much surveying has happened, when, and how far each "
        "survey has got through the pipeline.",
        "Sites": "Species abundance and per-site rollups, grouped by region, "
        "MPA and protection status.",
        "Deployments": "Where the footage is: ready to process, archived, or "
        "missing.",
        "Annotations": "Annotation coverage, and how the three sources compare.",
        "Species": "Per-species abundance and occurrence across sites and years.",
    }
    cols = st.columns(3)
    for i, (name, blurb) in enumerate(summaries.items()):
        page = PAGES.get(("Reporting", name))
        if page is None:
            continue
        with cols[i % 3]:
            built = REPORTING_VIEWS.get(name) is not None
            st.page_link(page, label=name if built else f"{name} (not built yet)")
            st.caption(blurb)


def render(section: str, name: str) -> None:
    """Render one view of one section, with the shared filters and tab strip."""
    _sticky_header_css()

    # Only the title and the filters are pinned. The tab strip is navigation:
    # it is used once on arrival, and keeping it out halves the height of the
    # sticky band.
    with st.container(key="report_header"):
        # The year slider gets the widest filter column. At the previous
        # ratio its track was 139px for 15 years, under 10px per year,
        # with two handles and their value labels on it, so it read as
        # stuck rather than fine-grained.
        # The shared grid: title, then four equal filter slots. See
        # `layout.header_columns`.
        head = layout.header_columns()
        # The reset button rides with the title rather than taking a filter
        # slot: it is not a filter, and in the row it left three filters and a
        # button sharing a grid meant for four of a kind.
        title_col, reset_col = head[0].columns(
            [0.72, 0.28], vertical_alignment="bottom"
        )
        with title_col:
            # "Reporting · Report home" says the same thing twice, so the
            # landing views drop the section prefix.
            heading = name if name.endswith(" home") else f"{section} · {name}"
            st.markdown(
                f'<div style="font-size:1.5rem;font-weight:700;line-height:1.2;'
                f'padding-bottom:.2rem">📊 {heading}</div>',
                unsafe_allow_html=True,
            )
        # Three filters into the first three slots; the fourth stays empty and
        # is where a fourth shared filter would go.
        ctx = build_context(list(head[1:4]) + [reset_col])

        # Two empty containers, filled later by the view: one for any filters
        # of its own, one for its chips. They are created here, inside the
        # sticky block, so both end up in the header band even though the view
        # that supplies them does not run until further down. See `layout`.
        #
        # Created filters-first, because the order they are created in is the
        # order they appear: every filter in the band, then the chips under all
        # of them.
        # Keyed here rather than nested inside once the view fills them: a
        # keyed container inside a plain one gave the strip a flex parent that
        # squeezed it to 11px while its chips stood 26px tall, so they hung out
        # of the header. One container, one key, one box.
        layout.measure_header()
        view_filters = st.container(key="view_filters")
        layout.set_slots(st.container(key="section_chips"), view_filters)

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

**The filters at the top apply to every {section} view**, and stay put as you
move between them. Survey year and MPA narrow which deployments are counted;
Source picks which annotation source the species numbers come from. The ↺
button returns all three to their defaults.

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
