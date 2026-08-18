"""Spyfish Data Tools. App entrypoint and navigation.

Uses `st.navigation` rather than Streamlit's automatic `pages/` discovery, so
that:

* shared chrome is written once here and appears on every page, instead of each
  page calling it,
* pages can be grouped and ordered deliberately,
* the reporting views each get their own nav entry and URL,
* the support page stays routable without appearing in the nav list.

Imports here are kept to streamlit and `support`, which itself imports only
streamlit. `st.navigation` runs this file on every page load, so anything it
imports becomes a single point of failure for the whole app, including the
support page that exists to survive breakage. The reporting views are imported
lazily inside `_reporting_view` for the same reason.

`render_contact_note()` runs after `nav.run()` so it lands at the bottom of the
sidebar, and is the only route to the support page. Do not add a link above
each view as well: it costs a row above the fold on every page to repeat what
the sidebar note says.
"""

from pathlib import Path

import streamlit as st
from cache_controls import render_sidebar_refresh
from support import render_contact_note

st.set_page_config(
    page_title="Spyfish Data Tools",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Text sizing for the whole app, applied here because `st.navigation` runs this
# file on every page load, so it reaches pages that never import `theme`.
#
# Guarded: the support page exists to survive breakage elsewhere, and must not
# be taken down by a styling import.
try:
    import theme as _theme

    _theme.apply_plotly_defaults()
    st.markdown(_theme.UI_TEXT_CSS, unsafe_allow_html=True)
except Exception:  # noqa: BLE001, styling must never break navigation
    pass

# Icon per view name. The same seven names appear in both sections, asking a
# different question in each, so they share an icon.
VIEW_ICONS = {
    "Report home": "📊",
    "Operations home": "🔄",
    "MPA": "🛡️",
    "Surveys": "🗓️",
    "Sites": "📍",
    "Deployments": "📺",
    "Annotations": "✏️",
    "Species": "🐟",
    "Metadata error review": "🔍",
    "Species search": "🔎",
}


def _view(section: str, name: str):
    """Build a page callable for one view of one section.

    The import is inside the function so a broken reporting module cannot stop
    the entrypoint, and therefore cannot take the support page down with it.
    """

    def page() -> None:
        from doc_report import shell

        shell.register_home()
        shell.render(section, name)

    page.__name__ = f"{section.lower()}_{name.lower().replace(' ', '_')}"
    return page


def home():
    st.title("🐟 Spyfish Aotearoa Data Tools")
    st.caption(
        "A collection of tools for rangers and scientists working with "
        "Spyfish Aotearoa data."
    )

    st.divider()
    st.subheader("Quick links")

    cols = st.columns(2, border=True)
    with cols[0]:
        if reporting_home_page is not None:
            st.page_link(reporting_home_page, label="Reporting", icon="📊")
            st.caption("Surveys, sites, media coverage and species, in one report.")
        else:
            st.caption("📊 Reporting is unavailable, see the error banner.")
    with cols[1]:
        if operations_home_page is not None:
            st.page_link(operations_home_page, label="Operations", icon="🔄")
            st.caption(
                "Pipeline state, footage coverage, annotation progress and "
                "data quality."
            )
        else:
            st.caption("🔄 Operations is unavailable, see the error banner.")

    cols = st.columns(2, border=True)
    with cols[0]:
        if metadata_page is not None:
            st.page_link(metadata_page, label="Metadata error review", icon="🔍")
            st.caption("Review validation errors and data quality issues.")
        else:
            st.caption("🔍 Metadata error review is unavailable, see the error banner.")
    with cols[1]:
        st.page_link(videos_page, label="View Deployment Videos", icon="📺")
        st.caption("View videos from the deployments.")

    st.divider()
    st.markdown(
        "For more info about Spyfish Aotearoa check here: "
        "https://spyfish.notion.site/overview  \n"
        "For any issues please write to Kalindi or add your issues here: "
        "https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/issues"
    )


home_page = st.Page(home, title="Home", icon="🐟", default=True)

# One page per (section, view). Built from the registries in `shell`, so adding
# a view there is all it takes for it to appear here.
#
# Guarded: importing `shell` pulls in plotly, the view modules and the spyfish
# config (which raises at import time if config.yaml is broken). Unguarded,
# any of those failures takes down the whole app, including the support page
# that exists to survive exactly this. On failure the report sections are
# dropped from the nav, an error banner says so, and everything else still
# runs.
try:
    from doc_report.shell import OPERATIONS_VIEWS, REPORTING_VIEWS  # noqa: E402

    _SHELL_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001, reporting must never kill the app
    OPERATIONS_VIEWS, REPORTING_VIEWS = {}, {}
    _SHELL_ERROR = exc

SECTION_VIEWS = {
    "Reporting": list(REPORTING_VIEWS),
    "Operations": list(OPERATIONS_VIEWS),
}


def _url_path(section: str, name: str) -> str:
    """URL for a view. The landing view of a section owns the section's URL.

    Without this the front pages would sit at /reporting-report-home, which is
    both long and worse than the obvious /reporting.
    """
    if name.endswith(" home"):
        return section.lower()
    return f"{section.lower()}-{name.lower().replace(' ', '-')}"


MISSING_PAGES: list = []


def _file_page(path: str, **kwargs):
    """`st.Page` for a file, or None if the file is gone.

    `st.Page` raises on a missing file, and it is constructed before
    `nav.run()` and `render_contact_note()` — so an unguarded one kills the app
    with a traceback and without the banner saying who to contact. Returning
    None keeps every other page working and names the missing file instead.
    """
    if not (Path(__file__).parent / path).exists():
        MISSING_PAGES.append(path)
        return None
    return st.Page(path, **kwargs)


section_pages: dict = {}
view_pages: dict = {}
for _section, _names in SECTION_VIEWS.items():
    _pages = []
    for _name in _names:
        _page = st.Page(
            _view(_section, _name),
            title=_name,
            icon=VIEW_ICONS.get(_name, "•"),
            url_path=_url_path(_section, _name),
        )
        _pages.append(_page)
        view_pages[(_section, _name)] = _page
    section_pages[_section] = _pages

# `.get`: in the fallback mode above these pages do not exist, and the home
# page's quick links skip them rather than crashing the landing page too.
reporting_home_page = view_pages.get(("Reporting", "Report home"))
operations_home_page = view_pages.get(("Operations", "Operations home"))
metadata_page = view_pages.get(("Operations", "Metadata error review"))

videos_page = _file_page(
    "pages/📺_View_Deployment_Videos.py", title="Deployment Videos", icon="📺"
)
# Unreleased dashboard concepts, gated on TEST_DASHBOARD_PASSWORD by the page
# itself. Named for what it is rather than for the dataset behind it, so it can
# hold the next draft too.
dashboard_test_page = _file_page(
    "pages/_advanced/🐚_Mussel_Insights.py",
    title="Dashboard test",
    icon="🧪",
    url_path="dashboard-test",
)
# Model Metrics sits in Operations, not Tools: how the model is performing is a
# question about the state of the pipeline, like every other view in that
# section. Spliced in below, after Species and before Metadata error review.
model_page = _file_page("pages/📊_Model_Metrics.py", title="Model Metrics", icon="📊")
if model_page is not None and section_pages.get("Operations"):
    _ops = section_pages["Operations"]
    _before = view_pages.get(("Operations", "Metadata error review"))
    _ops.insert(_ops.index(_before) if _before in _ops else len(_ops), model_page)
# Moved out of `pages/_advanced/` and into Tools with the rest of them. It had
# an "In development" group to itself, which was a header and a horizontal rule
# spent on one link.
substrate_page = _file_page(
    "pages/🪨_Substrate_Cover.py",
    title="Substrate Cover",
    icon="🪨",
    url_path="substrate-cover",
)
# Registered so the URL resolves, but left out of `st.navigation` groups so it
# does not appear in the sidebar. The contact note at the bottom of the sidebar
# is the way in.
support_page = _file_page(
    "pages/0_🆘_Support.py",
    title="Error page",
    icon="🆘",
    url_path="support",
)

# Streamlit's own grouped nav: section headers are always visible, so there is
# nothing to expand and every page is one click away. A custom two-level nav was
# tried and removed; the extra nesting cost a click per page and bought nothing.
#
# Home sits in an unnamed section so it renders without a header, above the
# groups. It is the app landing page, not part of the report.
# Pre-rebuild versions, restored from git and listed so the old visualisations
# can be compared against the new ones. Not maintained: they read the same data
# their own way, so where one disagrees with a Reporting view, the Reporting
# view is the one that has been checked.
#
# Only what still shows something the rebuild does not. The archived Sites,
# Experiments, Model Metrics, segmented Reporting and Programme Health pages
# were checked against their live versions and matched, so they were dropped.
# Git history holds them if they are ever wanted again. Programme Health at
# `app/pages/📈_Health_Dashboard.py`.
# The last one left, `_archive/ML_vs_Expert.py`, is no longer a nav entry: it is
# a view inside Dashboard test, next to the current version of the same
# comparison, which is where anyone would want to read the two against each
# other. That page runs the file directly, so nothing here has to register it.
ARCHIVE_PAGES: list = []
archive_pages: list = []

# Sections with no pages (the fallback mode) are left out entirely, an empty
# nav group renders as a bare header pointing at nothing.
_nav_sections: dict = {"": [home_page]}
for _section in ("Reporting", "Operations"):
    if section_pages.get(_section):
        _nav_sections[_section] = section_pages[_section]


# `_file_page` returns None for a file that is no longer there, so every group
# is filtered before it reaches `st.navigation`, which rejects a None entry.
def _present(pages: list) -> list:
    return [page for page in pages if page is not None]


nav = st.navigation(
    {
        **_nav_sections,
        # Everything that is not one of the two report sections: the
        # exploratory pages, the operational tools, the pre-rebuild pages kept
        # for comparison, and the support page. One group rather than four,
        # because four headers for eleven links is mostly headers.
        #
        # `support_page` is registered so /support resolves. Streamlit routes
        # only pages that are in the nav, but `render_contact_note` hides this
        # link with CSS. The red box at the bottom of the sidebar is the way in.
        "Tools": _present(
            [
                dashboard_test_page,
                substrate_page,
                videos_page,
                *archive_pages,
                support_page,
            ]
        ),
    },
    # Two sections of seven views each puts the nav over Streamlit's collapse
    # threshold, which hid 15 of the 24 links behind a "View 15 more" link,
    # including all of Operations. The grouped nav only works if the groups are
    # visible.
    expanded=True,
)

# Views link to each other, and `st.page_link` needs the page object rather than
# a URL string for function-defined pages. The entrypoint owns those objects, so
# it hands them over here. Keys are `(section, view name)`.
if _SHELL_ERROR is None:
    from doc_report import shell as _shell

    _shell.register_pages(view_pages)
else:
    st.error(
        "The reporting section failed to load and has been removed from the "
        "navigation, the rest of the app still works. "
        f"`{type(_SHELL_ERROR).__name__}: {_SHELL_ERROR}`"
    )

nav.run()
# Both go under the page list, in this order: the control first, the "who to
# call when it breaks" note last. Once here rather than per page, which is why
# the report views had no refresh button.
render_sidebar_refresh()
render_contact_note()
