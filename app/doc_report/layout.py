"""Section headings, the chip strip that jumps between them, and the slots in
the sticky header that both are drawn into.

The Experiments page uses chips to *switch* between independent demos, only
one is on screen at a time, which suits a page of unrelated experiments. A
report view is not that: its sections are a sequence meant to be read and
scanned together, and hiding four of five behind a chip would cost more than it
saves.

So these chips navigate rather than filter. Each one is an anchor link to a
heading further down the page; every section stays rendered, and the strip is a
table of contents you can also scroll past.

**Slots.** The chips sit on the title row of the sticky header, and a view's
own filters go to the sidebar under the shared ones. But the header and the
sidebar filter block are drawn by `shell.render` *before* the view runs, and
only the view knows its own section titles and its own filters. So the shell
creates two empty containers and registers them here; `chips()` and
`extra_filters()` render into those containers whenever the view gets round to
calling them. Streamlit places output by container, not by call order, so the
strip lands in the header even though it is asked for later.

Anchors are explicit, not Streamlit's auto-slug. The auto-slug is derived from
the heading text, so "Bad / excluded deployments per MPA" and any heading with
punctuation or a stray double space produce an anchor that is easy to get wrong
from the calling side, and a wrong anchor fails silently, as a link that
scrolls nowhere. Passing the same string to `chips()` and `section()` means one
place decides.
"""

import re
from contextlib import contextmanager

import streamlit as st
import streamlit.components.v1 as components

# Set once per run by `shell.render`. A view that renders chips or filters
# outside the report shell (or before the shell has built the header) falls back
# to drawing them inline, which is wrong-looking but not broken.
_SLOTS: dict = {"chips": None, "filters": None}


def set_slots(chips_container, filters_container) -> None:
    """Called by `shell.render` with the two containers inside the header."""
    _SLOTS["chips"] = chips_container
    _SLOTS["filters"] = filters_container


_CHIP_CSS = """
<style>
  /* Streamlit hands the strip's element container a height measured before the
     pills exist, so the box stays 11px tall while the chips inside it stand
     26px, and the overflow hangs through the header's bottom edge and over
     the first line of the page. Forced back to auto so the box is as tall as
     what it holds. */
  .st-key-section_chips,
  .st-key-section_chips [data-testid="stElementContainer"],
  .st-key-section_chips [data-testid="stMarkdown"] {
    height: auto !important;
    /* `height: auto` alone does not win. Streamlit's own layout keeps the box
       at the height it measured, and the pills overflow it. `min-height` is
       not part of that calculation, so this is what actually reserves the
       room: one pill's height, plus a row's worth for each extra line the
       strip wraps onto. */
    min-height: 1.9rem;
  }
  /* Flex, not a line of inline-blocks. An inline-block sits on the text
     baseline, so a pill taller than its line box hangs below it, the chips
     overflowed the header and the band's bottom edge cut through them. As flex
     items they are measured by their own height, and the row grows to fit.
     Right-justified: the chips share the title row and hang off its right
     edge, so a short strip does not leave a hole next to the title. */
  .st-key-section_chips [data-testid="stMarkdownContainer"] p {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: .2rem .3rem;
    margin: 0;
  }
  .st-key-section_chips a {
    padding: .12rem .6rem;
    border: 1px solid rgba(128,128,128,.35);
    border-radius: 1rem;
    font-size: .82rem;
    text-decoration: none;
    color: inherit;
  }
  .st-key-section_chips a:hover {
    border-color: rgba(128,128,128,.75);
    background: rgba(128,128,128,.12);
  }
  /* Where a chip's target comes to rest.
     Streamlit's own toolbar is 60px and drawn OVER the page; the sticky header
     sits under it and now carries the chips too, so one measured height covers
     both. Streamlit puts the anchor id on the heading element itself, so that
     is what the browser scrolls to and what has to carry the margin, the
     wrapper selector is kept for versions that put it elsewhere. */
  h1[id], h2[id], h3[id], h4[id],
  [data-testid="stHeadingWithActionElements"] {
    scroll-margin-top: calc(
      var(--header-height, 3.75rem)
      + var(--report-header-h, 7rem)
      + 1rem
    );
  }
  /* The measuring iframe below is zero-height but still a block, so it would
     take a full block gap out of the page for nothing. Out of flow, not
     hidden: a `display: none` iframe may never load, and the script inside it
     is what publishes the height everything above depends on. */
  [data-testid="stElementContainer"]:has(> [data-testid="stIFrame"]) {
    position: absolute;
    height: 0;
    visibility: hidden;
  }
</style>
"""

# Measures the sticky header, filters, chips and any view-specific filters,
# and republishes its height as a CSS custom property.
#
# `st.components.v1.html` renders into a same-origin iframe, so `window.parent`
# reaches the real document. This is the only way to get a rendered height into
# CSS: `calc()` cannot ask how tall another element is, and hardcoding a number
# is what put the headings behind the filter bar in the first place, the
# header is ~90px at desktop width and half as tall again once it wraps.
#
# The ResizeObserver is on `body`, so a window resize, a filter row that wraps,
# or a view with more chips than fit on one line all re-measure.
_MEASURE_JS = """
<script>
  const doc = window.parent.document;
  const root = doc.documentElement;
  const sync = () => {
    const header = doc.querySelector(
      'div[data-testid="stLayoutWrapper"]:has(> .st-key-report_header)');
    if (header) {
      root.style.setProperty('--report-header-h', header.offsetHeight + 'px');
    }
  };
  sync();
  new ResizeObserver(sync).observe(doc.body);
</script>
"""


def slug(title: str) -> str:
    """Anchor id for a section title. Same rule on both sides of the link."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def measure_header() -> None:
    """Publish the sticky header's height as a CSS variable, once per run.

    Called by `shell.render` for every view, not by `chips()`: a view with no
    chip strip still has a sticky header, and still has headings an anchor
    could land under. When this only ran alongside the chips, Report home never
    measured anything and fell back to the hardcoded guess.
    """
    components.html(_MEASURE_JS, height=0)


def chips(titles: list[str]) -> None:
    """Links to each section of the current view, drawn in the sticky header.

    Call once, anywhere in a view, with the section titles in the order they
    appear, the header slot puts them in the right place regardless. A title
    listed here that no `section()` call matches renders a dead link, which is
    the reason both sides share `slug()`.
    """
    titles = [t for t in titles if t]
    if len(titles) < 2:
        # One section is not a table of contents.
        return
    # Injected outside the strip, not in it. A stylesheet is an element like
    # any other to Streamlit: inside the keyed container it took a block gap of
    # its own, which pushed the links 16px below the container's top and left
    # them hanging out of its bottom, through the header's edge.
    st.markdown(_CHIP_CSS, unsafe_allow_html=True)

    slot = _SLOTS["chips"] or st.container(key="section_chips")
    with slot:
        st.markdown(
            " ".join(f"[{title}](#{slug(title)})" for title in titles),
            unsafe_allow_html=True,
        )


@contextmanager
def extra_filters(count: int = 3):
    """Yield `count` sidebar containers for a view's own filters.

    The shared filters live in the sidebar, so a view's extras go directly
    under them rather than in a band of their own on the page. The slot is the
    container `shell.render` created there; the fallback (a view rendered
    outside the shell) still lands in the sidebar rather than mid-page.

    Yields a list so existing callers written against header columns
    (`with filter_cols[0]:`) keep working unchanged — in the sidebar the
    "columns" simply stack.
    """
    slot = _SLOTS["filters"] or st.sidebar.container(key="view_filters")
    with slot:
        yield [st.container() for _ in range(count)]


def section(title: str, **kwargs) -> None:
    """`st.subheader` with the anchor `chips()` expects."""
    st.subheader(title, anchor=slug(title), **kwargs)
