"""Site-level data loading and filtering, shared by the views that need it.

Both the Sites view and the MPA view read the same tables and have to agree on
what a filtered row is. When these helpers lived in `sites.py`, `mpa.py` had to
import from a sibling *view* to get at them, which put a rendering module on the
import path of another rendering module for no reason.

They live here instead: views import from this module, and no view imports
another. `data.py` is the equivalent for the deployment and annotation frames
that every view reads; this one is the site-level layer on top of it.

Nothing here renders. If a function calls `st` for anything other than caching,
it belongs in a view.
"""

import math
import sys
from pathlib import Path

# The shared `ecology_data` module lives in app/, which is not on sys.path when
# Streamlit runs a page from a subfolder. parents[1] is app/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from ecology_data import (  # noqa: E402
    SOURCE_PRIORITY,
    add_display_names,
    add_drop_id_columns,
    join_site_metadata,
    load_common_names,
    load_effort,
    load_maxn,
    load_sites,
)
from utils import CACHE_TTL_SECONDS, check_password  # noqa: E402

from . import data as report_data  # noqa: E402

UNKNOWN = "unknown"

# ── Coordinates gate ──────────────────────────────────────────────────────────
#
# Site coordinates joined to abundance are effectively a ranked fishing map,
# and for rare species a poaching aid, see the sensitive-data rule in
# CLAUDE.md. Every view that wants coordinates goes through these two
# functions, so a deploy cannot forget a gate that lives in one caller.

_MAP_SECRET = (
    "MAP_PASSWORD"  # the secret's *name*, not a value  # pragma: allowlist secret
)


def coords_allowed() -> bool:
    """Whether this session has unlocked site coordinates.

    Pure query, renders nothing, so data loaders can call it. The flag is set
    by `render_map_gate()` below; the session key is the one
    `utils.check_password` writes, so the two cannot disagree.
    """
    return bool(st.session_state.get(f"_pw_ok_{_MAP_SECRET}", False))


def render_map_gate() -> bool:
    """Sidebar control to unlock the site map. Returns the same as coords_allowed.

    MAP_PASSWORD, not APP_PASSWORD, map access is granted separately, so
    unlocking one never unlocks the other. The control lives in the sidebar so
    a view can render it before its data loads, wherever its map sits on the
    page.
    """
    with st.sidebar:
        st.header("Site map")
        st.caption(
            "Coordinates are restricted, site positions joined to abundance "
            "read as a fishing map. Everything else works without unlocking."
        )
        if st.toggle(
            "Show map (needs password)", value=coords_allowed(), key="_map_gate_toggle"
        ):
            return check_password(_MAP_SECRET, label="Map password")
    return False


# ── Data ──────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_site_view(include_coords: bool) -> pd.DataFrame:
    """One row per (drop, species, source), enriched with site metadata.

    Cached on `include_coords` so the unauthenticated view never even holds
    coordinates in its cache entry.
    """
    return add_display_names(
        join_site_metadata(
            add_drop_id_columns(load_maxn()), load_sites(include_coords)
        ),
        load_common_names(),
    )


def best_per_drop_species(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each deployment's best source: expert beats citsci beats ml.

    Without this, a deployment reviewed by both an expert and volunteers is
    counted twice in every site total.

    Per DEPLOYMENT, not per (drop, species), the same rule as
    `data._apply_source`, so the Sites/MPA views and the rest of the report
    cannot disagree. The winning source supplies all of the drop's rows; a
    species only a lesser source recorded on that drop is treated as not
    confirmed, not filled in from the lesser source. Input is `load_maxn()`
    (one row per drop × species × source), so the output stays one row per
    (drop, species).
    """
    ranked = df.assign(_rank=df["annotated_by"].map(SOURCE_PRIORITY).fillna(99))
    best = ranked.groupby("drop_id")["_rank"].transform("min")
    return ranked[ranked["_rank"] == best].drop(columns="_rank")


def split_reserves(series: pd.Series) -> list:
    """`data.split_reserves`, sorted, for use as a filter's option list.

    The splitting itself lives in the data layer, where every other copy of it
    now goes. This wrapper exists only for the ordering: a filter's options have
    to be in a stable order, and a set is not.
    """
    return sorted(report_data.split_reserves(series))


def apply_filters(frame: pd.DataFrame, *, years, regions, reserves, protections):
    """Apply the page filters to any enriched frame, sightings or effort alike.

    Effort must be filtered by the same predicates as the sightings, and NOT by
    "which sites happen to appear in the sightings". Deriving effort from the
    sightings would exclude any analysed site that saw nothing, which is precisely
    the zeros the abundance means depend on.
    """
    out = frame
    if years:
        out = out[out["survey_year"].between(*years) | out["survey_year"].isna()]
    if regions:
        out = out[out["region"].isin(regions)]
    if reserves:
        by_reserve = split_reserve_rows(out)
        out = out.loc[
            by_reserve.loc[by_reserve["reserve"].isin(reserves)].index.unique()
        ]
    if protections:
        out = out[out["protection_status"].isin(protections)]
    return out


def _fit_zoom(lat_span: float, lon_span: float) -> float:
    """A zoom level that fits a bounding box of this size.

    Web maps double their scale per zoom level, so the level that fits a span is
    logarithmic in it. 360 degrees of longitude is the whole world at zoom 0;
    the latitude span is compared against 170 because the projection is taller
    than it is wide. The wider of the two constraints wins, and a small pad
    stops points sitting on the edge.

    Clamped at 12 so a single site does not zoom to street level, and at 3.5 so
    a nationwide spread stays recognisable as New Zealand.
    """
    lat_span = max(lat_span, 0.02)
    lon_span = max(lon_span, 0.02)
    zoom = min(math.log2(360 / lon_span), math.log2(170 / lat_span))
    return float(min(max(zoom - 0.7, 3.5), 12))


def split_reserve_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (original row × marine reserve), keeping the original index.

    A site listed against two reserves contributes to both. That means reserve
    totals deliberately do NOT sum to the overall total, a deployment between
    two reserves is counted once under each, because "how many deployments does
    this reserve have" is the question being asked.
    """
    out = df.copy()
    out["reserve"] = out["link_to_marine_reserve"].fillna("").str.split(",")
    out = out.explode("reserve")
    out["reserve"] = out["reserve"].str.strip()
    return out[out["reserve"] != ""]


def filtered_site_frames(ctx: dict | None = None) -> dict:
    """Frames filtered by the report-wide sidebar filters.

    Species is deliberately left out: the panels these frames feed carry their
    own species picker, and `df_context` is the species-UNfiltered denominator
    ("deployments where nothing was seen" is measured against it).
    """
    # Locked by default; unlocked per session via `render_map_gate` (the
    # sidebar control on the MPA view). Never hardcode True here, coordinates
    # joined to abundance are a ranked fishing map.
    show_coords = coords_allowed()
    df_all = best_per_drop_species(load_site_view(show_coords))
    effort_all = join_site_metadata(
        add_drop_id_columns(load_effort()), load_sites(show_coords)
    )
    filters = dict(
        years=(ctx or {}).get("years"),
        regions=(ctx or {}).get("regions") or [],
        reserves=(ctx or {}).get("reserves") or [],
        protections=(ctx or {}).get("protections") or [],
    )
    df_context = apply_filters(df_all, **filters)
    return {
        "df_context": df_context,
        "effort_view": apply_filters(effort_all, **filters),
        "show_coords": show_coords,
        # Not applied to the frames (see the docstring), but passed along so
        # panels with their own species picker can honour the shared filter.
        "species": (ctx or {}).get("species") or [],
        **filters,
    }
