"""Species search — every observation timeline for one species.

Pulled out of the Experiments page into its own nav-visible page (Experiments
lives under ``_advanced/`` and is hidden). Shares the data layer with Experiments
via ``ecology_data`` so the loaders/enrichment have a single source of truth.

Deliberately self-contained beyond the data layer, so it's a clean base to grow
more search features onto.
"""

import streamlit as st
from ecology_data import (
    _SOURCE_PRIORITY,
    _enrich,
    load_common_names,
    load_maxn,
    load_sites,
    search_species_annotations,
)
from utils import render_contact_note, render_sidebar_refresh

st.set_page_config(page_title="Species Search", page_icon="🔎", layout="wide")
render_contact_note()
render_sidebar_refresh()

st.title("🔎 Species Search")
st.caption(
    "Pick a species — get every (deployment, time) where it was observed, "
    "grouped by source priority: expert > citsci > ml. "
    "Queries the annotations DB per species (not loaded upfront)."
)

# ── Load + enrich ───────────────────────────────────────────────────────────────

raw_maxn = load_maxn()
if raw_maxn.empty:
    st.info("No annotations in the database yet.")
    st.stop()

sites = load_sites()
common_names = load_common_names()
df_enriched = _enrich(raw_maxn, sites, common_names)

# ── Filters (year / reserve) ──────────────────────────────────────────────────────

fc1, fc2 = st.columns([2, 3])

years_series = df_enriched["survey_year"].dropna()
if not years_series.empty:
    y_min, y_max = int(years_series.min()), int(years_series.max())
    if y_min < y_max:
        year_range = fc1.slider(
            "Year range", y_min, y_max, (y_min, y_max), key="search_years"
        )
    else:
        fc1.markdown(f"**Year**\n\n{y_min}")
        year_range = (y_min, y_max)
else:
    year_range = None

all_reserves = sorted([r for r in df_enriched["reserve_code"].dropna().unique() if r])
selected_reserves = fc2.multiselect(
    "Reserves (empty = all)", options=all_reserves, key="search_reserves"
)

# All sources retained — the point of search is to see what each source recorded.
df_multi = df_enriched.copy()
if year_range is not None:
    df_multi = df_multi[
        df_multi["survey_year"].between(*year_range) | df_multi["survey_year"].isna()
    ]
if selected_reserves:
    df_multi = df_multi[df_multi["reserve_code"].isin(selected_reserves)]

st.info(
    "Uses **all sources** so you can see what each source recorded. "
    "The year/reserve filters above apply."
)

# ── Species picker ────────────────────────────────────────────────────────────────

# Drawn from species present in the current filtered view; UX uses display_name
# but the query keys on scientific_name.
spp_lookup = (
    df_multi[df_multi["scientific_name"].notna()][["scientific_name", "display_name"]]
    .drop_duplicates()
    .sort_values("display_name")
)
if spp_lookup.empty:
    st.warning("No species found in the current filter.")
    st.stop()

display_to_sci = dict(zip(spp_lookup["display_name"], spp_lookup["scientific_name"]))
species_label = st.selectbox(
    "Species (type to filter)",
    options=spp_lookup["display_name"].tolist(),
    key="search_species",
)
if not species_label:
    st.stop()
scientific_name = display_to_sci[species_label]

# ── Per-species query ───────────────────────────────────────────────────────────

obs = search_species_annotations(scientific_name)
if obs.empty:
    st.warning(f"No annotations found for {species_label}.")
    st.stop()

# Re-enrich keeps the join logic in one place rather than duplicating it.
obs = _enrich(obs, sites, common_names)

# Apply the same year/reserve filters that gate the picker.
if year_range:
    lo, hi = year_range
    obs = obs[obs["survey_year"].between(lo, hi) | obs["survey_year"].isna()]
if selected_reserves:
    obs = obs[obs["reserve_code"].isin(selected_reserves)]

if obs.empty:
    st.warning("No observations match the current year/reserve filter.")
    st.stop()

view = st.radio(
    "View",
    ["Peak per deployment", "All observations"],
    horizontal=True,
    key="search_view",
    help=(
        "Peak: one row per (deployment, source) — the time-window with the "
        "highest count, matching the canonical MaxN. "
        "All: every individual observation/time-window the source recorded "
        "(can be many rows per deployment for citsci and expert)."
    ),
)
if view == "Peak per deployment":
    # Keep the row with the highest max_interval per (drop_id, annotated_by).
    # Ties broken by smallest time_of_max_seconds (earliest peak).
    obs = obs.sort_values(
        ["max_interval", "time_of_max_seconds"],
        ascending=[False, True],
        na_position="last",
    ).drop_duplicates(subset=["drop_id", "annotated_by"], keep="first")

# Sort by source priority, then most recent first.
obs["_rank"] = obs["annotated_by"].map(_SOURCE_PRIORITY).fillna(99)
obs = obs.sort_values(
    ["_rank", "survey_date", "drop_id", "time_of_max_seconds"],
    ascending=[True, False, True, True],
    na_position="last",
)

# ── Summary metrics per source ────────────────────────────────────────────────────

counts = obs["annotated_by"].value_counts()
drops_per_source = obs.groupby("annotated_by")["drop_id"].nunique()
m_cols = st.columns(4)
m_cols[0].metric("Total observations", len(obs))
m_cols[1].metric(
    "Expert",
    f"{int(counts.get('expert', 0))} obs",
    f"{int(drops_per_source.get('expert', 0))} deployments",
    delta_color="off",
)
m_cols[2].metric(
    "CitSci",
    f"{int(counts.get('citsci', 0))} obs",
    f"{int(drops_per_source.get('citsci', 0))} deployments",
    delta_color="off",
)
m_cols[3].metric(
    "ML",
    f"{int(counts.get('ml', 0))} obs",
    f"{int(drops_per_source.get('ml', 0))} deployments",
    delta_color="off",
)

# ── Observation table ─────────────────────────────────────────────────────────────

obs_display = obs.copy()
obs_display["Date"] = obs_display["survey_date"].dt.strftime("%Y-%m-%d")

cols_to_show = [
    "annotated_by",
    "Date",
    "reserve_code",
    "site_id",
    "drop_id",
    "time_of_max",
    "max_interval",
    "confidence_agreement",
]
if "external_id" in obs_display.columns:
    cols_to_show.append("external_id")

display = obs_display[cols_to_show].rename(
    columns={
        "annotated_by": "Source",
        "reserve_code": "Reserve",
        "site_id": "Site",
        "drop_id": "Drop ID",
        "time_of_max": "Video time",
        "max_interval": "Count",
        "confidence_agreement": "Confidence",
        "external_id": "External ID",
    }
)

st.dataframe(
    display,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Source": st.column_config.TextColumn("Source", width="small"),
        "Date": st.column_config.TextColumn("Survey date", width="small"),
        "Reserve": st.column_config.TextColumn("Reserve", width="small"),
        "Site": st.column_config.TextColumn("Site", width="small"),
        "Drop ID": st.column_config.TextColumn("Drop ID"),
        "Video time": st.column_config.TextColumn(
            "Video time",
            width="small",
            help="HH:MM:SS within the deployment",
        ),
        "Count": st.column_config.NumberColumn("Count", width="small"),
        "Confidence": st.column_config.NumberColumn(
            "Confidence",
            format="%.2f",
            width="small",
        ),
        "External ID": st.column_config.TextColumn(
            "External ID",
            width="small",
            help="Model name (ml) or BIIGLE annotation ID (expert)",
        ),
    },
)

st.download_button(
    f"⬇ Download {species_label} observations (CSV)",
    data=display.to_csv(index=False).encode("utf-8"),
    file_name=f"{scientific_name.replace(' ', '_')}_observations.csv",
    mime="text/csv",
)
