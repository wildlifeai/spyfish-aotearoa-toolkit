"""Reporting - Species search: every observation of one species.

Uses the shared year / MPA / source filters at the top of the report rather
than carrying its own, filtering matches on MPA name, like every other view,
not on the DropID reserve code. The Source filter applies here too; pick
**All** to see every source's observations at once.

The per-species query goes to the annotations DB one species at a time rather
than loading everything upfront.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
from ecology_data import (  # noqa: E402
    SOURCE_PRIORITY,
    add_display_names,
    add_drop_id_columns,
    join_site_metadata,
    load_common_names,
    load_sites,
    search_species_annotations,
    source_bucket,
)

from . import data as report_data  # noqa: E402


def render(ctx: dict) -> None:
    st.caption(
        "Pick a species and get every (deployment, time) it was recorded at. "
        "The filters at the top apply, set **Source** to *All* to see what "
        "each source said about the same deployment."
    )

    ann = ctx["all_annotations"]
    if ann.empty:
        st.info("No annotations in the database yet.")
        return

    sites = load_sites()
    common_names = load_common_names()

    # The picker lists species present under the current filters, so choosing
    # one can never lead to an empty result for a reason that is off-screen.
    in_scope = ctx["annotations"]
    lookup = (
        in_scope[in_scope["scientific_name"].notna()][
            ["scientific_name", "display_name"]
        ]
        .drop_duplicates()
        .sort_values("display_name")
    )
    if lookup.empty:
        st.warning("No species match the current filters.")
        return

    display_to_sci = dict(zip(lookup["display_name"], lookup["scientific_name"]))
    species_label = st.selectbox(
        "Species (type to filter)",
        options=lookup["display_name"].tolist(),
        key="search_species",
    )
    if not species_label:
        return
    scientific_name = display_to_sci[species_label]

    obs = search_species_annotations(scientific_name)
    if obs.empty:
        st.warning(f"No annotations found for {species_label}.")
        return

    # Re-enriched through the same helpers as everything else, so the site join
    # and the display names are not a second implementation.
    obs = add_display_names(
        join_site_metadata(add_drop_id_columns(obs), sites), common_names
    )

    years = ctx.get("years")
    if years:
        lo, hi = years
        # Undated rows are kept, as they are in the shared context filter.
        obs = obs[obs["survey_year"].between(lo, hi) | obs["survey_year"].isna()]
    if ctx.get("reserves"):
        obs = obs[
            report_data.matches_reserves(obs["link_to_marine_reserve"], ctx["reserves"])
        ]
    # The shared rule, applied to raw observation rows rather than to a MaxN
    # summary: "Best available" keeps every row of the winning source per
    # deployment, a source can legitimately record several observations of one
    # species in one drop.
    obs = report_data._apply_source(obs, ctx["source"])

    if obs.empty:
        st.warning("No observations match the current filters.")
        return

    view = st.radio(
        "View",
        ["Peak per deployment", "All observations"],
        horizontal=True,
        key="search_view",
        help=(
            "Peak: one row per (deployment, source), the time-window with the "
            "highest count, matching the canonical MaxN. "
            "All: every individual observation/time-window the source recorded "
            "(can be many rows per deployment for citsci and expert)."
        ),
    )
    if view == "Peak per deployment":
        # Highest max_interval per (drop_id, annotated_by); ties go to the
        # earliest peak.
        obs = obs.sort_values(
            ["max_interval", "time_of_max_seconds"],
            ascending=[False, True],
            na_position="last",
        ).drop_duplicates(subset=["drop_id", "annotated_by"], keep="first")

    obs = obs.assign(
        _rank=obs["annotated_by"].map(SOURCE_PRIORITY).fillna(99)
    ).sort_values(
        ["_rank", "survey_date", "drop_id", "time_of_max_seconds"],
        ascending=[True, False, True, True],
        na_position="last",
    )

    # Bucketed, not raw `annotated_by`: ML rows carry the model name, so a
    # lookup on the literal key "ml" always showed 0 observations.
    bucket = source_bucket(obs["annotated_by"])
    counts = bucket.value_counts()
    drops_per_source = obs.groupby(bucket)["drop_id"].nunique()
    m_cols = st.columns(4)
    m_cols[0].metric("Total observations", len(obs))
    for slot, label in enumerate(("Expert", "CitSci", "ML"), start=1):
        m_cols[slot].metric(
            label,
            f"{int(counts.get(label, 0))} obs",
            f"{int(drops_per_source.get(label, 0))} deployments",
            delta_color="off",
        )

    display = obs.assign(Date=obs["survey_date"].dt.strftime("%Y-%m-%d"))
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
    if "external_id" in display.columns:
        cols_to_show.append("external_id")
    display = display[cols_to_show].rename(
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
        width="stretch",
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
