"""Chart layer for the Species view.

Everything the Species view draws, and nothing about how the page is put
together. `species.py` decides what the reader is asked and in what order;
these functions take a frame and render one chart or one panel.

The split is here rather than left in the view because the Experiments page has
several more species charts still to fold in. Each one arriving as another
inline block would have pushed the view past a thousand lines and buried the
sequence of the page in the drawing code.

The peak-across-intervals step is NOT repeated here: frames arrive already
aggregated by `data.species_maxn`, so a chart cannot take its own MaxN and
disagree with the number the view printed above it.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ecology_data import OTHER_PROTECTION, PROTECTED, UNPROTECTED, protection_group
from theme import protection_color_map, species_color_map

from spyfish.config.wrapper import config

from ..charting import (
    group_colors,
    group_dashes,
    protection_dashes,
    source_coverage_note,
    style,
    top_n_slider,
    year_axis,
)
from ..data import effort_per, real_species
from ..layout import section

# Protected / Unprotected come from `ecology_data.protection_group`, which is
# driven by `config.protected_statuses`, the same definition the MPA view's
# trend uses, so the two views cannot classify one deployment on opposite
# sides of the comparison. Anything unrecognised is left out rather than
# guessed at, because guessing wrong here invents the finding. Colour and dash
# come from the shared convention in `charting`, so this view and the MPA
# trends cannot disagree about which side is which.
GROUP_COLOURS = group_colors()
# Shape as well as colour, so protection is readable without relying on hue:
# circle inside an MPA, square outside. Matches the reserve-effect chart on
# the Experiments page.
GROUP_SYMBOLS = {PROTECTED: "circle", UNPROTECTED: "square"}
GROUP_DASHES = group_dashes()


def render_protection_boxes(comparable: pd.DataFrame) -> None:
    """The two box plots: total MaxN and richness, protected against not.

    One dot per deployment on purpose. A bar of two means hides that the two
    distributions overlap almost entirely, which is the thing a reader has to
    see before treating the gap as a finding.
    """
    left, right = st.columns(2)
    for column, metric, label in (
        (left, "abundance", "Total MaxN per deployment"),
        (right, "richness", "Species per deployment"),
    ):
        with column:
            st.markdown(f"**{label}**")
            fig = px.box(
                comparable,
                x="Group",
                y=metric,
                color="Group",
                points="all",
                hover_name="drop_id",
                category_orders={"Group": [PROTECTED, UNPROTECTED]},
                color_discrete_map=GROUP_COLOURS,
            )
            for trace in fig.data:
                trace.marker.symbol = GROUP_SYMBOLS.get(trace.name, "circle")
            style(fig, height=330, legend=False)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, key=f"species_box_{metric}")


def render_species_by_site(
    picked: pd.DataFrame,
    all_species: pd.DataFrame,
    all_species_mode: bool,
) -> None:
    """Where the chosen species are seen, and how much.

    Colour is frequency, not abundance: the share of a site's annotated
    deployments that recorded the species. Frequency is comparable between
    species, whereas MaxN is not, a site full of sweep and a site with one
    snapper are not read off the same colour scale sensibly. Mean MaxN is in the
    hover for when the magnitude is the question.
    """
    st.markdown("**Where these species are seen**")
    if all_species_mode:
        # "All species combined" made a degenerate heatmap: one column whose
        # every cell tends toward 100%, since almost every annotated deployment
        # records SOMETHING. Break out the most frequently seen species
        # instead, which is the question the chart is for.
        top = picked.groupby("scientific_name")["drop_id"].nunique().nlargest(12).index
        picked = picked[picked["scientific_name"].isin(top)].copy()
        picked["Species"] = (
            picked["display_name"]
            if "display_name" in picked.columns
            else picked["scientific_name"]
        )
        st.caption(
            "No species selected above, so this shows the **12 most frequently "
            "seen species**. Pick species above to choose your own."
        )

    # Denominator is every annotated deployment at the site, not just those
    # recording the species, so a blank cell means "looked and did not see it".
    deps_per_site = effort_per(all_species, "site_id", "deployments")
    grid = (
        picked.groupby(["site_id", "Species"])
        .agg(seen=("drop_id", "nunique"), mean_maxn=("maxn", "mean"))
        .reset_index()
        .merge(deps_per_site, on="site_id", how="left")
    )
    if grid.empty:
        st.info("No sites recorded the chosen species.")
        return
    grid["Frequency"] = (grid["seen"] / grid["deployments"]).round(3)
    grid["mean_maxn"] = grid["mean_maxn"].round(2)

    # Which sites appear is chosen by how much they show, since a heatmap with
    # 200 rows is unreadable and the tail is all sites with one deployment.
    # How they are ORDERED is alphabetical, so a site can be found by name
    # rather than hunted for by rank.
    top_sites = (
        grid.groupby("site_id")["Frequency"]
        .max()
        .sort_values(ascending=False)
        .head(25)
        .index
    )
    grid = grid[grid["site_id"].isin(top_sites)]
    # Plotly puts the first category at the bottom, so the list is reversed to
    # read A to Z downwards.
    site_order = sorted(grid["site_id"].unique(), reverse=True)
    species_order = sorted(grid["Species"].unique())

    # The deployment count rides on the site label — "SITE (n)" — rather than
    # in a third aligned panel: neither a bar chart nor a one-column heatmap
    # keeps its rows level with these heatmaps (each chart's bottom margin is
    # sized by its own x labels), and a count inside the label cannot drift.
    deps_by_site = grid.drop_duplicates("site_id").set_index("site_id")["deployments"]
    # Compact on the axis — "SLI_012 (2)" — with the words in the hover: the
    # hovertemplates below spell out what the bracketed number means.
    label_of = {s: f"{s} ({int(n)})" for s, n in deps_by_site.items()}
    grid = grid.assign(Site=grid["site_id"].map(label_of))
    site_order = [label_of[s] for s in site_order]

    height = max(320, 22 * grid["site_id"].nunique())
    left, right = st.columns(2)

    with left:
        st.markdown("**How often it is seen**")
        fig = px.density_heatmap(
            grid,
            x="Species",
            y="Site",
            z="Frequency",
            histfunc="max",
            color_continuous_scale=["#EAF2FB", "#0B3D6B"],
            range_color=[0, 1],
            category_orders={"Site": site_order, "Species": species_order},
        )
        style(
            fig,
            height=height,
            coloraxis_colorbar=dict(title="Seen in", tickformat=".0%"),
        )
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        fig.update_traces(
            hovertemplate="Site (annotated deployments): %{y}<br>%{x}<br>"
            "Seen in %{z:.0%} of them<extra></extra>"
        )
        st.plotly_chart(fig, key="species_by_site_freq")
        st.caption(
            "Share of that site's annotated deployments recording the species. "
            "Comparable between species, because every cell is a proportion of "
            "the same thing."
        )

    with right:
        st.markdown("**How many, when it is seen**")
        fig = px.density_heatmap(
            grid,
            x="Species",
            y="Site",
            z="mean_maxn",
            histfunc="max",
            color_continuous_scale=["#EAF3EC", "#1B7F4B"],
            category_orders={"Site": site_order, "Species": species_order},
        )
        style(
            fig,
            height=height,
            coloraxis_colorbar=dict(title="Mean MaxN"),
        )
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        fig.update_traces(
            hovertemplate="Site (annotated deployments): %{y}<br>%{x}<br>"
            "Mean MaxN when seen: %{z:.2f}<extra></extra>"
        )
        st.plotly_chart(fig, key="species_by_site_maxn")
        st.caption(
            "Mean MaxN across the deployments that recorded it. **One colour "
            "scale spans every selected species**, so a naturally abundant "
            "schooling species washes out a rarer one: read this down a column, "
            "not across a row."
        )

    st.caption(
        "A blank cell is a site that was annotated and did not record the "
        "species, which is a real absence rather than a gap. The two heatmaps "
        "answer different questions, and they disagree usefully: a species can "
        "be seen almost everywhere in ones and twos, or rarely but in numbers. "
        "**The number after each site name is its annotated deployments** — "
        "the denominator behind that row. A share resting on two deployments "
        "is far less certain than one resting on forty."
    )

    with st.expander("The numbers behind this"):
        # `Site` (the labelled axis value) is dropped in favour of the plain
        # site_id: the table has a real column for the deployment count, so
        # the label's annotation would repeat it.
        table = grid.drop(columns="Site").rename(
            columns={
                "site_id": "Site",
                "seen": "Deployments seen in",
                "deployments": "Deployments annotated",
                "mean_maxn": "Mean MaxN",
            }
        )
        st.dataframe(
            table[
                [
                    "Site",
                    "Species",
                    "Deployments seen in",
                    "Deployments annotated",
                    "Mean MaxN",
                    "Frequency",
                ]
            ].sort_values(["Site", "Species"]),
            hide_index=True,
            width="stretch",
            height=320,
            column_config={
                "Frequency": st.column_config.ProgressColumn(
                    "Frequency", min_value=0, max_value=100, format="%.0f%%"
                ),
                "Mean MaxN": st.column_config.NumberColumn(
                    "Mean MaxN",
                    help="Mean across deployments that recorded the species, "
                    "so it answers 'when present, how many'.",
                ),
            },
        )


# Which names are not species comes from config (`reporting.non_species_classes`),
# the same list the MPA view's diversity panels exclude, this module used to
# carry its own copy and the two had already drifted.

# Opened on by default. Three is enough to read at a glance, and these are the
# ones a New Zealand reader recognises without a key. The full indicator list
# lives in config (`reporting.indicator_species`).
DEFAULT_SPECIES = config.indicator_species[:3]

# Reader-facing blurbs for the indicator species. UI text, so it lives in code;
# which species COUNT as indicators is config's call, and a species missing
# from this dict simply shows without a blurb.
INDICATORS = {
    "Parapercis colias": "Blue cod, the South Island MPA indicator",
    "Pagrus auratus": "Snapper, the northern indicator, heavily fished",
    "Jasus edwardsii": "Rock lobster / kōura, slow to recover, classic "
    "MPA-response species",
    "Nemadactylus macropterus": "Tarakihi, commercially fished",
    "Chirodactylus spectabilis": "Red moki, reef resident, low mobility",
}


def render_species_over_time(
    per_species: pd.DataFrame, common: dict, with_sites: bool = True
) -> tuple:
    """Per-species trends, opening on the indicator species.

    Returns `(picked, all_species_mode)` so a caller can render
    `render_species_by_site` separately when `with_sites=False` — the front
    page does, to put the site map between the two.
    """
    section("Species over time")
    st.caption(
        "Abundance for chosen species by survey year, inside an MPA against "
        "outside. Indicator species are selected to begin with: their abundance "
        "is read as a signal of condition rather than as one more count."
    )

    real = real_species(per_species)
    real = real[real["survey_year"].notna()]
    if real.empty:
        st.info("No species annotations with a survey year.")
        return None, False

    available = sorted(real["scientific_name"].unique())
    chosen = st.multiselect(
        "Species",
        available,
        default=[s for s in DEFAULT_SPECIES if s in available],
        format_func=lambda s: common.get(s, s),
        help="`fish` is not offered: it is the binary detector's only class, "
        "not a species. Clear the selection to see every species combined.",
    )

    # With nothing selected, fall back to every species combined rather than an
    # empty chart. It is the same question asked of the whole community.
    all_species_mode = not chosen
    if all_species_mode:
        chosen = available
        st.caption(
            f"No species selected, so this is **all {len(available)} species "
            "combined**. Pick one or more to break it out."
        )
    else:
        notes = [
            f"**{common.get(s, s)}**, {INDICATORS[s]}"
            for s in chosen
            if s in INDICATORS
        ]
        if notes:
            st.caption("  \n".join(notes))

    picked = real[real["scientific_name"].isin(chosen)].copy()
    # `display_name` when the frame carries it, so this chart cannot label a
    # species differently from one drawn off the same frame elsewhere.
    picked["Species"] = (
        "All species"
        if all_species_mode
        else (
            picked["display_name"]
            if "display_name" in picked.columns
            else picked["scientific_name"].map(lambda s: common.get(s, s))
        )
    )
    picked = picked[picked["Group"].notna()]
    if picked.empty:
        st.info("No classified deployments recorded the chosen species.")
        return

    # The denominator is every classified deployment that year, not just those
    # recording the species. Dividing by the latter reports the mean among
    # deployments that already had it, which cannot fall and always looks like a
    # healthy population.
    classified = per_species[
        per_species["Group"].notna() & per_species["survey_year"].notna()
    ]
    deps_per_year = effort_per(
        classified.assign(Year=classified["survey_year"].astype("Int64")),
        ["Year", "Group"],
        "deployments",
    )

    totals = (
        picked.groupby([picked["survey_year"].astype("Int64"), "Species", "Group"])
        .agg(total=("maxn", "sum"), seen=("drop_id", "nunique"))
        .reset_index()
        .rename(columns={"survey_year": "Year"})
        .merge(deps_per_year, on=["Year", "Group"], how="left")
    )
    if totals.empty:
        st.info("No annotations for the chosen species in a dated survey.")
        return
    totals["Mean MaxN"] = (totals["total"] / totals["deployments"]).round(2)
    totals["Frequency"] = (totals["seen"] / totals["deployments"]).round(3)

    # Colour carries the species, dash and marker carry protection, so a species
    # stays the same colour on both sides and the pair reads as a pair.
    colour_map = species_color_map(
        [common.get(s, s) for s in available] + ["All species"]
    )

    left, right = st.columns(2)
    for column, metric, title, pct in (
        (left, "Mean MaxN", "Mean MaxN per deployment", False),
        (right, "Frequency", "Share of deployments recording it", True),
    ):
        with column:
            st.markdown(f"**{title}**")
            fig = px.line(
                totals,
                x="Year",
                y=metric,
                color="Species",
                line_dash="Group",
                markers=True,
                hover_data=["seen", "deployments"],
                symbol="Group",
                symbol_map=GROUP_SYMBOLS,
                category_orders={"Group": [PROTECTED, UNPROTECTED]},
                line_dash_map=GROUP_DASHES,
                # Built from every AVAILABLE species, not the selected ones, so
                # ticking a species off does not repaint the rest. Plotly ignores
                # map entries for series that are absent. "Same colour inside and
                # outside" only means something if the colour is stable to begin
                # with.
                color_discrete_map=colour_map,
            )
            style(
                fig,
                height=360,
                legend=dict(orientation="h", y=1.14, x=0, title_text=""),
                **({"yaxis_tickformat": ".0%"} if pct else {}),
            )
            year_axis(fig)
            st.plotly_chart(fig, key=f"species_time_{metric.replace(' ', '_')}")

    st.caption(
        "**Inside an MPA: solid line, circles. Outside: dotted, squares.** "
        "Frequency is the steadier of the two measures: mean MaxN moves with "
        "one busy deployment, whereas a species is either present or not."
    )

    if with_sites:
        st.divider()
        render_species_by_site(picked, per_species, all_species_mode)
    return picked, all_species_mode


def render_reserve_effect(per_species: pd.DataFrame, common: dict) -> None:
    """Paired species comparison: mean MaxN outside against inside an MPA.

    Ported from the Experiments page. One line per species connecting its mean
    outside an MPA to its mean inside, so the direction of the line is the
    effect. Up-sloping means more abundant inside.

    A species must clear a minimum number of deployments on BOTH sides. With one
    or two, a single busy drop decides the direction and the chart shows a
    strong effect that is really one lucky video.
    """
    section("MPA effect by species")
    st.caption(
        "Each line is one species: **square** = mean outside an MPA, "
        "**circle** = mean inside. An up-sloping line to the right means more "
        "abundant inside. Green is higher inside, red is higher outside, and "
        "stronger colour means a bigger difference."
    )

    real = real_species(per_species)
    real = real[real["Group"].notna()]
    if real.empty:
        st.info("No classified deployments to compare.")
        return

    min_deps = st.slider(
        "Minimum deployments per side",
        1,
        20,
        3,
        key="species_effect_min",
        help="A species needs at least this many deployments both inside and "
        "outside an MPA to appear. Below about three, one busy deployment "
        "decides the direction of the line.",
    )

    counts = (
        real.groupby(["scientific_name", "Group"])["drop_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    if not {PROTECTED, UNPROTECTED} <= set(counts.columns):
        st.warning("The selection needs deployments both inside and outside an MPA.")
        return
    keep = counts[
        (counts[PROTECTED] >= min_deps) & (counts[UNPROTECTED] >= min_deps)
    ].index
    if keep.empty:
        st.warning(
            f"No species have at least {min_deps} deployments on both sides. "
            "Lower the threshold or widen the filters."
        )
        return

    # Mean over deployments that RECORDED the species: "when present, how many".
    # How often it is present is the frequency table above.
    means = (
        real[real["scientific_name"].isin(keep)]
        .groupby(["scientific_name", "Group"])["maxn"]
        .mean()
        .unstack()
        .round(2)
    )
    means["effect"] = means[PROTECTED] - means[UNPROTECTED]
    means = means.sort_values("effect")

    fig = go.Figure()
    max_abs = max(abs(means["effect"].min()), abs(means["effect"].max())) or 1
    for name, row in means.iterrows():
        effect = row["effect"]
        strength = 0.35 + 0.55 * min(1.0, abs(effect) / max_abs)
        colour = (
            f"rgba(46, 125, 50, {strength})"
            if effect >= 0
            else f"rgba(198, 40, 40, {strength})"
        )
        label = common.get(name, name)
        fig.add_trace(
            go.Scatter(
                x=[row[UNPROTECTED], row[PROTECTED]],
                y=[label, label],
                mode="lines+markers",
                line={"color": colour, "width": 3},
                marker={
                    "size": [10, 13],
                    "color": colour,
                    "symbol": ["square", "circle"],
                },
                hovertemplate=(
                    f"<b>{label}</b><br>Outside: {row[UNPROTECTED]:.2f}"
                    f"<br>Inside: {row[PROTECTED]:.2f}"
                    f"<br>Effect: {effect:+.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    style(fig, height=max(320, len(means) * 24 + 120), hovermode="closest")
    fig.update_xaxes(
        title="Mean MaxN when present", zeroline=True, zerolinecolor="#CCCCCC"
    )
    fig.update_yaxes(title=None)
    st.plotly_chart(fig, key="species_reserve_effect")

    higher = int((means["effect"] > 0).sum())
    st.caption(
        f"**{higher} of {len(means)} species** are more abundant inside an MPA "
        "in this selection. Deployments are not evenly spread across sites, "
        "years or effort, so this is a lead to follow rather than a measured "
        "effect."
    )


# ── Ported from the Experiments page ──────────────────────────────────────
#
# Copied, not moved: the Experiments page still holds its own copy, so the
# two can be read against each other before either is retired. Widget keys
# carry a `rep_` prefix here, since Streamlit scopes keys per page and the
# originals keep the unprefixed names.


def render_detection_rate(df: pd.DataFrame) -> None:
    section("Detection rate")
    st.caption(
        "For each species: % of deployments where it was detected. "
        "Highlights survey coverage of common vs rare species. "
        "**Inside vs outside view** splits the same rate by reserve protection, "
        "a presence-axis complement to the Reserve effect slope (which is on abundance)."
    )

    # Null-species rows are absence records ("reviewed, nothing seen"): they
    # belong in the deployment denominator, but carry no species to rate — so
    # count deployments BEFORE dropping them.
    total_drops = df["drop_id"].nunique()
    if total_drops == 0:
        st.warning("No deployments after filters.")
        return
    df_full = df
    df = df[df["scientific_name"].notna()]

    view = st.radio(
        "View",
        ["Programme-wide", "Inside vs outside reserve"],
        horizontal=True,
        key="rep_det_view",
    )

    if view == "Programme-wide":
        rates = (
            df.groupby("display_name")["drop_id"]
            .nunique()
            .reset_index(name="n_deps_seen")
        )
        rates["pct_seen"] = (rates["n_deps_seen"] / total_drops * 100).round(2)
        rates["mean_maxn"] = (
            df.groupby("display_name")["maxn"]
            .mean()
            .round(2)
            .reindex(rates["display_name"])
            .values
        )

        top_n = top_n_slider("Show top N species", len(rates), 40, "rep_det_topn")
        rates = rates.nlargest(top_n, "pct_seen").sort_values(
            "pct_seen", ascending=True
        )

        def _rarity(pct):
            if pct >= 20:
                return "common (≥20%)"
            if pct >= 5:
                return "intermediate (5–20%)"
            return "rare (<5%)"

        rates["rarity"] = rates["pct_seen"].apply(_rarity)

        fig = px.bar(
            rates,
            x="pct_seen",
            y="display_name",
            color="rarity",
            color_discrete_map={
                "common (≥20%)": "#2E7D32",
                "intermediate (5–20%)": "#FBC02D",
                "rare (<5%)": "#C62828",
            },
            orientation="h",
            hover_data={"n_deps_seen": True, "mean_maxn": True},
            labels={
                "pct_seen": "% of deployments where seen",
                "display_name": "Species",
                "rarity": "Rarity",
            },
            height=max(380, len(rates) * 20 + 120),
        )
        style(
            fig,
            xaxis={"ticksuffix": "%"},
            yaxis_title=None,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            legend={"orientation": "h", "y": -0.05},
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Species shown", len(rates))
        c2.metric("Total deployments", total_drops)
        c3.metric("Common (≥20%)", int((rates["pct_seen"] >= 20).sum()))
        c4.metric("Rare (<5%)", int((rates["pct_seen"] < 5).sum()))
        return

    # ── Inside vs outside reserve view ──────────────────────────────────────
    # Denominators come from df_full so reviewed-but-empty deployments count;
    # the seen/rate side uses the named-species frame.
    df_full = df_full.assign(reserve=protection_group(df_full["protection_status"]))
    partial = int((df_full["reserve"] == OTHER_PROTECTION).sum())
    df_full = df_full[df_full["reserve"].isin([PROTECTED, UNPROTECTED])]
    if partial:
        st.caption(
            f"{partial:,} annotation rows are left out of this split: a partial "
            "or unclear protection regime is neither the reserve nor its "
            "control. They are still in the programme-wide view above."
        )
    if df_full.empty:
        st.warning("No deployments sit clearly inside or outside a reserve.")
        return
    # The rest of this chart was written against these two literals; mapping
    # here keeps the classification in one place without rewriting the chart.
    df_full["reserve"] = df_full["reserve"].map(
        {PROTECTED: "inside", UNPROTECTED: "outside"}
    )
    df = df_full[df_full["scientific_name"].notna()]

    # Per-class deployment totals (denominator for detection rate)
    drop_class = df_full.drop_duplicates("drop_id")[["drop_id", "reserve"]]
    total_per_class = drop_class.groupby("reserve")["drop_id"].nunique().to_dict()
    if "inside" not in total_per_class or "outside" not in total_per_class:
        st.warning(
            "Filters need to include sites both inside AND outside a reserve "
            "to use this view."
        )
        return

    # Detection rate per (species, reserve_class)
    seen = (
        df.groupby(["display_name", "reserve"])["drop_id"]
        .nunique()
        .reset_index(name="n_seen")
    )
    seen["total"] = seen["reserve"].map(total_per_class)
    seen["pct_seen"] = (seen["n_seen"] / seen["total"] * 100).round(2)

    # Fill in zero rows so every species has both inside and outside bars
    all_species = seen["display_name"].unique()
    full_grid = pd.MultiIndex.from_product(
        [all_species, ["inside", "outside"]],
        names=["display_name", "reserve"],
    ).to_frame(index=False)
    seen = full_grid.merge(seen, on=["display_name", "reserve"], how="left")
    seen["pct_seen"] = seen["pct_seen"].fillna(0)
    seen["n_seen"] = seen["n_seen"].fillna(0).astype(int)
    seen["total"] = seen["reserve"].map(total_per_class)

    # Detection-rate spread (inside − outside) drives the spotlight ordering
    spread = seen.pivot(
        index="display_name", columns="reserve", values="pct_seen"
    ).fillna(0)
    spread["abs_diff"] = (spread["inside"] - spread["outside"]).abs()
    spread["max_pct"] = spread[["inside", "outside"]].max(axis=1)

    top_n = top_n_slider(
        "Show top N species",
        len(spread),
        25,
        "rep_det_topn_io",
        help="Ranked by max detection rate across the two protection classes.",
    )
    ordering = spread.sort_values("max_pct", ascending=True).index.tolist()
    keep = spread.nlargest(top_n, "max_pct").index
    seen = seen[seen["display_name"].isin(keep)]
    species_order = [s for s in ordering if s in set(keep)]

    fig = px.bar(
        seen,
        x="pct_seen",
        y="display_name",
        color="reserve",
        barmode="group",
        # The shared group pair, keyed to this chart's lowercase categories:
        # "inside" is the protected side, "outside" the unprotected one.
        color_discrete_map={
            "inside": GROUP_COLOURS[PROTECTED],
            "outside": GROUP_COLOURS[UNPROTECTED],
        },
        orientation="h",
        category_orders={
            "display_name": species_order,
            "reserve": ["inside", "outside"],
        },
        hover_data={"n_seen": True, "total": True, "pct_seen": ":.2f"},
        labels={
            "pct_seen": "% of deployments where seen",
            "display_name": "Species",
            "reserve": "Protection",
        },
        height=max(380, len(species_order) * 28 + 120),
    )
    style(
        fig,
        xaxis={"ticksuffix": "%"},
        yaxis_title=None,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend_title_text="Protection",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics on the protection effect
    spread_visible = spread.loc[list(keep)].copy()
    spread_visible["effect"] = spread_visible["inside"] - spread_visible["outside"]
    n_higher_inside = int((spread_visible["effect"] > 0).sum())
    n_higher_outside = int((spread_visible["effect"] < 0).sum())
    n_equal = int((spread_visible["effect"] == 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Species shown", len(species_order))
    c2.metric("Higher inside reserve", n_higher_inside)
    c3.metric("Higher outside", n_higher_outside)
    c4.metric("Equal", n_equal)

    with st.expander("Detection rate table"):
        table = spread_visible.reset_index().rename(
            columns={
                "display_name": "Species",
                "inside": "% inside",
                "outside": "% outside",
                "effect": "Δ (inside − outside)",
            }
        )[["Species", "% inside", "% outside", "Δ (inside − outside)"]]
        st.dataframe(
            table.sort_values("Δ (inside − outside)", ascending=False),
            hide_index=True,
        )

    source_coverage_note(df, "these detection rates")


def render_freq_abundance(df: pd.DataFrame) -> None:
    section("Frequency vs abundance")
    st.caption(
        "Each dot = one species. **Frequency** (x) = % of deployments where seen. "
        "**Abundance** (y) = mean MaxN when seen. Median lines split species into four "
        "ecological strategies. Bubble size = total observed across the dataset."
    )

    df = df[df["scientific_name"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    total_drops = df["drop_id"].nunique()
    if total_drops == 0:
        st.warning("No deployments after filters.")
        return

    stats = (
        df.groupby("display_name")
        .agg(
            n_deps=("drop_id", "nunique"),
            mean_maxn=("maxn", "mean"),
            total_count=("maxn", "sum"),
        )
        .reset_index()
    )
    stats["frequency_pct"] = (stats["n_deps"] / total_drops * 100).round(2)
    stats["mean_maxn"] = stats["mean_maxn"].round(2)

    log_scale = st.checkbox(
        "Log-scale axes (recommended)", value=True, key="rep_freq_log"
    )

    fx, fy = float(stats["frequency_pct"].median()), float(stats["mean_maxn"].median())

    stats["strategy"] = stats.apply(
        lambda r: (
            "core"
            if r["frequency_pct"] >= fx and r["mean_maxn"] >= fy
            else (
                "transient"
                if r["frequency_pct"] >= fx
                else "patchy" if r["mean_maxn"] >= fy else "incidental"
            )
        ),
        axis=1,
    )

    strategy_colours = {
        "core": "#2E7D32",
        "transient": "#FF9800",
        "patchy": "#1976D2",
        "incidental": "#9E9E9E",
    }

    fig = px.scatter(
        stats,
        x="frequency_pct",
        y="mean_maxn",
        size="total_count",
        color="strategy",
        color_discrete_map=strategy_colours,
        hover_name="display_name",
        hover_data={
            "n_deps": True,
            "frequency_pct": ":.1f",
            "mean_maxn": ":.2f",
            "total_count": True,
            "strategy": True,
            "display_name": False,
        },
        labels={
            "frequency_pct": "Frequency (% of deployments seen)",
            "mean_maxn": "Mean MaxN when seen",
            "strategy": "Strategy",
        },
        size_max=40,
        height=560,
    )

    fig.add_hline(y=fy, line_dash="dash", line_color="#888", line_width=1)
    fig.add_vline(x=fx, line_dash="dash", line_color="#888", line_width=1)

    # Quadrant labels at corners
    x_min = max(0.1, stats["frequency_pct"].min() / 2)
    y_min = max(0.1, stats["mean_maxn"].min() / 2)
    x_max = stats["frequency_pct"].max() * 1.05
    y_max = stats["mean_maxn"].max() * 1.05
    quadrants = [
        ("Patchy", x_min, y_max, "#1976D2"),
        ("Core", x_max, y_max, "#2E7D32"),
        ("Incidental", x_min, y_min, "#9E9E9E"),
        ("Transient", x_max, y_min, "#FF9800"),
    ]
    for label, x, y, colour in quadrants:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font={"size": 12, "color": colour},
            xanchor="left" if x < fx else "right",
            yanchor="bottom" if y < fy else "top",
        )

    if log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    style(
        fig,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="white",
        legend={"orientation": "h", "y": -0.12},
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    fig.update_yaxes(gridcolor="#EEEEEE")
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(4)
    for col, strat in zip(cols, ["core", "patchy", "transient", "incidental"]):
        col.metric(strat.capitalize(), int((stats["strategy"] == strat).sum()))

    with st.expander("Species classification"):
        st.dataframe(
            stats.sort_values("mean_maxn", ascending=False)[
                [
                    "display_name",
                    "n_deps",
                    "frequency_pct",
                    "mean_maxn",
                    "total_count",
                    "strategy",
                ]
            ].rename(
                columns={
                    "display_name": "Species",
                    "n_deps": "Deployments seen",
                    "frequency_pct": "Frequency (%)",
                    "mean_maxn": "Mean MaxN",
                    "total_count": "Total observed",
                    "strategy": "Strategy",
                }
            ),
            hide_index=True,
        )


def render_cooccurrence(df: pd.DataFrame) -> None:
    # Species counts only: the unidentified bucket is N unknown species
    # under one label, so counting it as one both understates richness and
    # puts a meaningless row in the matrix.
    df = real_species(df)
    section("Co-occurrence")
    st.caption(
        "Cell = how often two species are seen at the same deployment, normalised by "
        "the rarer of the two species' total deployments (Jaccard-like). "
        "High values = species that travel together. Diagonal = self (always 1)."
    )

    df = df[df["scientific_name"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    min_deps = st.slider(
        "Min deployments to include a species", 2, 30, 5, key="rep_co_min"
    )
    species_counts = df.groupby("display_name")["drop_id"].nunique()
    keep = species_counts[species_counts >= min_deps].index
    df = df[df["display_name"].isin(keep)]
    if df["display_name"].nunique() < 2:
        st.warning(f"Need ≥ 2 species each observed at ≥ {min_deps} deployments.")
        return

    top_n = top_n_slider(
        "Show top N species (by occurrence)",
        df["display_name"].nunique(),
        25,
        "rep_co_topn",
    )
    top_species = species_counts[species_counts.index.isin(keep)].nlargest(top_n).index
    df = df[df["display_name"].isin(top_species)]

    presence = (
        df.assign(p=1)
        .pivot_table(
            index="drop_id",
            columns="display_name",
            values="p",
            aggfunc="max",
            fill_value=0,
        )
        .astype(int)
    )
    species_list = presence.columns.tolist()

    arr = presence.values
    co = arr.T @ arr
    totals = arr.sum(axis=0)
    min_totals = np.minimum.outer(totals, totals)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(min_totals > 0, co / min_totals, 0)

    matrix = pd.DataFrame(norm, index=species_list, columns=species_list)
    order = pd.Series(totals, index=species_list).sort_values(ascending=False).index
    matrix = matrix.loc[order, order]

    fig = px.imshow(
        matrix,
        color_continuous_scale="Viridis",
        zmin=0,
        zmax=1,
        aspect="auto",
        labels={"x": "Species B", "y": "Species A", "color": "Co-occurrence"},
        # Row budget plus a fixed allowance for the angled x labels: Plotly
        # auto-expands the bottom margin to fit them, and that expansion comes
        # out of the plot area — at 22px/row the labels were most of the chart
        # and the heatmap itself was a strip.
        height=max(480, len(order) * 30 + 260),
    )
    # A tick per species, forced. Plotly silently thins categorical ticks when
    # rows get tight — at top-11 it showed six labels — and a heatmap row
    # without its name is unreadable. `dtick=1` walks the category indices.
    fig.update_xaxes(tickangle=-60, tickmode="linear", dtick=1)
    fig.update_yaxes(tickmode="linear", dtick=1)
    style(fig, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, use_container_width=True)

    pairs = matrix.where(np.triu(np.ones(matrix.shape, dtype=bool), k=1)).stack()
    top_pairs = pairs.sort_values(ascending=False).head(15).reset_index()
    top_pairs.columns = ["Species A", "Species B", "Co-occurrence"]
    st.markdown("**Top 15 co-occurring species pairs**")
    st.dataframe(top_pairs, hide_index=True)


def render_species_depth(df: pd.DataFrame, dep: pd.DataFrame) -> None:
    """The depth range each species has been recorded over.

    The easy version of the species-by-depth question: one box per species
    across the depths of the deployments that recorded it. Depth lives on the
    deployments table, not the annotations, so it is merged on drop_id here.
    """
    section("Depth")
    depths = dep[["drop_id", "depth"]].copy()
    depths["depth"] = pd.to_numeric(depths["depth"], errors="coerce")
    depths = depths.dropna(subset=["depth"]).rename(columns={"depth": "dep_depth"})
    if depths.empty:
        st.info("No deployments in this selection carry a recorded depth.")
        return

    # One point per (deployment, species): the question is where the species
    # occurs, so a deployment counts once however many intervals saw it.
    seen = real_species(df[df["scientific_name"].notna()])
    seen = seen.drop_duplicates(["drop_id", "display_name"]).merge(
        depths, on="drop_id", how="inner"
    )
    if seen.empty:
        st.info("No annotated deployments in this selection carry a depth.")
        return

    st.caption(
        f"The depths each species has been recorded at — one marker per "
        f"deployment that saw it, box = the middle half. Depth is recorded on "
        f"**{depths['drop_id'].nunique():,} of {dep['drop_id'].nunique():,} "
        f"deployments** in the current selection; the rest cannot appear here."
    )

    top = seen["display_name"].value_counts().head(15).index
    plot = seen[seen["display_name"].isin(top)]
    order = (
        plot.groupby("display_name")["dep_depth"].median().sort_values().index.tolist()
    )
    fig = px.box(
        plot,
        x="dep_depth",
        y="display_name",
        orientation="h",
        points="all",
        category_orders={"display_name": order},
        labels={"dep_depth": "Depth (m)"},
        height=max(360, 30 * len(order) + 120),
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.55}, line={"width": 1.5})
    style(fig, legend=False)
    fig.update_yaxes(title=None)
    st.plotly_chart(fig, key="species_depth")
    st.caption(
        "Top 15 species by occurrence, ordered by median depth. A wide range "
        "partly reflects being recorded often — rarely seen species have not "
        "had the chance to show their full range."
    )


def render_species_accumulation(df: pd.DataFrame) -> None:
    # Species counts only: the unidentified bucket is N unknown species
    # under one label, so counting it as one both understates richness and
    # puts a meaningless row in the matrix.
    df = real_species(df)
    section("Accumulation")
    st.caption(
        "Cumulative unique species discovered as more deployments are surveyed. "
        "Resampled 50× over random deployment orderings, shaded band = 95% interval. "
        "When the curve flattens, additional surveys add little new biodiversity. "
        "If it's still rising sharply at the right edge, more surveys would still find new species."
    )

    df = df[df["scientific_name"].notna()].copy()
    drops_with_species = df[["drop_id", "display_name"]].drop_duplicates()
    if drops_with_species.empty:
        st.warning("No species data after filters.")
        return

    by_reserve = st.checkbox("Split by reserve", value=False, key="rep_acc_split")
    n_iter = 50

    def _accumulate(pairs_df: pd.DataFrame):
        """Resampled mean + 95% interval cumulative species count."""
        drop_to_species = (
            pairs_df.groupby("drop_id")["display_name"].apply(set).to_dict()
        )
        all_drops = list(drop_to_species.keys())
        n = len(all_drops)
        if n == 0:
            return None
        rng = np.random.default_rng(42)
        results = np.empty((n_iter, n), dtype=int)
        for i in range(n_iter):
            order = rng.permutation(all_drops)
            seen = set()
            for j, d in enumerate(order):
                seen.update(drop_to_species[d])
                results[i, j] = len(seen)
        return (
            np.arange(1, n + 1),
            results.mean(axis=0),
            np.percentile(results, 2.5, axis=0),
            np.percentile(results, 97.5, axis=0),
        )

    fig = go.Figure()
    palette = px.colors.qualitative.Safe

    if by_reserve:
        reserve_drops = df[df["reserve_code"].notna() & (df["reserve_code"] != "")]
        reserve_drops = reserve_drops[
            ["reserve_code", "drop_id", "display_name"]
        ].drop_duplicates()
        for i, (reserve, grp) in enumerate(reserve_drops.groupby("reserve_code")):
            pairs = grp[["drop_id", "display_name"]].drop_duplicates()
            if pairs["drop_id"].nunique() < 5:
                continue
            res = _accumulate(pairs)
            if res is None:
                continue
            x, mean_y, lo, hi = res
            colour = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor=(
                        colour.replace("rgb(", "rgba(").replace(")", ",0.15)")
                        if colour.startswith("rgb")
                        else colour
                    ),
                    line={"color": "rgba(0,0,0,0)"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean_y,
                    mode="lines",
                    line={"color": colour, "width": 2},
                    name=reserve,
                    hovertemplate=(
                        f"<b>{reserve}</b><br>"
                        "%{x} deps → %{y:.1f} species<extra></extra>"
                    ),
                )
            )
    else:
        res = _accumulate(drops_with_species)
        if res is None:
            st.warning("Not enough deployments.")
            return
        x, mean_y, lo, hi = res
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor="rgba(33, 150, 243, 0.2)",
                line={"color": "rgba(0,0,0,0)"},
                name="95% interval",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_y,
                mode="lines",
                line={"color": "#1976D2", "width": 2.5},
                name="Mean",
                hovertemplate="%{x} deployments → %{y:.1f} species<extra></extra>",
            )
        )

    style(
        fig,
        xaxis_title="Cumulative deployments surveyed",
        yaxis_title="Cumulative unique species",
        height=480,
        plot_bgcolor="white",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    fig.update_yaxes(gridcolor="#EEEEEE")
    st.plotly_chart(fig, use_container_width=True)

    # Saturation hint (overall curve only)
    if not by_reserve:
        tail = mean_y[-min(10, len(mean_y)) :]
        head = (
            mean_y[-min(20, len(mean_y)) : -min(10, len(mean_y))]
            if len(mean_y) >= 20
            else None
        )
        slope_tail = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
        slope_head = (
            ((head[-1] - head[0]) / max(len(head) - 1, 1)) if head is not None else None
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Deployments included", drops_with_species["drop_id"].nunique())
        c2.metric("Unique species found", drops_with_species["display_name"].nunique())
        c3.metric(
            "Last 10 deps: new species/dep",
            f"{slope_tail:.2f}",
            delta=(
                f"{slope_tail - slope_head:+.2f}" if slope_head is not None else None
            ),
            delta_color="off",
            help="< 0.5 = curve flattening; close to zero = saturated.",
        )


def render_yearly_trend(df: pd.DataFrame) -> None:
    section("Year trend")
    st.caption(
        "Mean peak MaxN per site per survey year. "
        "Surveys are 1–2 year cadence, not seasonal."
    )

    df = df[df["scientific_name"].notna() & df["survey_year"].notna()]
    all_species = sorted(df["display_name"].dropna().unique())
    if not all_species:
        st.warning("No species data after filters.")
        return

    # Default to Snapper (Pagrus auratus), iconic NZ BUV species with rich
    # historical data, falling back to Blue cod, then alphabetical first.
    default_idx = 0
    for preferred in ("Pagrus auratus", "Parapercis colias"):
        for i, name in enumerate(all_species):
            if preferred in name:
                default_idx = i
                break
        else:
            continue
        break
    species = st.selectbox(
        "Species", all_species, index=default_idx, key="rep_trend_spp"
    )
    df = df[df["display_name"] == species]

    trend = (
        df.groupby(["site_id", "survey_year", "protection_status"], dropna=False)[
            "maxn"
        ]
        .mean()
        .reset_index()
        .rename(columns={"maxn": "mean_maxn"})
    )
    trend["mean_maxn"] = trend["mean_maxn"].round(2)
    trend["survey_year"] = trend["survey_year"].astype(int)

    if trend.empty:
        st.warning(f"No data for {species}.")
        return

    fig = px.line(
        trend.sort_values("survey_year"),
        x="survey_year",
        y="mean_maxn",
        color="site_id",
        line_dash="protection_status",
        # Solid inside a reserve, dotted outside. Without the map Plotly hands
        # out dash styles in whatever order the statuses arrive, so the reserve
        # was solid on one filter setting and dashed on the next.
        line_dash_map=protection_dashes(trend["protection_status"].dropna().unique()),
        color_discrete_sequence=px.colors.qualitative.Safe,
        markers=True,
        hover_data={"protection_status": True, "mean_maxn": True},
        labels={
            "survey_year": "Survey year",
            "mean_maxn": f"Mean MaxN, {species}",
            "site_id": "Site",
            "protection_status": "Protection",
        },
        height=450,
    )
    style(
        fig,
        legend_title_text="Site",
        xaxis={"dtick": 1, "tickformat": "d"},
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Solid = protected, dashed = unprotected. Each line = one site.")

    with st.expander("Data"):
        st.dataframe(trend.sort_values(["site_id", "survey_year"]), hide_index=True)


def render_species_abundance(df, df_context, effort_view) -> None:
    """Mean MaxN and frequency per species, against the effort behind them.

    Moved off the Sites view: it is a statement about species, not about
    sites, and it was the only thing on that page not keyed by site.

    Three frames, on purpose. `df` is the species-filtered sightings,
    `df_context` the same without the species filter (so "nothing was seen
    here" is not read as "not THIS species here"), and `effort_view` every
    surveyed deployment, which is the denominator: dividing by sightings
    alone would discard the zeros and overstate every rare species.
    """
    section("Species abundance")
    st.caption(
        "**Mean MaxN** is the abundance index: total MaxN divided by every deployment "
        "surveyed, *including the ones where the species was not seen*. Dividing only "
        "by sightings would discard the zeros and overstate rare species. "
        "**Frequency** is the share of deployments the species appeared in at all, "
        "how widespread it is, as opposed to how many there are."
    )

    effort_total = int(effort_view["drop_id"].nunique())
    totals = (
        df.groupby("display_name")
        .agg(seen_in=("drop_id", "nunique"), total_maxn=("maxn", "sum"))
        .reset_index()
        .rename(columns={"display_name": "Species"})
    )
    totals["mean_maxn"] = totals["total_maxn"] / max(effort_total, 1)
    totals["frequency"] = totals["seen_in"] / max(effort_total, 1)
    totals = totals.sort_values("mean_maxn", ascending=False)

    left, right = st.columns([1, 1])
    with left:
        st.dataframe(
            totals[["Species", "mean_maxn", "frequency", "seen_in"]],
            width="stretch",
            hide_index=True,
            column_config={
                "Species": st.column_config.TextColumn("Species", width="large"),
                "mean_maxn": st.column_config.NumberColumn("Mean MaxN", format="%.2f"),
                "frequency": st.column_config.NumberColumn(
                    "Frequency", format="percent"
                ),
                "seen_in": st.column_config.NumberColumn("Seen in"),
            },
        )
        # Analysed drops that produced no sighting of anything, the zeros that every
        # mean above is divided by. Sightings come from df_context so a species filter
        # does not turn "no species here" into "not this species here"; the analysed
        # set comes from effort_view so it carries the SAME year/region/protection
        # filters as the sightings and the caption's denominator. Filtering the two
        # sides differently made blank_pct overshoot (past 100% with a year range
        # active), and going through "sites seen in the sightings" dropped fully
        # blank sites, exactly the zeros this caption exists to count.
        # Named rows only: a null-species row means "reviewed, nothing seen",
        # so a deployment carrying only null rows is exactly the blank this
        # count exists to report, not a sighting.
        seen_any = set(
            df_context.loc[df_context["scientific_name"].notna(), "drop_id"].dropna()
        )
        analysed_drops = set(effort_view["drop_id"].dropna())
        blank_drops = len(analysed_drops - seen_any)
        blank_pct = blank_drops / max(effort_total, 1)

        st.caption(
            f"Across **{effort_total:,}** analysed deployments in this selection, "
            f"**{blank_drops:,}** of them ({blank_pct:.0%}) recorded nothing at all. "
            "Those blanks are real zeros and are included in every mean above."
        )

    with right:
        # Split by protection so the reserve effect is visible per species rather than
        # averaged away. Effort differs sharply between protection classes, so each
        # bar is divided by its OWN group's deployment count, a single shared
        # denominator would make whichever class was surveyed more look richer.
        grouped = (
            df.groupby(["display_name", "protection_status"])["maxn"]
            .sum()
            .reset_index()
        )
        group_effort = effort_per(effort_view, "protection_status", "effort")
        grouped = grouped.merge(group_effort, on="protection_status", how="left")
        grouped["mean_maxn"] = grouped["maxn"] / grouped["effort"].clip(lower=1)

        # Ranked by mean MaxN across the whole selection, so the chart shows the
        # most abundant species rather than an arbitrary slice.
        n_top = 10
        top_species = totals.head(n_top)["Species"]
        plot_df = grouped[grouped["display_name"].isin(top_species)]
        st.markdown(
            f"**Top {min(n_top, len(top_species))} most abundant species**, "
            "split by protection status"
        )

        fig = px.bar(
            plot_df,
            x="mean_maxn",
            y="display_name",
            color="protection_status",
            color_discrete_map=protection_color_map(plot_df["protection_status"]),
            orientation="h",
            barmode="group",
            height=max(360, 34 * plot_df["display_name"].nunique()),
            # Plotly puts the first category at the BOTTOM of a horizontal
            # bar chart, so this runs least abundant first and the most
            # abundant species ends up on top.
            category_orders={"display_name": list(top_species)},
        )
        fig.update_traces(marker_cornerradius=4)
        style(
            fig,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "title": None},
        )
        fig.update_xaxes(title="Mean MaxN per deployment")
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, use_container_width=True)

    # "By MPA", "Deployments by protection status" and "MPA populations" used to
    # sit below here. They moved to the MPA view: everything left on this page is
    # per-site, while those three are per-area, which is a different question.

    st.divider()
