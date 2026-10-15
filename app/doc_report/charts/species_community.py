"""Community charts for the Species view: which species, and how they sit together.

How often each species is detected, how common against how abundant, which
species turn up together, at what depth, whether the survey effort has found
most of what is there, and mean abundance against the effort behind it.
`doc_report/species.py` decides what the reader is asked; these functions take a frame
and draw.

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
from theme import (
    GRID_ON_WHITE,
    MODEL_LINE,
    PLOT_BG_WHITE,
    RARITY_COLORS,
    REFERENCE_LINE,
    STRATEGY_COLORS,
    protection_color_map,
)

from ..charting import source_coverage_note, style, top_n_slider
from ..data import effort_per, real_species
from ..layout import section
from .species import GROUP_COLOURS

# ── Ported from the Experiments page ──────────────────────────────────────
#
# Ported from the old Experiments page, which has since been retired, so
# these are now the only copies. The `rep_` prefix on the widget keys is a
# leftover of that move: harmless, and renaming keys would reset anyone's
# saved selections for no gain.


def render_detection_rate(df: pd.DataFrame, df_all: pd.DataFrame | None = None) -> None:
    section("Detection rate")
    st.caption(
        "For each species: % of deployments where it was detected. "
        "Highlights survey coverage of common vs rare species. "
        "**Inside vs outside view** splits the same rate by MPA protection, "
        "a presence-axis complement to the MPA effect slope (which is on abundance)."
    )

    # The DENOMINATOR is every annotated deployment, from the
    # species-UNfiltered frame: with a species ticked in the sidebar, a
    # denominator built from `df` would collapse to the deployments that
    # recorded that species and every rate would read 100%. Numerators (the
    # species rows) stay on `df`, which is the filter narrowing which BARS
    # show, not what they are divided by.
    denominators = df_all if df_all is not None else df
    # Null-species rows are absence records ("reviewed, nothing seen"): they
    # belong in the deployment denominator, but carry no species to rate, so
    # count deployments BEFORE dropping them.
    total_drops = denominators["drop_id"].nunique()
    if total_drops == 0:
        st.warning("No deployments after filters.")
        return
    df_full = denominators
    df = df[df["scientific_name"].notna()]

    view = st.radio(
        "View",
        ["Programme-wide", "Inside vs outside MPA"],
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
            color_discrete_map=RARITY_COLORS,
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
            "or unclear protection regime is neither the MPA nor its "
            "control. They are still in the programme-wide view above."
        )
    if df_full.empty:
        st.warning("No deployments sit clearly inside or outside an MPA.")
        return
    # The rest of this chart was written against these two literals; mapping
    # here keeps the classification in one place without rewriting the chart.
    df_full["reserve"] = df_full["reserve"].map(
        {PROTECTED: "inside", UNPROTECTED: "outside"}
    )
    # Numerators keep the sidebar species filter; they are classified
    # independently of the denominator frame.
    df = df.assign(reserve=protection_group(df["protection_status"]))
    df = df[df["reserve"].isin([PROTECTED, UNPROTECTED])]
    df["reserve"] = df["reserve"].map({PROTECTED: "inside", UNPROTECTED: "outside"})

    # Per-class deployment totals (denominator for detection rate)
    drop_class = df_full.drop_duplicates("drop_id")[["drop_id", "reserve"]]
    total_per_class = drop_class.groupby("reserve")["drop_id"].nunique().to_dict()
    if "inside" not in total_per_class or "outside" not in total_per_class:
        st.warning(
            "Filters need to include sites both inside AND outside an MPA "
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
    c2.metric("Higher inside MPA", n_higher_inside)
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
        delta = "Δ (inside − outside)"
        # Diverging scale on the effect, centred on zero: green = detected
        # more often inside the MPA, red = more often outside. The rates
        # themselves get the plain blue magnitude scale.
        span = max(float(table[delta].abs().max() or 1), 1.0)
        st.dataframe(
            table.sort_values(delta, ascending=False)
            .style.background_gradient(
                cmap="Blues", subset=["% inside", "% outside"], vmin=0, vmax=100
            )
            .background_gradient(cmap="RdYlGn", subset=[delta], vmin=-span, vmax=span)
            .format("{:.1f}", subset=["% inside", "% outside", delta]),
            hide_index=True,
            column_config={
                "Species": st.column_config.TextColumn("Species", pinned=True)
            },
        )

    source_coverage_note(df, "these detection rates")


def render_freq_abundance(df: pd.DataFrame) -> None:
    section("Frequency vs abundance")
    st.caption(
        "Each dot = one species. **Frequency** (x) = % of deployments where seen. "
        "**Abundance** (y) = mean MaxN when seen. Median lines split species into four "
        "ecological strategies. Bubble size = total observed across the dataset."
    )

    # Deployments are counted BEFORE the null-species rows go: those are
    # absence records ("reviewed, nothing seen"), and they belong in the
    # frequency denominator. Counting after dropping them gave this chart a
    # smaller denominator than the detection-rate panel beside it, so the two
    # showed different frequencies for the same species on the same page.
    total_drops = df["drop_id"].nunique()
    if total_drops == 0:
        st.warning("No deployments after filters.")
        return

    df = df[df["scientific_name"].notna()]
    if df.empty:
        st.warning("No data after filters.")
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

    fig = px.scatter(
        stats,
        x="frequency_pct",
        y="mean_maxn",
        size="total_count",
        color="strategy",
        color_discrete_map=STRATEGY_COLORS,
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

    fig.add_hline(y=fy, line_dash="dash", line_color=REFERENCE_LINE, line_width=1)
    fig.add_vline(x=fx, line_dash="dash", line_color=REFERENCE_LINE, line_width=1)

    # Quadrant labels at corners
    x_min = max(0.1, stats["frequency_pct"].min() / 2)
    y_min = max(0.1, stats["mean_maxn"].min() / 2)
    x_max = stats["frequency_pct"].max() * 1.05
    y_max = stats["mean_maxn"].max() * 1.05
    quadrants = [
        ("Patchy", x_min, y_max),
        ("Core", x_max, y_max),
        ("Incidental", x_min, y_min),
        ("Transient", x_max, y_min),
    ]
    for label, x, y in quadrants:
        colour = STRATEGY_COLORS[label.lower()]
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
        plot_bgcolor=PLOT_BG_WHITE,
        legend={"orientation": "h", "y": -0.12},
    )
    fig.update_xaxes(gridcolor=GRID_ON_WHITE)
    fig.update_yaxes(gridcolor=GRID_ON_WHITE)
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
            column_config={
                "Species": st.column_config.TextColumn("Species", pinned=True)
            },
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
    st.dataframe(
        top_pairs.style.background_gradient(
            cmap="Blues", subset=["Co-occurrence"], vmin=0, vmax=1
        ).format("{:.2f}", subset=["Co-occurrence"]),
        hide_index=True,
        column_config={
            "Species A": st.column_config.TextColumn("Species A", pinned=True)
        },
    )


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
        f"The depths each species has been recorded at, one marker per "
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
        "partly reflects being recorded often, rarely seen species have not "
        "had the chance to show their full range."
    )


def render_species_accumulation(df: pd.DataFrame) -> None:
    # Species counts only: the unidentified bucket is N unknown species
    # under one label, so counting it as one both understates richness and
    # puts a meaningless row in the matrix.
    df = real_species(df)
    section("Accumulation")
    st.caption(
        "Cumulative unique species discovered as more deployments are "
        "surveyed, averaged over 50 random deployment orderings. All "
        "species, whatever the sidebar Species filter says. When the curve "
        "flattens, additional surveys add little new biodiversity; still "
        "rising at the right edge means more surveys would find more "
        "species."
    )

    df = df[df["scientific_name"].notna()].copy()
    drops_with_species = df[["drop_id", "display_name"]].drop_duplicates()
    if drops_with_species.empty:
        st.warning("No species data after filters.")
        return

    by_reserve = st.checkbox("Split by MPA", value=False, key="rep_acc_split")
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
            # Mean line only. The resampled 95% ribbons were removed on
            # review (2026-10-02): with one ribbon per MPA they overlapped
            # into unreadable grey.
            x, mean_y, lo, hi = res
            colour = palette[i % len(palette)]
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
                x=x,
                y=mean_y,
                mode="lines",
                line={"color": MODEL_LINE, "width": 2.5},
                name="Mean",
                hovertemplate="%{x} deployments → %{y:.1f} species<extra></extra>",
            )
        )

    style(
        fig,
        xaxis_title="Cumulative deployments surveyed",
        yaxis_title="Cumulative unique species",
        height=480,
        plot_bgcolor=PLOT_BG_WHITE,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    fig.update_xaxes(gridcolor=GRID_ON_WHITE)
    fig.update_yaxes(gridcolor=GRID_ON_WHITE)
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
                "Species": st.column_config.TextColumn(
                    "Species", width="large", pinned=True
                ),
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
