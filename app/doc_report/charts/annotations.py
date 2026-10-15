"""Chart layer for the Annotations views.

One function per chart, taking the frame it draws.

First occupant is the timing chart: the only chart in the report that reads
*when* in a deployment something happened rather than how much of it there was.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from theme import (
    ANCHOR_LINE,
    ARRIVAL_PEAK_COLORS,
    GRID_ON_WHITE,
    GUIDE_LINE,
    MARKER_OUTLINE,
    MINOR_GUIDE_LINE,
    MODEL_LINE,
    PLOT_BG_WHITE,
    REFERENCE_LINE,
    ZERO_LINE_ON_WHITE,
)

from ..charting import style, top_n_slider
from ..data import CURVE_EXPERT, CURVE_ML
from ..layout import section

# Deployments are 30 minutes, and the ML model scores every 10 seconds, so 180
# intervals cover one. Used for the axis range and the reference lines.
DEPLOYMENT_MINUTES = 30

MEASURES = ARRIVAL_PEAK_COLORS


def render_arrival_and_peak(times: pd.DataFrame) -> None:
    """Two timings per species: when it first appeared, and when it peaked.

    Ported from the Experiments page's bait-arrival curve, which drew one
    distribution and labelled it arrival while actually plotting the time of
    peak count. The two are now separate measurements, from the sources that
    can actually answer them (see `data.arrival_and_peak`), so the gap between
    them is visible rather than being the error bar on a single number.
    """
    section("Arrival and MaxN time")
    st.caption(
        "**Arrival** is the first 10-second interval the model detected the "
        "species: how quickly it responded to the bait. **MaxN time** is when "
        "its count peaked, from the best source available for that deployment. "
        "A species can arrive early and peak much later, and the distance "
        "between the two boxes is how long it took to gather."
    )

    if times.empty:
        st.info("No annotations carry a usable timestamp.")
        return

    long = pd.concat(
        [
            times[["display_name", column]]
            .rename(columns={column: "seconds"})
            .assign(Measure=label)
            for column, label in (("arrival_s", "Arrival"), ("peak_s", "MaxN time"))
        ]
    ).dropna(subset=["seconds", "display_name"])
    if long.empty:
        st.info("No annotations carry a usable timestamp.")
        return
    long["minutes"] = long["seconds"] / 60

    # Ranked by how many timings the species has, so the chart shows the
    # species there is enough evidence to say anything about.
    counts = long.groupby("display_name").size()
    top_n = st.slider(
        "Top N species (by timed observations)",
        5,
        max(5, len(counts)),
        min(12, len(counts)),
        key="rep_timing_topn",
    )
    keep = counts.nlargest(top_n).index
    long = long[long["display_name"].isin(keep)]

    # Ordered by arrival, so the quickest responders to the bait are together
    # at one end. Falls back to whatever timing exists for a species with no
    # ML detection at all.
    arrival_median = (
        long[long["Measure"] == "Arrival"].groupby("display_name")["minutes"].median()
    )
    order = (
        arrival_median.reindex(keep)
        .fillna(long.groupby("display_name")["minutes"].median())
        .sort_values()
        .index.tolist()
    )

    # Box, not violin. A violin's width is a kernel density estimate, which
    # needs enough points to mean anything and is read as "more fish" by anyone
    # who has not been told otherwise. A box says exactly four things — median,
    # quartiles, range — and says them the same way at any sample size.
    fig = px.box(
        long,
        x="minutes",
        y="display_name",
        color="Measure",
        orientation="h",
        points="outliers",
        category_orders={"display_name": order, "Measure": list(MEASURES)},
        color_discrete_map=MEASURES,
        labels={"minutes": "Time in deployment (min)", "display_name": "Species"},
        height=max(400, len(order) * 55 + 120),
    )
    for minute in range(5, DEPLOYMENT_MINUTES, 5):
        fig.add_vline(
            x=minute, line_dash="dot", line_color=MINOR_GUIDE_LINE, line_width=1
        )

    style(
        fig,
        legend="top",
        xaxis={"range": [0, DEPLOYMENT_MINUTES], "dtick": 5},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    fig.update_yaxes(title=None)
    st.plotly_chart(fig, key="rep_arrival_peak")

    paired = times.dropna(subset=["arrival_s", "peak_s"])
    if not paired.empty:
        gap = (paired["peak_s"] - paired["arrival_s"]) / 60
        st.caption(
            f"Where both are known ({len(paired):,} deployment-species pairs), the "
            f"peak comes **{gap.median():.0f} minutes** after arrival at the "
            f"median, and later than arrival in "
            f"{(gap > 0).mean():.0%} of them."
        )

    summary = (
        long.groupby(["display_name", "Measure"])["minutes"]
        .agg(["count", "median", "mean"])
        .round(1)
        .reset_index()
        .rename(
            columns={
                "display_name": "Species",
                "count": "Timed observations",
                "median": "Median (min)",
                "mean": "Mean (min)",
            }
        )
    )
    with st.expander("The numbers behind this"):
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption(
            "Arrival counts are smaller than MaxN-time counts: only the model "
            "scores every interval, so only deployments it has run on can "
            "report an arrival at all."
        )


def render_discovery_curve(curve: pd.DataFrame) -> None:
    """Cumulative-MaxN discovery curves: is 30 minutes long enough?

    The deployment-length check the DOC BUV reports run before trusting MaxN
    (Brough et al. 2018 plot it for blue cod and blue moki and read the
    plateau as "30 minutes is adequate"). Each line is the mean, across
    deployment-species pairs, of the best count seen so far as a share of the
    final MaxN. Early plateau = the peak is established early and the
    deployment length is not limiting; a line still climbing into the final
    minutes = MaxN is censored by the cutoff.
    """
    # Renamed from "Discovery curve" (2026-10-02): that name collided with
    # the Species view's across-deployments accumulation chart, and the two
    # answer different questions. The DOC reports' own term stays in the
    # caption.
    section("Deployment length check")
    st.caption(
        "Is 30 minutes of footage enough? For each deployment and species: "
        "the best count seen so far, as a share of the deployment's final "
        "MaxN, minute by minute (the DOC BUV reports call this a discovery "
        "curve). Every line reaches 100% at the end **by construction**; "
        "the finding is how early it gets close. A species whose curve "
        "plateaus early has its peak well inside the deployment, one still "
        "climbing in the final minutes would likely count higher with "
        "longer footage."
    )

    if curve.empty:
        st.info(
            "No interval-level count data in this selection, so there is no "
            "count path to draw."
        )
        return

    # Two count paths, never blended: the model's 10-second interval scores,
    # and the legacy expert reviews that counted interval by interval through
    # the whole video (the DOC toolbox method). The model has a known habit of
    # missing brief appearances, so the expert curve is the check on whether a
    # late-climbing ML curve is fish behaviour or detector noise.
    available = [s for s in (CURVE_ML, CURVE_EXPERT) if s in set(curve["Source"])]
    if len(available) > 1:
        chosen = st.radio(
            "Count path",
            available,
            horizontal=True,
            key="rep_discovery_source",
        )
    else:
        chosen = available[0]
        st.caption(f"Only **{chosen}** carries interval-level counts here.")
    curve = curve[curve["Source"] == chosen]

    # Pair = one (deployment, species) curve; species are ranked by how many
    # pairs stand behind their line, same policy as the timing chart above.
    pair_counts = curve.groupby("display_name")["drop_id"].nunique()
    top_n = top_n_slider(
        "Top N species (by deployments with interval counts)",
        len(pair_counts),
        6,
        "rep_discovery_topn",
    )
    keep = pair_counts.nlargest(top_n).index

    ALL = "All species"
    shown = pd.concat(
        [
            curve.assign(Species=ALL),
            curve[curve["display_name"].isin(keep)].rename(
                columns={"display_name": "Species"}
            ),
        ]
    )
    stats = (
        shown.groupby(["Species", "minute"])["share"]
        .agg(total="sum", n="count")
        .reset_index()
    )
    stats["mean_pct"] = stats["total"] / stats["n"] * 100

    order = [ALL] + [s for s in keep]
    fig = px.line(
        stats,
        x="minute",
        y="mean_pct",
        color="Species",
        category_orders={"Species": order},
        hover_data={"n": True, "mean_pct": ":.1f"},
        labels={
            "minute": "Time in deployment (min)",
            "mean_pct": "Share of final MaxN reached",
            "n": "Pairs",
        },
        height=460,
    )
    # The overall line anchors the chart; a neutral dark keeps it readable
    # under however many species colours sit on top of it.
    for trace in fig.data:
        if trace.name == ALL:
            trace.line.color = ANCHOR_LINE
            trace.line.width = 3
    fig.add_hline(y=95, line_dash="dot", line_color=GUIDE_LINE, line_width=1)

    style(
        fig,
        legend="top",
        xaxis={"range": [0.5, DEPLOYMENT_MINUTES], "dtick": 5},
        yaxis={"range": [0, 102], "ticksuffix": "%"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    st.plotly_chart(fig, key="rep_discovery_curve")

    overall = stats[stats["Species"] == ALL].sort_values("minute")
    settled = overall[overall["mean_pct"] >= 95]
    n_pairs = int(pair_counts.sum())
    if not settled.empty:
        st.caption(
            f"Across **{n_pairs:,} deployment-species pairs**, the mean curve "
            f"passes 95% of the final MaxN at **minute "
            f"{int(settled['minute'].iloc[0])}** (dotted line)."
        )
    st.caption(
        "Read with care: the curve is conditional on the species having been "
        "detected at all, and it cannot see past the end of the video. If "
        "counts were still rising at the cutoff, the true MaxN is higher than "
        "any source reports; only longer test deployments can rule that out."
    )


def render_source_calibration(df: pd.DataFrame) -> None:
    section("Calibration")
    st.caption(
        "Each point = one (deployment, species) where both sources observed. "
        "Diagonal = perfect agreement. Above 1:1 = X-source undercounts; below = overcounts. "
        "R² and slope quantify systematic bias."
    )
    st.info(
        "This experiment uses **all sources** regardless of the global source filter, "
        "it needs both sources for every comparison point."
    )

    available = sorted(df["annotated_by"].dropna().unique())
    if len(available) < 2:
        st.info("Need annotations from at least two sources to compare.")
        return

    c1, c2 = st.columns(2)
    src_x = c1.selectbox(
        "X axis (typically ML)",
        available,
        index=available.index("ml") if "ml" in available else 0,
        key="rep_cal_x",
    )
    remaining = [s for s in available if s != src_x]
    src_y = c2.selectbox(
        "Y axis (typically ground truth)",
        remaining,
        index=remaining.index("expert") if "expert" in remaining else 0,
        key="rep_cal_y",
    )

    a = df[df["annotated_by"] == src_x][["drop_id", "display_name", "maxn"]].rename(
        columns={"maxn": "x"}
    )
    b = df[df["annotated_by"] == src_y][["drop_id", "display_name", "maxn"]].rename(
        columns={"maxn": "y"}
    )
    merged = a.merge(b, on=["drop_id", "display_name"])
    if merged.empty:
        st.info(f"No deployments have both **{src_x}** and **{src_y}** annotations.")
        return

    n = len(merged)
    x_vals, y_vals = merged["x"].astype(float), merged["y"].astype(float)
    if x_vals.std() > 0:
        denom = (x_vals**2).sum()
        slope = (x_vals * y_vals).sum() / denom if denom > 0 else 0
        # R² of the SAME through-origin fit the slope comes from (uncentred
        # total sum of squares), not Pearson r² — mixing the two put an
        # ordinary-regression number next to an origin-fit slope, so the pair
        # did not describe one model.
        ss_tot = (y_vals**2).sum()
        resid = y_vals - slope * x_vals
        r2 = 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else float("nan")
    else:
        r2, slope = float("nan"), float("nan")

    bias = (y_vals - x_vals).mean()
    mae = (y_vals - x_vals).abs().mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Points compared", n)
    m2.metric(
        "R² (origin fit)",
        f"{r2:.3f}",
        help="Variance explained by the through-origin line the slope "
        "describes. Can go negative when that line fits worse than y = 0.",
    )
    m3.metric("Slope (origin)", f"{slope:.2f}")
    m4.metric(
        "Mean bias (y − x)",
        f"{bias:+.2f}",
        help="Positive = x-source underestimates; negative = overestimates.",
    )
    m5.metric("Mean abs error", f"{mae:.2f}")

    fig = px.scatter(
        merged,
        x="x",
        y="y",
        color="display_name",
        hover_data={"drop_id": True, "x": True, "y": True, "display_name": False},
        labels={
            "x": f"MaxN, {src_x}",
            "y": f"MaxN, {src_y}",
            "display_name": "Species",
        },
        height=520,
    )
    fig.update_traces(
        marker={
            "size": 9,
            "opacity": 0.7,
            "line": {"width": 0.5, "color": MARKER_OUTLINE},
        }
    )

    max_v = max(x_vals.max(), y_vals.max()) + 1
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_v,
        y1=max_v,
        line={"dash": "dash", "color": REFERENCE_LINE, "width": 1.5},
    )
    fig.add_annotation(
        x=max_v,
        y=max_v,
        text="  1:1",
        showarrow=False,
        xanchor="left",
        font={"color": REFERENCE_LINE},
    )
    if slope == slope:
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_v,
            y1=max_v * slope,
            line={"color": MODEL_LINE, "width": 2},
        )
        fig.add_annotation(
            x=max_v,
            y=max_v * slope,
            text=f"  fit (b={slope:.2f})",
            showarrow=False,
            xanchor="left",
            font={"color": MODEL_LINE},
        )

    style(
        fig,
        legend={"orientation": "v", "x": 1.02, "y": 1},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor=PLOT_BG_WHITE,
    )
    fig.update_xaxes(
        gridcolor=GRID_ON_WHITE, zeroline=True, zerolinecolor=ZERO_LINE_ON_WHITE
    )
    fig.update_yaxes(
        gridcolor=GRID_ON_WHITE, zeroline=True, zerolinecolor=ZERO_LINE_ON_WHITE
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Worst-calibrated points"):
        merged["abs_diff"] = (merged["y"] - merged["x"]).abs()
        # Relative error: |Δ| / max(x, y). Symmetric and bounded in [0, 1].
        # A 23-vs-25 mismatch (9% rel) ranks below a 1-vs-3 mismatch (67% rel),
        # the latter is a much bigger calibration problem proportionally even
        # though the raw difference is smaller.
        denom = merged[["x", "y"]].max(axis=1).replace(0, np.nan)
        merged["rel_diff_pct"] = (merged["abs_diff"] / denom * 100).round(1).fillna(0)
        st.dataframe(
            merged.nlargest(15, "rel_diff_pct").rename(
                columns={
                    "display_name": "Species",
                    "x": f"MaxN ({src_x})",
                    "y": f"MaxN ({src_y})",
                    "abs_diff": "|Δ|",
                    "rel_diff_pct": "% diff",
                }
            ),
            hide_index=True,
            column_config={
                "% diff": st.column_config.NumberColumn(
                    "% diff",
                    format="%.1f%%",
                    help="|MaxN_x − MaxN_y| / max(MaxN_x, MaxN_y), "
                    "relative error, bounded in 0–100%.",
                ),
            },
        )
