"""Annotations view for the DOC reporting page.

Who annotated what, and whether the sources agree.

Three sources write annotations: the model (`annotated_by` = a model name),
Zooniverse volunteers (`citsci`) and experts (`expert`), the latter covering both legacy annotations and BIIGLE review. There is no merge
step: where expert annotations exist they win, and the other two stay for
auditing.

The comparison panels need deployments carrying *both* ML and expert
annotations, which is a much smaller set than either source alone. They say so
rather than implying a sample size they do not have.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from ecology_data import source_bucket
from theme import SOURCE_COLORS

from .charting import style
from .charts.annotations import render_source_calibration
from .data import (
    ALL_SOURCES,
    INTERVAL_COUNT,
    deployment_maxn,
    experiments_frame,
)
from .layout import chips, section


def _paired(ann: pd.DataFrame) -> pd.DataFrame:
    """Per-deployment ML and expert totals, for deployments carrying both.

    The peak-then-sum lives in `data.deployment_maxn`, keyed by source as well
    as deployment, so this view and the Species view cannot disagree about what
    a deployment's MaxN is.
    """
    tagged = ann.assign(Source=source_bucket(ann["annotated_by"]))
    per_dep = deployment_maxn(tagged, ("Source",)).unstack(fill_value=0)
    if not {"ML", "Expert"} <= set(per_dep.columns):
        return pd.DataFrame(columns=["ML", "Expert"])
    # Only the two compared columns: unstack also yields CitSci, which would
    # otherwise ride along into the scatter and the gap table as zeros.
    both = per_dep[["ML", "Expert"]]
    return both[(both["ML"] > 0) & (both["Expert"] > 0)]


# Review order in the pipeline: the model runs first, volunteers check after,
# the expert last. Each pair compares an earlier source against the later one
# that re-reviewed the same deployment.
_REVIEW_PAIRS = (("ML", "CitSci"), ("CitSci", "Expert"), ("ML", "Expert"))


def _source_disagreement(ann: pd.DataFrame) -> pd.DataFrame:
    """Per-deployment total MaxN for every source pair that both reviewed it.

    "Reviewed" comes from row presence — a null-species row is a review that
    found nothing — while the totals come from named rows only. That keeps the
    two states apart: a source that reviewed and saw nothing compares as 0,
    a source that never looked is absent from the pair entirely.
    """
    tagged = ann.assign(Source=source_bucket(ann["annotated_by"]))
    reviewed = tagged.groupby(["drop_id", "Source"]).size().rename("rows")
    totals = deployment_maxn(tagged, ("Source",)).rename("total")
    wide = (
        reviewed.to_frame().join(totals).fillna({"total": 0})["total"].unstack("Source")
    )

    frames = []
    for earlier, later in _REVIEW_PAIRS:
        if not {earlier, later} <= set(wide.columns):
            continue
        both = wide[[earlier, later]].dropna()
        if both.empty:
            continue
        pair = both.rename(columns={earlier: "earlier", later: "later"}).reset_index()
        pair["Pair"] = f"{earlier} → {later}"
        frames.append(pair)
    if not frames:
        return pd.DataFrame(
            columns=[
                "drop_id",
                "earlier",
                "later",
                "Pair",
                "gap",
                "severity",
                "empty_vs_seen",
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    out["gap"] = (out["earlier"] - out["later"]).abs()
    # The UoA comparison's severity: squared gap scaled by the larger total,
    # so 0 vs 8 outranks 30 vs 38 even though the gaps match.
    out["severity"] = out["gap"] ** 2 / out[["earlier", "later"]].max(axis=1).clip(
        lower=1
    )
    out["empty_vs_seen"] = ((out["earlier"] == 0) ^ (out["later"] == 0)) & (
        out[["earlier", "later"]].max(axis=1) >= 3
    )
    return out


def render(ctx: dict) -> None:
    """Render the Annotations view from the shared context."""
    ann = ctx["annotations"]
    dep = ctx["deployments"]
    if ann.empty:
        st.warning("No annotations match the current filters.")
        return

    chips(
        [
            "Deployments by source",
            "Species observed",
            "Source disagreement",
            "ML against expert",
            "Calibration",
        ]
    )

    tagged = ann.assign(Source=source_bucket(ann["annotated_by"]))
    by_source = tagged.groupby("Source")["drop_id"].nunique()

    kpis = st.columns(5)
    kpis[0].metric(
        "Annotations",
        f"{len(ann):,}",
        help="Rows in the annotations database: one per species observation, "
        "from any source.",
    )
    kpis[1].metric(
        "Deployments annotated",
        f"{ann['drop_id'].nunique():,}",
        help=f"Distinct deployments carrying at least one annotation, out of "
        f"{len(dep):,} held.",
    )
    kpis[2].metric(
        "Expert",
        f"{int(by_source.get('Expert', 0)):,}",
        help="Deployments with expert annotations, whether legacy or from BIIGLE. Expert wins wherever "
        "sources disagree, so this is the number the reporting rests on.",
    )
    kpis[3].metric(
        "ML",
        f"{int(by_source.get('ML', 0)):,}",
        help="Deployments with model annotations, counting any model version.",
    )
    kpis[4].metric(
        "Species",
        f"{ann['scientific_name'].nunique():,}",
        help="Distinct species names recorded across all sources.",
    )

    st.divider()

    # ── Coverage by source ───────────────────────────────────────────────────
    left, right = st.columns([1, 1.4])

    with left:
        section("Deployments by source")
        st.caption(
            "Sources overlap: a deployment can carry more than one, so these "
            "do not sum to the total annotated."
        )
        counts = by_source.rename_axis("Source").reset_index(name="Deployments")
        fig = px.bar(
            counts.sort_values("Deployments"),
            x="Deployments",
            y="Source",
            orientation="h",
            text="Deployments",
            color="Source",
            color_discrete_map={
                "Expert": SOURCE_COLORS["expert"],
                "CitSci": SOURCE_COLORS["citsci"],
                "ML": SOURCE_COLORS["ml"],
            },
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        style(fig, height=240, legend=False)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, key="ann_sources")

    with right:
        section("Species observed")
        st.caption(
            "**Seen in** is frequency of occurrence, and it is the more robust "
            "number: a total can come from one lucky deployment, a frequency "
            "cannot. **Best drop** is the highest MaxN the species reached in "
            "any single deployment."
        )
        n_deps = ann["drop_id"].nunique()
        species = (
            # `max` is right here: the best a species reached in any single
            # interval of any deployment is its peak, which is what MaxN means.
            ann.groupby("scientific_name")
            .agg(best=(INTERVAL_COUNT, "max"), seen=("drop_id", "nunique"))
            .reset_index()
            .rename(columns={"scientific_name": "Species"})
        )
        species["Best drop"] = species["best"].astype(int)
        species["Seen in"] = species["seen"].astype(str) + f" / {n_deps}"
        species["Frequency"] = species["seen"] / max(n_deps, 1) * 100
        species = species.sort_values("Frequency", ascending=False)
        st.dataframe(
            species[["Species", "Best drop", "Seen in", "Frequency"]],
            hide_index=True,
            width="stretch",
            height=300,
            column_config={
                "Best drop": st.column_config.NumberColumn(
                    "Best drop",
                    help="Highest MaxN this species reached in any single "
                    "deployment.",
                ),
                "Seen in": st.column_config.TextColumn(
                    "Seen in",
                    help="Deployments recording this species at all, out of "
                    "those with any annotation.",
                ),
                "Frequency": st.column_config.ProgressColumn(
                    "Frequency",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                    help="Share of annotated deployments where the species was "
                    "present. More robust than a total: a total can be one "
                    "lucky drop, a frequency cannot.",
                ),
            },
        )

    st.divider()

    # ── Source disagreement ──────────────────────────────────────────────────
    section("Source disagreement")
    st.caption(
        "The pipeline reviews in order — ML first, volunteers after, the expert "
        "last — so each pair compares an earlier source against the later one "
        "that re-reviewed the same deployment. Totals are class-agnostic (each "
        "source's species peaks, summed), because the binary ML model cannot be "
        "compared species by species. **Empty vs seen** is the sharpest "
        "conflict: one source reviewed and recorded nothing while the other "
        "counted 3 or more — either a missed review or a wrong one, and under "
        "expert-wins a wrong empty silently suppresses the other source's data."
    )
    # Source-unfiltered on purpose, like the ML-against-expert panel below:
    # comparing sources needs all of them present.
    dis = _source_disagreement(ctx["annotations_all_sources"])
    flagged = dis[(dis["gap"] >= 2) | dis["empty_vs_seen"]].sort_values(
        "severity", ascending=False
    )

    d = st.columns(3)
    d[0].metric(
        "Pairs compared",
        f"{len(dis):,}",
        help="Deployment × source-pair combinations where both sources "
        "reviewed the deployment.",
    )
    d[1].metric(
        "Disagreements",
        f"{len(flagged):,}",
        help="Pairs whose totals differ by 2 or more, or empty-vs-seen " "conflicts.",
    )
    d[2].metric(
        "Empty vs seen",
        f"{int(dis['empty_vs_seen'].sum()) if not dis.empty else 0:,}",
        help="One source recorded nothing, the other counted 3 or more. "
        "These deserve a second review first.",
    )

    if dis.empty:
        st.info(
            "**No deployment has been reviewed by two sources yet**, so there "
            "is nothing to compare. This panel fills in as coverage overlaps."
        )
    elif flagged.empty:
        st.success(
            "Every deployment reviewed by two sources got comparable totals — "
            "no disagreements at the current filters."
        )
    else:
        st.dataframe(
            flagged[
                [
                    "Pair",
                    "drop_id",
                    "earlier",
                    "later",
                    "gap",
                    "severity",
                    "empty_vs_seen",
                ]
            ],
            hide_index=True,
            width="stretch",
            height=min(400, 60 + 35 * len(flagged)),
            column_config={
                "Pair": st.column_config.TextColumn("Pair", width="small"),
                "drop_id": st.column_config.TextColumn("DropID", width="large"),
                "earlier": st.column_config.NumberColumn(
                    "Earlier total",
                    format="%.0f",
                    help="Total MaxN from the pair's earlier source.",
                ),
                "later": st.column_config.NumberColumn(
                    "Later total",
                    format="%.0f",
                    help="Total MaxN from the pair's later source.",
                ),
                "gap": st.column_config.NumberColumn("Gap", format="%.0f"),
                "severity": st.column_config.NumberColumn(
                    "Severity",
                    format="%.1f",
                    help="Squared gap over the larger total: 0 vs 8 outranks "
                    "30 vs 38 even though the gaps match.",
                ),
                "empty_vs_seen": st.column_config.CheckboxColumn(
                    "Empty vs seen",
                    help="One side recorded nothing while the other counted "
                    "3 or more.",
                ),
            },
        )

    st.divider()

    # ── ML against expert ────────────────────────────────────────────────────
    section("ML against expert")
    # The source-UNfiltered frame, on purpose: this panel compares two sources,
    # so it must see both. Under the shared Source filter one side has already
    # been dropped. Best available removes ML wherever expert covered the
    # drop, and any single-source choice removes the other entirely.
    paired = _paired(ctx["annotations_all_sources"])
    if ctx["source"] != ALL_SOURCES:
        st.caption(
            "The Source filter is deliberately ignored here, comparing ML "
            "against expert needs both sources present."
        )
    if paired.empty:
        st.info(
            "**No deployments carry both ML and expert annotations**, so there "
            "is nothing to compare. The two sources currently cover different "
            "deployments. This panel fills in as the overlap grows."
        )
        return

    st.caption(
        f"Only the **{len(paired)} deployments carrying both** can be compared. "
        "That is far too few to state model accuracy from, it shows the "
        "comparison working, not how good the model is. A deployment's total is "
        "each species' peak count, summed across species."
    )

    corr = paired["Expert"].corr(paired["ML"]) if len(paired) > 2 else float("nan")
    under = (paired["ML"] < paired["Expert"]).mean() * 100

    m = st.columns(3)
    m[0].metric(
        "Paired deployments",
        f"{len(paired):,}",
        help="Deployments carrying annotations from both ML and expert.",
    )
    m[1].metric(
        "Agreement",
        "—" if pd.isna(corr) else f"{corr:.2f}",
        help="Pearson correlation between the ML and expert totals for the "
        "same deployment. Needs at least three paired deployments.",
    )
    m[2].metric(
        "ML reads low",
        f"{under:.0f}%",
        help="Share of paired deployments where ML counted fewer fish than the "
        "expert. A high rate with good correlation means the model ranks "
        "deployments correctly but under-counts.",
    )

    scatter, breakdown = st.columns([1, 1])

    with scatter:
        plot = paired.reset_index()
        fig = px.scatter(
            plot,
            x="Expert",
            y="ML",
            hover_name="drop_id",
            labels={"Expert": "expert total MaxN", "ML": "ML total MaxN"},
        )
        lim = float(max(plot["Expert"].max(), plot["ML"].max())) * 1.1 + 1
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=lim,
            y1=lim,
            line=dict(dash="dash", color="#B9C4D6", width=1.5),
        )
        fig.add_annotation(
            x=lim * 0.72,
            y=lim * 0.82,
            text="agree",
            showarrow=False,
            font=dict(color="#8794A8"),
        )
        fig.update_traces(marker=dict(size=11, color=SOURCE_COLORS["ml"], opacity=0.85))
        style(fig, height=330)
        st.plotly_chart(fig, key="ann_scatter")
        st.caption(
            "Points below the dashed line are deployments where ML counted "
            "fewer fish than the expert."
        )

    with breakdown:
        st.markdown("**Where agreement breaks down**")
        gap = paired.assign(gap=(paired["Expert"] - paired["ML"]).abs())
        gap = gap.sort_values("gap", ascending=False).head(8)
        for row in gap.itertuples():
            colour = "#D9603B" if row.gap >= 10 else "#E8A33D"
            direction = "ML low" if row.ML < row.Expert else "ML high"
            st.markdown(
                f'<div style="display:flex;gap:.6rem;align-items:flex-start;'
                f'padding:.35rem 0;border-bottom:1px solid #F0F3F8">'
                f'<div style="width:8px;height:8px;border-radius:50%;'
                f'background:{colour};margin-top:.4rem;flex:0 0 8px"></div>'
                f'<div style="font-size:.8rem;color:#2B3A55">{row.Index}<br>'
                f'<span style="color:#7A879C">expert {int(row.Expert)} · '
                f"ML {int(row.ML)} · {direction}</span></div>"
                f'<div style="margin-left:auto;font-size:.75rem;color:#94A0B4">'
                f"{int(row.gap)}</div></div>",
                unsafe_allow_html=True,
            )

    # ── Ported from the Experiments page ─────────────────────────────────────
    #
    # When in a deployment something was seen, rather than how much of it there
    # was. the retired Experiments sandbox drew a single timing here.
    #
    # The UNFILTERED annotations, like the ML-against-expert panel above: a
    # calibration needs both sources present, and the Source filter has already
    # dropped one of them.
    st.divider()
    render_source_calibration(experiments_frame(ctx["annotations_all_sources"]))
