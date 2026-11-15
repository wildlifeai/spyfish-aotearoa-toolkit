"""Mussel farm vs control from the ML counts, day by day, checked against the
hand counts where they exist. The render body of the Mussel Insights
"Farm vs control (ML)" view.

Every 6-minute interval in ``Annotations_CLEANED_v2.csv`` (including the
placeholder rows of days nobody counted, all of 21 Jan) is measured in the
ML raw CSVs in exactly its window, with the frozen rule: detector confidence,
then the 2-s persistence rule. Per deployment the mean interval MaxN, then
farm minus control per pair (same day, site, depth, time of day). The pairing
and the summary are uoa/scripts/two_stage/farm_pairs.py, the same code the
farm_vs_control.py report runs, so the view and the report cannot disagree.

Days in TRAINING_DAYS are footage the models trained on and read flattering;
every other day is footage they never saw.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from theme import PLOT_LAYOUT

from uoa.app.data import (
    DEFAULT_PERSISTENCE_SEC,
    DEPTH_LABEL,
    SITE_NAMES,
    TRAINING_DAYS,
    active_species_map,
    chapter_from_symlink,
    default_conf,
    load_expert_rows,
    load_ml_raw,
    maxn_in_window,
    window_coverage,
)
from uoa.scripts.two_stage.farm_pairs import (
    deployment_summary,
    detection_ratio,
    pair_table,
    summary,
)

ALL_FISH = "All fish"
# What the 18/19 Jan check (maxn_compare.py, 2026-11-11) says ML can be read
# for. Sweep is inflated by boxes on the seabed and the farm lines; snapper and
# jack mackerel are too thin in the training data.
READABLE = {ALL_FISH, "Parore"}
FULL_COVERAGE = 0.999
TREAT_COLORS = {"farm": "#1B7F4B", "control": "#B98A4B"}
DAY_COLORS = {
    "18 Jan": "#0072B2",
    "19 Jan": "#E69F00",
    "20 Jan": "#CC79A7",
    "21 Jan": "#009E73",
}
# A count can be scaled up only where ML sees the species whenever the
# observers did; below this share of intervals it misses fish outright, and no
# factor brings those back.
CORRECTABLE_RECALL = 0.9


def day_label(day: str) -> str:
    seen = "trained on" if day in TRAINING_DAYS else "unseen by the model"
    return f"{int(day[6:])} Jan ({seen})"


@st.cache_data(show_spinner="Measuring ML MaxN in every 6-minute interval…")
def interval_table(conf: float, persistence_sec: float) -> pd.DataFrame:
    """One row per interval x category: ML MaxN, expert MaxN (NaN where nobody
    counted). Only intervals whose whole window has ML raw CSVs."""
    expert, maxn_cols = load_expert_rows()
    raw_by_key, _ = load_ml_raw()
    deps = {dep for dep, _ in raw_by_key}
    classes = set()
    for df in raw_by_key.values():
        if df is not None and not df.empty:
            classes.update(df["class"].dropna().unique())
    species = active_species_map(classes)  # {expert column: ML class}
    categories = [(ALL_FISH, None, maxn_cols)] + [
        (col.replace(" MaxN", ""), cls, [col]) for col, cls in species.items()
    ]

    # the expert side of each category, summed over its columns, once
    expert_of = {
        name: expert[cols].fillna(0).sum(axis=1) for name, _, cols in categories
    }

    rows = []
    for idx, r in expert[expert["deployment_id"].isin(deps)].iterrows():
        dep = r["deployment_id"]
        chap = chapter_from_symlink(r.get("video_symlink"))
        start = r.get("start_sec_in_chapter")
        if chap is None or pd.isna(start):
            continue
        start = float(start)
        if window_coverage(raw_by_key, dep, chap, start) < FULL_COVERAGE:
            continue
        counted = str(r.get("unannotated")).strip().lower() != "true"
        for name, cls, _ in categories:
            rows.append(
                {
                    "deployment_id": dep,
                    "category": name,
                    "ml": maxn_in_window(
                        raw_by_key,
                        dep,
                        chap,
                        start,
                        conf,
                        class_name=cls,
                        persistence_sec=persistence_sec,
                    ),
                    "expert": expert_of[name].at[idx] if counted else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def render_body(key_prefix: str = "fc") -> None:
    conf = default_conf()
    iv = interval_table(conf, DEFAULT_PERSISTENCE_SEC)
    if iv.empty:
        st.warning(
            "No ML raw CSVs that cover an interval. They are read from "
            "`deployment_data/<deployment>/annotations/` (synced from S3)."
        )
        return

    cats = list(dict.fromkeys(iv["category"]))
    cat = st.radio(
        "Count",
        cats,
        horizontal=True,
        key=f"{key_prefix}_cat",
        format_func=lambda c: c if c in READABLE else f"{c} (not reliable)",
    )
    if cat not in READABLE:
        st.warning(
            f"The 18/19 Jan check says ML {cat.lower()} counts cannot be read yet: "
            "sweep is inflated by boxes on the seabed and the farm lines, snapper "
            "and jack mackerel are too thin in the training data. Shown for "
            "inspection only."
        )

    dep = deployment_summary(iv[iv["category"] == cat])
    pairs = pair_table(dep)
    summ = summary(pairs).set_index("day")
    days = sorted(dep["day"].unique())

    # ── One card per day ────────────────────────────────────────────────────
    cols = st.columns(len(days))
    for col, day in zip(cols, days):
        with col, st.container(border=True):
            st.markdown(f"**{day_label(day)}**")
            if day not in summ.index:
                st.caption("No farm/control pairs on this day.")
                continue
            s = summ.loc[day]
            st.metric(
                "ML: farm vs control",
                f"{s['ml_farm_mean']:.1f} vs {s['ml_control_mean']:.1f}",
                help="Mean MaxN per 6-minute interval, averaged over the paired "
                "farm and control deployments.",
            )
            line = f"Farm higher in {s['ml_farm_higher']} pairs"
            if pd.notna(s.get("expert_farm_higher")):
                line += (
                    f" · hand counts {s['expert_farm_mean']:.1f} vs "
                    f"{s['expert_control_mean']:.1f}, farm higher in "
                    f"{s['expert_farm_higher']}"
                )
                if pd.notna(s.get("same_direction")):
                    line += f" · ML picks the same side in {s['same_direction']}"
            else:
                line += " · no hand counts on this day"
            st.caption(line)

    # ── Deployments, farm vs control, ML beside hand counts ─────────────────
    long = pd.concat(
        [
            dep.assign(source="ML", value=dep["ml_mean"]),
            dep.assign(source="Hand count", value=dep["expert_mean"]),
        ]
    ).dropna(subset=["value"])
    long["Day"] = long["day"].map(day_label)
    long["Depth"] = long["depth"].map(DEPTH_LABEL)
    long["Site"] = long["site"].map(lambda s: SITE_NAMES.get(s, s))
    order = {"Day": [day_label(d) for d in days], "source": ["ML", "Hand count"]}

    with st.container(border=True):
        st.markdown(
            f"**{cat}: every deployment** "
            "<span style='color:#66748C'>(mean MaxN per 6-minute interval; "
            "farms without a control, at Rat Island and Whanganui, included)</span>",
            unsafe_allow_html=True,
        )
        fig = px.strip(
            long,
            x="Day",
            y="value",
            color="treatment",
            facet_col="source",
            category_orders=order,
            color_discrete_map=TREAT_COLORS,
            hover_data=["deployment_id", "Depth", "Site"],
        )
        fig.update_traces(marker=dict(size=7, opacity=0.75))
        fig.update_layout(**PLOT_LAYOUT, height=360, legend_title_text="")
        fig.update_yaxes(title="mean MaxN per interval", col=1)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, key=f"{key_prefix}_strip")

    with st.container(border=True):
        st.markdown(
            f"**{cat}: surface vs benthic** "
            "<span style='color:#66748C'>(ML, every day)</span>",
            unsafe_allow_html=True,
        )
        ml_only = long[long["source"] == "ML"]
        fig = px.box(
            ml_only,
            x="Depth",
            y="value",
            color="treatment",
            facet_col="Day",
            points="all",
            category_orders=order,
            color_discrete_map=TREAT_COLORS,
            hover_data=["deployment_id", "Site"],
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7), line=dict(width=1.5))
        fig.update_layout(**PLOT_LAYOUT, height=320, legend_title_text="")
        fig.update_yaxes(title="mean MaxN per interval", col=1)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, key=f"{key_prefix}_depth")

    render_reading_guide(iv, dep, pairs, cat, key_prefix)

    with st.expander("Every pair"):
        show = pairs.rename(
            columns={
                "deployment_id_farm": "farm",
                "deployment_id_control": "control",
                "ml_mean_farm": "ML farm",
                "ml_mean_control": "ML control",
                "expert_mean_farm": "hand farm",
                "expert_mean_control": "hand control",
            }
        )[
            [
                "day",
                "farm",
                "control",
                "ML farm",
                "ML control",
                "hand farm",
                "hand control",
            ]
        ]
        st.dataframe(show.round(2), hide_index=True, width="stretch")

    st.caption(
        f"ML MaxN is measured in each observer interval's own 6-minute window, "
        f"detections at confidence ≥ {conf:g}, a fish counting only if it "
        f"stays {DEFAULT_PERSISTENCE_SEC:g} s (the frozen UoA rule). Intervals "
        "whose window the ML run does not fully cover are left out. ML counts about "
        "half the fish the observers count, so read direction and ranking, not "
        "absolute numbers. Pairs share day, site, depth and time of day; only Esk "
        "Point and Motukopake have controls."
    )


def short_day(day: str) -> str:
    return f"{int(day[6:])} Jan"


def render_reading_guide(iv, dep, pairs, cat, key_prefix) -> None:
    """Three graphs for what the ML numbers can and cannot say: the farm effect
    against the observers', the share of fish ML finds, and the unseen days'
    farm effect corrected by that share."""
    st.markdown("#### How far the ML numbers can be read")
    counted = pairs.dropna(subset=["expert_mean_diff"])
    sub = iv[iv["category"] == cat]
    trained = sub[sub["deployment_id"].str.split("_").str[2].isin(TRAINING_DAYS)]
    ratio = detection_ratio(trained).set_index("depth")
    c1, c2 = st.columns(2)

    # 1. The farm effect, ML against hand counts, one point per pair
    with c1, st.container(border=True):
        st.markdown(
            "**1. Same direction, about half the size** "
            "<span style='color:#66748C'>(farm minus control, one point per pair)</span>",
            unsafe_allow_html=True,
        )
        if counted.empty:
            st.info("No hand-counted pairs for this count.")
        else:
            x, y = counted["expert_mean_diff"], counted["ml_mean_diff"]
            both = counted[(x != 0) & (y != 0)]
            same = int(
                (
                    np.sign(both["expert_mean_diff"]) == np.sign(both["ml_mean_diff"])
                ).sum()
            )
            slope = float((x * y).sum() / (x * x).sum()) if (x * x).sum() else np.nan
            d = counted.assign(
                Day=counted["day"].map(short_day),
                Depth=counted["depth"].map(DEPTH_LABEL),
                pair=counted["site"]
                + " "
                + counted["depth"]
                + " "
                + counted["time_of_day"],
            )
            fig = px.scatter(
                d,
                x="expert_mean_diff",
                y="ml_mean_diff",
                color="Day",
                symbol="Depth",
                color_discrete_map=DAY_COLORS,
                hover_data=["pair", "deployment_id_farm", "deployment_id_control"],
                labels={
                    "expert_mean_diff": "hand count: farm − control",
                    "ml_mean_diff": "ML: farm − control",
                },
            )
            lim = float(max(x.abs().max(), y.abs().max(), 1)) * 1.1
            fig.add_shape(
                type="line",
                x0=-lim,
                y0=-lim,
                x1=lim,
                y1=lim,
                line=dict(dash="dash", color="grey"),
            )
            fig.add_shape(
                type="line",
                x0=-lim,
                y0=-lim * slope,
                x1=lim,
                y1=lim * slope,
                line=dict(color="#16233F", width=1),
            )
            fig.add_hline(y=0, line=dict(color="#B9C4D6", width=1))
            fig.add_vline(x=0, line=dict(color="#B9C4D6", width=1))
            fig.update_traces(marker=dict(size=10, opacity=0.85))
            fig.update_layout(**PLOT_LAYOUT, height=340, legend_title_text="")
            st.plotly_chart(fig, key=f"{key_prefix}_effect")
            st.caption(
                f"Top right = both say the farm had more. ML picks the observers' side "
                f"in **{same} of {len(both)}** pairs where both saw a difference. "
                f"Solid line: ML ≈ **{slope:.2f} ×** the observers' difference; dashed: "
                "equal. Hand counts exist on the trained days only."
            )

    # 2. The share of the fish ML finds, per depth
    with c2, st.container(border=True):
        st.markdown(
            "**2. How many of the fish ML finds** "
            "<span style='color:#66748C'>(hand-counted intervals, trained days)</span>",
            unsafe_allow_html=True,
        )
        if ratio.empty:
            st.info("No hand-counted fish for this count.")
        else:
            r = ratio.reset_index()
            r["Depth"] = r["depth"].map(DEPTH_LABEL)
            r["correctable"] = r["presence_recall"] >= CORRECTABLE_RECALL
            fig = go.Figure(
                go.Bar(
                    x=r["Depth"],
                    y=r["ratio"] * 100,
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=(r["hi"] - r["ratio"]) * 100,
                        arrayminus=(r["ratio"] - r["lo"]) * 100,
                        color="#16233F",
                    ),
                    marker_color=[
                        "#1B7F4B" if c else "#B9C4D6" for c in r["correctable"]
                    ],
                    text=[
                        f"{v:.0%}<br><sup>sees it in {p:.0%} of intervals</sup>"
                        for v, p in zip(r["ratio"], r["presence_recall"])
                    ],
                    textposition="inside",
                )
            )
            fig.add_hline(y=100, line=dict(dash="dash", color="grey"))
            fig.update_layout(**PLOT_LAYOUT, height=340, showlegend=False)
            fig.update_yaxes(
                title="ML count as % of the observers'", rangemode="tozero"
            )
            st.plotly_chart(fig, key=f"{key_prefix}_ratio")
            st.caption(
                "Bar: ML MaxN summed over every hand-counted interval, as a share of the "
                "observers' sum; whiskers: 95% range, resampling whole deployments. "
                f"Green = ML saw the species in at least {CORRECTABLE_RECALL:.0%} of the "
                "intervals the observers did, so a factor can correct the count; grey = it "
                "misses fish outright, and no factor brings those back."
            )

    # 3. The unseen days: the farm effect as ML saw it, and corrected
    unseen = pairs[~pairs["day"].isin(TRAINING_DAYS)]
    if unseen.empty:
        return
    with st.container(border=True):
        st.markdown(
            "**3. The unseen days: farm minus control, ML and corrected** "
            "<span style='color:#66748C'>(one bar pair per farm/control pair)</span>",
            unsafe_allow_html=True,
        )
        rows = []
        for _, pr in unseen.iterrows():
            label = f"{short_day(pr['day'])} · {SITE_NAMES.get(pr['site'], pr['site'])} · {DEPTH_LABEL.get(pr['depth'], pr['depth'])} · {pr['time_of_day']}"
            diff = pr["ml_mean_diff"]
            rows.append(
                {"pair": label, "kind": "ML", "value": diff, "lo": diff, "hi": diff}
            )
            st_ = ratio.loc[pr["depth"]] if pr["depth"] in ratio.index else None
            if (
                st_ is not None
                and st_["presence_recall"] >= CORRECTABLE_RECALL
                and st_["ratio"] > 0
            ):
                rows.append(
                    {
                        "pair": label,
                        "kind": "corrected",
                        "value": diff / st_["ratio"],
                        "lo": diff / st_["hi"],
                        "hi": diff / st_["lo"],
                    }
                )
        b = pd.DataFrame(rows)
        fig = go.Figure()
        for kind, color in (("ML", "#B9C4D6"), ("corrected", "#1B7F4B")):
            g = b[b["kind"] == kind]
            if g.empty:
                continue
            fig.add_bar(
                y=g["pair"],
                x=g["value"],
                name=kind,
                orientation="h",
                marker_color=color,
                error_x=(
                    dict(
                        type="data",
                        symmetric=False,
                        array=g["hi"] - g["value"],
                        arrayminus=g["value"] - g["lo"],
                        color="#16233F",
                    )
                    if kind == "corrected"
                    else None
                ),
            )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=60 + 34 * b["pair"].nunique(),
            barmode="group",
            legend=dict(orientation="h", y=1.08, x=0),
        )
        fig.update_xaxes(title="farm − control, mean MaxN per interval")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, key=f"{key_prefix}_corrected")
        st.caption(
            "Corrected = the ML difference divided by the share of fish ML finds at that "
            "depth (graph 2), the whisker its 95% range. Only depths where ML sees the "
            "species reliably are corrected. Provisional: the share is measured on the "
            "days the models trained on, where ML looks better than it is; 20 Jan, "
            "hand-counted and never trained on, is where it gets tested."
        )
