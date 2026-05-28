"""
Programme health dashboard — high-level view for DOC and programme leads.

Answers: how is the programme going overall, which surveys have problems,
where are we in the annotation pipeline?
"""

import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import render_contact_note, render_sidebar_refresh, sync_db_if_needed

from spyfish.config.base import CitSciStatus, ExpertStatus, MlStatus, VideoPresence
from spyfish.config.wrapper import config
from spyfish.utils import extract_survey_id

st.set_page_config(page_title="Programme Health", page_icon="📈", layout="wide")
st.title("📈 Programme Health")
render_contact_note()
render_sidebar_refresh()

# ── Data loading ──────────────────────────────────────────────────────────────

_PIPELINE_ORDER = [
    "expert_complete",
    "expert_uploaded",
    "citsci_complete",
    "ml_complete",
    "ml_running",
    "ml_ready",
    "ml_pending",
    "excluded",
    "metadata_error",
    "validation_error",
]


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame | None:
    try:
        with sqlite3.connect(config.db_path) as conn:
            df = pd.read_sql("SELECT * FROM deployments", conn)
            sites = pd.read_sql("SELECT site_id, protection_status FROM sites", conn)
    except Exception as e:
        st.error(f"Could not load database: {e}")
        return None

    if df.empty:
        return df

    df["is_bad_deployment"] = df["is_bad_deployment"].astype(bool)
    df["survey_id"] = extract_survey_id(df["drop_id"])

    # Parse year and reserve code from drop_id (format: RESERVE_YYYYMMDD_BUV_…)
    parts = df["drop_id"].str.split("_", expand=True)
    df["survey_year"] = pd.to_numeric(parts[1].str[:4], errors="coerce")
    df["reserve_code"] = parts[0]
    df["site_id"] = parts[3] + "_" + parts[4]

    df = df.merge(sites, on="site_id", how="left")
    df["protection_status"] = df["protection_status"].fillna("unknown")

    for col in ("expert_annotations", "ml_annotations", "citsci_annotations"):
        df[col] = df[col].fillna(0).astype(int)

    df["ok"] = df["ingest_status"] == "ok"
    df["has_video"] = df["video_presence"] == VideoPresence.PRESENT
    df["ml_done"] = df["ml_status"] == MlStatus.COMPLETE
    df["citsci_done"] = df["citsci_status"] == CitSciStatus.COMPLETE
    df["expert_done"] = df["expert_status"] == ExpertStatus.COMPLETE
    df["fully_annotated"] = df["expert_done"]

    return df


# Sync the DB from S3 before loading. Kept OUT of the cached load_data() so the
# sync isn't skipped when load_data returns a cached frame; sync_db_if_needed is
# itself cached (ttl=None) so this still runs only once per session.
sync_db_if_needed()
df = load_data()
if df is None or df.empty:
    st.info("No deployments in the database yet.")
    st.stop()

ok = df[df["ok"]]

# ── KPI row ───────────────────────────────────────────────────────────────────

st.header("Summary")
k = st.columns(7)
k[0].metric("Total deployments", len(df))
k[1].metric("Surveys", df["survey_id"].nunique())
k[2].metric("Bad / excluded", int(df["is_bad_deployment"].sum() + (~df["ok"]).sum()))
k[3].metric("Videos present", int(df["has_video"].sum()))
# Data-presence counts: a drop with annotations from a source has, by
# definition, made it through the earlier pipeline stages for that source.
# No ingest_status filter — see funnel section below for the same principle.
k[4].metric("ML complete", int(df["ml_done"].sum()))
k[5].metric("CitSci complete", int(df["citsci_done"].sum()))
k[6].metric("Expert complete", int(df["expert_done"].sum()))

st.divider()

# ── Row 1: Deployments per year  |  Bad deployments per survey ───────────────

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Deployments per year")
    year_counts = (
        df.groupby(["survey_year", "ingest_status"]).size().reset_index(name="count")
    )
    # Colour: ok = teal, bad/excluded = amber/red
    status_colours = {
        "ok": "#26A69A",
        "excluded": "#FF9800",
        "metadata_error": "#EF5350",
        "validation_error": "#EF5350",
        "removed": "#9E9E9E",
    }
    fig_year = px.bar(
        year_counts.sort_values("survey_year"),
        x="survey_year",
        y="count",
        color="ingest_status",
        color_discrete_map=status_colours,
        labels={
            "survey_year": "Year",
            "count": "Deployments",
            "ingest_status": "Ingest status",
        },
        barmode="stack",
        height=320,
    )
    fig_year.update_layout(
        xaxis={"dtick": 1, "tickformat": "d"},
        legend_title_text="Status",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig_year, use_container_width=True)

with col_b:
    st.subheader("Bad / excluded deployments per survey")
    bad_per_survey = (
        df.groupby("survey_id")
        .agg(
            total=("drop_id", "count"),
            bad=("is_bad_deployment", "sum"),
            not_ok=(
                "ok",
                lambda x: (~x).sum(),
            ),
        )
        .reset_index()
    )
    bad_per_survey["bad_pct"] = (
        bad_per_survey["bad"] / bad_per_survey["total"] * 100
    ).round(1)
    bad_surveys = bad_per_survey[bad_per_survey["bad"] > 0].copy()
    if bad_surveys.empty:
        st.success("No bad deployments recorded.")
    else:
        # Parse RESERVE_YYYYMMDD_BUV → reserve code + chronological order
        sid_parts = bad_surveys["survey_id"].str.split("_", expand=True)
        bad_surveys["reserve"] = sid_parts[0]
        bad_surveys["survey_date"] = pd.to_datetime(
            sid_parts[1], format="%Y%m%d", errors="coerce"
        )
        bad_surveys = bad_surveys.sort_values("survey_date")

        fig_bad = px.bar(
            bad_surveys,
            x="survey_id",
            y="bad_pct",
            color="bad",
            color_continuous_scale="Oranges",
            text="bad_pct",
            hover_data={
                "total": True,
                "bad": True,
                "survey_date": "|%Y-%m-%d",
                "reserve": True,
                "survey_id": False,
            },
            labels={
                "survey_id": "Survey",
                "bad_pct": "% bad deployments",
                "bad": "Count",
                "reserve": "Reserve",
            },
            category_orders={"survey_id": bad_surveys["survey_id"].tolist()},
            height=340,
        )
        fig_bad.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
        # 10% reference line — surveys above this warrant a closer look
        fig_bad.add_hline(
            y=10,
            line_dash="dash",
            line_color="#EF5350",
            line_width=1,
            annotation_text="10% threshold",
            annotation_position="right",
            annotation_font_color="#EF5350",
        )
        fig_bad.update_layout(
            xaxis_tickangle=-40,
            coloraxis_colorbar_title="Bad count",
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
        )
        st.plotly_chart(fig_bad, use_container_width=True)

st.divider()

# ── Row 2: Pipeline summary (left, stacked) | Annotation depth (right, tall) ─
#
# Annotation depth per survey is naturally a long vertical bar list (one bar
# per survey, often 30+). Stacking three smaller charts in the left column —
# Pipeline funnel, Video presence by year, Deployments by protection status —
# fills the height that would otherwise be empty next to it.

col_c, col_d = st.columns(2)

with col_c:
    # ── Pipeline funnel ──────────────────────────────────────────────────────
    st.subheader("Pipeline funnel")
    st.caption(
        "Each stage counts deployments by data presence — a drop with "
        "annotations from a source must have made it through the earlier "
        "stages for that source. Ingest issues (excluded / validation_error) "
        "discovered after the fact don't unwind that progress."
    )
    n_total = len(df)
    n_video = int(df["has_video"].sum())
    n_ml = int(df["ml_done"].sum())
    n_citsci = int(df["citsci_done"].sum())
    n_expert = int(df["expert_done"].sum())

    funnel_df = pd.DataFrame(
        {
            "Stage": [
                "Ingested",
                "Video present",
                "ML complete",
                "CitSci complete",
                "Expert complete",
            ],
            "Count": [n_total, n_video, n_ml, n_citsci, n_expert],
        }
    )
    fig_funnel = go.Figure(
        go.Funnel(
            y=funnel_df["Stage"],
            x=funnel_df["Count"],
            textinfo="value+percent initial",
            marker_color=["#26A69A", "#42A5F5", "#AB47BC", "#FF7043", "#66BB6A"],
        )
    )
    fig_funnel.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=320,
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    # ── Video presence by year ───────────────────────────────────────────────
    st.subheader("Video presence by year")
    vp_counts = (
        df[df["ok"]]
        .groupby(["survey_year", "video_presence"])
        .size()
        .reset_index(name="count")
    )
    vp_colours = {
        VideoPresence.PRESENT: "#26A69A",
        VideoPresence.ABSENT: "#EF5350",
        "archived": "#FF9800",
        "no_video_bad_dep": "#9E9E9E",
    }
    fig_vp = px.bar(
        vp_counts.sort_values("survey_year"),
        x="survey_year",
        y="count",
        color="video_presence",
        color_discrete_map=vp_colours,
        barmode="stack",
        labels={
            "survey_year": "Year",
            "count": "Deployments",
            "video_presence": "Video status",
        },
        height=300,
    )
    fig_vp.update_layout(
        xaxis={"dtick": 1, "tickformat": "d"},
        legend_title_text="Video status",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig_vp, use_container_width=True)

    # ── Deployments by protection status ─────────────────────────────────────
    st.subheader("Deployments by protection status")
    prot_counts = (
        df[df["ok"]]
        .groupby(["protection_status", "ingest_status"])
        .size()
        .reset_index(name="count")
    )
    _PROT_PALETTE = {
        "marine reserve": "#2196F3",
        "reserve": "#2196F3",
        "inside": "#2196F3",
        "partial": "#FF9800",
        "buffer": "#FF9800",
        "fished": "#F44336",
        "outside": "#F44336",
        "unprotected": "#F44336",
        "unknown": "#9E9E9E",
    }
    prot_cmap = {
        s: next((v for k, v in _PROT_PALETTE.items() if k in s.lower()), "#9E9E9E")
        for s in prot_counts["protection_status"].unique()
    }
    fig_prot = px.bar(
        prot_counts.sort_values("count", ascending=True),
        x="count",
        y="protection_status",
        color="protection_status",
        color_discrete_map=prot_cmap,
        orientation="h",
        labels={"count": "Deployments", "protection_status": "Protection status"},
        height=300,
    )
    fig_prot.update_layout(
        showlegend=False,
        yaxis_title=None,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig_prot, use_container_width=True)

with col_d:
    st.subheader("Annotation depth per survey")
    st.caption(
        "Each bar = one survey, 100% wide. Segments show what fraction of ok deployments "
        "have reached each annotation depth. Wider green = deeper annotation."
    )

    # Build mutually-exclusive depth categories (each deployment lands in one).
    # Order matters: a drop with expert lands in 'expert', not 'ml'.
    # No ok filter — annotations prove the drop reached that stage.
    depth_drops = df.copy()
    depth_drops["depth"] = "none"
    depth_drops.loc[depth_drops["ml_done"], "depth"] = "ml"
    depth_drops.loc[depth_drops["citsci_done"], "depth"] = "citsci"
    depth_drops.loc[depth_drops["expert_done"], "depth"] = "expert"

    depth_counts = (
        depth_drops.groupby(["survey_id", "depth"]).size().reset_index(name="count")
    )
    survey_totals = depth_drops.groupby("survey_id").size().rename("total")
    depth_counts = depth_counts.merge(survey_totals, on="survey_id")
    depth_counts["pct"] = (depth_counts["count"] / depth_counts["total"] * 100).round(1)

    # Sort surveys: most-complete (highest % expert) first so the eye reads top-down
    survey_order = (
        depth_counts[depth_counts["depth"] == "expert"]
        .set_index("survey_id")["pct"]
        .reindex(survey_totals.index, fill_value=0)
        .sort_values(
            ascending=True
        )  # bottom = most complete (plot reads bottom-to-top)
        .index.tolist()
    )

    depth_order = ["none", "ml", "citsci", "expert"]
    depth_colours = {
        "none": "#E0E0E0",  # grey — not started
        "ml": "#CE93D8",  # light purple — automated only
        "citsci": "#FFAB91",  # peach — community validated
        "expert": "#81C784",  # green — expert ground truth
    }
    depth_labels = {
        "none": "No annotations",
        "ml": "ML only",
        "citsci": "ML + CitSci",
        "expert": "Expert complete",
    }
    depth_counts["depth_label"] = depth_counts["depth"].map(depth_labels)

    fig_ann = px.bar(
        depth_counts,
        x="pct",
        y="survey_id",
        color="depth",
        color_discrete_map=depth_colours,
        category_orders={"survey_id": survey_order, "depth": depth_order},
        orientation="h",
        custom_data=["depth_label", "count", "total"],
        labels={"pct": "% of ok deployments", "survey_id": "Survey"},
        height=max(320, len(survey_order) * 18 + 80),
    )
    fig_ann.update_traces(
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}: "
        "%{customdata[1]} / %{customdata[2]} (%{x:.1f}%)<extra></extra>",
    )
    fig_ann.update_layout(
        barmode="stack",
        legend_title_text="Depth",
        legend={"traceorder": "normal"},
        xaxis={"ticksuffix": "%", "range": [0, 100]},
        yaxis_title=None,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    # Rename legend entries via for_each_trace
    fig_ann.for_each_trace(lambda t: t.update(name=depth_labels.get(t.name, t.name)))
    st.plotly_chart(fig_ann, use_container_width=True)

st.divider()

# ── Survey detail table ───────────────────────────────────────────────────────

st.subheader("Survey summary table")
survey_table = (
    df.groupby("survey_id")
    .agg(
        year=("survey_year", "first"),
        reserve=("reserve_code", "first"),
        total=("drop_id", "count"),
        ok=("ok", "sum"),
        bad=("is_bad_deployment", "sum"),
        with_video=("has_video", "sum"),
        ml_complete=("ml_done", "sum"),
        citsci_complete=("citsci_done", "sum"),
        expert_complete=("expert_done", "sum"),
    )
    .reset_index()
    .sort_values(["year", "survey_id"], ascending=[False, True])
)
survey_table["bad_pct"] = (
    survey_table["bad"] / survey_table["total"].replace(0, pd.NA) * 100
).round(1)
survey_table["annotation_depth"] = survey_table.apply(
    lambda r: (
        "expert"
        if r["expert_complete"] > 0
        else (
            "citsci"
            if r["citsci_complete"] > 0
            else ("ml" if r["ml_complete"] > 0 else "none")
        )
    ),
    axis=1,
)

st.dataframe(
    survey_table,
    hide_index=True,
    width="stretch",
    column_config={
        "survey_id": st.column_config.TextColumn("Survey"),
        "year": st.column_config.NumberColumn("Year", format="%d", width="small"),
        "reserve": st.column_config.TextColumn("Reserve", width="small"),
        "total": st.column_config.NumberColumn("Total", width="small"),
        "ok": st.column_config.NumberColumn("OK", width="small"),
        "bad": st.column_config.NumberColumn("Bad", width="small"),
        "bad_pct": st.column_config.NumberColumn(
            "Bad %", format="%.1f%%", width="small"
        ),
        "with_video": st.column_config.NumberColumn("Videos", width="small"),
        "ml_complete": st.column_config.ProgressColumn(
            "ML done",
            min_value=0,
            max_value=int(survey_table["ok"].max() or 1),
            format="%d",
        ),
        "citsci_complete": st.column_config.ProgressColumn(
            "CitSci done",
            min_value=0,
            max_value=int(survey_table["ok"].max() or 1),
            format="%d",
        ),
        "expert_complete": st.column_config.ProgressColumn(
            "Expert done",
            min_value=0,
            max_value=int(survey_table["ok"].max() or 1),
            format="%d",
        ),
        "annotation_depth": st.column_config.TextColumn("Best source", width="small"),
    },
)
