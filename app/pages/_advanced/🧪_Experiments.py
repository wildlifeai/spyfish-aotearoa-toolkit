"""
Experiments — read-only sandbox for ecological data views.

Architecture:
    1. Load raw MaxN + sites + common names (cached).
    2. Enrich once: parse drop_id → site_id / reserve_code / survey_year;
       join sites for protection_status; attach display_name from class_map.
    3. Render global filters (source / year range / reserves) at the top.
    4. Build a `ctx` dict with the filtered + multi-source dataframes and the
       project-wide protection-status colour map.
    5. Dispatch to a single experiment function which receives `ctx`.

Each experiment is a self-contained function that takes `ctx` and renders.
"""

import json
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import render_sidebar_refresh

from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager

st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")
st.title("🧪 Experiments")
render_sidebar_refresh()

# ── Constants ─────────────────────────────────────────────────────────────────

_SOURCE_PRIORITY = {"expert": 0, "citsci": 1, "ml": 2}
_BEST_KEY = "_best"  # sentinel value for "best available" source


# ── Helpers ───────────────────────────────────────────────────────────────────


def _prot_rank(p: str) -> int:
    p = (p or "").lower()
    if any(k in p for k in ("reserve", "inside")):
        return 0
    if any(k in p for k in ("partial", "buffer")):
        return 1
    if any(k in p for k in ("fished", "outside", "unprotected")):
        return 2
    return 3


def _protection_color_map(statuses: list) -> dict:
    # Blues (cool) = protected, warm = unprotected.
    palette = [
        (("marine reserve", "full reserve"), "#0D47A1"),
        (("reserve", "inside"), "#1976D2"),
        (("partial reserve", "partially protected"), "#42A5F5"),
        (("partial", "buffer"), "#90CAF9"),
        (("fished area", "heavily fished"), "#B71C1C"),
        (("fished", "outside", "unprotected"), "#EF5350"),
        (("open", "no protection"), "#FF7043"),
        (("unknown",), "#BDBDBD"),
    ]
    result = {}
    for s in statuses:
        low = (s or "").lower()
        colour = "#BDBDBD"
        for keys, hex_colour in palette:
            if any(k in low for k in keys):
                colour = hex_colour
                break
        result[s] = colour
    return result


def _best_source(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the highest-priority source per (drop_id, scientific_name)."""
    df = df.copy()
    df["_rank"] = df["annotated_by"].map(_SOURCE_PRIORITY).fillna(99)
    df = df.sort_values("_rank")
    df = df.drop_duplicates(subset=["drop_id", "scientific_name"], keep="first")
    return df.drop(columns=["_rank"])


# ── Data loading (cached) ─────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_maxn() -> pd.DataFrame:
    return AnnotationDatabaseManager().get_maxn_summary()


@st.cache_data(ttl=300)
def search_species_annotations(scientific_name: str) -> pd.DataFrame:
    """Every raw annotation row for one species — cached per species.

    Unlike `get_maxn_summary()` (peak per drop only), this returns every
    time-window observation so the species-search experiment can list
    every timestamp the species was seen. Cached by argument so each
    species selection is fetched once per session.
    """
    with sqlite3.connect(config.annotations_db_path) as conn:
        return pd.read_sql(
            "SELECT drop_id, scientific_name, time_of_max, time_of_max_seconds, "
            "max_interval, annotated_by, confidence_agreement, external_id "
            "FROM annotations WHERE scientific_name = ? "
            "ORDER BY drop_id, time_of_max_seconds",
            conn,
            params=(scientific_name,),
        )


@st.cache_data(ttl=300)
def load_sites() -> pd.DataFrame:
    with sqlite3.connect(config.db_path) as conn:
        return pd.read_sql(
            "SELECT site_id, site_name, protection_status FROM sites", conn
        )


@st.cache_data(ttl=3600)
def load_common_names() -> dict:
    """scientific_name → 'Common name (Scientific name)' from class_map.json.

    Returns empty dict when the file is missing or for legacy/generic entries.
    """
    path = config.class_map_path
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        entry["scientific_name"]: f"{entry['common_name']} ({entry['scientific_name']})"
        for entry in data.values()
        if entry.get("scientific_name")
        and entry.get("common_name")
        and entry["common_name"].lower() not in ("fish", "bait", "unknown")
        and entry["common_name"] != entry["scientific_name"]
    }


# ── Enrichment ────────────────────────────────────────────────────────────────


def _enrich(df: pd.DataFrame, sites: pd.DataFrame, common_names: dict) -> pd.DataFrame:
    """Parse drop_id segments, join sites, attach display_name. Done once globally."""
    parts = df["drop_id"].str.split("_", expand=True)
    df = df.copy()
    df["reserve_code"] = parts.get(0, pd.Series("", index=df.index)).fillna("")
    df["survey_date"] = pd.to_datetime(parts[1], format="%Y%m%d", errors="coerce")
    df["survey_year"] = df["survey_date"].dt.year
    p3 = parts.get(3, pd.Series("", index=df.index)).fillna("")
    p4 = parts.get(4, pd.Series("", index=df.index)).fillna("")
    df["site_id"] = p3 + "_" + p4
    df["site_id"] = df["site_id"].replace("_", pd.NA)
    df = df.merge(
        sites[["site_id", "site_name", "protection_status"]], on="site_id", how="left"
    )
    df["site_name"] = df["site_name"].fillna(df["site_id"])
    df["protection_status"] = df["protection_status"].fillna("unknown")
    df["display_name"] = df["scientific_name"].map(
        lambda s: common_names.get(s, s) if pd.notna(s) else s
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  Experiments — each takes `ctx` and renders to the current Streamlit context.
#
#  ctx keys:
#    "df"        — filtered + single-source view (best available, or one source)
#    "df_multi"  — filtered (year/reserve) but with all sources retained;
#                  used by Calibration & Disagreement which compare two sources
#    "prot_cmap" — project-wide protection_status → colour map (consistent
#                  colours across experiments)
# ═════════════════════════════════════════════════════════════════════════════


# ── RESERVE EFFECT ───────────────────────────────────────────────────────────


def experiment_reserve_slope(ctx):
    df = ctx["df"]
    st.subheader("Reserve effect: paired species comparison")
    st.caption(
        "For each species: line connects mean MaxN **outside** (left dot) to mean MaxN "
        "**inside** reserve (right dot). Up-sloping = more abundant inside — the classic "
        "reserve effect. Colour intensity = magnitude."
    )

    df = df[df["scientific_name"].notna()].copy()
    if df.empty:
        st.warning("No data after filters.")
        return

    df["reserve"] = df["protection_status"].apply(
        lambda p: "inside" if _prot_rank(p) <= 1 else "outside"
    )

    min_deps = st.slider(
        "Min deployments per side",
        1,
        20,
        3,
        key="slope_min",
        help="A species needs at least this many deployments inside AND outside to be shown.",
    )

    counts = (
        df.groupby(["display_name", "reserve"])["drop_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    if "inside" not in counts.columns or "outside" not in counts.columns:
        st.warning("Filters need to include sites both inside AND outside a reserve.")
        return

    keep = counts[
        (counts["inside"] >= min_deps) & (counts["outside"] >= min_deps)
    ].index
    df = df[df["display_name"].isin(keep)]
    if df.empty:
        st.warning(f"No species have ≥ {min_deps} deployments on both sides.")
        return

    means = df.groupby(["display_name", "reserve"])["maxn"].mean().round(2).unstack()
    means["effect"] = means["inside"] - means["outside"]
    means["pct_change"] = (
        (means["effect"] / means["outside"].replace(0, np.nan)) * 100
    ).round(1)
    means = means.sort_values("effect", ascending=True)

    fig = go.Figure()
    max_abs = max(abs(means["effect"].min()), abs(means["effect"].max())) or 1

    for species, row in means.iterrows():
        eff = row["effect"]
        intensity = min(1.0, abs(eff) / max_abs)
        if eff >= 0:
            colour = f"rgba(46, 125, 50, {0.35 + 0.55 * intensity})"
        else:
            colour = f"rgba(198, 40, 40, {0.35 + 0.55 * intensity})"
        fig.add_trace(
            go.Scatter(
                x=[row["outside"], row["inside"]],
                y=[species, species],
                mode="lines+markers",
                line={"color": colour, "width": 3},
                marker={"size": [9, 12], "color": colour},
                hovertemplate=(
                    f"<b>{species}</b><br>"
                    f"Outside: {row['outside']:.2f}<br>"
                    f"Inside: {row['inside']:.2f}<br>"
                    f"Effect: {eff:+.2f} ({row['pct_change']:+.1f}%)<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis_title="Mean Peak MaxN  (small dot = outside,  large dot = inside)",
        yaxis_title=None,
        height=max(350, len(means) * 24 + 120),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        hovermode="closest",
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE", zeroline=True, zerolinecolor="#CCCCCC")
    fig.update_yaxes(gridcolor="#F5F5F5")
    st.plotly_chart(fig, use_container_width=True)

    n_pos = int((means["effect"] > 0).sum())
    n_neg = int((means["effect"] < 0).sum())
    n_tot = len(means)
    cols = st.columns(3)
    cols[0].metric("Species shown", n_tot)
    cols[1].metric("Higher inside reserve", f"{n_pos} ({n_pos / n_tot * 100:.0f}%)")
    cols[2].metric("Higher outside", f"{n_neg} ({n_neg / n_tot * 100:.0f}%)")

    with st.expander("Effect table"):
        st.dataframe(
            means.reset_index()
            .rename(
                columns={
                    "display_name": "Species",
                    "outside": "Mean MaxN — outside",
                    "inside": "Mean MaxN — inside",
                    "effect": "Effect (Δ MaxN)",
                    "pct_change": "% change",
                }
            )
            .sort_values("Effect (Δ MaxN)", ascending=False),
            hide_index=True,
        )


def experiment_reserve_trends(ctx):
    df = ctx["df"]
    st.subheader("Reserve trends — recovering or declining?")
    st.caption(
        "For each reserve: how is the chosen metric changing year-on-year? "
        "Lines show the metric over time; the table fits a linear trend per reserve "
        "and labels each as **↑ Recovering / → Stable / ↓ Declining**."
    )

    df = df[
        df["scientific_name"].notna()
        & df["survey_year"].notna()
        & df["reserve_code"].notna()
        & (df["reserve_code"] != "")
    ].copy()
    if df.empty:
        st.warning("No data after filters.")
        return

    metric_choice = st.radio(
        "Metric",
        [
            "Mean MaxN per deployment",
            "Species richness per deployment",
            "Reserve effect ratio (inside ÷ outside)",
            "Flagship species MaxN",
        ],
        horizontal=False,
        key="trends_metric",
    )

    flagship_selected = None
    if metric_choice == "Flagship species MaxN":
        all_species = sorted(df["display_name"].unique())
        # Default to Snapper + Blue cod if present
        defaults = [
            s for s in all_species if "Pagrus auratus" in s or "Parapercis colias" in s
        ]
        flagship_selected = st.multiselect(
            "Flagship species (empty = falls back to all-species mean)",
            options=all_species,
            default=defaults,
            key="trends_flagship",
        )

    min_years = st.slider(
        "Minimum years of data per reserve",
        2,
        8,
        3,
        key="trends_min_years",
        help="Reserves with fewer years are shown but not labelled with a trend direction.",
    )

    df["survey_year"] = df["survey_year"].astype(int)

    # ── Compute per-(reserve, year) metric ───────────────────────────────────
    def _agg_total_maxn_per_dep(grp):
        """Per (reserve, year): total MaxN per drop, then mean across drops."""
        per_drop = grp.groupby("drop_id")["maxn"].sum()
        return per_drop.mean()

    def _agg_richness_per_dep(grp):
        per_drop = grp.groupby("drop_id")["scientific_name"].nunique()
        return per_drop.mean()

    def _agg_flagship_maxn(grp, species_list):
        sub = grp[grp["display_name"].isin(species_list)]
        if sub.empty:
            return 0.0
        per_drop = sub.groupby("drop_id")["maxn"].sum()
        return per_drop.mean()

    if metric_choice == "Mean MaxN per deployment":
        series = (
            df.groupby(["reserve_code", "survey_year"])
            .apply(_agg_total_maxn_per_dep, include_groups=False)
            .reset_index(name="metric")
        )
        y_title = "Mean MaxN per deployment"
    elif metric_choice == "Species richness per deployment":
        series = (
            df.groupby(["reserve_code", "survey_year"])
            .apply(_agg_richness_per_dep, include_groups=False)
            .reset_index(name="metric")
        )
        y_title = "Mean species per deployment"
    elif metric_choice == "Reserve effect ratio (inside ÷ outside)":
        df["_side"] = df["protection_status"].apply(
            lambda p: "inside" if _prot_rank(p) <= 1 else "outside"
        )
        by_side = (
            df.groupby(["reserve_code", "survey_year", "_side"])
            .apply(_agg_total_maxn_per_dep, include_groups=False)
            .unstack("_side")
        )
        if "inside" not in by_side.columns or "outside" not in by_side.columns:
            st.warning(
                "Need both inside-reserve and outside-reserve deployments "
                "to compute this ratio."
            )
            return
        by_side["metric"] = by_side["inside"] / by_side["outside"].replace(0, np.nan)
        series = by_side["metric"].reset_index()
        series = series.dropna(subset=["metric"])
        y_title = "MaxN ratio (inside ÷ outside)"
    else:  # Flagship species MaxN
        if not flagship_selected:
            st.info("Pick at least one flagship species.")
            return
        series = (
            df.groupby(["reserve_code", "survey_year"])
            .apply(
                lambda g: _agg_flagship_maxn(g, flagship_selected),
                include_groups=False,
            )
            .reset_index(name="metric")
        )
        y_title = f"Mean flagship MaxN per deployment ({len(flagship_selected)} spp)"

    if series.empty:
        st.warning("No data after applying the metric.")
        return

    # ── Majority protection status per reserve (drives colour) ──────────────
    reserve_prot = (
        df.groupby("reserve_code")["protection_status"]
        .agg(lambda x: x.value_counts().index[0] if len(x) else "unknown")
        .to_dict()
    )
    series["protection"] = series["reserve_code"].map(reserve_prot).fillna("unknown")

    # ── Linear trend per reserve ────────────────────────────────────────────
    trend_rows = []
    for reserve, grp in series.groupby("reserve_code"):
        grp = grp.dropna(subset=["metric"]).sort_values("survey_year")
        n_years = grp["survey_year"].nunique()
        prot = reserve_prot.get(reserve, "unknown")

        if n_years < min_years:
            first_year = int(grp["survey_year"].min()) if not grp.empty else None
            latest_year = int(grp["survey_year"].max()) if not grp.empty else None
            first_metric = (
                round(float(grp["metric"].iloc[0]), 2) if not grp.empty else None
            )
            latest_metric = (
                round(float(grp["metric"].iloc[-1]), 2) if not grp.empty else None
            )
            trend_rows.append(
                {
                    "Reserve": reserve,
                    "Protection": prot,
                    "n_years": int(n_years),
                    "slope": np.nan,
                    "direction": "insufficient data",
                    "first_year": first_year,
                    "latest_year": latest_year,
                    "first_metric": first_metric,
                    "latest_metric": latest_metric,
                }
            )
            continue

        slope, _ = np.polyfit(grp["survey_year"].values, grp["metric"].values, 1)
        # Stability threshold: 5% of the mean of the metric (per-year basis)
        eps = 0.05 * abs(grp["metric"].mean()) if grp["metric"].mean() else 0
        if slope > eps:
            direction = "↑ Recovering"
        elif slope < -eps:
            direction = "↓ Declining"
        else:
            direction = "→ Stable"

        trend_rows.append(
            {
                "Reserve": reserve,
                "Protection": prot,
                "n_years": int(n_years),
                "slope": round(float(slope), 3),
                "direction": direction,
                "first_year": int(grp["survey_year"].min()),
                "latest_year": int(grp["survey_year"].max()),
                "first_metric": round(float(grp["metric"].iloc[0]), 2),
                "latest_metric": round(float(grp["metric"].iloc[-1]), 2),
            }
        )

    trends_df = pd.DataFrame(trend_rows).sort_values(
        "slope", ascending=False, na_position="last"
    )

    # ── Line chart ──────────────────────────────────────────────────────────
    fig = px.line(
        series.sort_values(["reserve_code", "survey_year"]),
        x="survey_year",
        y="metric",
        color="reserve_code",
        markers=True,
        hover_data={"protection": True, "metric": ":.2f"},
        labels={
            "survey_year": "Survey year",
            "metric": y_title,
            "reserve_code": "Reserve",
            "protection": "Protection",
        },
        height=480,
    )
    fig.update_layout(
        legend_title_text="Reserve",
        xaxis={"dtick": 1},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    fig.update_yaxes(gridcolor="#EEEEEE")

    if metric_choice == "Reserve effect ratio (inside ÷ outside)":
        # Reference line at ratio = 1 (parity between inside and outside)
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="#888",
            line_width=1,
            annotation_text="parity",
            annotation_position="right",
        )

    st.plotly_chart(fig, use_container_width=True)

    # ── Honest caveat ───────────────────────────────────────────────────────
    st.caption(
        "Trend lines fit a simple linear regression per reserve; 3+ years of data "
        "required for a direction label. Real recovery often takes 5–10+ years and "
        "isn't linear — treat short-term trends as hints, not conclusions."
    )

    # ── Summary metrics ─────────────────────────────────────────────────────
    n_up = int((trends_df["direction"] == "↑ Recovering").sum())
    n_down = int((trends_df["direction"] == "↓ Declining").sum())
    n_flat = int((trends_df["direction"] == "→ Stable").sum())
    n_insuff = int((trends_df["direction"] == "insufficient data").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("↑ Recovering", n_up)
    c2.metric("→ Stable", n_flat)
    c3.metric("↓ Declining", n_down)
    c4.metric("Insufficient data", n_insuff)

    st.markdown("**Per-reserve trend table**")
    st.dataframe(
        trends_df.rename(
            columns={
                "n_years": "Years",
                "slope": "Slope (/yr)",
                "direction": "Trend",
                "first_year": "First year",
                "latest_year": "Latest year",
                "first_metric": "First value",
                "latest_metric": "Latest value",
            }
        ),
        hide_index=True,
    )


def experiment_yearly_trend(ctx):
    df = ctx["df"]
    st.subheader("Year-on-year MaxN per site")
    st.caption(
        "Mean peak MaxN per site per survey year. "
        "Surveys are 1–2 year cadence, not seasonal."
    )

    df = df[df["scientific_name"].notna() & df["survey_year"].notna()]
    all_species = sorted(df["display_name"].dropna().unique())
    if not all_species:
        st.warning("No species data after filters.")
        return

    # Default to Snapper (Pagrus auratus) — iconic NZ BUV species with rich
    # historical data — falling back to Blue cod, then alphabetical first.
    default_idx = 0
    for preferred in ("Pagrus auratus", "Parapercis colias"):
        for i, name in enumerate(all_species):
            if preferred in name:
                default_idx = i
                break
        else:
            continue
        break
    species = st.selectbox("Species", all_species, index=default_idx, key="trend_spp")
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
        color_discrete_sequence=px.colors.qualitative.Safe,
        markers=True,
        hover_data={"protection_status": True, "mean_maxn": True},
        labels={
            "survey_year": "Survey year",
            "mean_maxn": f"Mean MaxN — {species}",
            "site_id": "Site",
            "protection_status": "Protection",
        },
        height=450,
    )
    fig.update_layout(
        legend_title_text="Site",
        xaxis={"dtick": 1},
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Solid = protected, dashed = unprotected. Each line = one site.")

    with st.expander("Data"):
        st.dataframe(trend.sort_values(["site_id", "survey_year"]), hide_index=True)


# ── PROGRAMME OVERVIEW ───────────────────────────────────────────────────────


def experiment_detection_rate(ctx):
    df = ctx["df"]
    st.subheader("Species detection rate")
    st.caption(
        "For each species: % of deployments where it was detected. "
        "Highlights survey coverage of common vs rare species. "
        "**Inside vs outside view** splits the same rate by reserve protection — "
        "a presence-axis complement to the Reserve effect slope (which is on abundance)."
    )

    df = df[df["scientific_name"].notna()]
    total_drops = df["drop_id"].nunique()
    if total_drops == 0:
        st.warning("No deployments after filters.")
        return

    view = st.radio(
        "View",
        ["Programme-wide", "Inside vs outside reserve"],
        horizontal=True,
        key="det_view",
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

        top_n = st.slider(
            "Show top N species",
            5,
            max(5, len(rates)),
            min(40, len(rates)),
            key="det_topn",
        )
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
        fig.update_layout(
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
    df = df.copy()
    df["reserve"] = df["protection_status"].apply(
        lambda p: "inside" if _prot_rank(p) <= 1 else "outside"
    )

    # Per-class deployment totals (denominator for detection rate)
    drop_class = df.drop_duplicates("drop_id")[["drop_id", "reserve"]]
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

    top_n = st.slider(
        "Show top N species",
        5,
        max(5, len(spread)),
        min(25, len(spread)),
        key="det_topn_io",
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
        color_discrete_map={"inside": "#1976D2", "outside": "#EF5350"},
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
    fig.update_layout(
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


def experiment_community_composition(ctx):
    df = ctx["df"]
    st.subheader("Community composition by reserve")
    st.caption(
        "Top-N species relative abundance per reserve. Shows which species "
        "dominate where — colour patterns reveal community similarities and "
        "differences across reserves."
    )

    df = df[df["scientific_name"].notna() & df["site_id"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    top_n = st.slider("Top N species per reserve", 3, 12, 6, key="comm_top")

    tot = df.groupby(["reserve_code", "display_name"])["maxn"].sum().reset_index()
    reserve_totals = tot.groupby("reserve_code")["maxn"].sum().rename("reserve_total")
    tot = tot.merge(reserve_totals, on="reserve_code")
    tot["pct"] = (tot["maxn"] / tot["reserve_total"] * 100).round(1)

    keep_rows = []
    for reserve, grp in tot.groupby("reserve_code"):
        top = grp.nlargest(top_n, "pct")
        other_pct = grp["pct"].sum() - top["pct"].sum()
        keep_rows.append(top)
        if other_pct > 0:
            keep_rows.append(
                pd.DataFrame(
                    [
                        {
                            "reserve_code": reserve,
                            "display_name": "Other",
                            "maxn": 0,
                            "reserve_total": 0,
                            "pct": round(other_pct, 1),
                        }
                    ]
                )
            )
    plot_df = pd.concat(keep_rows, ignore_index=True)

    reserve_order = reserve_totals.sort_values(ascending=False).index.tolist()

    fig = px.bar(
        plot_df,
        x="pct",
        y="reserve_code",
        color="display_name",
        orientation="h",
        category_orders={"reserve_code": reserve_order},
        labels={
            "pct": "% of MaxN within reserve",
            "reserve_code": "Reserve",
            "display_name": "Species",
        },
        height=max(320, len(reserve_order) * 32 + 80),
    )
    fig.update_layout(
        barmode="stack",
        xaxis={"ticksuffix": "%", "range": [0, 100]},
        legend={"font": {"size": 9}},
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Composition table"):
        sorted_df = plot_df.sort_values(
            ["reserve_code", "pct"], ascending=[True, False]
        )
        st.dataframe(sorted_df, hide_index=True)


def experiment_site_leaderboard(ctx):
    df = ctx["df"]
    st.subheader("Site leaderboard")
    st.caption(
        "Which sites have the most fish, or the most variety? Colour = protection."
    )

    df = df[df["site_id"].notna() & df["scientific_name"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    metric = st.radio(
        "Metric",
        ["Sum of MaxN", "Mean MaxN per deployment", "Species richness"],
        horizontal=True,
        key="leader_metric",
    )

    if metric == "Sum of MaxN":
        board = (
            df.groupby(["site_id", "site_name", "protection_status"], dropna=False)[
                "maxn"
            ]
            .sum()
            .reset_index()
            .rename(columns={"maxn": "value"})
        )
        ylabel = "Total MaxN (all species)"
    elif metric == "Mean MaxN per deployment":
        per_dep = (
            df.groupby(
                ["site_id", "site_name", "protection_status", "drop_id"], dropna=False
            )["maxn"]
            .sum()
            .reset_index()
        )
        board = (
            per_dep.groupby(
                ["site_id", "site_name", "protection_status"], dropna=False
            )["maxn"]
            .mean()
            .reset_index()
            .rename(columns={"maxn": "value"})
        )
        board["value"] = board["value"].round(2)
        ylabel = "Mean total MaxN per deployment"
    else:
        board = (
            df.groupby(["site_id", "site_name", "protection_status"], dropna=False)[
                "scientific_name"
            ]
            .nunique()
            .reset_index()
            .rename(columns={"scientific_name": "value"})
        )
        ylabel = "Unique species observed"

    top_n = st.slider(
        "Show top N sites",
        5,
        max(5, len(board)),
        min(30, len(board)),
        key="leader_topn",
    )
    board = board.nlargest(top_n, "value").sort_values("value", ascending=True)

    fig = px.bar(
        board,
        x="value",
        y="site_id",
        color="protection_status",
        color_discrete_map=ctx["prot_cmap"],
        orientation="h",
        labels={"value": ylabel, "site_id": "Site", "protection_status": "Protection"},
        height=max(350, len(board) * 26 + 100),
    )
    fig.update_layout(
        legend_title_text="Protection status",
        yaxis_title=None,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)

    n_deps = (
        df.groupby("site_id", dropna=False)["drop_id"].nunique().rename("n_deployments")
    )
    board = board.merge(n_deps, on="site_id", how="left")
    with st.expander("Table"):
        st.dataframe(board.sort_values("value", ascending=False), hide_index=True)


# ── SPECIES RELATIONSHIPS ────────────────────────────────────────────────────


def experiment_cooccurrence(ctx):
    df = ctx["df"]
    st.subheader("Species co-occurrence")
    st.caption(
        "Cell = how often two species are seen at the same deployment, normalised by "
        "the rarer of the two species' total deployments (Jaccard-like). "
        "High values = species that travel together. Diagonal = self (always 1)."
    )

    df = df[df["scientific_name"].notna()]
    if df.empty:
        st.warning("No data after filters.")
        return

    min_deps = st.slider("Min deployments to include a species", 2, 30, 5, key="co_min")
    species_counts = df.groupby("display_name")["drop_id"].nunique()
    keep = species_counts[species_counts >= min_deps].index
    df = df[df["display_name"].isin(keep)]
    if df["display_name"].nunique() < 2:
        st.warning(f"Need ≥ 2 species each observed at ≥ {min_deps} deployments.")
        return

    top_n = st.slider(
        "Show top N species (by occurrence)",
        5,
        max(5, df["display_name"].nunique()),
        min(25, df["display_name"].nunique()),
        key="co_topn",
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
        height=max(400, len(order) * 22 + 120),
    )
    fig.update_xaxes(tickangle=-60, tickfont={"size": 9})
    fig.update_yaxes(tickfont={"size": 9})
    fig.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, use_container_width=True)

    pairs = matrix.where(np.triu(np.ones(matrix.shape, dtype=bool), k=1)).stack()
    top_pairs = pairs.sort_values(ascending=False).head(15).reset_index()
    top_pairs.columns = ["Species A", "Species B", "Co-occurrence"]
    st.markdown("**Top 15 co-occurring species pairs**")
    st.dataframe(top_pairs, hide_index=True)


# ── SOURCE QUALITY (uses multi-source df) ────────────────────────────────────


def experiment_source_calibration(ctx):
    df = ctx["df_multi"]
    st.subheader("Source calibration: scatter")
    st.caption(
        "Each point = one (deployment, species) where both sources observed. "
        "Diagonal = perfect agreement. Above 1:1 = X-source undercounts; below = overcounts. "
        "R² and slope quantify systematic bias."
    )
    st.info(
        "This experiment uses **all sources** regardless of the global source filter — "
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
        key="cal_x",
    )
    remaining = [s for s in available if s != src_x]
    src_y = c2.selectbox(
        "Y axis (typically ground truth)",
        remaining,
        index=remaining.index("expert") if "expert" in remaining else 0,
        key="cal_y",
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
        r = np.corrcoef(x_vals, y_vals)[0, 1]
        r2 = r**2
        denom = (x_vals**2).sum()
        slope = (x_vals * y_vals).sum() / denom if denom > 0 else 0
    else:
        r2, slope = float("nan"), float("nan")

    bias = (y_vals - x_vals).mean()
    mae = (y_vals - x_vals).abs().mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Points compared", n)
    m2.metric("R²", f"{r2:.3f}")
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
            "x": f"MaxN — {src_x}",
            "y": f"MaxN — {src_y}",
            "display_name": "Species",
        },
        height=520,
    )
    fig.update_traces(
        marker={"size": 9, "opacity": 0.7, "line": {"width": 0.5, "color": "white"}}
    )

    max_v = max(x_vals.max(), y_vals.max()) + 1
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_v,
        y1=max_v,
        line={"dash": "dash", "color": "#888888", "width": 1.5},
    )
    fig.add_annotation(
        x=max_v,
        y=max_v,
        text="  1:1",
        showarrow=False,
        xanchor="left",
        font={"color": "#888888"},
    )
    if slope == slope:
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_v,
            y1=max_v * slope,
            line={"color": "#1976D2", "width": 2},
        )
        fig.add_annotation(
            x=max_v,
            y=max_v * slope,
            text=f"  fit (b={slope:.2f})",
            showarrow=False,
            xanchor="left",
            font={"color": "#1976D2"},
        )

    fig.update_layout(
        legend={"orientation": "v", "x": 1.02, "y": 1, "font": {"size": 9}},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE", zeroline=True, zerolinecolor="#CCCCCC")
    fig.update_yaxes(gridcolor="#EEEEEE", zeroline=True, zerolinecolor="#CCCCCC")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Worst-calibrated points"):
        merged["abs_diff"] = (merged["y"] - merged["x"]).abs()
        # Relative error: |Δ| / max(x, y). Symmetric and bounded in [0, 1].
        # A 23-vs-25 mismatch (9% rel) ranks below a 1-vs-3 mismatch (67% rel) —
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
                    help="|MaxN_x − MaxN_y| / max(MaxN_x, MaxN_y) — "
                    "relative error, bounded in 0–100%.",
                ),
            },
        )


def experiment_diversity(ctx):
    df = ctx["df"]
    st.subheader("Diversity per reserve")
    st.caption(
        "**Shannon (H)** rewards evenness — a reserve with 10 evenly-distributed species "
        "scores higher than one with 30 species dominated by one. "
        "**Simpson (1−D)** = probability that two random observations are different species. "
        "**Evenness** = Shannon normalised by ln(species count); 1 = perfectly even."
    )

    df = df[
        df["scientific_name"].notna()
        & df["reserve_code"].notna()
        & (df["reserve_code"] != "")
    ].copy()
    if df.empty:
        st.warning("No data after filters.")
        return

    tot = df.groupby(["reserve_code", "display_name"])["maxn"].sum().reset_index()

    def _shannon(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def _simpson(counts):
        p = counts / counts.sum()
        return float(1 - (p**2).sum())

    rows = []
    for reserve, grp in tot.groupby("reserve_code"):
        counts = grp["maxn"].values
        if counts.sum() == 0:
            continue
        n_spp = int(grp["display_name"].nunique())
        h = _shannon(counts)
        rows.append(
            {
                "Reserve": reserve,
                "Species": n_spp,
                "Total MaxN": int(counts.sum()),
                "Shannon (H)": round(h, 3),
                "Simpson (1-D)": round(_simpson(counts), 3),
                "Evenness": round(h / np.log(n_spp), 3) if n_spp > 1 else 0.0,
            }
        )
    if not rows:
        st.warning("Not enough data to compute diversity.")
        return

    div_df = pd.DataFrame(rows)

    # Dominant protection status per reserve, for colouring
    reserve_prot = (
        df.groupby("reserve_code")["protection_status"]
        .agg(lambda x: x.value_counts().index[0] if len(x) else "unknown")
        .rename("protection_status")
    )
    div_df = div_df.merge(reserve_prot, left_on="Reserve", right_index=True, how="left")
    div_df["protection_status"] = div_df["protection_status"].fillna("unknown")

    metric_choice = st.radio(
        "Diversity index",
        ["Shannon (H)", "Simpson (1-D)", "Evenness"],
        horizontal=True,
        key="div_metric",
    )

    fig = px.bar(
        div_df.sort_values(metric_choice, ascending=True),
        x=metric_choice,
        y="Reserve",
        color="protection_status",
        color_discrete_map=ctx["prot_cmap"],
        orientation="h",
        hover_data={
            "Species": True,
            "Total MaxN": True,
            "Shannon (H)": ":.3f",
            "Simpson (1-D)": ":.3f",
        },
        labels={"protection_status": "Protection"},
        height=max(350, len(div_df) * 32 + 100),
    )
    fig.update_layout(
        legend_title_text="Protection status",
        yaxis_title=None,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full table"):
        st.dataframe(
            div_df.sort_values(metric_choice, ascending=False), hide_index=True
        )


def experiment_species_accumulation(ctx):
    df = ctx["df"]
    st.subheader("Species accumulation curve")
    st.caption(
        "Cumulative unique species discovered as more deployments are surveyed. "
        "Resampled 50× over random deployment orderings — shaded band = 95% interval. "
        "When the curve flattens, additional surveys add little new biodiversity. "
        "If it's still rising sharply at the right edge, more surveys would still find new species."
    )

    df = df[df["scientific_name"].notna()].copy()
    drops_with_species = df[["drop_id", "display_name"]].drop_duplicates()
    if drops_with_species.empty:
        st.warning("No species data after filters.")
        return

    by_reserve = st.checkbox("Split by reserve", value=False, key="acc_split")
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

    fig.update_layout(
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


# ── SPECIES DEEP-DIVE ────────────────────────────────────────────────────────


def experiment_bait_arrival(ctx):
    df = ctx["df"]
    st.subheader("Bait arrival curves")
    st.caption(
        "Distribution of `time_of_max_seconds` per species — when during the 30-min "
        "deployment did each species' **peak count** occur? Early peaks = quick responders "
        "to bait; later peaks = slower / more curious species. "
        "Each violin = one species; box inside = quartiles, line = median."
    )
    st.warning(
        "**Caveat — this shows time of PEAK, not arrival.** A species that arrives at "
        "t=2 min but peaks at t=20 min appears here as 'late'. And because the global "
        "source filter applies (expert > citsci > ml), the timestamp comes from "
        "whichever source 'won' for each (drop, species) — not necessarily the earliest "
        "observation across sources. For true arrival times, we'd need a separate "
        "viz that queries the first observation per drop regardless of source (see todo)."
    )

    df = df[df["scientific_name"].notna() & df["time_of_max_seconds"].notna()].copy()
    if df.empty:
        st.warning("No annotations have a usable timestamp.")
        return

    df["time_min"] = df["time_of_max_seconds"] / 60

    top_n = st.slider(
        "Top N species (by observation count)",
        5,
        max(5, df["display_name"].nunique()),
        min(12, df["display_name"].nunique()),
        key="bait_topn",
    )
    counts = df.groupby("display_name").size().nlargest(top_n)
    df = df[df["display_name"].isin(counts.index)]
    if df.empty:
        st.warning("No species after filters.")
        return

    medians = df.groupby("display_name")["time_min"].median().sort_values()
    species_order = medians.index.tolist()

    fig = px.violin(
        df,
        x="time_min",
        y="display_name",
        orientation="h",
        points=False,
        category_orders={"display_name": species_order},
        labels={"time_min": "Time in deployment (min)", "display_name": "Species"},
        height=max(400, len(species_order) * 40 + 100),
    )
    fig.update_traces(
        box_visible=True,
        meanline_visible=True,
        line={"color": "#1976D2"},
        fillcolor="rgba(25, 118, 210, 0.25)",
    )

    # Reference lines: 5-min markers across the 30-min deployment
    for x in (5, 10, 15, 20, 25):
        fig.add_vline(x=x, line_dash="dot", line_color="#DDDDDD", line_width=1)

    overall_median = df["time_min"].median()
    fig.add_vline(
        x=overall_median,
        line_dash="dash",
        line_color="#FF7043",
        line_width=2,
        annotation_text=f"overall median {overall_median:.1f} min",
        annotation_position="top",
    )

    fig.update_layout(
        xaxis={"range": [0, 30], "dtick": 5},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    st.plotly_chart(fig, use_container_width=True)

    summary = pd.DataFrame(
        {
            "Species": species_order,
            "n_obs": [int(counts[s]) for s in species_order],
            "Median (min)": [round(float(medians[s]), 1) for s in species_order],
            "Mean (min)": [
                round(float(df[df["display_name"] == s]["time_min"].mean()), 1)
                for s in species_order
            ],
            "Std (min)": [
                round(float(df[df["display_name"] == s]["time_min"].std() or 0), 1)
                for s in species_order
            ],
        }
    )
    with st.expander("Arrival statistics"):
        st.dataframe(summary, hide_index=True)


def experiment_freq_abundance(ctx):
    df = ctx["df"]
    st.subheader("Frequency × abundance")
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

    log_scale = st.checkbox("Log-scale axes (recommended)", value=True, key="freq_log")

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

    fig.update_layout(
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


def experiment_species_search(ctx):
    df_multi = ctx["df_multi"]
    sites = ctx["sites"]
    common_names = ctx["common_names"]

    st.subheader("Species search: every observation timeline")
    st.caption(
        "Pick a species — get every (drop, time) where it was observed, "
        "grouped by source priority: expert > citsci > ml. "
        "Queries the annotations DB per species (not loaded upfront)."
    )
    st.info(
        "Uses **all sources** regardless of the global source filter — the point "
        "is to see what each source recorded. Year/reserve filters do apply."
    )

    # Species picker: drawn from species that exist in current filtered view,
    # using display_name for UX but keying on scientific_name for the query.
    spp_lookup = (
        df_multi[df_multi["scientific_name"].notna()][
            ["scientific_name", "display_name"]
        ]
        .drop_duplicates()
        .sort_values("display_name")
    )
    if spp_lookup.empty:
        st.warning("No species found in the current filter.")
        return

    display_to_sci = dict(
        zip(spp_lookup["display_name"], spp_lookup["scientific_name"])
    )
    species_label = st.selectbox(
        "Species (type to filter)",
        options=spp_lookup["display_name"].tolist(),
        key="search_species",
    )
    if not species_label:
        return
    scientific_name = display_to_sci[species_label]

    # Per-species query — fast, scales with the species' own row count
    obs = search_species_annotations(scientific_name)
    if obs.empty:
        st.warning(f"No annotations found for {species_label}.")
        return

    # Attach site / reserve / date metadata using the existing _enrich pipeline.
    # Re-enrich keeps the join logic in one place rather than duplicating it.
    obs = _enrich(obs, sites, common_names)

    # Apply the same year/reserve filters that gate the rest of the page
    if ctx["year_range"]:
        lo, hi = ctx["year_range"]
        obs = obs[obs["survey_year"].between(lo, hi) | obs["survey_year"].isna()]
    if ctx["reserves"]:
        obs = obs[obs["reserve_code"].isin(ctx["reserves"])]

    if obs.empty:
        st.warning("No observations match the current year/reserve filter.")
        return

    # View toggle: peak per (drop, source) vs every observation
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

    # Sort by source priority, then most recent first
    obs["_rank"] = obs["annotated_by"].map(_SOURCE_PRIORITY).fillna(99)
    obs = obs.sort_values(
        ["_rank", "survey_date", "drop_id", "time_of_max_seconds"],
        ascending=[True, False, True, True],
        na_position="last",
    )

    # Summary metrics per source
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

    # Formatted date column for display
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


# ═════════════════════════════════════════════════════════════════════════════
#  Main flow: load → enrich → global controls → filter → dispatch
# ═════════════════════════════════════════════════════════════════════════════


raw_maxn = load_maxn()
if raw_maxn.empty:
    st.info("No annotations in the database yet.")
    st.stop()

sites = load_sites()
common_names = load_common_names()
df_enriched = _enrich(raw_maxn, sites, common_names)

# ── Global filters row ────────────────────────────────────────────────────────

st.markdown("**Global filters** &mdash; apply to every experiment below.")
fc1, fc2, fc3 = st.columns([2, 2, 3])

# Source picker — show coverage so empty charts aren't a mystery
source_counts = df_enriched.groupby("annotated_by")["drop_id"].nunique().to_dict()
source_keys = [_BEST_KEY] + sorted(source_counts.keys())
source_labels = {_BEST_KEY: "Best available (expert > citsci > ml)"}
for s, n in sorted(source_counts.items(), key=lambda kv: -kv[1]):
    source_labels[s] = f"{s} ({n} deps)"

source_choice = fc1.selectbox(
    "Source",
    options=source_keys,
    format_func=lambda k: source_labels.get(k, k),
    key="global_source",
)

# Year range
years_series = df_enriched["survey_year"].dropna()
if not years_series.empty:
    y_min, y_max = int(years_series.min()), int(years_series.max())
    if y_min < y_max:
        year_range = fc2.slider(
            "Year range", y_min, y_max, (y_min, y_max), key="global_years"
        )
    else:
        fc2.markdown(f"**Year**\n\n{y_min}")
        year_range = (y_min, y_max)
else:
    year_range = None

# Reserve filter
all_reserves = sorted([r for r in df_enriched["reserve_code"].dropna().unique() if r])
selected_reserves = fc3.multiselect(
    "Reserves (empty = all)",
    options=all_reserves,
    key="global_reserves",
)

# ── Apply filters ─────────────────────────────────────────────────────────────

# df_multi = year + reserve filters applied, all sources kept (calibration/disagreement)
df_multi = df_enriched.copy()
if year_range is not None:
    df_multi = df_multi[
        df_multi["survey_year"].between(*year_range) | df_multi["survey_year"].isna()
    ]
if selected_reserves:
    df_multi = df_multi[df_multi["reserve_code"].isin(selected_reserves)]

# df = df_multi with source filter applied (used by all other experiments)
if source_choice == _BEST_KEY:
    df = _best_source(df_multi)
else:
    df = df_multi[df_multi["annotated_by"] == source_choice].copy()

# Project-wide protection-status colour map (consistent across experiments)
prot_cmap = _protection_color_map(
    sorted(df_enriched["protection_status"].dropna().unique())
)

ctx = {
    "df": df,
    "df_multi": df_multi,
    "prot_cmap": prot_cmap,
    "sites": sites,
    "common_names": common_names,
    "year_range": year_range,
    "reserves": selected_reserves,
}

# ── Experiment dispatcher ─────────────────────────────────────────────────────

# Order: reserve-effect group → programme overview → species relationships
#        → source quality → distribution view.
EXPERIMENTS = {
    # Reserve effect
    "Reserve effect": experiment_reserve_slope,
    "Reserve trends": experiment_reserve_trends,
    "Year trend": experiment_yearly_trend,
    # Programme overview
    "Detection rate": experiment_detection_rate,
    "Composition": experiment_community_composition,
    "Diversity": experiment_diversity,
    "Leaderboard": experiment_site_leaderboard,
    "Accumulation": experiment_species_accumulation,
    # Species deep-dive
    "Species search": experiment_species_search,
    "Bait arrival": experiment_bait_arrival,
    "Freq × abundance": experiment_freq_abundance,
    "Co-occurrence": experiment_cooccurrence,
    # Source quality
    "Calibration": experiment_source_calibration,
}

st.divider()
choice = st.pills(
    "Experiment",
    options=list(EXPERIMENTS.keys()),
    default="Reserve effect",
    key="exp_nav",
)
st.divider()

# Show row count summary for context
n_drops = df["drop_id"].nunique() if not df.empty else 0
n_species = df["display_name"].nunique() if not df.empty else 0
st.caption(
    f"Current filter: {len(df):,} annotation rows · "
    f"{n_drops:,} deployments · {n_species:,} species"
)

if choice:
    EXPERIMENTS[choice](ctx)
