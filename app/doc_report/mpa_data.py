"""Data layer for the MPA view.

The aggregation behind `charts/mpa.py`, pulled out of the drawing so the
numbers can be pinned without a Streamlit run. Frame in, frame out: nothing
here renders, and nothing calls plotly.

The MPA panels read three shapes of frame, and the functions say which:

* the site frames from `site_data.filtered_site_frames` (`df_context`, one row
  per (drop, species) with `maxn`, and `effort_view`, one row per analysed
  deployment), for the populations panel and its map;
* the `data.experiments_frame` shape (one row per (drop, source, species) with
  `maxn`), for composition, trends and the inside/outside balance;
* the deployments and annotations frames, for the habitat tables.

Every "per deployment" here divides by `data.effort_per`, the same definition
the rest of the report uses.
"""

import numpy as np
import pandas as pd
from ecology_data import (
    OTHER_PROTECTION,
    PROTECTED,
    UNPROTECTED,
    UNRECORDED_PROTECTION,
    protection_group,
)
from theme import protection_sort_key

from spyfish.config.wrapper import config

from .data import effort_per, real_species, richness_by, richness_rollup

# The label for deployments whose site carries no protection metadata, on the
# per-status trend. The MPA view's name for the shared constant: the one
# spelling lives in `ecology_data`, next to the other protection groups, so
# the Species charts' unknown group and this line cannot drift apart again.
NOT_RECORDED = UNRECORDED_PROTECTION


# ── MPA populations: species picker ───────────────────────────────────────────


def species_choices(frame: pd.DataFrame) -> tuple[dict, list, list]:
    """The populations picker's options: (sci → display, indicator, other).

    Indicator species come from config (reporting.indicator_species),
    intersected with what is present so the default is never empty, and keep
    the config's order. `real_species` also keeps the unidentified bucket out
    of the picker: it is N unknown species wearing one label, not a species to
    map. Everything else is the alphabetical long tail.
    """
    present = real_species(frame)
    sci_to_display = (
        present.dropna(subset=["display_name"])
        .drop_duplicates("scientific_name")
        .set_index("scientific_name")["display_name"]
        .to_dict()
    )
    indicator_present = [s for s in config.indicator_species if s in sci_to_display]
    other_present = sorted(s for s in sci_to_display if s not in indicator_present)
    return sci_to_display, indicator_present, other_present


# ── MPA populations: trend by protection ──────────────────────────────────────


def trend_protection(status: pd.Series, by_status: bool) -> pd.Series:
    """The line each deployment belongs to on the populations trend.

    Per status (`by_status`), every recorded regime is its own line. The site
    join writes a literal "unknown" where a site has no protection metadata
    (all legacy-import sites); merged with NaN so the chart shows one
    "Not recorded" line, not two spellings.

    Otherwise the shared bucketing (config.protected_statuses; unknown → None),
    so this trend and the Species view's comparison classify every deployment
    the same way. Unrecorded deployments then sit on neither line.
    """
    if by_status:
        return status.replace("unknown", None).fillna(NOT_RECORDED)
    return protection_group(status)


def trend_measure(trend_src: pd.DataFrame, map_sci: str | None) -> pd.DataFrame:
    """Per (year, protection line) numerator for the populations trend.

    `trend_src` is the dated `df_context` with a `protection` column. For a
    species (`map_sci`), the summed MaxN as `total`. Without one, species
    richness as column `value`: the distinct real species recorded in that
    cell, pooled over its deployments (`data.richness_by`, the one place
    richness is counted). Richness is the one many-species measure kept
    (Shannon/Simpson/Pielou removed on review 2026-10-02): "how many species"
    needs no key to read. Absence records (NaN species) count nothing, so a
    cell holding only absences reads 0; `protection_trend` fills surveyed
    cells with no row at all with 0 as well.
    """
    keys = ["survey_year", "protection"]
    if map_sci:
        subject = trend_src[trend_src["scientific_name"] == map_sci]
        return subject.groupby(keys)["maxn"].sum().reset_index(name="total")
    counted = richness_by(trend_src, keys)
    return counted[keys].assign(value=counted["species"].astype(float))


def protection_trend(
    df_context: pd.DataFrame,
    effort_view: pd.DataFrame,
    map_sci: str | None,
    by_status: bool,
) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """The populations trend: (trend, classes, effort with `protection`).

    `trend` has one row per (survey_year, protection) over every year in the
    surveyed range, with `value` and `analysed`. Empty when no deployment is
    dated. `classes` is the lines
    in drawing order; the effort frame is returned so the caller can describe
    the 'Not recorded' line from the same classification.
    """
    trend_effort = effort_view.copy()
    trend_effort["protection"] = trend_protection(
        trend_effort["protection_status"], by_status
    )
    effort_by_year = effort_per(
        trend_effort.dropna(subset=["survey_year"]), ["survey_year", "protection"]
    )

    trend_src = df_context.dropna(subset=["survey_year"]).copy()
    trend_src["protection"] = trend_protection(
        trend_src["protection_status"], by_status
    )
    measured = trend_measure(trend_src, map_sci)

    # A complete year × protection grid, so years with no survey show as a gap in the
    # line rather than being closed up, a compressed axis silently implies continuous
    # monitoring that did not happen.
    years_present = sorted(effort_by_year["survey_year"].dropna().unique())
    # Which lines the grid holds: the two sides, or every status with either a
    # measurement or effort — a status surveyed but never sighted still gets a
    # line of real zeros rather than vanishing.
    classes = (
        sorted(
            set(measured["protection"].dropna())
            | set(effort_by_year["protection"].dropna()),
            key=protection_sort_key,
        )
        if by_status
        else ["Protected", "Unprotected"]
    )
    if not years_present:
        return pd.DataFrame(), classes, trend_effort

    full_years = range(int(min(years_present)), int(max(years_present)) + 1)
    grid = pd.MultiIndex.from_product(
        [list(full_years), classes],
        names=["survey_year", "protection"],
    ).to_frame(index=False)
    trend = grid.merge(measured, on=["survey_year", "protection"], how="left").merge(
        effort_by_year, on=["survey_year", "protection"], how="left"
    )
    if map_sci:
        # Only years that were actually surveyed get a value; a surveyed year with
        # no sighting is a real zero, an unsurveyed year stays NaN and breaks the line.
        trend["value"] = (trend["total"].fillna(0) / trend["analysed"]).where(
            trend["analysed"].notna()
        )
    else:
        # Same rule as a single species: a surveyed year where every analysed
        # deployment was reviewed and nothing was seen is zero species, not a
        # gap. Only an unsurveyed year breaks the line.
        trend["value"] = trend["value"].fillna(0).where(trend["analysed"].notna())
    return trend, classes, trend_effort


# ── MPA populations: site map ─────────────────────────────────────────────────


def site_values(
    df_context: pd.DataFrame, site_geo: pd.DataFrame, map_sci: str | None
) -> pd.DataFrame:
    """`site_geo` (one row per mapped site, with `analysed`) plus `value`.

    For a species, mean MaxN per analysed deployment: total MaxN at the site
    divided by every deployment analysed there, not only those where the
    species appeared. Without one, species richness per site. Either way a
    site where nothing was recorded keeps its row with a value of 0.
    """
    if map_sci:
        per_site = (
            df_context[df_context["scientific_name"] == map_sci]
            .groupby("site_id")["maxn"]
            .sum()
            .reset_index(name="total_maxn")
        )
        mapped = site_geo.merge(per_site, on="site_id", how="left")
        mapped["total_maxn"] = mapped["total_maxn"].fillna(0)
        mapped["value"] = mapped["total_maxn"] / mapped["analysed"].clip(lower=1)
        return mapped
    # Pooled richness per site (`data.richness_by`): distinct real species
    # recorded there, the generic and unidentified classes dropped first.
    counted = richness_by(df_context, "site_id")
    scores = counted[["site_id"]].assign(value=counted["species"].astype(float))
    mapped = site_geo.merge(scores, on="site_id", how="left")
    mapped["value"] = mapped["value"].fillna(0)
    return mapped


def bubble_reference(
    df_context: pd.DataFrame,
    site_effort: pd.DataFrame,
    mapped: pd.DataFrame,
    map_sci: str | None,
) -> float:
    """The value the map's largest bubble stands for.

    Absolute size reference, so a mean MaxN of 13 is the same circle whichever
    species is showing. Plotly Express rescales to whatever the current maximum
    happens to be, which silently makes a rare species look as abundant as a
    common one. The reference is the largest per-site value across ALL species
    (or all sites, for a diversity metric) in the current filter selection, it
    changes when the filters change, but never when the radio changes.

    A zero or missing maximum falls back to 1.0.
    """
    if map_sci:
        all_species_per_site = (
            df_context.groupby(["site_id", "scientific_name"])["maxn"]
            .sum()
            .reset_index()
            .merge(site_effort, on="site_id", how="left")
        )
        all_species_per_site["per_dep"] = all_species_per_site["maxn"] / (
            all_species_per_site["analysed"].fillna(1).clip(lower=1)
        )
        return float(all_species_per_site["per_dep"].max() or 1.0)
    return float(mapped["value"].max() or 1.0)


# ── Composition ───────────────────────────────────────────────────────────────


def composition_shares(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list]:
    """Top-N species share of summed MaxN per MPA: (rows, MPA order).

    Each MPA keeps its `top_n` species by share; the remainder becomes one
    "Other" row (only when it is above zero). MPAs are ordered by total MaxN,
    largest first. `df` must already hold named species with a site.
    """
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
    return plot_df, reserve_order


# ── Reserve trends ────────────────────────────────────────────────────────────

RICHNESS_METRIC = "Species richness per deployment"
SELECTED_METRIC = "Selected species MaxN"

RECOVERING = "↑ Recovering"
STABLE = "→ Stable"
DECLINING = "↓ Declining"
INSUFFICIENT = "insufficient data"


def reserve_trend_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows a reserve trend can use: a year and an MPA.

    Absence records (NaN species, "reviewed, nothing seen") are KEPT: they
    are deployments, and every per-deployment mean below divides by them.
    Dropping them meant a reserve where half the drops saw nothing read the
    same as one where every drop was busy.
    """
    return frame[
        frame["survey_year"].notna()
        & frame["reserve_code"].notna()
        & (frame["reserve_code"] != "")
    ].copy()


def split_sides(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the rows clearly inside or outside an MPA: (rows, partial count).

    Which side of the boundary each deployment sits on. Computed for every
    metric, not just the ratio: a reserve is surveyed inside AND outside, and
    collapsing both into one line per reserve was the bug this replaced. The
    old code labelled the whole reserve by whichever side happened to carry
    more annotation rows, so AHE (43% inside) read "No protection" while SLI
    (58% inside) read "Marine Reserve" — the same kind of place, labelled
    oppositely, and liable to flip when the year filter moved.

    Adds `_side` and casts `survey_year` to int. The count is the annotation
    rows dropped for a partial regime; unrecorded protection drops silently.
    """
    frame = frame.copy()
    frame["survey_year"] = frame["survey_year"].astype(int)
    frame = frame.assign(_side=protection_group(frame["protection_status"]))
    partial = int((frame["_side"] == OTHER_PROTECTION).sum())
    return frame[frame["_side"].isin([PROTECTED, UNPROTECTED])], partial


def _agg_selected_maxn(grp, species_list):
    """Mean summed MaxN of `species_list` per deployment, over EVERY deployment.

    A deployment that recorded none of them contributes 0, so 1 drop with 4
    snapper among 10 reads 0.4, not 4.0. Averaging only over the drops that
    saw the species cannot fall and always looks like a healthy population.
    """
    drops = grp["drop_id"].unique()
    sub = grp[grp["display_name"].isin(species_list)]
    per_drop = sub.groupby("drop_id")["maxn"].sum().reindex(drops, fill_value=0)
    return per_drop.mean()


def reserve_trend_series(
    sided: pd.DataFrame, metric: str, species: list | None = None
) -> pd.DataFrame:
    """Per (reserve_code, survey_year, Side) value of `metric`, as `metric`.

    Richness is mean distinct real species per deployment, from the shared
    `data.richness_rollup` with the cell's own deployments as the effort, so
    it is the same number as the Sites and MPA tables' `species_per_dep`.
    The frame has been through `unify_unidentified`, so an ML-only
    deployment carries one "Unidentified" row; that is N unknown species
    wearing one label, and the rollup drops it, so such a deployment reads
    0, not 1. The selected-species metric is mean summed MaxN of `species`
    per deployment. Both divide by every deployment in the (reserve, year,
    side) cell, absence records included, so a deployment that saw nothing
    counts as 0. `sided` comes from `split_sides`.
    """
    keys = ["reserve_code", "survey_year", "_side"]
    if metric == RICHNESS_METRIC:
        rolled = richness_rollup(sided, sided, by=keys)
        series = (
            rolled[keys]
            .assign(metric=rolled["species_per_dep"])
            .sort_values(keys)
            .reset_index(drop=True)
        )
    else:
        series = (
            sided.groupby(keys)
            .apply(lambda g: _agg_selected_maxn(g, species), include_groups=False)
            .reset_index(name="metric")
        )
    return series.rename(columns={"_side": "Side"})


def linear_trends(series: pd.DataFrame, min_years: int) -> pd.DataFrame:
    """One row per (MPA, Side): fitted slope and a direction label.

    A straight-line fit of `metric` on `survey_year`. Stable means the slope is
    within 5% of the series mean per year. Fewer than `min_years` distinct
    years gets no slope and reads "insufficient data". Steepest rise first.
    """
    trend_rows = []
    for (reserve, side), grp in series.groupby(["reserve_code", "Side"]):
        grp = grp.dropna(subset=["metric"]).sort_values("survey_year")
        n_years = grp["survey_year"].nunique()
        prot = side

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
                    "MPA": reserve,
                    "Side": prot,
                    "n_years": int(n_years),
                    "slope": np.nan,
                    "direction": INSUFFICIENT,
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
            direction = RECOVERING
        elif slope < -eps:
            direction = DECLINING
        else:
            direction = STABLE

        trend_rows.append(
            {
                "MPA": reserve,
                "Side": prot,
                "n_years": int(n_years),
                "slope": round(float(slope), 3),
                "direction": direction,
                "first_year": int(grp["survey_year"].min()),
                "latest_year": int(grp["survey_year"].max()),
                "first_metric": round(float(grp["metric"].iloc[0]), 2),
                "latest_metric": round(float(grp["metric"].iloc[-1]), 2),
            }
        )

    return pd.DataFrame(trend_rows).sort_values(
        "slope", ascending=False, na_position="last"
    )


# ── Inside vs outside ─────────────────────────────────────────────────────────


def inside_share(df: pd.DataFrame) -> pd.DataFrame:
    """Deployments per MPA on each side of the boundary, least-inside first.

    Indexed by `reserve_code`, one column per side (Protected, Unprotected,
    Other) plus `total` and `share_inside`. All three groups here, unlike the
    comparisons: this describes how the surveying was spread, and the
    partial-regime deployments are part of that picture even though no
    comparison can use them. Unrecorded protection is left out, and MPAs with
    no deployments drop.
    """
    sided = df.assign(Side=protection_group(df["protection_status"])).dropna(
        subset=["Side"]
    )
    per_reserve = (
        sided.groupby(["reserve_code", "Side"])["drop_id"]
        .nunique()
        .unstack("Side")
        .fillna(0)
    )
    for column in (PROTECTED, UNPROTECTED, OTHER_PROTECTION):
        if column not in per_reserve.columns:
            per_reserve[column] = 0
    per_reserve["total"] = (
        per_reserve[PROTECTED]
        + per_reserve[UNPROTECTED]
        + per_reserve[OTHER_PROTECTION]
    )
    per_reserve = per_reserve[per_reserve["total"] > 0]
    per_reserve["share_inside"] = per_reserve[PROTECTED] / per_reserve["total"]
    return per_reserve.sort_values("share_inside")


# ── Habitat ───────────────────────────────────────────────────────────────────


def explode_habitat(dep: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per (drop_id, habitat class) from a comma-joined NZMHCS column.

    A deployment listing several classes counts under each: the question is
    "what does this class hold", not a partition of effort.
    """
    rows = dep[dep[column].notna() & (dep[column] != "")][["drop_id", column]]
    if rows.empty:
        return rows.assign(habitat=pd.NA)
    out = rows.assign(habitat=rows[column].str.split(", ")).explode("habitat")
    return out[out["habitat"].notna()]


def species_peaks(ann: pd.DataFrame) -> pd.DataFrame:
    """Peak count per (drop_id, display_name), as `peak`, real species only.

    Peak per (drop, species) first, so a species seen in many intervals of one
    video is one deployment with one MaxN, same as everywhere else in the
    report. Without a `max_interval` column the peak is NA.
    """
    seen = real_species(ann[ann["scientific_name"].notna()])
    if "max_interval" in seen.columns:
        return (
            seen.groupby(["drop_id", "display_name"])["max_interval"]
            .max()
            .reset_index(name="peak")
        )
    return seen.drop_duplicates(["drop_id", "display_name"]).assign(peak=pd.NA)[
        ["drop_id", "display_name", "peak"]
    ]


def substrate_tables(
    abiotic: pd.DataFrame, joined: pd.DataFrame, denom_ann: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """(detection %, mean peak, species columns) per substrate class.

    `joined` is `abiotic` merged with `species_peaks`. Denominators
    ("annotated deps" per class) come from `denom_ann`, the species-UNfiltered
    frame, or a sidebar species pick would shrink them to the deployments
    recording that species and every rate would read 100%. Only classes with
    five or more annotated deployments, and the seven most-recorded species.
    Both frames are empty when no class qualifies.
    """
    denom = abiotic[abiotic["drop_id"].isin(set(denom_ann["drop_id"]))]
    deps_per_class = denom.groupby("habitat")["drop_id"].nunique()
    keep_classes = deps_per_class[deps_per_class >= 5].index
    top = joined["display_name"].value_counts().head(7).index
    det_rows = []
    maxn_rows = []
    for hab in sorted(keep_classes):
        drops = set(denom.loc[denom["habitat"] == hab, "drop_id"])
        sub = joined[joined["drop_id"].isin(drops)]
        det = {"Substrate": hab, "Annotated deps": len(drops)}
        amt = {"Substrate": hab}
        for sp in top:
            sp_rows = sub[sub["display_name"] == sp]
            det[sp] = round(100 * sp_rows["drop_id"].nunique() / len(drops))
            mean_peak = sp_rows["peak"].mean()
            # Never recorded = 0, not NaN: matplotlib paints NaN as the
            # colormap's "bad" colour, which pandas flattens to BLACK, so a
            # missing species read as the biggest value on the heatmap.
            amt[sp] = 0.0 if pd.isna(mean_peak) else round(float(mean_peak), 1)
        det_rows.append(det)
        maxn_rows.append(amt)
    return pd.DataFrame(det_rows), pd.DataFrame(maxn_rows), list(top)
