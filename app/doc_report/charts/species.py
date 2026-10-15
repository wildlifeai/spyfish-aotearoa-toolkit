"""Species charts: does protection make a difference, and where and when.

Protected against unprotected, per deployment and per species: how far apart
the two sides sit, how sure the data is about that gap once survey and region
are held level, and which species the reserve effect shows up in. Then which
sites a species turns up at, how its numbers move over the years inside and
outside protection, and the per-site yearly trend the Sites view borrows.
The community charts (detection rate, co-occurrence, accumulation, ...) are in
`species_community.py`. `doc_report/species.py` decides what the reader is
asked; these functions take a frame and draw.

The peak-across-intervals step is NOT repeated here: frames arrive already
aggregated by `data.species_maxn`, so a chart cannot take its own MaxN and
disagree with the number the view printed above it.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ecology_data import (
    PROTECTED,
    UNPROTECTED,
    UNRECORDED_PROTECTION,
    protection_group,
)
from theme import (
    ABUNDANCE_SCALE,
    ANCHOR_LINE,
    DETECTION_FREQUENCY_SCALE,
    EFFECT_COLOURS,
    GUIDE_LINE,
    INTERVAL_FILL,
    species_color_map,
)

from spyfish.config.wrapper import config

from ..charting import (
    focus_inside_outside,
    group_colors,
    group_dashes,
    protection_dashes,
    style,
    top_n_slider,
    year_axis,
)
from ..data import (
    effort_per,
    real_species,
    species_value_grid,
    stratified_diff_ci,
)
from ..layout import section

# ── Protection-group styling ──────────────────────────────────────────────
#
# Shared by the protection charts, the time charts and the detection-rate
# panel in `species_community.py`, so they cannot disagree about which side
# is which, or about how a deployment with no protection metadata is drawn.

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

# Drops whose site has no protection metadata (above all legacy sites that are
# not in the SharePoint sites list) used to be dropped from the time charts
# entirely, which cut every pre-2011 year out of "Species over time". They are
# a group of their own instead: visibly on the chart, impossible to mistake
# for either side of the comparison. The label is the shared
# `ecology_data.UNRECORDED_PROTECTION`, the same one the MPA trend draws.
GROUP_SYMBOLS[UNRECORDED_PROTECTION] = "diamond-open"
GROUP_DASHES[UNRECORDED_PROTECTION] = "dashdot"


# ── Does protection make a difference ─────────────────────────────────────


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


RICHNESS_MEASURE = "Species per deployment"


def render_protection_effect_ci(
    per_dep: pd.DataFrame, per_species: pd.DataFrame, common: dict
) -> None:
    """The protection effect per indicator species, measured within each MPA.

    Three deliberate choices, each fixing a way the pooled two-number version
    misled:

    * **Indicator species, one row each.** The panel used to compare total
      MaxN summed across species, a number this report elsewhere calls
      meaningless ("adding a snapper to a rock lobster"). It also buried the
      species that answer the question: the config indicator list exists
      because snapper, blue cod and rock lobster are the heavily fished
      species a reserve is expected to change, and a community total mixes
      their signal with wrasses nobody takes.
    * **Within each MPA.** Pooling compares places as well as regimes, see
      `data.stratified_diff_ci`.
    * **The spread is shown, not just the pooled number.** One dot per MPA
      below the table, because a pooled difference resting on MPAs that
      disagree is a different finding from one they all share.
    """
    if per_dep.empty or per_species.empty:
        return
    st.markdown("**Is the difference bigger than sampling luck?**")

    if "reserve_code" not in per_dep.columns:
        st.info("No MPA code on these deployments, so the comparison cannot be split.")
        return

    stratum_choice = st.radio(
        "Compare inside against outside within",
        ["Each MPA", "Each MPA and survey year"],
        horizontal=True,
        key="species_ci_stratum",
        help="Inside and outside are only compared where they share the "
        "chosen context, so a difference between places (or seasons) cannot "
        "arrive looking like a difference from protection. Adding the year "
        "controls more and compares less: contexts without both sides drop "
        "out.",
    )
    stratum_col = "_stratum"
    frame = per_dep.copy()
    if stratum_choice == "Each MPA":
        frame[stratum_col] = frame["reserve_code"]
    else:
        frame[stratum_col] = (
            frame["reserve_code"].astype(str)
            + " "
            + frame["survey_year"].astype("Int64").astype(str)
        )

    # Indicator species the data actually holds; the rest of the config list
    # is named below rather than silently absent.
    available = set(per_species["scientific_name"].dropna())
    indicators = [s for s in config.indicator_species if s in available]
    missing = [s for s in config.indicator_species if s not in available]

    grid = species_value_grid(
        frame[["drop_id", "Group", stratum_col]], per_species, indicators
    )

    measures = [
        (common.get(name, name), grid[grid["scientific_name"] == name], "maxn")
        for name in indicators
    ]
    # Richness rides along as the one community measure that is not a
    # cross-species sum: it counts species, so it stays comparable.
    measures.append((RICHNESS_MEASURE, frame, "richness"))

    rows, detail = [], {}
    for label, data, value_col in measures:
        result = stratified_diff_ci(
            data,
            value_col=value_col,
            group_col="Group",
            stratum_col=stratum_col,
            inside=PROTECTED,
            outside=UNPROTECTED,
        )
        if result["n_strata"] == 0:
            continue
        rows.append(
            {
                "Measure": label,
                # Same weights as the difference, so the three columns agree:
                # Inside − Outside is exactly Difference.
                "Inside": result["mean_inside"],
                "Outside": result["mean_outside"],
                "Difference": result["diff"],
                "95% range": f"{result['lo']:+.2f} to {result['hi']:+.2f}",
                "Other way": result["share_other_way"] * 100,
                "Beyond luck": result["lo"] > 0 or result["hi"] < 0,
                "Areas": result["n_strata"],
            }
        )
        detail[label] = result

    if not rows:
        st.info(
            "No MPA has at least two annotated deployments both inside and "
            "outside, so there is nothing to compare within. Widen the "
            "filters, or switch the context above back to **Each MPA**."
        )
        return

    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Measure": st.column_config.TextColumn("Measure", pinned=True),
            "Inside": st.column_config.NumberColumn(
                "Inside",
                format="%.2f",
                help="Mean per deployment inside an MPA, weighted the same way "
                "as the difference, so Inside minus Outside is exactly the "
                "Difference column.",
            ),
            "Outside": st.column_config.NumberColumn(
                "Outside", format="%.2f", help="Mean per deployment outside an MPA."
            ),
            "Difference": st.column_config.NumberColumn(
                "Inside − outside",
                format="%+.2f",
                help="Each MPA's own inside-minus-outside difference, combined "
                "with weight by how many deployments stand behind it.",
            ),
            "95% range": st.column_config.TextColumn(
                "95% range",
                help="Where the difference lands when deployments are "
                "resampled 2,000 times within each MPA and side. A range "
                "spanning zero means the sign of the difference is not "
                "settled.",
            ),
            "Other way": st.column_config.NumberColumn(
                "Other way",
                format="%.1f%%",
                help="Share of resamples whose difference landed on the "
                "opposite side of zero. The continuous version of the tick "
                "box beside it: 0.1% is a direction established, 2.4% is one "
                "held by a hair, 30% is no direction at all.",
            ),
            "Beyond luck": st.column_config.CheckboxColumn(
                "Beyond luck",
                help="Ticked when the whole 95% range sits on one side of "
                "zero, which is the same thing as 'Other way' being under "
                "2.5%. Read the two together: a tick with 2.4% beside it is "
                "a knife-edge, not a result.",
            ),
            "Areas": st.column_config.NumberColumn(
                "Areas",
                format="%d",
                help="MPAs (or MPA-years) with at least two deployments on "
                "both sides. Others are left out.",
            ),
        },
    )

    settled = table[table["Beyond luck"]]
    if settled.empty:
        st.caption(
            "**Nothing here is settled yet.** Every range spans zero, so for "
            "each measure the data cannot say which side is higher: surveying "
            "the same MPAs again could produce the opposite sign."
        )
    else:
        names = ", ".join(
            f"**{r.Measure}** ({r.Difference:+.2f})" for r in settled.itertuples()
        )
        st.caption(
            f"Clear of zero in this selection: {names}. The rest span zero, so "
            "their direction is not settled."
        )
    if missing:
        st.caption(
            "Not recorded in this selection, so not compared: "
            + ", ".join(f"*{common.get(m, m)}*" for m in missing)
            + "."
        )

    # ── The spread behind the pooled number ──────────────────────────────────
    pick = st.selectbox(
        "Show each area's own difference for",
        list(detail),
        key="species_ci_detail",
        help="The pooled difference above is a weighted average of these. "
        "Areas pointing in opposite directions mean the effect is not "
        "programme-wide, whatever the pooled number says.",
    )
    result = detail[pick]
    strata = result["strata"].copy()
    label_col = strata.columns[0]
    strata["Weight"] = (
        strata["n_inside"]
        * strata["n_outside"]
        / (strata["n_inside"] + strata["n_outside"])
    )
    fig = px.scatter(
        strata,
        x="diff",
        y=label_col,
        size="Weight",
        color=strata["diff"] > 0,
        color_discrete_map={
            True: GROUP_COLOURS[PROTECTED],
            False: EFFECT_COLOURS["Higher outside"],
        },
        hover_data={
            "n_inside": True,
            "n_outside": True,
            "mean_inside": ":.2f",
            "mean_outside": ":.2f",
        },
        labels={"diff": f"{pick}: inside − outside", label_col: ""},
        height=max(260, len(strata) * 26 + 140),
    )
    # The pooled estimate and its range, drawn over the areas it came from.
    fig.add_vrect(
        x0=result["lo"],
        x1=result["hi"],
        fillcolor=INTERVAL_FILL,
        opacity=0.13,
        line_width=0,
    )
    fig.add_vline(x=result["diff"], line_color=ANCHOR_LINE, line_width=2)
    fig.add_vline(x=0, line_dash="dot", line_color=GUIDE_LINE, line_width=1)
    style(fig, legend=False, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, key="species_ci_strata")
    st.caption(
        f"One dot per area, sized by how many deployments stand behind it. "
        f"Blue is higher inside the MPA, red higher outside. The dark line is "
        f"the pooled difference ({result['diff']:+.2f}) and the grey band its "
        f"95% range; the dotted line is no difference. "
        f"{result['n_skipped']:,} area(s) are left out for lacking two "
        f"deployments on both sides."
    )
    st.caption(
        "One assumption remains, and it flatters the interval: deployments "
        "are treated as independent, so replicate drops at one site count for "
        "more than they should. Read the ranges as optimistic."
    )


def render_reserve_effect(
    per_species: pd.DataFrame, common: dict, per_dep: pd.DataFrame | None = None
) -> None:
    """Every species' protection effect, with the uncertainty on each one.

    Was a slope chart: each species' mean outside joined to its mean inside,
    the line's direction being the effect. Two things were wrong with it and
    neither was fixable in the drawing:

    * **The means were presence-only.** `species_maxn` carries a row only
      where a species was recorded, so "mean MaxN" was the mean among
      deployments that already had it — a number that cannot fall when a
      species gets rarer, because the deployments it vanishes from leave the
      average instead of entering it as zeros.
    * **Nothing showed uncertainty**, so a species resting on four
      deployments drew the same confident line as one resting at four
      hundred, and the longest line read as the strongest finding.

    Now a forest plot of the same comparison the headline table makes:
    zero-filled means, differenced within each MPA (see
    `data.stratified_diff_ci`), one dot per species with its 95% interval.
    Colour marks whether the interval clears zero, so the eye is drawn to
    what the data settles rather than to whatever is longest.

    `per_dep` supplies the deployment universe for the zero fill. Without it
    the function falls back to the deployments present in `per_species`,
    which is every annotated deployment and is what the front page passes.
    """
    section("MPA effect by species")
    st.caption(
        "One dot per species: how many more (or fewer) individuals a "
        "deployment inside an MPA records at peak, compared with one outside "
        "the **same** MPA. The bar is the 95% range. Coloured where the range "
        "clears zero, grey where it spans zero and the direction is therefore "
        "unsettled."
    )

    real = real_species(per_species)
    real = real[real["Group"].notna()]
    if real.empty:
        st.info("No classified deployments to compare.")
        return
    if "reserve_code" not in real.columns:
        st.info("No MPA code on these deployments, so the comparison cannot be split.")
        return

    # The deployment universe for the zero fill: every classified deployment,
    # not only those recording the species.
    if per_dep is None or per_dep.empty:
        deployments = real[["drop_id", "Group", "reserve_code"]].drop_duplicates(
            "drop_id"
        )
    else:
        deployments = per_dep.loc[
            per_dep["Group"].isin([PROTECTED, UNPROTECTED]),
            ["drop_id", "Group", "reserve_code"],
        ].drop_duplicates("drop_id")
    if deployments.empty:
        st.info("No deployments both inside and outside an MPA in this selection.")
        return

    seen = (
        real.groupby("scientific_name")["drop_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    min_deps = st.slider(
        "Minimum deployments recording the species",
        2,
        50,
        10,
        key="species_effect_min",
        help="A species seen in only a handful of deployments gets an "
        "interval too wide to say anything, and crowds out the ones that can "
        "be read. Raising this shows fewer, better-evidenced species.",
    )
    eligible = seen[seen >= min_deps]
    if eligible.empty:
        st.warning(
            f"No species appear in at least {min_deps} deployments. Lower the "
            "threshold or widen the filters."
        )
        return
    # Capped as well as thresholded: the interval is bootstrapped per species
    # per MPA, so an uncapped list is both unreadable and slow.
    top_n = top_n_slider(
        "Show top N species (by deployments recording them)",
        len(eligible),
        15,
        "species_effect_topn",
    )
    chosen = list(eligible.head(top_n).index)

    grid = species_value_grid(deployments, real, chosen)
    rows = []
    for name in chosen:
        result = stratified_diff_ci(
            grid[grid["scientific_name"] == name],
            value_col="maxn",
            group_col="Group",
            stratum_col="reserve_code",
            inside=PROTECTED,
            outside=UNPROTECTED,
        )
        if result["n_strata"] == 0:
            continue
        verdict = (
            "Higher inside"
            if result["lo"] > 0
            else "Higher outside" if result["hi"] < 0 else "Not settled"
        )
        rows.append(
            {
                "Species": common.get(name, name),
                "effect": result["diff"],
                "lo": result["lo"],
                "hi": result["hi"],
                "inside": result["mean_inside"],
                "outside": result["mean_outside"],
                "areas": result["n_strata"],
                "other_way": result["share_other_way"] * 100,
                "Verdict": verdict,
            }
        )

    if not rows:
        st.warning(
            "No species has two deployments both inside and outside the same "
            "MPA, so no effect can be measured within an area."
        )
        return

    effects = pd.DataFrame(rows).sort_values("effect")
    fig = go.Figure()
    for verdict, colour in EFFECT_COLOURS.items():
        part = effects[effects["Verdict"] == verdict]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["effect"],
                y=part["Species"],
                mode="markers",
                name=verdict,
                marker={"size": 11, "color": colour, "symbol": "circle"},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": part["hi"] - part["effect"],
                    "arrayminus": part["effect"] - part["lo"],
                    "color": colour,
                    "thickness": 2,
                    "width": 0,
                },
                customdata=part[
                    ["inside", "outside", "lo", "hi", "areas", "other_way"]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>Inside: %{customdata[0]:.2f}"
                    "<br>Outside: %{customdata[1]:.2f}"
                    "<br>Difference: %{x:+.2f}"
                    "<br>95% range: %{customdata[2]:+.2f} to %{customdata[3]:+.2f}"
                    "<br>Resamples the other way: %{customdata[5]:.1f}%"
                    "<br>Across %{customdata[4]} MPAs<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line_dash="dot", line_color=GUIDE_LINE, line_width=1)
    style(
        fig,
        height=max(340, len(effects) * 28 + 140),
        legend="top",
        hovermode="closest",
    )
    fig.update_xaxes(
        title="Inside − outside, mean MaxN per deployment",
        zeroline=False,
    )
    fig.update_yaxes(title=None)
    st.plotly_chart(fig, key="species_reserve_effect")

    inside_n = int((effects["Verdict"] == "Higher inside").sum())
    outside_n = int((effects["Verdict"] == "Higher outside").sum())
    unsettled = int((effects["Verdict"] == "Not settled").sum())
    dropped = len(seen) - len(eligible)
    st.caption(
        f"Of {len(effects)} species shown, **{inside_n} are clearly higher "
        f"inside** an MPA and **{outside_n} clearly higher outside**; "
        f"{unsettled} have a range spanning zero, so their direction is not "
        f"settled. Means include the deployments where the species was not "
        f"seen, as real zeros, and each difference is measured within an MPA "
        f"before being combined."
        + (
            f" {dropped} rarer species are below the deployment threshold."
            if dropped
            else ""
        )
    )


# ── Where and when ────────────────────────────────────────────────────────


def render_species_by_site(
    picked: pd.DataFrame,
    all_species: pd.DataFrame,
    all_species_mode: bool,
    mpa_selected: bool = False,
) -> None:
    """Where the chosen species are seen, and how much.

    Colour is frequency, not abundance: the share of a site's annotated
    deployments that recorded the species. Frequency is comparable between
    species, whereas MaxN is not, a site full of sweep and a site with one
    snapper are not read off the same colour scale sensibly. Mean MaxN is in the
    hover for when the magnitude is the question.

    Species-view only: the front page used to show a cut-down version, and
    now links here instead.
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

    # ── What a row is ───────────────────────────────────────────────────────
    # Sites are the finest cut, and often too fine: 592 of ~1,000 annotated
    # sites hold exactly ONE deployment, so their frequency can only be 0%
    # or 100% and the panel turns into a chequerboard. Grouping by MPA and
    # protection pools those singletons into ~28 groups that carry ten or
    # more deployments each, where a percentage means something.
    row_choice = "Site"
    if "reserve_code" in picked.columns:
        row_choice = st.radio(
            "Rows",
            ["MPA and protection", "Site"],
            horizontal=True,
            key="species_site_rows",
            help="Most sites hold a single deployment, so per-site "
            "frequencies are nearly all 0% or 100%. Pooling by MPA and "
            "protection gives each row enough deployments for a share to "
            "be meaningful; sites stay available for the fine detail.",
        )

    def with_row(frame: pd.DataFrame) -> pd.DataFrame:
        """Attach the chosen row key as `_row`."""
        frame = frame.copy()
        if row_choice == "Site":
            frame["_row"] = frame["site_id"]
            return frame
        groups = protection_group(frame["protection_status"]).fillna(
            UNRECORDED_PROTECTION
        )
        frame["_row"] = frame["reserve_code"].fillna("(no code)") + " · " + groups
        return frame

    picked = with_row(picked)
    all_species = with_row(all_species)
    row_noun = "site" if row_choice == "Site" else "group"

    # Denominator is every annotated deployment in the row, not just those
    # recording the species, so a zero cell means "looked and did not see it".
    deps_per_site = effort_per(all_species, "_row", "deployments")
    observed = (
        picked.groupby(["_row", "Species"])
        .agg(seen=("drop_id", "nunique"), mean_maxn=("maxn", "mean"))
        .reset_index()
    )
    if observed.empty:
        st.info("No sites recorded the chosen species.")
        return

    # ── Which sites are rows ────────────────────────────────────────────────
    # The row set is every ANNOTATED site in scope, not only the sites that
    # recorded the species. Building it from the sightings (as this did) made
    # every row contain at least one sighting, so a single-species selection
    # could never show an absence and the map read "seen everywhere".
    #
    # With an MPA picked in the sidebar: all of its sites, the complete
    # picture for that area. Programme-wide there are ~1,000 sites, so a
    # stable random sample of 25 stands in (fixed seed, so it does not
    # reshuffle on every click). Random rather than the old "top 25 by
    # frequency", which picked exactly the sites where the species shows
    # most and so guaranteed a wall of high percentages.
    row_universe = sorted(deps_per_site["_row"].dropna().unique())
    species_shown = sorted(observed["Species"].unique())
    # Pooled rows are few enough to show whole; only the site cut needs
    # thinning, and only when no MPA narrows it.
    sampled = False
    keep_rows = row_universe
    if row_choice == "Site" and not mpa_selected and len(row_universe) > 25:
        import random

        keep_rows = sorted(random.Random(42).sample(row_universe, 25))
        sampled = True

    # Full (row x species) grid, so a row that never recorded the species
    # is a real zero rather than a missing one.
    grid = (
        pd.MultiIndex.from_product(
            [keep_rows, species_shown], names=["_row", "Species"]
        )
        .to_frame(index=False)
        .merge(observed, on=["_row", "Species"], how="left")
        .merge(deps_per_site, on="_row", how="left")
    )
    grid["seen"] = grid["seen"].fillna(0)
    grid = grid[grid["deployments"].notna() & (grid["deployments"] > 0)]
    if grid.empty:
        st.info(f"No annotated {row_noun}s in this selection.")
        return
    grid["Frequency"] = (grid["seen"] / grid["deployments"]).round(3)
    # mean_maxn stays NaN where the species was never recorded: the
    # abundance panel leaves those cells blank, "when present, how many" has
    # no answer where it was never present.
    grid["mean_maxn"] = grid["mean_maxn"].round(2)

    hit_rows = int((grid.groupby("_row")["seen"].max() > 0).sum())
    if sampled:
        st.caption(
            f"A random **25 of the {len(row_universe)} annotated sites** in "
            "this selection (the same 25 between clicks), so no species' "
            f"hotspots decide the picture. {hit_rows} of them recorded "
            "something from the selection. **Pick an MPA in the sidebar to "
            "see all of its sites instead**, or switch the rows to MPA and "
            "protection above."
        )
    elif len(keep_rows) > 40:
        st.caption(
            f"All **{len(keep_rows)} annotated {row_noun}s** in this "
            f"selection; {hit_rows} recorded something from it. Scroll the "
            "panels."
        )
    # Plotly puts the first category at the bottom, so the list is reversed to
    # read A to Z downwards.
    row_order = sorted(grid["_row"].unique(), reverse=True)
    species_order = sorted(grid["Species"].unique())

    # The deployment count rides on the row label — "SITE (n)" — rather than
    # in a third aligned panel: neither a bar chart nor a one-column heatmap
    # keeps its rows level with these heatmaps (each chart's bottom margin is
    # sized by its own x labels), and a count inside the label cannot drift.
    deps_by_row = grid.drop_duplicates("_row").set_index("_row")["deployments"]
    # Compact on the axis — "SLI_012 (2)" — with the words in the hover: the
    # hovertemplates below spell out what the bracketed number means.
    label_of = {s: f"{s} ({int(n)})" for s, n in deps_by_row.items()}
    grid = grid.assign(Site=grid["_row"].map(label_of))
    site_order = [label_of[s] for s in row_order]

    # Row height shrinks once the list is long, so an MPA with 200 sites is
    # a scrollable panel rather than a 4,000px wall.
    n_rows = grid["_row"].nunique()
    height = max(320, (22 if n_rows <= 40 else 14) * n_rows)
    left, right = st.columns(2)

    with left:
        st.markdown("**How often it is seen**")
        fig = px.density_heatmap(
            grid,
            x="Species",
            y="Site",
            z="Frequency",
            histfunc="max",
            color_continuous_scale=DETECTION_FREQUENCY_SCALE,
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
            hovertemplate=f"{row_noun.capitalize()} (annotated deployments): "
            "%{y}<br>%{x}<br>Seen in %{z:.0%} of them<extra></extra>"
        )
        st.plotly_chart(fig, key="species_by_site_freq")
        st.caption(
            "Share of that site's annotated deployments recording the "
            "species. Comparable between species, because every cell is "
            "a proportion of the same thing."
        )

    with right:
        st.markdown("**How many, when it is seen**")
        fig = px.density_heatmap(
            grid,
            x="Species",
            y="Site",
            z="mean_maxn",
            histfunc="max",
            color_continuous_scale=ABUNDANCE_SCALE,
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
            hovertemplate=f"{row_noun.capitalize()} (annotated deployments): "
            "%{y}<br>%{x}<br>Mean MaxN when seen: %{z:.2f}<extra></extra>"
        )
        st.plotly_chart(fig, key="species_by_site_maxn")
        st.caption(
            "Mean MaxN across the deployments that recorded it. **One colour "
            "scale spans every selected species**, so a naturally abundant "
            "schooling species washes out a rarer one: read this down a column, "
            "not across a row."
        )

    st.caption(
        "Every annotated site in the selection is a row, including the ones "
        "that never recorded the species: on the left they read **0%** (the "
        "palest shade), on the right they are **blank**, since 'how many "
        "when present' has no answer where it was never present. Those "
        "zeros are real absences, not gaps. The two panels disagree "
        "usefully: a species can be seen almost everywhere in ones and twos, "
        "or rarely but in numbers. **The number after each site name is its "
        "annotated deployments**, the denominator behind that row. A value "
        "resting on two deployments is far less certain than one resting on "
        "forty."
    )

    with st.expander("The numbers behind this"):
        # `Site` (the labelled axis value) is dropped in favour of the plain
        # site_id: the table has a real column for the deployment count, so
        # the label's annotation would repeat it.
        table = grid.drop(columns="Site").rename(
            columns={
                "_row": "Site",
                "seen": "Deployments seen in",
                "deployments": "Deployments annotated",
                "mean_maxn": "Mean MaxN",
            }
        )
        view = table[
            [
                "Site",
                "Species",
                "Deployments seen in",
                "Deployments annotated",
                "Mean MaxN",
                "Frequency",
            ]
        ].sort_values(["Site", "Species"])
        st.dataframe(
            # Heatmap shading on the abundance column, matching the chart
            # beside it; `background_gradient` picks black or white text per
            # cell so it stays readable in either theme.
            view.style.background_gradient(
                cmap="Greens", subset=["Mean MaxN"], vmin=0
            ).format("{:.2f}", subset=["Mean MaxN"]),
            hide_index=True,
            width="stretch",
            height=320,
            column_config={
                # Pinned so the row keeps its identity when the table scrolls
                # sideways on a narrow screen.
                "Site": st.column_config.TextColumn("Site", pinned=True),
                # Frequency is stored 0-1; the bar's scale and the printed
                # number must both say so (max_value=100 rendered every bar
                # as a sliver reading "0%").
                "Frequency": st.column_config.ProgressColumn(
                    "Frequency", min_value=0, max_value=1, format="percent"
                ),
            },
        )


# Which names are not species comes from config (`reporting.non_species_classes`),
# the same list the MPA view's species panels exclude, this module used to
# carry its own copy and the two had already drifted.

# Opened on by default. Three is enough to read at a glance, and these are the
# ones a New Zealand reader recognises without a key. The full indicator list
# lives in config (`reporting.indicator_species`).
DEFAULT_SPECIES = config.indicator_species[:3]


def render_species_over_time(
    per_species: pd.DataFrame,
    common: dict,
    with_sites: bool = True,
    species_filter: list | None = None,
    per_species_all: pd.DataFrame | None = None,
    mpa_selected: bool = False,
) -> tuple:
    """Per-species trends, following the sidebar species filter.

    No picker of its own: the sidebar Species filter is the one control
    (`species_filter` is `ctx["species"]`, and the frame arriving here is
    already narrowed by it). With the filter empty the chart opens on
    snapper, one species, because three lines per species (inside /
    outside / unknown) make a multi-species default unreadable.

    Returns `(picked, all_species_mode)` so a caller can render
    `render_species_by_site` separately when `with_sites=False` (the front
    page passes False and skips the heatmaps entirely). Every early exit
    returns `(None, False)`, the same shape, so a caller can unpack without
    checking first.
    """
    section("Species over time")
    st.caption(
        "Abundance for chosen species by survey year, inside an MPA against "
        "outside. Pick species with the sidebar **Species** filter."
    )

    real = real_species(per_species)
    real = real[real["survey_year"].notna()]
    if real.empty:
        st.info("No species annotations with a survey year.")
        return None, False
    # Unknown protection becomes its own group rather than vanishing: the
    # legacy imports live at sites the SharePoint sites list does not know,
    # and dropping Group-less rows here cut those years off the time axis.
    real = real.assign(Group=real["Group"].fillna(UNRECORDED_PROTECTION))

    # NaN species are absence records; they stay in `classified` below (a
    # reviewed-empty deployment belongs in the per-year denominator) but they
    # are not a species anyone can pick.
    available = sorted(real["scientific_name"].dropna().unique())
    all_species_mode = False
    if species_filter:
        # The sidebar already narrowed the frame; show everything it kept.
        chosen = available
    else:
        # Filter empty: open on snapper (first in config's indicator list),
        # falling back to every species combined where snapper is absent.
        chosen = [s for s in DEFAULT_SPECIES if s in available][:1]
        if not chosen:
            all_species_mode = True
            chosen = available
            st.caption(
                f"This is **all {len(available)} species combined**, snapper "
                "is not in this selection. Use the sidebar Species filter to "
                "break out particular species."
            )

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
    if picked.empty:
        st.info("No classified deployments recorded the chosen species.")
        return None, False

    # The denominator is every classified deployment that year, not just those
    # recording the species. Dividing by the latter reports the mean among
    # deployments that already had it, which cannot fall and always looks like a
    # healthy population. From the species-UNfiltered frame (`per_species_all`)
    # when the caller has one: a denominator built from the filtered frame
    # collapses to the deployments that recorded the picked species, which
    # pinned the Frequency panel at 100% whenever the sidebar filter was on.
    # Same fill as `real` above: an unknown-protection deployment counts in
    # the Unknown line's denominator, not in nobody's.
    denom_src = per_species_all if per_species_all is not None else per_species
    classified = denom_src[denom_src["survey_year"].notna()].assign(
        Group=lambda f: f["Group"].fillna(UNRECORDED_PROTECTION)
    )
    deps_per_year = effort_per(
        classified.assign(Year=classified["survey_year"].astype("Int64")),
        ["Year", "Group"],
        "deployments",
    )

    totals = (
        picked.groupby([picked["survey_year"].astype("Int64"), "Species", "Group"])
        .agg(
            total=("maxn", "sum"),
            seen=("drop_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"survey_year": "Year"})
        .merge(deps_per_year, on=["Year", "Group"], how="left")
    )
    if totals.empty:
        st.info("No annotations for the chosen species in a dated survey.")
        return None, False
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
                category_orders={
                    "Group": [PROTECTED, UNPROTECTED, UNRECORDED_PROTECTION]
                },
                line_dash_map=GROUP_DASHES,
                # Built from every AVAILABLE species, not the selected ones, so
                # ticking a species off does not repaint the rest. Plotly ignores
                # map entries for series that are absent. "Same colour inside and
                # outside" only means something if the colour is stable to begin
                # with.
                color_discrete_map=colour_map,
            )
            # Only the inside/outside pairs start selected; "Other" and
            # "Not recorded" sit greyed out in the legend, one click
            # from visible. Autorange would then shrink the axis to the
            # visible lines and hide that the greyed ones reach further back,
            # so the axis is pinned to the span of ALL the data instead.
            focus_inside_outside(fig)
            yr = totals["Year"].dropna()
            if not yr.empty:
                fig.update_xaxes(range=[int(yr.min()) - 0.5, int(yr.max()) + 0.5])
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
        "Partial regimes ('Other') and drops with unknown protection start "
        "greyed out in the legend, click them to add their lines. Frequency "
        "is the steadier of the two measures: mean MaxN moves with one busy "
        "deployment, whereas a species is either present or not."
    )
    # Say what the greyed-out unknown lines hold, with numbers, so an axis
    # reaching further back than the visible lines is explained on the page.
    unknown_drops = classified.loc[
        classified["Group"] == UNRECORDED_PROTECTION, "drop_id"
    ].nunique()
    if unknown_drops:
        u_years = classified.loc[
            classified["Group"] == UNRECORDED_PROTECTION, "survey_year"
        ]
        st.info(
            f"**{unknown_drops:,} deployments in this selection have no "
            "protection status**: their sites are not in the SharePoint sites "
            "list, which is where protection metadata lives. Most are legacy "
            f"imports, they span {int(u_years.min())}–{int(u_years.max())} "
            "and are the only data in the earliest years, which is why the "
            "axis reaches further back than the visible lines. Click "
            f"'{UNRECORDED_PROTECTION}' in the legend to see them; once the sites "
            "gain SharePoint records they move onto their proper side "
            "automatically."
        )

    if with_sites:
        st.divider()
        # The heatmaps' per-site denominators come from the unfiltered frame
        # for the same reason as `deps_per_year` above.
        render_species_by_site(
            picked, denom_src, all_species_mode, mpa_selected=mpa_selected
        )
    return picked, all_species_mode


def render_yearly_trend(df: pd.DataFrame, species_filter: list | None = None) -> None:
    """One line per site over the years, for a single species.

    One species at a time by construction: a line per (site x species) is
    unreadable. The sidebar Species filter chooses it where it names exactly
    one; with several picked, or none, a selectbox picks among them, opening
    on snapper.
    """
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

    if species_filter and len(all_species) == 1:
        # The sidebar filter already narrowed to one species; a selectbox
        # offering that single option is a control with nothing to choose.
        species = all_species[0]
        st.caption(f"Species: **{species}**, from the sidebar Species filter.")
    else:
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
        st.dataframe(
            trend.sort_values(["site_id", "survey_year"]),
            hide_index=True,
            column_config={"site_id": st.column_config.TextColumn("Site", pinned=True)},
        )
