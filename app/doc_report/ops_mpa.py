"""Operations - MPA: how each marine protected area is doing on data quality.

The Reporting side of MPA answers what each area holds. This side answers what
state the data from it is in, starting with what each area loses in the field.

Deliberately separate from `mpa.py`: that module is the Reporting view, and the
two share no rendering. What they do share, how a deployment is attributed to
an area, comes from `site_data.split_reserve_rows`.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from .charting import style
from .layout import section
from .site_data import split_reserve_rows


def bad_by_mpa(dep: pd.DataFrame) -> pd.DataFrame:
    """Deployments held, lost and the share lost, per marine protected area.

    A deployment whose site sits between two areas is counted under **both**, so
    these rows do not sum to the programme total. That is the same rule the
    Reporting per-MPA rollup uses, and it comes from the same helper, the
    question is what each area holds, not how the programme divides up.
    """
    per_mpa = (
        split_reserve_rows(
            dep.assign(_bad=dep["is_bad_deployment"].fillna(0).astype(bool))
        )
        .groupby("reserve")
        .agg(
            total=("drop_id", "nunique"),
            bad=("_bad", "sum"),
            surveys=("survey_id", "nunique"),
        )
        .reset_index()
    )
    per_mpa["bad_pct"] = (per_mpa["bad"] / per_mpa["total"] * 100).round(1)
    return per_mpa


def render_bad_per_mpa(dep: pd.DataFrame) -> None:
    """Bad deployments per MPA, the per-survey chart, one level up.

    Same encoding as "Bad / excluded deployments per survey" on the Surveys
    view, on purpose: height is the share, colour depth is the count behind it.
    A survey is one visit, so a bad run there can be one bad week; an area
    losing a tenth of everything ever dropped in it is a standing problem with
    the site, the boat or the conditions.
    """
    section("Bad deployments")
    st.caption(
        "Bad deployments went wrong in the field and are not recoverable. "
        "Height is the share of that area's deployments lost; colour depth is "
        "the count behind it, and each label says how many surveys the share is "
        "drawn from. An area with no losses is left out. Sites between two "
        "areas count under both, so these do not sum to the programme total."
    )

    per_mpa = bad_by_mpa(dep)
    lost = per_mpa[per_mpa["bad"] > 0].sort_values("bad_pct", ascending=False)

    if lost.empty:
        st.success("No bad deployments recorded in the current selection.")
        return

    fig = px.bar(
        lost,
        x="reserve",
        y="bad_pct",
        color="bad",
        color_continuous_scale="Oranges",
        # The share alone hides how much is behind it. A 40% loss across one
        # survey is one bad day; the same share across six is a standing
        # problem with the area, and that is the difference the label carries.
        text=[
            f"{pct:.1f}% · {n} survey{'' if n == 1 else 's'}"
            for pct, n in zip(lost["bad_pct"], lost["surveys"])
        ],
        # Surveys is not in the hover: the bar label already carries it, and
        # the same number in two places reads as two different numbers.
        hover_data={"total": True, "bad": True, "reserve": False},
        labels={
            "reserve": "MPA",
            "bad_pct": "% bad deployments",
            "bad": "Count",
            "surveys": "Surveys",
            "total": "Deployments",
        },
        category_orders={"reserve": lost["reserve"].tolist()},
        height=380,
    )
    fig.update_traces(textposition="outside", cliponaxis=False, textangle=0)
    fig.add_hline(
        y=10,
        line_dash="dash",
        line_color="#EF5350",
        line_width=1,
        annotation_text="10% threshold",
        annotation_position="right",
        annotation_font_color="#EF5350",
    )
    style(
        fig,
        xaxis_tickangle=-40,
        coloraxis_colorbar_title="Bad count",
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="% bad deployments", ticksuffix="%")
    st.plotly_chart(fig, key="ops_bad_per_mpa")

    with st.expander("All MPAs, including those with no losses"):
        table = per_mpa.sort_values("bad_pct", ascending=False)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "reserve": st.column_config.TextColumn("MPA", width="large"),
                "total": st.column_config.NumberColumn("Deployments"),
                "surveys": st.column_config.NumberColumn("Surveys"),
                "bad": st.column_config.NumberColumn("Bad"),
                "bad_pct": st.column_config.NumberColumn("Bad %", format="%.1f%%"),
            },
            column_order=["reserve", "surveys", "total", "bad", "bad_pct"],
        )


def render(ctx: dict) -> None:
    st.caption(
        "Data quality per marine protected area. What each area holds is on the "
        "Reporting side."
    )

    dep = ctx["deployments"]
    if dep.empty:
        st.info("No deployments match the current filters.")
        return

    render_bad_per_mpa(dep)
