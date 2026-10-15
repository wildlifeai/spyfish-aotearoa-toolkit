"""Unit tests for the MPA view's data layer (`app/doc_report/mpa_data.py`).

These functions were inline in `charts/mpa.py`, between the widgets that pick
a species and the plotly calls that draw it. The rules worth pinning are the
ones a chart cannot show you are wrong about: which line a deployment with no
protection metadata lands on, whether an absence record counts as a species,
what "per deployment" divides by when a site saw nothing.

`app/` is not a package, so it goes on the path the same way the Streamlit
pages do.
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[2] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from doc_report import mpa_data  # noqa: E402
from ecology_data import OTHER_PROTECTION, PROTECTED, UNPROTECTED  # noqa: E402

from spyfish.config.base import NULL_DEPLOYMENT  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402

DROP_A = "KSF_20240124_BUV_KSF_085_01"
DROP_B = "KSF_20240124_BUV_KSF_085_02"
DROP_C = "KSF_20240124_BUV_KSF_086_01"
DROP_D = "TUK_20230310_BUV_TUK_012_01"
SITE_A = "KSF_085"
SITE_B = "KSF_086"

RESERVE = config.protected_statuses[0]
NO_PROTECTION = config.unprotected_statuses[0]
PARTIAL_STATUS = "Mataitai"
SNAPPER = "Pagrus auratus"
LOBSTER = "Jasus edwardsii"
BLUE_COD = "Parapercis colias"


def _sightings(rows) -> pd.DataFrame:
    """A `df_context`-shaped frame: one row per (drop, species) with `maxn`."""
    frame = pd.DataFrame(
        rows,
        columns=[
            "drop_id",
            "site_id",
            "scientific_name",
            "maxn",
            "survey_year",
            "protection_status",
        ],
    )
    return frame.assign(display_name=frame["scientific_name"])


def _effort(rows) -> pd.DataFrame:
    """An `effort_view`-shaped frame: one row per analysed deployment."""
    return pd.DataFrame(
        rows, columns=["drop_id", "site_id", "survey_year", "protection_status"]
    )


# ── species_choices ───────────────────────────────────────────────────────────


def test_species_choices_orders_indicators_by_config_then_alphabetical():
    frame = _sightings(
        [
            (DROP_A, SITE_A, "Zeus faber", 1, 2024, RESERVE),
            (DROP_A, SITE_A, LOBSTER, 1, 2024, RESERVE),
            (DROP_A, SITE_A, "Chrysophrys sp", 1, 2024, RESERVE),
            (DROP_A, SITE_A, SNAPPER, 1, 2024, RESERVE),
        ]
    )
    sci_to_display, indicators, others = mpa_data.species_choices(frame)
    expected = [s for s in config.indicator_species if s in (SNAPPER, LOBSTER)]
    assert indicators == expected
    assert others == ["Chrysophrys sp", "Zeus faber"]
    assert sci_to_display[SNAPPER] == SNAPPER


def test_species_choices_keeps_the_unidentified_bucket_and_absences_out():
    frame = _sightings(
        [
            (DROP_A, SITE_A, config.unidentified_label, 3, 2024, RESERVE),
            (DROP_A, SITE_A, NULL_DEPLOYMENT, 0, 2024, RESERVE),
            (DROP_B, SITE_A, None, 0, 2024, RESERVE),
            (DROP_A, SITE_A, SNAPPER, 1, 2024, RESERVE),
        ]
    )
    sci_to_display, _, others = mpa_data.species_choices(frame)
    assert list(sci_to_display) == [SNAPPER]
    assert others == []


# ── trend_protection ──────────────────────────────────────────────────────────


def test_trend_protection_per_status_merges_unknown_and_missing():
    status = pd.Series([RESERVE, "unknown", None, PARTIAL_STATUS])
    out = mpa_data.trend_protection(status, by_status=True)
    assert out.tolist() == [
        RESERVE,
        mpa_data.NOT_RECORDED,
        mpa_data.NOT_RECORDED,
        PARTIAL_STATUS,
    ]


def test_trend_protection_grouped_uses_the_shared_three_way_bucketing():
    status = pd.Series([RESERVE, NO_PROTECTION, PARTIAL_STATUS, "unknown"])
    out = mpa_data.trend_protection(status, by_status=False)
    assert out.tolist()[:3] == [PROTECTED, UNPROTECTED, OTHER_PROTECTION]
    # Not recorded sits on neither line rather than being guessed into one.
    assert pd.isna(out.iloc[3])


# ── trend_measure ─────────────────────────────────────────────────────────────
# Richness itself is counted once, in `data.richness_by`, and pinned in
# test_report_data.py; these check the MPA paths read the right column of it.


def test_trend_measure_richness_ignores_absences_and_unidentified():
    src = _sightings(
        [
            (DROP_A, SITE_A, SNAPPER, 2, 2024, RESERVE),
            (DROP_B, SITE_A, SNAPPER, 1, 2024, RESERVE),
            (DROP_B, SITE_A, LOBSTER, 1, 2024, RESERVE),
            (DROP_B, SITE_A, config.unidentified_label, 4, 2024, RESERVE),
            (DROP_C, SITE_B, None, 0, 2024, RESERVE),
        ]
    ).assign(protection=RESERVE)
    out = mpa_data.trend_measure(src, None)
    assert out.to_dict("records") == [
        {"survey_year": 2024, "protection": RESERVE, "value": 2.0}
    ]


def test_trend_measure_single_species_carries_the_summed_maxn():
    src = _sightings(
        [
            (DROP_A, SITE_A, SNAPPER, 2, 2024, RESERVE),
            (DROP_B, SITE_A, SNAPPER, 3, 2024, RESERVE),
            (DROP_B, SITE_A, LOBSTER, 9, 2024, RESERVE),
        ]
    ).assign(protection=RESERVE)
    out = mpa_data.trend_measure(src, SNAPPER)
    assert out["total"].iloc[0] == 5


# ── protection_trend ──────────────────────────────────────────────────────────


def test_protection_trend_divides_by_every_analysed_deployment():
    # Two analysed deployments in 2024, snapper on one: mean is 4 / 2.
    df = _sightings([(DROP_A, SITE_A, SNAPPER, 4, 2024, RESERVE)])
    effort = _effort([(DROP_A, SITE_A, 2024, RESERVE), (DROP_B, SITE_A, 2024, RESERVE)])
    trend, classes, _ = mpa_data.protection_trend(df, effort, SNAPPER, by_status=True)
    assert classes == [RESERVE]
    row = trend.iloc[0]
    assert row["value"] == 2.0 and row["analysed"] == 2


def test_protection_trend_fills_unsurveyed_years_with_nan_and_surveyed_with_zero():
    df = _sightings([(DROP_A, SITE_A, SNAPPER, 4, 2022, RESERVE)])
    effort = _effort([(DROP_A, SITE_A, 2022, RESERVE), (DROP_D, SITE_B, 2024, RESERVE)])
    trend, _, _ = mpa_data.protection_trend(df, effort, SNAPPER, by_status=True)
    by_year = trend.set_index("survey_year")["value"]
    assert by_year.index.tolist() == [2022, 2023, 2024]
    assert by_year[2022] == 4.0
    assert math.isnan(by_year[2023])  # not surveyed: the line breaks
    assert by_year[2024] == 0.0  # surveyed, none seen: a real zero


def test_protection_trend_keeps_a_status_with_effort_but_no_sighting():
    df = _sightings([(DROP_A, SITE_A, SNAPPER, 1, 2024, RESERVE)])
    effort = _effort(
        [(DROP_A, SITE_A, 2024, RESERVE), (DROP_B, SITE_B, 2024, NO_PROTECTION)]
    )
    _, classes, _ = mpa_data.protection_trend(df, effort, SNAPPER, by_status=True)
    assert set(classes) == {RESERVE, NO_PROTECTION}


def test_protection_trend_grouped_always_draws_both_sides():
    df = _sightings([(DROP_A, SITE_A, SNAPPER, 1, 2024, RESERVE)])
    effort = _effort([(DROP_A, SITE_A, 2024, RESERVE)])
    trend, classes, _ = mpa_data.protection_trend(df, effort, SNAPPER, by_status=False)
    assert classes == ["Protected", "Unprotected"]
    assert len(trend) == 2


def test_protection_trend_richness_absence_only_year_reads_zero():
    # "Reviewed, nothing seen" is zero species, not a gap: the surveyed cell
    # holding only absence records has no measured row, and is filled with 0.
    df = _sightings([(DROP_A, SITE_A, None, 0, 2024, RESERVE)])
    effort = _effort([(DROP_A, SITE_A, 2024, RESERVE)])
    trend, _, _ = mpa_data.protection_trend(df, effort, None, by_status=True)
    assert trend["value"].iloc[0] == 0


def test_protection_trend_richness_unsurveyed_year_stays_a_gap():
    df = _sightings(
        [
            (DROP_A, SITE_A, SNAPPER, 1, 2022, RESERVE),
            (DROP_B, SITE_A, SNAPPER, 1, 2024, RESERVE),
        ]
    )
    effort = _effort([(DROP_A, SITE_A, 2022, RESERVE), (DROP_B, SITE_A, 2024, RESERVE)])
    trend, _, _ = mpa_data.protection_trend(df, effort, None, by_status=True)
    by_year = trend.set_index("survey_year")["value"]
    assert by_year[2022] == 1 and by_year[2024] == 1
    assert math.isnan(by_year[2023])


def test_protection_trend_empty_when_nothing_is_dated():
    df = _sightings([(DROP_A, SITE_A, SNAPPER, 1, None, RESERVE)])
    effort = _effort([(DROP_A, SITE_A, None, "unknown")])
    trend, _, trend_effort = mpa_data.protection_trend(df, effort, SNAPPER, True)
    assert trend.empty
    # The effort frame still comes back classified, for the 'Not recorded' note.
    assert trend_effort["protection"].tolist() == [mpa_data.NOT_RECORDED]


# ── site_values / bubble_reference ────────────────────────────────────────────


def _site_geo(analysed: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {"site_id": list(analysed), "analysed": list(analysed.values())}
    )


def test_site_values_species_mean_counts_sites_that_saw_nothing():
    df = _sightings(
        [
            (DROP_A, SITE_A, SNAPPER, 6, 2024, RESERVE),
            (DROP_C, SITE_B, LOBSTER, 2, 2024, RESERVE),
        ]
    )
    out = mpa_data.site_values(df, _site_geo({SITE_A: 3, SITE_B: 1}), SNAPPER)
    assert out.set_index("site_id")["value"].to_dict() == {SITE_A: 2.0, SITE_B: 0.0}


def test_site_values_zero_effort_divides_by_one():
    df = _sightings([(DROP_A, SITE_A, SNAPPER, 3, 2024, RESERVE)])
    out = mpa_data.site_values(df, _site_geo({SITE_A: 0}), SNAPPER)
    assert out["value"].tolist() == [3.0]


def test_site_values_richness_skips_absences_and_keeps_empty_sites_at_zero():
    df = _sightings(
        [
            (DROP_A, SITE_A, SNAPPER, 1, 2024, RESERVE),
            (DROP_A, SITE_A, LOBSTER, 1, 2024, RESERVE),
            (DROP_B, SITE_A, SNAPPER, 2, 2024, RESERVE),
            (DROP_A, SITE_A, "fish", 5, 2024, RESERVE),
            (DROP_C, SITE_B, None, 0, 2024, RESERVE),
        ]
    )
    out = mpa_data.site_values(df, _site_geo({SITE_A: 2, SITE_B: 1}), None)
    assert out.set_index("site_id")["value"].to_dict() == {SITE_A: 2.0, SITE_B: 0}


def test_bubble_reference_is_fixed_across_species():
    df = _sightings(
        [
            (DROP_A, SITE_A, SNAPPER, 1, 2024, RESERVE),
            (DROP_A, SITE_A, BLUE_COD, 8, 2024, RESERVE),
        ]
    )
    effort = pd.DataFrame({"site_id": [SITE_A], "analysed": [2]})
    geo = _site_geo({SITE_A: 2})
    refs = {
        sci: mpa_data.bubble_reference(
            df, effort, mpa_data.site_values(df, geo, sci), sci
        )
        for sci in (SNAPPER, BLUE_COD)
    }
    # The largest per-deployment value over ALL species, whichever is picked.
    assert refs == {SNAPPER: 4.0, BLUE_COD: 4.0}


def test_bubble_reference_falls_back_to_one_when_nothing_was_seen():
    mapped = pd.DataFrame({"site_id": [SITE_A], "value": [0.0]})
    assert mpa_data.bubble_reference(pd.DataFrame(), None, mapped, None) == 1.0


# ── composition_shares ────────────────────────────────────────────────────────


def _exp(rows) -> pd.DataFrame:
    """An `experiments_frame`-shaped frame."""
    frame = pd.DataFrame(
        rows,
        columns=[
            "drop_id",
            "reserve_code",
            "scientific_name",
            "maxn",
            "survey_year",
            "protection_status",
        ],
    )
    return frame.assign(display_name=frame["scientific_name"])


def test_composition_shares_rolls_the_tail_into_other():
    df = _exp(
        [
            (DROP_A, "KSF", SNAPPER, 6, 2024, RESERVE),
            (DROP_A, "KSF", LOBSTER, 3, 2024, RESERVE),
            (DROP_A, "KSF", BLUE_COD, 1, 2024, RESERVE),
            (DROP_D, "TUK", SNAPPER, 20, 2023, RESERVE),
        ]
    )
    plot_df, order = mpa_data.composition_shares(df, top_n=1)
    ksf = plot_df[plot_df["reserve_code"] == "KSF"].set_index("display_name")["pct"]
    assert ksf.to_dict() == {SNAPPER: 60.0, "Other": 40.0}
    # A reserve whose top N is everything gets no Other row.
    assert plot_df[plot_df["reserve_code"] == "TUK"]["display_name"].tolist() == [
        SNAPPER
    ]
    assert order == ["TUK", "KSF"]


# ── reserve trends ────────────────────────────────────────────────────────────


def test_reserve_trend_rows_keeps_absences_drops_undated_and_reserveless_rows():
    """An absence record is a deployment that was looked at: it stays, so the
    per-deployment means below divide by it."""
    df = _exp(
        [
            (DROP_A, "KSF", SNAPPER, 1, 2024, RESERVE),
            (DROP_B, "KSF", None, 0, 2024, RESERVE),
            (DROP_C, "KSF", SNAPPER, 1, None, RESERVE),
            (DROP_D, "", SNAPPER, 1, 2023, RESERVE),
        ]
    )
    assert mpa_data.reserve_trend_rows(df)["drop_id"].tolist() == [DROP_A, DROP_B]


def test_split_sides_drops_partial_and_unrecorded_but_counts_only_partial():
    df = _exp(
        [
            (DROP_A, "KSF", SNAPPER, 1, 2024.0, RESERVE),
            (DROP_B, "KSF", SNAPPER, 1, 2024.0, NO_PROTECTION),
            (DROP_C, "KSF", SNAPPER, 1, 2024.0, PARTIAL_STATUS),
            (DROP_D, "TUK", SNAPPER, 1, 2023.0, "unknown"),
        ]
    )
    sided, partial = mpa_data.split_sides(df)
    assert partial == 1
    assert sided["_side"].tolist() == [PROTECTED, UNPROTECTED]
    assert sided["survey_year"].dtype.kind == "i"


def test_reserve_trend_series_richness_is_mean_species_per_deployment():
    df = _exp(
        [
            (DROP_A, "KSF", SNAPPER, 1, 2024, RESERVE),
            (DROP_A, "KSF", LOBSTER, 1, 2024, RESERVE),
            (DROP_B, "KSF", SNAPPER, 1, 2024, RESERVE),
        ]
    )
    sided, _ = mpa_data.split_sides(df)
    out = mpa_data.reserve_trend_series(sided, mpa_data.RICHNESS_METRIC)
    assert out.columns.tolist() == ["reserve_code", "survey_year", "Side", "metric"]
    assert out["metric"].tolist() == [1.5]


def test_reserve_trend_series_richness_does_not_count_the_unidentified_bucket():
    """An ML-only deployment carries one "Unidentified" row after
    `unify_unidentified`. It is not a species: that deployment reads 0, and
    still counts in the denominator."""
    bucket = config.unidentified_label
    df = _exp(
        [
            (DROP_A, "KSF", SNAPPER, 1, 2024, RESERVE),
            (DROP_A, "KSF", bucket, 5, 2024, RESERVE),
            (DROP_B, "KSF", bucket, 2, 2024, RESERVE),
        ]
    )
    sided, _ = mpa_data.split_sides(df)
    out = mpa_data.reserve_trend_series(sided, mpa_data.RICHNESS_METRIC)
    # (1 + 0) / 2, not (2 + 1) / 2
    assert out["metric"].tolist() == [0.5]


def test_reserve_trend_series_richness_counts_a_reviewed_empty_deployment_as_zero():
    df = _exp(
        [
            (DROP_A, "KSF", SNAPPER, 1, 2024, RESERVE),
            (DROP_A, "KSF", LOBSTER, 1, 2024, RESERVE),
            (DROP_B, "KSF", None, 0, 2024, RESERVE),  # reviewed, nothing seen
        ]
    )
    sided, _ = mpa_data.split_sides(mpa_data.reserve_trend_rows(df))
    out = mpa_data.reserve_trend_series(sided, mpa_data.RICHNESS_METRIC)
    assert out["metric"].tolist() == [1.0]  # (2 + 0) / 2


def test_reserve_trend_series_selected_maxn_divides_by_every_deployment():
    """1 of 2 deployments with 4 snapper reads 2.0, not 4.0; an absence
    record is one of those deployments."""
    df = _exp(
        [
            (DROP_A, "KSF", SNAPPER, 4, 2024, RESERVE),
            (DROP_B, "KSF", LOBSTER, 3, 2024, RESERVE),
            (DROP_C, "KSF", None, 0, 2024, NO_PROTECTION),
            (DROP_D, "KSF", SNAPPER, 6, 2024, NO_PROTECTION),
        ]
    )
    sided, _ = mpa_data.split_sides(mpa_data.reserve_trend_rows(df))
    out = mpa_data.reserve_trend_series(sided, mpa_data.SELECTED_METRIC, [SNAPPER])
    assert out.set_index("Side")["metric"].to_dict() == {
        PROTECTED: 2.0,
        UNPROTECTED: 3.0,
    }


def test_reserve_trend_series_selected_species_reads_zero_when_absent():
    df = _exp(
        [
            (DROP_A, "KSF", SNAPPER, 4, 2024, RESERVE),
            (DROP_B, "KSF", SNAPPER, 2, 2024, RESERVE),
            (DROP_C, "KSF", LOBSTER, 3, 2024, NO_PROTECTION),
        ]
    )
    sided, _ = mpa_data.split_sides(df)
    out = mpa_data.reserve_trend_series(sided, mpa_data.SELECTED_METRIC, [SNAPPER])
    assert out.set_index("Side")["metric"].to_dict() == {
        PROTECTED: 3.0,
        UNPROTECTED: 0.0,
    }


def _series(values: dict, reserve="KSF", side=PROTECTED) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reserve_code": reserve,
            "survey_year": list(values),
            "Side": side,
            "metric": list(values.values()),
        }
    )


@pytest.mark.parametrize(
    "values, direction",
    [
        ({2020: 1.0, 2021: 2.0, 2022: 3.0}, mpa_data.RECOVERING),
        ({2020: 3.0, 2021: 2.0, 2022: 1.0}, mpa_data.DECLINING),
        # Slope 0.04/yr on a mean of 1.04: inside the 5% band, so stable.
        ({2020: 1.0, 2021: 1.04, 2022: 1.08}, mpa_data.STABLE),
    ],
)
def test_linear_trends_direction(values, direction):
    out = mpa_data.linear_trends(_series(values), min_years=3)
    assert out["direction"].tolist() == [direction]


def test_linear_trends_too_few_years_has_no_slope_and_sorts_last():
    series = pd.concat(
        [
            _series({2020: 1.0, 2021: 5.0}, reserve="AHE"),
            _series({2020: 1.0, 2021: 2.0, 2022: 3.0}, reserve="KSF"),
        ]
    )
    out = mpa_data.linear_trends(series, min_years=3)
    assert out["MPA"].tolist() == ["KSF", "AHE"]
    short = out.iloc[1]
    assert short["direction"] == mpa_data.INSUFFICIENT
    assert math.isnan(short["slope"])
    assert (short["first_metric"], short["latest_metric"]) == (1.0, 5.0)


# ── inside_share ──────────────────────────────────────────────────────────────


def test_inside_share_counts_deployments_per_side_and_sorts_least_inside_first():
    df = _exp(
        [
            (DROP_A, "KSF", SNAPPER, 1, 2024, RESERVE),
            (DROP_A, "KSF", LOBSTER, 1, 2024, RESERVE),  # same drop, counted once
            (DROP_B, "KSF", SNAPPER, 1, 2024, NO_PROTECTION),
            (DROP_C, "KSF", SNAPPER, 1, 2024, PARTIAL_STATUS),
            (DROP_D, "TUK", SNAPPER, 1, 2023, NO_PROTECTION),
            ("TUK_20230310_BUV_TUK_013_01", "TUK", SNAPPER, 1, 2023, "unknown"),
        ]
    )
    out = mpa_data.inside_share(df)
    assert out.index.tolist() == ["TUK", "KSF"]
    ksf = out.loc["KSF"]
    assert (ksf[PROTECTED], ksf[UNPROTECTED], ksf[OTHER_PROTECTION]) == (1, 1, 1)
    assert ksf["total"] == 3
    assert ksf["share_inside"] == pytest.approx(1 / 3)
    # Unrecorded protection is not in the total.
    assert out.loc["TUK", "total"] == 1


# ── habitat ───────────────────────────────────────────────────────────────────


def test_explode_habitat_counts_a_deployment_under_each_class():
    dep = pd.DataFrame(
        {
            "drop_id": [DROP_A, DROP_B, DROP_C],
            "nzmhcs_abiotic": ["Sand, Reef", "", None],
        }
    )
    out = mpa_data.explode_habitat(dep, "nzmhcs_abiotic")
    assert list(zip(out["drop_id"], out["habitat"])) == [
        (DROP_A, "Sand"),
        (DROP_A, "Reef"),
    ]


def test_explode_habitat_empty_column_still_has_habitat():
    dep = pd.DataFrame({"drop_id": [DROP_A], "nzmhcs_biotic": [None]})
    out = mpa_data.explode_habitat(dep, "nzmhcs_biotic")
    assert out.empty and "habitat" in out.columns


def test_species_peaks_takes_the_peak_interval_and_drops_absences():
    ann = pd.DataFrame(
        {
            "drop_id": [DROP_A, DROP_A, DROP_B, DROP_B],
            "scientific_name": [SNAPPER, SNAPPER, None, "fish"],
            "max_interval": [2, 5, 0, 3],
        }
    ).assign(display_name=lambda d: d["scientific_name"])
    out = mpa_data.species_peaks(ann)
    assert out.to_dict("records") == [
        {"drop_id": DROP_A, "display_name": SNAPPER, "peak": 5}
    ]


def test_substrate_tables_use_unfiltered_denominator_and_five_deployment_floor():
    drops = [f"KSF_20240124_BUV_KSF_0{80 + i}_01" for i in range(6)]
    abiotic = pd.DataFrame(
        {
            "drop_id": drops,
            "habitat": ["Reef"] * 5 + ["Sand"],
        }
    )
    peaks = pd.DataFrame(
        {"drop_id": drops[:2], "display_name": SNAPPER, "peak": [2, 4]}
    )
    joined = abiotic.merge(peaks, on="drop_id", how="inner")
    denom_ann = pd.DataFrame({"drop_id": drops})
    det, maxn, cols = mpa_data.substrate_tables(abiotic, joined, denom_ann)
    assert cols == [SNAPPER]
    # Sand has one annotated deployment, below the floor of five.
    assert det["Substrate"].tolist() == ["Reef"]
    assert det.iloc[0]["Annotated deps"] == 5
    assert det.iloc[0][SNAPPER] == 40  # 2 of 5, not 2 of 2
    assert maxn.iloc[0][SNAPPER] == 3.0


def test_substrate_tables_empty_when_no_class_reaches_five():
    abiotic = pd.DataFrame({"drop_id": [DROP_A], "habitat": ["Reef"]})
    joined = abiotic.assign(display_name=SNAPPER, peak=1)
    det, maxn, _ = mpa_data.substrate_tables(
        abiotic, joined, pd.DataFrame({"drop_id": [DROP_A]})
    )
    assert det.empty and maxn.empty
