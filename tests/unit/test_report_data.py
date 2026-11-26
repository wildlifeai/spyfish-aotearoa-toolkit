"""Unit tests for the reporting data layer (`app/doc_report/data.py`).

Every reporting bug found during the dashboard rebuild lived here rather than
in the drawing: a protection label decided by majority vote, an arrival time
that was really a peak time, the ML catch-all counted as a species, MaxN summed
across intervals instead of peaked. None of them raised; each produced a
plausible chart with wrong numbers.

These functions are pure — frame in, frame out — so they are cheap to pin, and
pinning them is what stops the next silent version.

`app/` is not a package, so it goes on the path the same way the Streamlit
pages do.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[2] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from doc_report import data as report_data  # noqa: E402
from ecology_data import (  # noqa: E402
    OTHER_PROTECTION,
    PROTECTED,
    UNPROTECTED,
    protection_group,
)

DROP = "KSF_20240124_BUV_KSF_085_01"
OTHER_DROP = "KSF_20240124_BUV_KSF_085_02"


def _annotations(rows) -> pd.DataFrame:
    """An annotations frame in the shape `load_annotations` returns."""
    return pd.DataFrame(
        rows,
        columns=[
            "drop_id",
            "scientific_name",
            "max_interval",
            "annotated_by",
            "time_of_max_seconds",
        ],
    ).assign(display_name=lambda df: df["scientific_name"])


# ── MaxN is a peak, never a sum ──────────────────────────────────────────────


def test_species_maxn_takes_the_peak_across_intervals():
    """The whole point: one row per interval, and MaxN is the largest.

    Summing counts the same fish once per interval it appears in. On real data
    that turned 24 into 698.
    """
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 2, "expert", 10.0),
            (DROP, "Pagrus auratus", 7, "expert", 20.0),
            (DROP, "Pagrus auratus", 3, "expert", 30.0),
        ]
    )
    out = report_data.species_maxn(ann)
    assert len(out) == 1
    assert out.loc[0, "maxn"] == 7


def test_deployment_maxn_peaks_per_species_then_sums_across_them():
    """Two steps, in that order: peak within a species, add between species."""
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 2, "expert", 10.0),
            (DROP, "Pagrus auratus", 6, "expert", 20.0),
            (DROP, "Parapercis colias", 4, "expert", 20.0),
        ]
    )
    assert report_data.deployment_maxn(ann).loc[DROP] == 10


def test_species_maxn_keeps_reviewed_empty_deployments():
    """An absence record stays in the frame, as a zero.

    `groupby` drops NaN keys by default, so a "reviewed, nothing seen"
    deployment (null species, count 0) silently left this frame, and every
    mean built on it — the home-page protected/unprotected KPI, the Species
    view's mean abundance and richness, the protection box plots — was a mean
    over deployments where something was seen. Since empty deployments are
    plausibly more common outside reserves, the bias shrank the very reserve
    effect those charts exist to show.
    """
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 4, "expert", 10.0),
            (OTHER_DROP, None, 0, "expert", None),
        ]
    )
    out = report_data.species_maxn(ann)
    assert set(out["drop_id"]) == {DROP, OTHER_DROP}
    empty = out[out["drop_id"] == OTHER_DROP]
    assert len(empty) == 1
    assert empty["scientific_name"].isna().all()
    assert empty["maxn"].iloc[0] == 0
    # The downstream pattern every view uses: nunique ignores NaN, so the
    # reviewed-empty deployment counts with a richness of zero rather than
    # inventing a species or vanishing.
    richness = out.groupby("drop_id")["scientific_name"].nunique()
    assert richness.loc[OTHER_DROP] == 0
    assert richness.loc[DROP] == 1


def test_deployment_maxn_keeps_sources_apart():
    """Each source's account of a deployment stays its own.

    The ML-against-expert panel compares two numbers for one deployment; if the
    peak were taken across sources there would only ever be one.
    """
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 9, "ml", 10.0),
            (DROP, "Pagrus auratus", 2, "expert", 10.0),
        ]
    )
    totals = report_data.deployment_maxn(ann, ("annotated_by",))
    assert totals.loc[(DROP, "ml")] == 9
    assert totals.loc[(DROP, "expert")] == 2


# ── Protection: three groups, and nothing quietly in the wrong one ───────────


@pytest.mark.parametrize(
    "status, expected",
    [
        ("Type I MPA (Marine Reserve)", PROTECTED),
        # Decision 2026-10-01: High Protection Areas count as protected
        # alongside Type I MPAs; everything partial stays in Other.
        ("High Protection Area", PROTECTED),
        ("No protection", UNPROTECTED),
        ("Type II MPA", OTHER_PROTECTION),
        ("Taiapure", OTHER_PROTECTION),
        ("Mataitai", OTHER_PROTECTION),
        ("Fisheries Act closure areas", OTHER_PROTECTION),
        ("Seafloor Protection Area", OTHER_PROTECTION),
        ("Other", OTHER_PROTECTION),
    ],
)
def test_protection_group_classifies_every_real_status(status, expected):
    assert protection_group(pd.Series([status])).iloc[0] == expected


def test_protection_group_separates_unknown_from_partial():
    """ "We know it is partial" and "we do not know" are different answers.

    Both are excluded from the comparison, but only one of them is a data-entry
    problem, and the exclusion note counts them separately.
    """
    groups = protection_group(pd.Series([None, "", "unknown", "Taiapure"]))
    assert groups.isna().tolist() == [True, True, True, False]
    assert groups.iloc[3] == OTHER_PROTECTION


def test_protection_never_invents_a_side():
    """Anything not named in config lands in Other, never on a side.

    A status renamed upstream (SharePoint is outside this repo) must not
    silently join the reserve or the control.
    """
    assert (
        protection_group(pd.Series(["Type I MPA (renamed upstream)"])).iloc[0]
        == OTHER_PROTECTION
    )


# ── The unidentified bucket ──────────────────────────────────────────────────


def test_unify_unidentified_merges_every_catch_all_label():
    """`fish` and the BIIGLE label names are one bucket, not five species."""
    ann = _annotations(
        [
            (DROP, "fish", 3, "ml", 10.0),
            (DROP, "Fish: review required", 2, "expert", 10.0),
            (DROP, "Pagrus auratus", 1, "expert", 10.0),
        ]
    )
    out = report_data.unify_unidentified(ann)
    names = set(out["scientific_name"])
    assert names == {"Unidentified", "Pagrus auratus"}
    # The display name travels with it, or a chart labels the same bucket twice.
    assert set(out.loc[out["scientific_name"] == "Unidentified", "display_name"]) == {
        "Unidentified"
    }


def test_real_species_drops_the_bucket_but_not_real_species():
    """Richness counts species; the bucket is N unknown species under one label."""
    ann = report_data.unify_unidentified(
        _annotations(
            [
                (DROP, "fish", 3, "ml", 10.0),
                (DROP, "Pagrus auratus", 1, "expert", 10.0),
            ]
        )
    )
    assert report_data.real_species(ann)["scientific_name"].tolist() == [
        "Pagrus auratus"
    ]


# ── Arrival and MaxN time are two different measurements ─────────────────────


def test_arrival_comes_from_ml_and_peak_from_the_best_source():
    """Arrival is the first ML detection; MaxN time is when the count peaked.

    The chart these feed used to plot one timestamp and call it arrival. On
    real data the peak lands ~16 minutes after the arrival, so the two are not
    interchangeable.
    """
    ann = _annotations(
        [
            # ML scores every interval, so only it can say when something first
            # appeared.
            (DROP, "Pagrus auratus", 1, "ml", 120.0),
            (DROP, "Pagrus auratus", 5, "ml", 600.0),
            # Expert wins for the peak, and reviewed a later frame.
            (DROP, "Pagrus auratus", 9, "expert", 1200.0),
        ]
    )
    row = report_data.arrival_and_peak(ann).iloc[0]
    assert row["arrival_s"] == 120.0
    assert row["peak_s"] == 1200.0
    assert row["peak_source"] == "expert"


def test_peak_is_the_time_of_the_largest_count_not_the_first_row():
    """Order in the frame must not decide which interval is the peak."""
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 2, "expert", 60.0),
            (DROP, "Pagrus auratus", 8, "expert", 900.0),
            (DROP, "Pagrus auratus", 3, "expert", 1500.0),
        ]
    )
    assert report_data.arrival_and_peak(ann).iloc[0]["peak_s"] == 900.0


def test_arrival_is_absent_rather_than_guessed_without_ml():
    """A deployment no model has run on has no arrival time, and says so."""
    ann = _annotations([(DROP, "Pagrus auratus", 4, "expert", 300.0)])
    row = report_data.arrival_and_peak(ann).iloc[0]
    assert pd.isna(row["arrival_s"])
    assert row["peak_s"] == 300.0


# ── Discovery curve: is the deployment long enough for MaxN? ────────────────


def test_discovery_curve_is_the_share_of_final_maxn_reached():
    """Best count so far over final MaxN, held flat between detections."""
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 1, "ml", 30.0),  # minute 1
            (DROP, "Pagrus auratus", 2, "ml", 300.0),  # minute 6
            (DROP, "Pagrus auratus", 4, "ml", 900.0),  # minute 16
        ]
    )
    curve = report_data.discovery_curve(ann)
    assert set(curve["Source"]) == {report_data.CURVE_ML}
    one = curve[curve["drop_id"] == DROP].set_index("minute")["share"]
    assert len(one) == 30
    assert one.loc[1] == 0.25
    assert one.loc[5] == 0.25  # no new peak: the best so far holds
    assert one.loc[6] == 0.5
    assert one.loc[15] == 0.5
    assert one.loc[16] == 1.0
    assert one.loc[30] == 1.0


def test_discovery_curve_excludes_frame_and_clip_sources():
    """BIIGLE frame reviews and citsci clips describe selected moments.

    A curve built from them would mostly reflect which moments were picked,
    so only the model's interval scores and the legacy full-video reviews
    qualify. A timestamped BIIGLE expert row (numeric external_id) stays out.
    """
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 9, "expert", 10.0),
            (DROP, "Pagrus auratus", 7, "citsci", 10.0),
        ]
    ).assign(external_id="51297286", time_of_max=None)
    assert report_data.discovery_curve(ann).empty


def test_discovery_curve_builds_the_expert_path_from_legacy_times():
    """Legacy reviews carry interval times only as text; they still curve.

    `time_of_max_seconds` was never backfilled for the legacy import, so the
    expert curve parses the `HH:MM:SS` text column instead. The paths stay
    separate per source: an ML row for the same deployment must not blend
    into the expert account.
    """
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 1, "expert", None),
            (DROP, "Pagrus auratus", 3, "expert", None),
            (DROP, "Pagrus auratus", 2, "ml", 60.0),
        ]
    ).assign(
        external_id=["legacy", "legacy", None],
        time_of_max=["00:02:00", "00:25:30", None],
    )
    curve = report_data.discovery_curve(ann)
    expert = curve[curve["Source"] == report_data.CURVE_EXPERT].set_index("minute")[
        "share"
    ]
    assert len(expert) == 30
    assert expert.loc[3] == pytest.approx(1 / 3)  # count 1 at 2 min in
    assert expert.loc[25] == pytest.approx(1 / 3)
    assert expert.loc[26] == 1.0  # count 3 at 25.5 min in
    # The ML row forms its own one-row path, not part of the expert curve:
    # nothing before its minute-2 detection, the full peak from there on.
    ml = curve[curve["Source"] == report_data.CURVE_ML].set_index("minute")["share"]
    assert ml.loc[1] == 0.0
    assert (ml.loc[2:] == 1.0).all()


def test_discovery_curve_reads_excel_coerced_times_as_minutes_first():
    """A nonzero first field in hand legacy times is minutes, not hours.

    Excel swallowed the workbooks' MM:SS entries as HH:MM, so "23:45" (23
    minutes 45 seconds in) arrived as the cell 23:45:00. No BUV video is 23
    hours long, so a nonzero first field on a hand-entered row can only be
    minutes (Kalindi's rule, 2026-10-02) — and a deployment holding such a
    row went through the coercion wholesale, so its "00:17:00"-shaped rows
    are 17 SECONDS in, not 17 minutes. A deployment without any such row
    keeps the plain HH:MM:SS reading.
    """
    clean_drop = "TON_20211026_BUV_TON_001_01"
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 1, "expert", None),
            (DROP, "Pagrus auratus", 2, "expert", None),
            (clean_drop, "Pagrus auratus", 2, "expert", None),
        ]
    ).assign(
        external_id="legacy",
        time_of_max=["00:17:00", "23:45:00", "00:10:00"],
    )
    curve = report_data.discovery_curve(ann)
    expert = curve[curve["Source"] == report_data.CURVE_EXPERT]

    mangled = expert[expert["drop_id"] == DROP].set_index("minute")["share"]
    assert mangled.loc[1] == 0.5  # "00:17:00" -> 17 s -> minute 1
    assert mangled.loc[23] == 0.5
    assert mangled.loc[24] == 1.0  # "23:45:00" -> 23 min 45 s -> minute 24

    clean = expert[expert["drop_id"] == clean_drop].set_index("minute")["share"]
    assert clean.loc[10] == 0.0
    assert clean.loc[11] == 1.0  # "00:10:00" -> 10 minutes -> minute 11


def test_discovery_curve_keeps_true_times_inside_a_coerced_deployment():
    """A nonzero seconds field is proof the row was never Excel-coerced.

    The coercion always produces :00 seconds, so "00:25:30" is genuine
    HH:MM:SS (25.5 minutes in) even when it sits in a deployment flagged by
    a mangled sibling row — one typo must not corrupt a sheet's real times.
    """
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 1, "expert", None),
            (DROP, "Pagrus auratus", 2, "expert", None),
        ]
    ).assign(
        external_id="legacy",
        time_of_max=["23:45:00", "00:25:30"],
    )
    curve = report_data.discovery_curve(ann)
    one = curve[curve["Source"] == report_data.CURVE_EXPERT].set_index("minute")[
        "share"
    ]
    assert one.loc[24] == 0.5  # the coerced row: 23 min 45 s
    assert one.loc[26] == 1.0  # the real row: 25.5 minutes, untouched


def test_discovery_curve_clips_overrun_timestamps_into_the_last_minute():
    """A detection past the nominal cutoff lands in minute 30, not minute 32.

    Videos run slightly long (encoder drift), and a 31st minute on the axis
    would imply a sampling window no other deployment has.
    """
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 2, "ml", 30.0),
            (DROP, "Pagrus auratus", 5, "ml", 1890.0),  # 31.5 minutes in
        ]
    )
    curve = report_data.discovery_curve(ann)
    assert curve["minute"].max() == 30
    one = curve[curve["drop_id"] == DROP].set_index("minute")["share"]
    assert one.loc[29] == 0.4
    assert one.loc[30] == 1.0


# ── Bootstrap interval for the protected/unprotected gap ────────────────────


def test_bootstrap_diff_ci_brackets_an_obvious_gap():
    """Two clearly separated groups give an interval away from zero."""
    diff, lo, hi = report_data.bootstrap_diff_ci([5.0] * 30, [1.0] * 30)
    assert diff == 4.0
    assert lo == 4.0 and hi == 4.0  # no within-group variance, no wobble


def test_bootstrap_diff_ci_straddles_zero_when_groups_match():
    """Identical distributions: the gap is sampling luck, and the interval
    says so by containing zero."""
    values = list(range(20))
    diff, lo, hi = report_data.bootstrap_diff_ci(values, values)
    assert diff == 0.0
    assert lo < 0 < hi


def test_bootstrap_diff_ci_refuses_tiny_samples():
    """One deployment is not a distribution; NaNs instead of fake precision."""
    diff, lo, hi = report_data.bootstrap_diff_ci([3.0], [1.0, 2.0])
    assert diff != diff and lo != lo and hi != hi


def test_bootstrap_diff_ci_is_reproducible():
    """Fixed seed: the page shows the same interval on every rerun."""
    a, b = [1, 4, 2, 8, 0, 3], [2, 2, 1, 5, 9, 0]
    assert report_data.bootstrap_diff_ci(a, b) == report_data.bootstrap_diff_ci(a, b)


# ── Stratified comparison: within each area, never across them ──────────────


def _strata_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["drop_id", "Group", "reserve_code", "value"])


def _stratified(frame, **kw):
    return report_data.stratified_diff_ci(
        frame,
        value_col="value",
        group_col="Group",
        stratum_col="reserve_code",
        inside=PROTECTED,
        outside=UNPROTECTED,
        **kw,
    )


def test_stratified_diff_ci_does_not_compare_across_areas():
    """Simpson's paradox guard: the pooled sign must follow the areas.

    Inside beats outside by 1 at BOTH areas, but one area sits at a much
    higher abundance overall and is sampled mostly outside. Comparing all
    inside values against all outside values gives a NEGATIVE difference —
    the area difference wearing protection's clothes. Measured within each
    area first, the answer is +1, which is what the data actually says.
    """
    rows = []
    # Low-abundance area, sampled mostly inside.
    for i in range(6):
        rows.append((f"AAA_d{i}", PROTECTED, "AAA", 3.0))
    for i in range(2):
        rows.append((f"AAA_o{i}", UNPROTECTED, "AAA", 2.0))
    # High-abundance area, sampled mostly outside.
    for i in range(2):
        rows.append((f"BBB_d{i}", PROTECTED, "BBB", 11.0))
    for i in range(6):
        rows.append((f"BBB_o{i}", UNPROTECTED, "BBB", 10.0))
    frame = _strata_frame(rows)

    naive, _, _ = report_data.bootstrap_diff_ci(
        frame.loc[frame["Group"] == PROTECTED, "value"],
        frame.loc[frame["Group"] == UNPROTECTED, "value"],
    )
    assert naive < 0  # the trap

    result = _stratified(frame)
    assert result["diff"] == pytest.approx(1.0)
    assert result["n_strata"] == 2


def test_stratified_diff_ci_means_agree_with_the_difference():
    """Inside minus Outside must equal the Difference it is shown beside.

    The two means carry the same weights as the difference. Weighting them
    per side instead let the pair contradict it: on real data blue cod read
    "2.06 inside, 2.97 outside, difference -0.02", which is the
    stratification working but reads as a bug.
    """
    rows = []
    for i in range(6):
        rows.append((f"AAA_d{i}", PROTECTED, "AAA", 3.0))
    for i in range(2):
        rows.append((f"AAA_o{i}", UNPROTECTED, "AAA", 2.0))
    for i in range(2):
        rows.append((f"BBB_d{i}", PROTECTED, "BBB", 11.0))
    for i in range(6):
        rows.append((f"BBB_o{i}", UNPROTECTED, "BBB", 10.0))
    result = _stratified(_strata_frame(rows))
    assert result["mean_inside"] - result["mean_outside"] == pytest.approx(
        result["diff"]
    )


def test_stratified_diff_ci_reports_how_marginal_the_verdict_is():
    """`share_other_way` is the continuous reading of the tick box.

    The binary is a cliff, and on real data snapper sits on it: a lower bound
    of 0.00 flips with resampling noise. The share of resamples landing the
    other side of zero says whether a direction is established or held by a
    hair, and must agree with the interval at the 2.5% boundary.
    """
    rows = []
    for i in range(30):  # a wide, clear separation
        rows.append((f"AAA_d{i}", PROTECTED, "AAA", 10.0))
        rows.append((f"AAA_o{i}", UNPROTECTED, "AAA", 1.0))
    clear = _stratified(_strata_frame(rows))
    assert clear["share_other_way"] == 0.0
    assert clear["lo"] > 0

    # Identical groups: the difference is noise, so about half the resamples
    # land either side.
    same = [(f"AAA_d{i}", PROTECTED, "AAA", float(i)) for i in range(20)]
    same += [(f"AAA_o{i}", UNPROTECTED, "AAA", float(i)) for i in range(20)]
    noise = _stratified(_strata_frame(same))
    assert 0.3 < noise["share_other_way"] < 0.7
    assert noise["lo"] < 0 < noise["hi"]

    # The two readings agree: interval clear of zero iff the share is under
    # 2.5%, which is what lets the table show them side by side.
    for result in (clear, noise):
        settled = result["lo"] > 0 or result["hi"] < 0
        assert settled == (result["share_other_way"] < 0.025)


def test_stratified_diff_ci_shares_one_resample_count():
    """Every caller gets the same interval for the same data.

    Two panels on one page disagreed about snapper because one passed 1,000
    resamples and the other 2,000, and its interval sits on zero. The default
    is a module constant and callers are not expected to override it.
    """
    rows = [(f"AAA_d{i}", PROTECTED, "AAA", float(i % 5)) for i in range(12)] + [
        (f"AAA_o{i}", UNPROTECTED, "AAA", float(i % 4)) for i in range(12)
    ]
    frame = _strata_frame(rows)
    default = _stratified(frame)
    explicit = _stratified(frame, n_boot=report_data.N_BOOT)
    assert default["lo"] == explicit["lo"]
    assert default["hi"] == explicit["hi"]


def test_stratified_diff_ci_weights_by_precision():
    """A thin area moves the pooled number less than a well-sampled one.

    Weight is n_in*n_out/(n_in+n_out), the reciprocal of the difference's
    variance, so the pooled value sits nearer the area with more deployments
    behind it rather than at the midpoint of the two differences.
    """
    rows = []
    for i in range(20):  # well sampled, difference of +1
        rows.append((f"AAA_d{i}", PROTECTED, "AAA", 2.0))
        rows.append((f"AAA_o{i}", UNPROTECTED, "AAA", 1.0))
    for i in range(2):  # thin, difference of +9
        rows.append((f"BBB_d{i}", PROTECTED, "BBB", 10.0))
        rows.append((f"BBB_o{i}", UNPROTECTED, "BBB", 1.0))
    result = _stratified(_strata_frame(rows))
    # Weights are 10 and 1, so the pooled value is (10*1 + 1*9) / 11.
    assert result["diff"] == pytest.approx(19 / 11)
    assert result["diff"] < 5.0  # not the unweighted midpoint


def test_stratified_diff_ci_skips_one_sided_areas_and_counts_them():
    """An area surveyed on one side only cannot answer the question.

    It is excluded and reported, never folded into whichever side it has.
    """
    rows = [
        ("AAA_d0", PROTECTED, "AAA", 4.0),
        ("AAA_d1", PROTECTED, "AAA", 2.0),
        ("AAA_o0", UNPROTECTED, "AAA", 1.0),
        ("AAA_o1", UNPROTECTED, "AAA", 1.0),
        ("CCC_d0", PROTECTED, "CCC", 99.0),  # inside only
        ("CCC_d1", PROTECTED, "CCC", 99.0),
    ]
    result = _stratified(_strata_frame(rows))
    assert result["n_strata"] == 1
    assert result["n_skipped"] == 1
    assert result["strata"]["reserve_code"].tolist() == ["AAA"]
    assert result["diff"] == pytest.approx(2.0)  # CCC's 99s never enter


def test_stratified_diff_ci_reports_nothing_comparable():
    """No area with both sides: NaNs, not a number built from one side."""
    rows = [
        ("AAA_d0", PROTECTED, "AAA", 4.0),
        ("AAA_d1", PROTECTED, "AAA", 2.0),
        ("BBB_o0", UNPROTECTED, "BBB", 1.0),
        ("BBB_o1", UNPROTECTED, "BBB", 1.0),
    ]
    result = _stratified(_strata_frame(rows))
    assert result["n_strata"] == 0
    assert result["diff"] != result["diff"]
    assert result["n_skipped"] == 2


def test_species_value_grid_fills_absences_with_zero():
    """Every deployment carries a value for every named species.

    `species_maxn` only has a row where the species was recorded, so a mean
    off it is conditional on presence. The grid is what makes "mean snapper
    per deployment" include the deployments with no snapper.
    """
    deployments = pd.DataFrame(
        {"drop_id": [DROP, OTHER_DROP], "Group": [PROTECTED, UNPROTECTED]}
    )
    per_species = pd.DataFrame(
        {
            "drop_id": [DROP],
            "scientific_name": ["Pagrus auratus"],
            "maxn": [6.0],
        }
    )
    grid = report_data.species_value_grid(
        deployments, per_species, ["Pagrus auratus", "Parapercis colias"]
    )
    assert len(grid) == 4  # 2 deployments x 2 species
    snapper = grid[grid["scientific_name"] == "Pagrus auratus"].set_index("drop_id")
    assert snapper.loc[DROP, "maxn"] == 6.0
    assert snapper.loc[OTHER_DROP, "maxn"] == 0.0
    # A species nobody recorded is all zeros, not absent rows.
    cod = grid[grid["scientific_name"] == "Parapercis colias"]
    assert (cod["maxn"] == 0.0).all()
    # The grouping columns survive, so the result is ready to compare.
    assert set(grid["Group"]) == {PROTECTED, UNPROTECTED}


# ── Reserve names ────────────────────────────────────────────────────────────


def test_split_reserves_handles_sites_between_two_areas():
    """A site between two MPAs carries both, in either order."""
    series = pd.Series(
        ["Tonga Island", "Tonga Island, Horoirangi", "Horoirangi, Tonga Island", None]
    )
    assert report_data.split_reserves(series) == {"Tonga Island", "Horoirangi"}


# ── The Experiments-shaped frame ─────────────────────────────────────────────


def test_experiments_frame_keeps_absence_rows():
    """A null-species row is "reviewed, nothing seen".

    Dropping those removes the deployments that make up the denominator of
    every detection rate, which quietly inflates them.
    """
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 3, "expert", 10.0),
            (OTHER_DROP, None, 0, "expert", None),
        ]
    )
    frame = report_data.experiments_frame(ann)
    assert frame["drop_id"].nunique() == 2


# ── Metadata validation: names that do not exist, spellings that differ ──────
#
# These guard the two checks added on 2026-08-18. Both catch mistakes that were
# previously invisible: an unknown reserve name became a new marine reserve with
# its own row in every per-area chart, and an unrecognised protection status
# dropped into "Other" and left the reserve comparison without a word.


def _validation():
    from spyfish.validation.validation_strategies import (
        validate_multi_value_foreign_keys,
        validate_values,
    )

    return validate_multi_value_foreign_keys, validate_values


RESERVES = pd.DataFrame(
    {
        "Title": ["Tonga Island", "Cape Rodney", "Tawharanui"],
        "SurveyLocationAcronym": ["TON", "CRP", "TAW"],
    }
)
RESERVE_RULES = {"dataset": RESERVES, "file_name": "Marine Reserves.csv"}


def test_unknown_reserve_name_is_an_error():
    check, _ = _validation()
    sites = pd.DataFrame({"LinkToMarineReserve": ["Tonga Isand"]})
    rules = {
        "file_name": "BUV Survey Sites.csv",
        "multi_foreign_keys": [
            {
                "column": "LinkToMarineReserve",
                "target": "reserves",
                "target_column": "Title",
                "separator": ",",
            }
        ],
    }
    errors = check(sites, rules, {"reserves": RESERVE_RULES})
    assert len(errors) == 1
    assert "Tonga Isand" in errors[0].ErrorMessage


def test_a_site_between_two_reserves_is_not_an_error():
    """The reason this could not be a plain foreign key."""
    check, _ = _validation()
    sites = pd.DataFrame({"LinkToMarineReserve": ["Tonga Island, Cape Rodney"]})
    rules = {
        "file_name": "BUV Survey Sites.csv",
        "multi_foreign_keys": [
            {
                "column": "LinkToMarineReserve",
                "target": "reserves",
                "target_column": "Title",
                "separator": ",",
            }
        ],
    }
    assert check(sites, rules, {"reserves": RESERVE_RULES}) == []


def test_dropid_prefix_is_checked_against_the_acronym_list():
    """And a survey visiting another reserve's sites is not a false positive.

    `CRP_20220407_BUV_TAW_049_01` is a Cape Rodney survey at Tawharanui sites.
    Comparing the two embedded codes flags 24 of those; comparing the prefix to
    the acronym list asks the question that has an answer.
    """
    check, _ = _validation()
    deployments = pd.DataFrame(
        {
            "DropID": [
                "TON_20241125_BUV_TON_006_01",
                "CRP_20220407_BUV_TAW_049_01",
                "ZZZ_20240101_BUV_ZZZ_001_01",
            ]
        }
    )
    rules = {
        "file_name": "BUV Deployment.csv",
        "multi_foreign_keys": [
            {
                "column": "DropID",
                "target": "reserves",
                "target_column": "SurveyLocationAcronym",
                "extract": "^([A-Za-z]{3})_",
            }
        ],
    }
    errors = check(deployments, rules, {"reserves": RESERVE_RULES})
    assert len(errors) == 1
    assert "ZZZ" in errors[0].ErrorMessage


def test_a_new_spelling_is_reported_as_a_spelling_not_a_new_category():
    _, check_values = _validation()
    sites = pd.DataFrame({"ProtectionStatus": ["NO PROTECTION", "Marine Park"]})
    rules = {
        "values": [
            {
                "column": "ProtectionStatus",
                "rule": "one_of",
                "allowed": ["No protection", "Other"],
            }
        ]
    }
    messages = [e.ErrorMessage for e in check_values(sites, rules, "sites.csv")]
    assert len(messages) == 2
    assert "only in case or spacing" in messages[0]
    assert "not a recognised value" in messages[1]


def test_known_statuses_normalise_to_one_spelling():
    """Any casing of a known status stores as the canonical one."""
    from spyfish.database.manager import _clean_protection_status

    assert _clean_protection_status("NO PROTECTION") == "No protection"
    assert _clean_protection_status("no  protection ") == "No protection"
    assert (
        _clean_protection_status("Type I MPA (MARINE RESERVE)")
        == "Type I MPA (Marine Reserve)"
    )
    # Unrecognised values are left alone for validation to flag, not guessed at.
    assert _clean_protection_status("Marine Park") == "Marine Park"


def test_build_context_bad_deployment_toggle():
    """The bad-deployment toggle: off by default, and when on their rows
    leave every annotation frame but not the deployments frame.

    Unusable footage can neither record a species nor assert an absence, so
    its annotations belong in neither numerators nor denominators; the
    deployment itself stays visible in coverage counts.
    """
    dep = pd.DataFrame(
        {
            "drop_id": [DROP, OTHER_DROP],
            "is_bad_deployment": [False, True],
            "survey_year": [2024, 2024],
            "region": ["R", "R"],
            "protection_status": ["No protection", "No protection"],
            "link_to_marine_reserve": ["X", "X"],
            "ingest_status": ["ok", "ok"],
        }
    )
    ann = _annotations(
        [
            (DROP, "Pagrus auratus", 2, "expert", 10.0),
            (OTHER_DROP, "Pagrus auratus", 9, "expert", 10.0),
        ]
    ).assign(
        survey_year=2024,
        region="R",
        protection_status="No protection",
        link_to_marine_reserve="X",
    )
    # Default: counted. The flag is unreliable (see claude_docs/todo.md),
    # so the report includes them unless the sidebar toggle says otherwise.
    kept = report_data.build_context(
        dep, ann, None, [], ingested_only=False, source=report_data.ALL_SOURCES
    )
    assert OTHER_DROP in set(kept["annotations"]["drop_id"])

    ctx = report_data.build_context(
        dep,
        ann,
        None,
        [],
        ingested_only=False,
        source=report_data.ALL_SOURCES,
        exclude_bad_deployments=True,
    )
    assert OTHER_DROP not in set(ctx["annotations"]["drop_id"])
    assert OTHER_DROP not in set(ctx["annotations_all_species"]["drop_id"])
    assert OTHER_DROP in set(ctx["deployments"]["drop_id"])
    assert DROP in set(ctx["annotations"]["drop_id"])


# ── The one richness rollup ──────────────────────────────────────────────────


def test_richness_rollup_keeps_groups_with_effort_but_no_sightings():
    """A surveyed group with nothing seen is a row, not a gap.

    The per-site and per-MPA tables used to carry their own copies of this
    rollup, and they disagreed: the per-site one left-merged effort, so 25
    sites that were analysed and recorded nothing vanished from the table
    and from the averages. Effort with zero richness is data.
    """
    sightings = _annotations([(DROP, "Pagrus auratus", 4, "expert", 10.0)]).assign(
        site_id="KSF_085"
    )
    effort = pd.DataFrame(
        {"drop_id": [DROP, OTHER_DROP], "site_id": ["KSF_085", "KSF_086"]}
    )
    out = report_data.richness_rollup(sightings, effort, by="site_id")
    assert set(out["site_id"]) == {"KSF_085", "KSF_086"}
    empty = out[out["site_id"] == "KSF_086"].iloc[0]
    assert empty["species"] == 0
    assert empty["analysed"] == 1
    assert empty["species_per_dep"] == 0


def test_richness_rollup_excludes_the_unidentified_bucket():
    """Richness counts species, so `fish` must not add one.

    Applied inside the rollup rather than left to each caller: a caller that
    forgot `real_species` got a richness one too high, silently.
    """
    sightings = report_data.unify_unidentified(
        _annotations(
            [
                (DROP, "fish", 3, "ml", 10.0),
                (DROP, "Pagrus auratus", 1, "ml", 10.0),
            ]
        ).assign(site_id="KSF_085")
    )
    effort = pd.DataFrame({"drop_id": [DROP], "site_id": ["KSF_085"]})
    out = report_data.richness_rollup(sightings, effort, by="site_id")
    assert out.iloc[0]["species"] == 1
    assert out.iloc[0]["species_per_dep"] == 1


def test_richness_by_counts_species_not_their_abundance():
    """Pooled richness is how many species, not how many fish.

    Two drops at one site: snapper on both, lobster on one, and an absence
    record on a third. Pooled `species` is 2 whatever the MaxN values are;
    `richness_sum` is 2 + 1 + 0, the per-deployment count summed, with the
    absence record counting nothing. A multi-key grouping works the same.
    """
    third = "KSF_20240124_BUV_KSF_086_01"
    sightings = pd.concat(
        [
            _annotations(
                [
                    (DROP, "Pagrus auratus", 5, "expert", 10.0),
                    (DROP, "Jasus edwardsii", 12, "expert", 10.0),
                    (OTHER_DROP, "Pagrus auratus", 1, "expert", 10.0),
                ]
            ),
            _annotations([(third, None, 0, "expert", 0.0)]),
        ]
    ).assign(site_id="KSF_085", survey_year=2024)
    out = report_data.richness_by(sightings, "site_id").iloc[0]
    assert (out["species"], out["richness_sum"]) == (2, 3)
    multi = report_data.richness_by(sightings, ["site_id", "survey_year"]).iloc[0]
    assert (multi["species"], multi["richness_sum"]) == (2, 3)


# ── Two expert accounts of one deployment ────────────────────────────────────


def _with_external_id(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "drop_id",
            "scientific_name",
            "max_interval",
            "annotated_by",
            "time_of_max_seconds",
            "external_id",
        ],
    ).assign(display_name=lambda df: df["scientific_name"])


def test_best_available_prefers_biigle_review_over_legacy_review():
    """Legacy (historical CSV) and BIIGLE expert rows are both `expert`.

    Shown together they would double count and merge two accounts of one
    deployment. The fresh BIIGLE review wins; the legacy rows stay in the DB
    for the comparison and for the "Expert" and "All" filters.
    """
    ann = _with_external_id(
        [
            (DROP, "Pagrus auratus", 5, "expert", 10.0, "legacy"),
            (DROP, "Pagrus auratus", 2, "expert", 10.0, "biigle_volume_7"),
            (DROP, "Pagrus auratus", 9, "ml", 10.0, "species_20260822"),
            (OTHER_DROP, "Pagrus auratus", 4, "expert", 10.0, "legacy"),
        ]
    )
    out = report_data._apply_source(ann, report_data.BEST_AVAILABLE)
    by_drop = out.groupby("drop_id")["external_id"].agg(set).to_dict()
    assert by_drop == {DROP: {"biigle_volume_7"}, OTHER_DROP: {"legacy"}}


def test_source_rank_without_external_id_column_falls_back_to_annotated_by():
    from ecology_data import SOURCE_PRIORITY, source_rank

    ann = _annotations([(DROP, "Pagrus auratus", 5, "expert", 10.0)])
    assert source_rank(ann).tolist() == [SOURCE_PRIORITY["expert"]]
