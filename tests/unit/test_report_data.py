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
        ("No protection", UNPROTECTED),
        # The two that used to count as protected in config while the charts'
        # own classifier read them as outside.
        ("High Protection Area", OTHER_PROTECTION),
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
