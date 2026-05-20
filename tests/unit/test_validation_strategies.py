"""
Tests for validation/validation_strategies.py — pure data validation functions.

Covers:
  - validate_required: missing values detected
  - validate_unique: duplicates flagged
  - validate_formats: regex pattern matching
  - validate_values: range checking with allowed_values bypass
  - validate_relationships: template matching, allow_null, allowed_values
  - validate_foreign_keys: missing reference, missing column
  - CleanRowTracker: tracks which rows had errors
"""

import pandas as pd

from spyfish.validation.validation_strategies import (
    CleanRowTracker,
    validate_foreign_keys,
    validate_formats,
    validate_relationships,
    validate_required,
    validate_unique,
    validate_values,
)

DROP_COL = "DropID"
SURVEY_COL = "SurveyID"


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── validate_required ────────────────────────────────────────────────────────


def test_required_catches_missing():
    df = _df(
        [
            {
                DROP_COL: "KSF_20240124_BUV_KSF_085_01",
                SURVEY_COL: "KSF_20240124_BUV",
                "Name": "",
            },
            {
                DROP_COL: "KSF_20240124_BUV_KSF_085_02",
                SURVEY_COL: "KSF_20240124_BUV",
                "Name": "Alice",
            },
        ]
    )
    rules = {"required": ["Name"], "info_columns": []}
    errors = validate_required(df, rules, "test.csv")
    assert len(errors) == 1
    assert "Missing value" in errors[0].ErrorMessage
    assert errors[0].ColumnName == "Name"


def test_required_no_errors_when_all_present():
    df = _df([{DROP_COL: "D1", SURVEY_COL: "S1", "Name": "Bob"}])
    errors = validate_required(
        df, {"required": ["Name"], "info_columns": []}, "test.csv"
    )
    assert len(errors) == 0


def test_required_catches_nan():
    df = _df([{DROP_COL: "D1", SURVEY_COL: "S1", "Name": None}])
    errors = validate_required(
        df, {"required": ["Name"], "info_columns": []}, "test.csv"
    )
    assert len(errors) == 1


# ── validate_unique ──────────────────────────────────────────────────────────


def test_unique_catches_duplicates():
    df = _df(
        [
            {DROP_COL: "KSF_01", SURVEY_COL: "S1"},
            {DROP_COL: "KSF_01", SURVEY_COL: "S1"},
            {DROP_COL: "KSF_02", SURVEY_COL: "S1"},
        ]
    )
    rules = {"unique": [DROP_COL], "file_name": "test.csv"}
    errors = validate_unique(df, rules)
    assert len(errors) == 2  # both duplicate rows flagged


def test_unique_no_errors_when_unique():
    df = _df([{DROP_COL: "A"}, {DROP_COL: "B"}])
    errors = validate_unique(df, {"unique": [DROP_COL], "file_name": "test.csv"})
    assert len(errors) == 0


# ── validate_formats ─────────────────────────────────────────────────────────


def test_formats_catches_invalid_pattern():
    df = _df(
        [
            {DROP_COL: "KSF_20240124_BUV_KSF_085_01", SURVEY_COL: "KSF_20240124_BUV"},
            {DROP_COL: "BAD_FORMAT", SURVEY_COL: "ALSO_BAD"},
        ]
    )
    patterns = {SURVEY_COL: r"^[A-Z]{3}_\d{8}_BUV$"}
    rules = {"formats": [SURVEY_COL], "file_name": "test.csv"}
    errors = validate_formats(df, rules, patterns, "test.csv")
    assert len(errors) == 1
    assert errors[0].InvalidValue == "ALSO_BAD"


def test_formats_passes_valid():
    df = _df([{DROP_COL: "D1", SURVEY_COL: "KSF_20240124_BUV"}])
    patterns = {SURVEY_COL: r"^[A-Z]{3}_\d{8}_BUV$"}
    errors = validate_formats(df, {"formats": [SURVEY_COL]}, patterns, "test.csv")
    assert len(errors) == 0


def test_formats_skips_empty_values():
    df = _df([{DROP_COL: "D1", SURVEY_COL: ""}])
    patterns = {SURVEY_COL: r"^[A-Z]{3}_\d{8}_BUV$"}
    errors = validate_formats(df, {"formats": [SURVEY_COL]}, patterns, "test.csv")
    assert len(errors) == 0  # empty = not checked, not invalid


# ── validate_values ──────────────────────────────────────────────────────────


def test_values_catches_out_of_range():
    df = _df(
        [
            {DROP_COL: "D1", SURVEY_COL: "S1", "Latitude": -50},
            {DROP_COL: "D2", SURVEY_COL: "S1", "Latitude": -40},
        ]
    )
    rules = {
        "values": [{"column": "Latitude", "rule": "value_range", "range": [-46, -36]}]
    }
    errors = validate_values(df, rules, "test.csv")
    assert len(errors) == 1  # -50 is out of range
    assert "Latitude" in errors[0].ErrorMessage


def test_values_allows_allowed_values():
    df = _df([{DROP_COL: "D1", SURVEY_COL: "S1", "Latitude": 0}])
    rules = {
        "values": [
            {
                "column": "Latitude",
                "rule": "value_range",
                "range": [-46, -36],
                "allowed_values": [0],
            }
        ]
    }
    errors = validate_values(df, rules, "test.csv")
    assert len(errors) == 0  # 0 is outside range but in allowed_values


# ── validate_foreign_keys ────────────────────────────────────────────────────


def test_foreign_keys_catches_missing_reference():
    df = _df(
        [
            {DROP_COL: "D1", SURVEY_COL: "S1", "SiteID": "GOOD_SITE"},
            {DROP_COL: "D2", SURVEY_COL: "S1", "SiteID": "BAD_SITE"},
        ]
    )
    rules = {"foreign_keys": {"sites": "SiteID"}, "file_name": "deployments.csv"}
    all_rules = {
        "sites": {
            "file_name": "sites.csv",
            "dataset": _df([{"SiteID": "GOOD_SITE"}]),
        }
    }
    errors = validate_foreign_keys(df, rules, all_rules)
    assert len(errors) == 1
    assert "BAD_SITE" in errors[0].ErrorMessage


def test_foreign_keys_no_errors_when_all_present():
    df = _df([{DROP_COL: "D1", "SiteID": "S1"}])
    all_rules = {
        "sites": {"file_name": "sites.csv", "dataset": _df([{"SiteID": "S1"}])}
    }
    errors = validate_foreign_keys(
        df, {"foreign_keys": {"sites": "SiteID"}, "file_name": "dep.csv"}, all_rules
    )
    assert len(errors) == 0


# ── validate_relationships ───────────────────────────────────────────────────


def test_relationships_template_match():
    df = _df(
        [
            {DROP_COL: "D1", SURVEY_COL: "S1", "FileName": "D1.mp4"},
            {DROP_COL: "D2", SURVEY_COL: "S1", "FileName": "WRONG.mp4"},
        ]
    )
    rules = {
        "relationships": [
            {"column": "FileName", "rule": "equals", "template": "{DropID}.mp4"}
        ]
    }
    errors = validate_relationships(df, rules, "test.csv")
    assert len(errors) == 1
    assert "WRONG.mp4" in errors[0].ErrorMessage


def test_relationships_allow_null():
    df = _df([{DROP_COL: "D1", SURVEY_COL: "S1", "FileName": None}])
    rules = {
        "relationships": [
            {
                "column": "FileName",
                "rule": "equals",
                "template": "{DropID}.mp4",
                "allow_null": True,
            }
        ]
    }
    errors = validate_relationships(df, rules, "test.csv")
    assert len(errors) == 0


def test_relationships_allowed_values_bypass():
    df = _df(
        [{DROP_COL: "D1", SURVEY_COL: "S1", "FileName": "NO VIDEO BAD DEPLOYMENT"}]
    )
    rules = {
        "relationships": [
            {
                "column": "FileName",
                "rule": "equals",
                "template": "{DropID}.mp4",
                "allowed_values": ["NO VIDEO BAD DEPLOYMENT"],
            }
        ]
    }
    errors = validate_relationships(df, rules, "test.csv")
    assert len(errors) == 0


# ── CleanRowTracker ──────────────────────────────────────────────────────────


def test_clean_row_tracker():
    df = _df([{"A": 1}, {"A": 2}, {"A": 3}])
    tracker = CleanRowTracker()
    tracker.initialize_dataset("test", df)
    assert tracker.get_clean_indices("test") == {0, 1, 2}

    tracker.mark_row_as_error(1, "test")
    assert tracker.get_clean_indices("test") == {0, 2}


def test_clean_row_tracker_unknown_dataset():
    tracker = CleanRowTracker()
    assert tracker.get_clean_indices("nonexistent") == set()
