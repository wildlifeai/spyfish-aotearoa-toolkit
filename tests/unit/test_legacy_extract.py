"""
Tests for legacy expert CSV parsing — the all-NULL placeholder skip.

The legacy CSV is a database export that wrote the literal string "NULL"
into every data field for deployments with no expert annotations. Pandas
reads "NULL" as missing, so those rows used to become null-species absence
records ("expert reviewed, saw nothing") — which the source never asserted.
They must be skipped and reported for status repair instead.
"""

import io

import pandas as pd

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.orchestrator.legacy_extract import legacy_time_reading, parse_legacy_rows

DROP = "KSF_20240124_BUV_KSF_085_01"
PLACEHOLDER_DROP = "SLI_20220228_BUV_SLI_053_01"

HEADER = "DropID,ScientificName,TimeOfMax,MaxInterval,AnnotatedBy,IntervalAnnotation,ConfidenceAgreement"


def _read(csv_text: str) -> pd.DataFrame:
    # Through pd.read_csv, not DataFrame(...), so the literal "NULL"/"NA"
    # strings hit the same missing-value coercion as the real ingest.
    return pd.read_csv(io.StringIO(csv_text))


def test_all_null_row_becomes_an_absence_record():
    """The export's all-NULL marker means "expert reviewed, saw nothing"
    (2026-08-22), so it becomes a NULL_DEPLOYMENT row rather than being
    dropped. It must NOT become an ordinary observation: max_interval stays 0
    and the species is the sentinel, so no chart can read it as a fish."""
    df = _read(
        f"{HEADER}\n"
        f"{PLACEHOLDER_DROP},NULL,NULL,NULL,expert,30,NA\n"
        f"{DROP},Pagrus auratus,00:05:10,3,expert,30,NA\n"
    )
    annotations, placeholders = parse_legacy_rows(df)

    assert placeholders == [PLACEHOLDER_DROP]
    assert len(annotations) == 2

    absence = next(a for a in annotations if a["drop_id"] == PLACEHOLDER_DROP)
    assert absence["scientific_name"] == NULL_DEPLOYMENT
    assert absence["max_interval"] == 0
    assert absence["time_of_max_seconds"] is None
    # Scoped to 'legacy' so a re-ingest replaces it instead of duplicating,
    # and so it can never be mistaken for a BIIGLE-synced expert review.
    assert absence["external_id"] == "legacy"

    real = next(a for a in annotations if a["drop_id"] == DROP)
    assert real["scientific_name"] == "Pagrus auratus"


def test_normal_row_maps_all_fields():
    df = _read(f"{HEADER}\n{DROP},Pagrus auratus,00:05:10,3,expert,30,0.8\n")
    annotations, placeholders = parse_legacy_rows(df)

    assert placeholders == []
    (ann,) = annotations
    assert ann["max_interval"] == 3
    assert ann["time_of_max"] == "00:05:10"
    assert ann["confidence_agreement"] == 0.8
    assert ann["annotated_by"] == "expert"
    assert ann["external_id"] == "legacy"


def test_partially_filled_row_is_kept_not_skipped():
    """Only the all-NULL signature marks a placeholder. A row with ANY of the
    three data fields present is a real (if incomplete) record."""
    df = _read(f"{HEADER}\n{DROP},Pagrus auratus,NULL,NULL,expert,30,NA\n")
    annotations, placeholders = parse_legacy_rows(df)

    assert placeholders == []
    (ann,) = annotations
    assert ann["scientific_name"] == "Pagrus auratus"
    assert ann["time_of_max"] is None
    assert ann["max_interval"] == 0


def test_na_confidence_becomes_none():
    df = _read(f"{HEADER}\n{DROP},Pagrus auratus,00:05:10,3,expert,30,NA\n")
    annotations, _ = parse_legacy_rows(df)
    assert annotations[0]["confidence_agreement"] is None


def test_time_of_max_seconds_derived_at_ingest_with_excel_repair():
    """Seconds are written alongside the text, repaired where Excel mangled it.

    The rule (Kalindi, 2026-10-02): hand legacy times have no milliseconds,
    and a nonzero first field can only be MINUTES — Excel swallowed the
    workbooks' MM:SS entries as HH:MM, so "23:45" (23 min 45 s) arrived as
    23:45:00. A deployment holding such a row was coerced wholesale, so its
    00:SS:00-shaped rows are seconds; a nonzero THIRD field is immune (the
    coercion always yields :00). A clean deployment parses as written.
    """
    df = _read(
        f"{HEADER}\n"
        f"{DROP},Pagrus auratus,23:45:00,2,expert,30,NA\n"  # 23 min 45 s
        f"{DROP},Pagrus auratus,00:17:00,1,expert,30,NA\n"  # 17 s (coerced dep)
        f"{DROP},Pagrus auratus,00:25:30,3,expert,30,NA\n"  # immune: 25.5 min
        f"{PLACEHOLDER_DROP},Parapercis colias,00:10:00,1,expert,30,NA\n"  # 10 min
    )
    annotations, _ = parse_legacy_rows(df)
    seconds = [a["time_of_max_seconds"] for a in annotations]
    assert seconds == [23 * 60 + 45, 17.0, 25 * 60 + 30, 600.0]
    # The text stays verbatim for audit; only the derived column is repaired.
    assert annotations[0]["time_of_max"] == "23:45:00"


def test_time_of_max_seconds_is_none_when_text_is_missing():
    df = _read(f"{HEADER}\n{DROP},Pagrus auratus,NULL,2,expert,30,NA\n")
    annotations, _ = parse_legacy_rows(df)
    assert annotations[0]["time_of_max_seconds"] is None


def test_time_repair_survives_surrounding_whitespace():
    """A padded cell must get the same repair as its unpadded neighbour.

    The regex is anchored, `to_timedelta` is not, so " 23:45:00" used to parse
    as written (23 hours, unrepaired) while "23:45:00" in the same deployment
    was repaired to 23 minutes 45 seconds.
    """
    out = legacy_time_reading(
        pd.Series([DROP, DROP, DROP]),
        pd.Series([" 23:45:00", "23:45:00", "00:17:00 "]),
    )
    assert out["seconds"].tolist() == [23 * 60 + 45, 23 * 60 + 45, 17.0]
    assert out["repaired"].tolist() == [True, True, True]


def test_rows_without_a_drop_id_do_not_form_one_pseudo_deployment():
    """NaN DropIDs must not share a coercion flag: an orphan "00:17:00" is
    read as written (17 minutes) even when another orphan was coerced."""
    out = legacy_time_reading(
        pd.Series([None, None]),
        pd.Series(["23:45:00", "00:17:00"]),
    )
    assert out["seconds"].tolist() == [23 * 60 + 45, 17 * 60]
