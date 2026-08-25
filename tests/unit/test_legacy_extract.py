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
from spyfish.orchestrator.legacy_extract import parse_legacy_rows

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
