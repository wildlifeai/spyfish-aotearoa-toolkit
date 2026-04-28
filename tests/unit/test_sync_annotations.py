"""
Tests for biigle/sync_annotations.py helper functions.

Covers the pure data-transformation layer — no Biigle API calls.
  - _extract_timestamp_from_filename: image frame, video clip, no match
  - _map_biigle_to_spyfish_schema: species name cleaning, annotation ID fallback
  - aggregate_raw_to_maxn_rows: MaxN counting, multiple species, sorting
"""

import pandas as pd

from spyfish.biigle.sync_annotations import (
    _extract_timestamp_from_filename,
    _map_biigle_to_spyfish_schema,
    aggregate_raw_to_maxn_rows,
)

# ── _extract_timestamp_from_filename ─────────────────────────────────────────


def test_extract_timestamp_image_frame():
    row = pd.Series({"filename": "KSF_20240124_BUV_KSF_085_01__frame_125.5s.jpg"})
    result = _extract_timestamp_from_filename(row, "filename")
    assert result == "00:02:05.500"


def test_extract_timestamp_video_clip():
    row = pd.Series({"filename": "KSF_085_01_clip_60.0s.mp4", "frames": "nan"})
    result = _extract_timestamp_from_filename(row, "filename")
    assert result == "00:01:00.000"


def test_extract_timestamp_video_clip_with_frame_offset():
    row = pd.Series({"filename": "KSF_085_01_clip_60.0s.mp4", "frames": "[2.5]"})
    result = _extract_timestamp_from_filename(row, "filename")
    assert result == "00:01:02.500"  # 60 + 2.5


def test_extract_timestamp_no_match():
    row = pd.Series({"filename": "random_file.csv"})
    assert _extract_timestamp_from_filename(row, "filename") is None


def test_extract_timestamp_missing_column():
    row = pd.Series({"other_col": "test"})
    assert _extract_timestamp_from_filename(row, "filename") is None


def test_extract_timestamp_empty_fname_col():
    row = pd.Series({"filename": "test.jpg"})
    assert _extract_timestamp_from_filename(row, "") is None


# ── _map_biigle_to_spyfish_schema ────────────────────────────────────────────


def test_map_strips_common_name_prefix():
    """'Kina - Evechinus chloroticus' → 'Evechinus chloroticus'"""
    row = pd.Series(
        {"label_name": "Kina - Evechinus chloroticus", "annotation_id": "42"}
    )
    key, mapped = _map_biigle_to_spyfish_schema(
        row,
        "label_name",
        "KSF_20240124_BUV_KSF_085_01",
        "00:01:00.000",
        "frame_60.0s.jpg",
    )
    assert mapped["scientific_name"] == "Evechinus chloroticus"
    assert key == ("00:01:00.000", "Evechinus chloroticus")


def test_map_plain_species_name():
    row = pd.Series({"label_name": "Pagrus auratus", "annotation_id": "99"})
    _, mapped = _map_biigle_to_spyfish_schema(
        row, "label_name", "DROP_01", "00:00:05.000", "frame_5.0s.jpg"
    )
    assert mapped["scientific_name"] == "Pagrus auratus"
    assert mapped["external_id"] == "99"
    assert mapped["annotated_by"] == "expert"
    assert mapped["max_interval"] == 0  # incremented during aggregation


def test_map_annotation_id_fallback():
    """Falls back to 'id' when 'annotation_id' is missing."""
    row = pd.Series({"label_name": "Pagrus auratus", "id": "77"})
    _, mapped = _map_biigle_to_spyfish_schema(
        row, "label_name", "DROP_01", None, "image_uuid_001.jpg"
    )
    assert mapped["external_id"] == "77"


def test_map_null_timestamp_uses_frame_key():
    """When timestamp is None (e.g. UUID-named image volumes), frame_key is used as time_of_max."""
    _, mapped = _map_biigle_to_spyfish_schema(
        pd.Series({"label_name": "Fish", "annotation_id": "1"}),
        "label_name",
        "DROP_01",
        None,
        "image_uuid_001.jpg",
    )
    assert mapped["time_of_max"] == "image_uuid_001.jpg"


# ── aggregate_raw_to_maxn_rows ───────────────────────────────────────────────────


def _make_biigle_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame matching the shape aggregate_raw_to_maxn_rows expects."""
    return pd.DataFrame(rows)


def test_aggregate_counts_per_timestamp_species():
    """Two annotations of the same species at the same timestamp → max_interval=2."""
    df = _make_biigle_df(
        [
            {
                "filename": "drop__frame_10.0s.jpg",
                "label_name": "Pagrus auratus",
                "annotation_id": "1",
            },
            {
                "filename": "drop__frame_10.0s.jpg",
                "label_name": "Pagrus auratus",
                "annotation_id": "2",
            },
        ]
    )
    result = aggregate_raw_to_maxn_rows(df, "KSF_20240124_BUV_KSF_085_01")
    assert len(result) == 1
    assert result[0]["max_interval"] == 2
    assert result[0]["scientific_name"] == "Pagrus auratus"


def test_aggregate_multiple_species():
    df = _make_biigle_df(
        [
            {
                "filename": "drop__frame_10.0s.jpg",
                "label_name": "Pagrus auratus",
                "annotation_id": "1",
            },
            {
                "filename": "drop__frame_10.0s.jpg",
                "label_name": "Notolabrus fucicola",
                "annotation_id": "2",
            },
        ]
    )
    result = aggregate_raw_to_maxn_rows(df, "DROP_01")
    assert len(result) == 2
    species = {r["scientific_name"] for r in result}
    assert species == {"Pagrus auratus", "Notolabrus fucicola"}
    assert all(r["max_interval"] == 1 for r in result)


def test_aggregate_multiple_timestamps():
    df = _make_biigle_df(
        [
            {
                "filename": "drop__frame_10.0s.jpg",
                "label_name": "Pagrus auratus",
                "annotation_id": "1",
            },
            {
                "filename": "drop__frame_20.0s.jpg",
                "label_name": "Pagrus auratus",
                "annotation_id": "2",
            },
        ]
    )
    result = aggregate_raw_to_maxn_rows(df, "DROP_01")
    assert len(result) == 2  # same species, different timestamps
    assert result[0]["time_of_max"] == "00:00:10.000"  # sorted
    assert result[1]["time_of_max"] == "00:00:20.000"


def test_aggregate_empty_df():
    result = aggregate_raw_to_maxn_rows(
        pd.DataFrame(columns=["filename", "label_name"]), "DROP_01"
    )
    assert result == []


def test_aggregate_strips_common_name():
    df = _make_biigle_df(
        [
            {
                "filename": "drop__frame_5.0s.jpg",
                "label_name": "Snapper - Pagrus auratus",
                "annotation_id": "1",
            },
        ]
    )
    result = aggregate_raw_to_maxn_rows(df, "DROP_01")
    assert result[0]["scientific_name"] == "Pagrus auratus"
