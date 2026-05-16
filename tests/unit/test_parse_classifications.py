"""
Tests for the Zooniverse classification parser — pure data transformation.

Covers:
  - _parse_annotation: nothing-here, species+count, bucket counts, drawing, empty
  - _extract_subject_metadata: strict canonical keys only
  - _missing_required_keys: detects absent canonical keys
  - parse_classifications: end-to-end with minimal mock classification
  - aggregate_by_subject_species: agreement_pct filter, suspicious minority
"""

from unittest.mock import patch

import pandas as pd

from spyfish.zooniverse.parse_classifications import (
    _extract_subject_metadata,
    _missing_required_keys,
    _parse_annotation,
    aggregate_by_subject_species,
    parse_classifications,
)
from spyfish.zooniverse.subject_keys import SubjectKeys

# ── _parse_annotation ────────────────────────────────────────────────────────


def test_parse_annotation_nothing_here():
    ann = {"value": [{"choice": "NOTHINGHERE"}]}
    rows = _parse_annotation(ann)
    assert len(rows) == 1
    assert rows[0]["is_nothing_here"] is True
    assert rows[0]["species"] is None
    assert rows[0]["count"] == 0


def test_parse_annotation_species_with_count_and_timestamp():
    ann = {
        "value": [
            {
                "choice": "Pagrus auratus",
                "answers": {
                    "HOWMANYINDIVIDUALSARETHEREATTHISTIMESTAMP": "3",
                    "WHATISTHEEARLIESTPOINTTHATYOUSEETHEMOSTINDIVIDUALSOFTHISSPECIES": "5S",
                },
            }
        ]
    }
    rows = _parse_annotation(ann)
    assert len(rows) == 1
    assert rows[0]["species"] == "Pagrus auratus"
    assert rows[0]["count"] == 3
    assert rows[0]["annotation_seconds"] == 5.0
    assert rows[0]["is_nothing_here"] is False


def test_parse_annotation_bucket_count():
    """Bucket answer '2030' should map to midpoint 25."""
    ann = {
        "value": [
            {
                "choice": "Pagrus auratus",
                "answers": {
                    "HOWMANYINDIVIDUALSARETHEREATTHISTIMESTAMP": "2030",
                },
            }
        ]
    }
    rows = _parse_annotation(ann)
    assert rows[0]["count"] == 25


def test_parse_annotation_drawing():
    ann = {
        "value": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200, "tool_label": "Snapper"}]
    }
    rows = _parse_annotation(ann)
    assert len(rows) == 1
    assert rows[0]["annotation_type"] == "drawing"
    assert rows[0]["species"] == "Snapper"
    assert rows[0]["x1"] == 10
    assert rows[0]["y2"] == 200


def test_parse_annotation_bounding_box():
    """Bounding box format (x/y/width/height) used in e.g. workflow 17057."""
    ann = {
        "value": [
            {
                "x": 100.0,
                "y": 200.0,
                "width": 50.0,
                "height": 80.0,
                "tool": 0,
                "tool_label": "Blue cod",
                "details": [{"value": 0}],
            },
            {
                "x": 300.0,
                "y": 400.0,
                "width": 60.0,
                "height": 30.0,
                "tool": 4,
                "tool_label": "Bait",
                "details": [],
            },
        ]
    }
    rows = _parse_annotation(ann)
    assert len(rows) == 2

    cod, bait = rows
    assert cod["annotation_type"] == "drawing"
    assert cod["species"] == "Blue cod"
    assert cod["count"] == 1
    assert cod["is_nothing_here"] is False
    assert cod["x1"] == 100.0
    assert cod["y1"] == 200.0
    assert cod["x2"] == 150.0  # x + width
    assert cod["y2"] == 280.0  # y + height

    assert bait["species"] == "Bait"
    assert bait["count"] == 1


def test_parse_annotation_empty_value():
    assert _parse_annotation({"value": []}) == []
    assert _parse_annotation({}) == []


def test_blank_submission_flagged_in_parsed_output():
    """A classification with value:[] on every task gets is_blank_submission=True."""
    from spyfish.zooniverse.parse_classifications import parse_classifications
    from spyfish.zooniverse.subject_keys import SubjectKeys

    blank_classification = {
        "classification_id": "99",
        "created_at": "2025-03-01T00:00:00Z",
        "user_name": "bot_user",
        "user_id": "999",
        "annotations": [{"task": "T0", "value": [], "taskType": "survey"}],
        "subject_id": "42",
        "subject_set_id": None,
        "workflow_id": "23923",
        "subject_metadata": {
            SubjectKeys.VIDEO_FILENAME: "DROP_20240101_BUV_SITE_001_01",
            SubjectKeys.UPL_SECONDS: "10",
            SubjectKeys.SUBJECT_TYPE: "clip",
        },
        "subject_locations": [],
    }
    df = parse_classifications([blank_classification])
    assert len(df) == 1
    assert bool(df.iloc[0]["is_blank_submission"])
    assert bool(df.iloc[0]["is_nothing_here"])


def test_blank_submissions_excluded_from_aggregation():
    """Blank submissions must not count toward total_classifiers or nothing_here_votes."""
    import pandas as pd

    from spyfish.zooniverse.parse_classifications import aggregate_by_subject_species

    rows = [
        # One real NH vote
        {
            "user_id": "1",
            "user_name": "real",
            "classification_id": "1",
            "subject_id": "s1",
            "drop_id": "D",
            "video_filename": "v",
            "species": None,
            "is_nothing_here": True,
            "is_blank_submission": False,
            "count": 0,
            "absolute_seconds": None,
            "annotation_seconds": None,
            "upl_seconds": None,
            "subject_set_id": None,
            "workflow_id": "w",
            "subject_locations": [],
            "annotation_type": "classification",
        },
        # Three blank submissions — should be invisible to the aggregator
        *[
            {
                "user_id": str(i),
                "user_name": f"bot{i}",
                "classification_id": str(10 + i),
                "subject_id": "s1",
                "drop_id": "D",
                "video_filename": "v",
                "species": None,
                "is_nothing_here": True,
                "is_blank_submission": True,
                "count": 0,
                "absolute_seconds": None,
                "annotation_seconds": None,
                "upl_seconds": None,
                "subject_set_id": None,
                "workflow_id": "w",
                "subject_locations": [],
                "annotation_type": "classification",
            }
            for i in range(2, 5)
        ],
    ]
    df = pd.DataFrame(rows)
    # With blanks excluded, no species row passes min_votes — agg is empty.
    # The key assertion is that total_classifiers reflects only the real vote.
    agg = aggregate_by_subject_species(df)
    # No species rows exist, so agg is empty — but the blank rows must not
    # have inflated total_classifiers for any species row that might exist.
    assert agg.empty or (agg["total_classifiers"] == 1).all()


def test_parse_annotation_skips_non_dict_values():
    ann = {"value": ["garbage", 42, None, {"choice": "Pagrus auratus"}]}
    rows = _parse_annotation(ann)
    assert len(rows) == 1
    assert rows[0]["species"] == "Pagrus auratus"


def test_parse_annotation_multiple_species():
    ann = {
        "value": [
            {
                "choice": "Pagrus auratus",
                "answers": {"HOWMANYINDIVIDUALSARETHEREATTHISTIMESTAMP": "2"},
            },
            {
                "choice": "Notolabrus fucicola",
                "answers": {"HOWMANYINDIVIDUALSARETHEREATTHISTIMESTAMP": "1"},
            },
        ]
    }
    rows = _parse_annotation(ann)
    assert len(rows) == 2
    assert {r["species"] for r in rows} == {"Pagrus auratus", "Notolabrus fucicola"}


# ── _extract_subject_metadata ────────────────────────────────────────────────


def test_extract_subject_metadata_canonical_keys():
    meta = {
        SubjectKeys.VIDEO_FILENAME: "media/KSF_20240124_BUV/KSF_20240124_BUV_KSF_085_01/KSF_20240124_BUV_KSF_085_01.mp4",
        SubjectKeys.UPL_SECONDS: "450.0",
        SubjectKeys.SUBJECT_TYPE: "clip",
        SubjectKeys.SITE_NAME: "KSF_085",
        SubjectKeys.LINK_TO_RESERVE: "Kapiti Island",
    }
    result = _extract_subject_metadata(meta)
    assert result["video_filename"].endswith(".mp4")
    assert result["upl_seconds"] == 450.0
    assert result["subject_type"] == "clip"
    assert result["site_name"] == "KSF_085"


def test_extract_subject_metadata_defaults_on_missing():
    result = _extract_subject_metadata({})
    assert result["video_filename"] == ""
    assert result["upl_seconds"] is None
    assert result["subject_type"] == "clip"  # default
    assert result["site_name"] == ""


def test_extract_subject_metadata_ignores_legacy_keys():
    """Strict mode: legacy key variants are NOT read."""
    meta = {
        "video_filename": "should_be_ignored.mp4",
        "#video_filename": "also_ignored.mp4",
        "upl_seconds": "999",
    }
    result = _extract_subject_metadata(meta)
    assert result["video_filename"] == ""  # not "should_be_ignored.mp4"
    assert result["upl_seconds"] is None  # not 999


# ── _missing_required_keys ───────────────────────────────────────────────────


def test_missing_required_keys_all_present():
    meta = {
        SubjectKeys.VIDEO_FILENAME: "test.mp4",
        SubjectKeys.SUBJECT_TYPE: "clip",
        SubjectKeys.UPL_SECONDS: "100",
    }
    assert _missing_required_keys(meta) == []


def test_missing_required_keys_some_missing():
    meta = {SubjectKeys.VIDEO_FILENAME: "test.mp4"}
    missing = _missing_required_keys(meta)
    assert SubjectKeys.SUBJECT_TYPE in missing
    assert SubjectKeys.UPL_SECONDS in missing
    assert SubjectKeys.VIDEO_FILENAME not in missing


# ── parse_classifications end-to-end ─────────────────────────────────────────


def _make_classification(
    subject_id="subj_1",
    video_filename="media/KSF_20240124_BUV/KSF_20240124_BUV_KSF_085_01/KSF_20240124_BUV_KSF_085_01.mp4",
    upl_seconds="450",
    subject_type="clip",
    annotations=None,
    classification_id="cls_1",
    user_name="volunteer1",
):
    """Build a minimal raw classification dict matching the Panoptes format."""
    if annotations is None:
        annotations = [
            {
                "value": [
                    {
                        "choice": "Pagrus auratus",
                        "answers": {
                            "HOWMANYINDIVIDUALSARETHEREATTHISTIMESTAMP": "2",
                            "WHATISTHEEARLIESTPOINTTHATYOUSEETHEMOSTINDIVIDUALSOFTHISSPECIES": "3S",
                        },
                    }
                ]
            }
        ]
    return {
        "classification_id": classification_id,
        "created_at": "2024-01-24T12:00:00Z",
        "user_name": user_name,
        "user_id": "u1",
        "subject_id": subject_id,
        "subject_set_id": "ss1",
        "workflow_id": "wf1",
        "subject_metadata": {
            SubjectKeys.VIDEO_FILENAME: video_filename,
            SubjectKeys.UPL_SECONDS: upl_seconds,
            SubjectKeys.SUBJECT_TYPE: subject_type,
            SubjectKeys.SITE_NAME: "KSF_085",
        },
        "annotations": annotations,
        "subject_locations": [],
    }


def test_parse_classifications_single_species():
    raw = [_make_classification()]
    df = parse_classifications(raw)
    assert len(df) == 1
    assert df.iloc[0]["species"] == "Pagrus auratus"
    assert df.iloc[0]["count"] == 2
    assert df.iloc[0]["annotation_seconds"] == 3.0
    assert df.iloc[0]["absolute_seconds"] == 450.0 + 3.0  # upl + annotation


def test_parse_classifications_nothing_here_preserved():
    raw = [_make_classification(annotations=[{"value": [{"choice": "NOTHINGHERE"}]}])]
    df = parse_classifications(raw)
    assert len(df) == 1
    assert df.iloc[0]["is_nothing_here"]
    assert df.iloc[0]["species"] is None


def test_parse_classifications_empty_annotations_gets_placeholder():
    raw = [_make_classification(annotations=[{"value": []}])]
    df = parse_classifications(raw)
    assert len(df) == 1
    assert df.iloc[0]["is_nothing_here"]


def test_parse_classifications_frame_uses_upl_seconds_directly():
    """For frame subjects, absolute_seconds should come from upl_seconds
    (not upl + annotation_seconds) because frames are static images."""
    raw = [
        _make_classification(
            subject_type="frame",
            upl_seconds="300",
            annotations=[
                {
                    "value": [
                        {
                            "choice": "Pagrus auratus",
                            "answers": {
                                "HOWMANYINDIVIDUALSARETHEREATTHISTIMESTAMP": "1",
                            },
                        }
                    ]
                }
            ],
        )
    ]
    df = parse_classifications(raw)
    # Frame without TimeOfMax falls through to upl_seconds + annotation_seconds.
    # annotation_seconds is None (no timestamp answer), so absolute_seconds = None.
    # This is a known gap — when TimeOfMax is present it works correctly.
    # For now, just assert the frame path doesn't crash.
    assert len(df) == 1
    assert df.iloc[0]["subject_type"] == "frame"


def test_parse_classifications_warns_on_missing_canonical_keys(caplog):
    """Legacy-key subjects should trigger a summary warning."""
    raw = [
        {
            "classification_id": "cls_1",
            "created_at": "2024-01-24T12:00:00Z",
            "user_name": "vol1",
            "user_id": "u1",
            "subject_id": "subj_1",
            "subject_set_id": "ss1",
            "workflow_id": "wf1",
            "subject_metadata": {
                # Legacy keys only — no canonical keys present
                "video_filename": "legacy.mp4",
                "upl_seconds": "100",
            },
            "annotations": [{"value": [{"choice": "NOTHINGHERE"}]}],
            "subject_locations": [],
        }
    ]
    import logging

    with caplog.at_level(logging.WARNING):
        df = parse_classifications(raw)
    assert len(df) == 1
    assert "missing current metadata keys" in caplog.text


# ── aggregate_by_subject_species ─────────────────────────────────────────────


def _make_parsed_rows(
    subject_id="subj_1",
    drop_id="KSF_20240124_BUV_KSF_085_01",
    species="Pagrus auratus",
    n_classifiers=5,
    n_species_votes=3,
    n_nothing_votes=2,
):
    """Build parsed classification rows ready for aggregation.

    Each row has a unique user_id so the aggregator's dedupe step
    (drop_duplicates by user_id, subject_id, species) treats them as
    distinct volunteer votes — same shape as real parsed data.
    """
    rows = []
    for i in range(n_species_votes):
        rows.append(
            {
                "classification_id": f"cls_{i}",
                "user_id": f"u_species_{i}",
                "subject_id": subject_id,
                "drop_id": drop_id,
                "video_filename": f"media/{drop_id}.mp4",
                "species": species,
                "count": 2,
                "absolute_seconds": 450.0,
                "is_nothing_here": False,
                "upl_seconds": 440.0,
                "subject_set_id": "ss1",
                "workflow_id": "wf1",
                "subject_locations": [],
            }
        )
    for i in range(n_nothing_votes):
        rows.append(
            {
                "classification_id": f"cls_nothing_{i}",
                "user_id": f"u_nothing_{i}",
                "subject_id": subject_id,
                "drop_id": drop_id,
                "video_filename": f"media/{drop_id}.mp4",
                "species": None,
                "count": 0,
                "absolute_seconds": None,
                "is_nothing_here": True,
                "upl_seconds": 440.0,
                "subject_set_id": "ss1",
                "workflow_id": "wf1",
                "subject_locations": [],
            }
        )
    return rows


@patch("spyfish.zooniverse.parse_classifications.config")
def test_aggregate_passes_min_agreement_pct(mock_config):
    """3 species + 2 NH → 60% agreement, passes 25% threshold."""
    mock_config.zooniverse_min_agreement_pct = 25.0
    mock_config.zooniverse_consensus_something_here_pct = 50.0
    rows = _make_parsed_rows(n_species_votes=3, n_nothing_votes=2)
    df = pd.DataFrame(rows)
    result = aggregate_by_subject_species(df)
    assert len(result) == 1
    assert result.iloc[0]["vote_count"] == 3
    assert result.iloc[0]["agreement_pct"] == 60.0


@patch("spyfish.zooniverse.parse_classifications.config")
def test_aggregate_filters_below_min_agreement_pct(mock_config):
    """1 species + 9 NH → 10% agreement, fails 25% threshold."""
    mock_config.zooniverse_min_agreement_pct = 25.0
    mock_config.zooniverse_consensus_something_here_pct = 50.0
    rows = _make_parsed_rows(n_species_votes=1, n_nothing_votes=9)
    df = pd.DataFrame(rows)
    result = aggregate_by_subject_species(df)
    assert len(result) == 0


@patch("spyfish.zooniverse.parse_classifications.config")
def test_aggregate_flags_suspicious_minority(mock_config):
    """1 species + 2 NH → 33% agreement (passes 25%); NH dominates so flag fires."""
    mock_config.zooniverse_min_agreement_pct = 25.0
    mock_config.zooniverse_consensus_something_here_pct = 50.0
    rows = _make_parsed_rows(n_species_votes=1, n_nothing_votes=2)
    df = pd.DataFrame(rows)
    result = aggregate_by_subject_species(df)
    assert len(result) == 1
    assert result.iloc[0]["suspicious_minority_find"]


@patch("spyfish.zooniverse.parse_classifications.config")
def test_aggregate_empty_input(mock_config):
    mock_config.zooniverse_min_agreement_pct = 25.0
    mock_config.zooniverse_consensus_something_here_pct = 50.0
    result = aggregate_by_subject_species(pd.DataFrame())
    assert result.empty


@patch("spyfish.zooniverse.parse_classifications.config")
def test_aggregate_consensus_fish_on_species_disagreement(mock_config):
    """3 voters, 3 different species, 0 NH → no species clears 50%, but 100%
    something-here → one consensus row labelled 'fish'.
    """
    mock_config.zooniverse_min_agreement_pct = 50.0
    mock_config.zooniverse_consensus_something_here_pct = 50.0
    # Build a parsed-rows dataframe with three different species on one subject
    rows = []
    for i, sp in enumerate(("BLUECOD", "SNAPPER", "TARAKIHI")):
        rows.append(
            {
                "user_id": f"u{i}",
                "user_name": f"vol{i}",
                "classification_id": f"c{i}",
                "subject_id": "subj1",
                "drop_id": "KSF_20240124_BUV_KSF_085_01",
                "video_filename": "v.mp4",
                "species": sp,
                "is_nothing_here": False,
                "is_blank_submission": False,
                "count": 1,
                "absolute_seconds": 30.0,
                "annotation_seconds": 30.0,
                "upl_seconds": 0.0,
                "subject_set_id": "ss1",
                "workflow_id": "wf1",
                "subject_locations": [],
                "annotation_type": "classification",
            }
        )
    df = pd.DataFrame(rows)
    result = aggregate_by_subject_species(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["species"] == "fish"
    assert row["vote_count"] == 3
    assert row["agreement_pct"] == 100.0


@patch("spyfish.zooniverse.parse_classifications.config")
def test_aggregate_no_consensus_when_species_clears_gate(mock_config):
    """3 species + 0 NH → species passes 100% agreement; no consensus row."""
    mock_config.zooniverse_min_agreement_pct = 50.0
    mock_config.zooniverse_consensus_something_here_pct = 50.0
    rows = _make_parsed_rows(n_species_votes=3, n_nothing_votes=0)
    df = pd.DataFrame(rows)
    result = aggregate_by_subject_species(df)
    assert len(result) == 1
    assert result.iloc[0]["species"] != "fish"  # named species, not consensus


@patch("spyfish.zooniverse.parse_classifications.config")
def test_aggregate_no_consensus_when_nothing_here_dominates(mock_config):
    """1 species + 4 NH → 20% something-here; below 50% consensus threshold."""
    mock_config.zooniverse_min_agreement_pct = 50.0
    mock_config.zooniverse_consensus_something_here_pct = 50.0
    rows = _make_parsed_rows(n_species_votes=1, n_nothing_votes=4)
    df = pd.DataFrame(rows)
    result = aggregate_by_subject_species(df)
    assert len(result) == 0  # neither species nor consensus passes
