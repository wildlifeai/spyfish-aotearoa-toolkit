"""Validation uploads: metadata per kind, set names, safe re-runs. Panoptes is mocked."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spyfish.config.wrapper import config
from spyfish.zooniverse import upload as up

DROP_ID = "KSF_20240124_BUV_KSF_085_06"


@pytest.fixture
def panoptes(monkeypatch):
    """Mocked panoptes_client names inside upload.py; no subject set exists yet."""
    monkeypatch.setenv("ZOONIVERSE_USER", "tester")
    monkeypatch.setenv("ZOONIVERSE_PASSWORD", "secret")
    saved = []

    def _subject():
        s = MagicMock()
        s.metadata = {}
        saved.append(s)
        return s

    with (
        patch.object(up, "Panoptes"),
        patch.object(up, "Project") as project,
        patch.object(up, "SubjectSet") as subject_set,
        patch.object(up, "Subject", side_effect=_subject),
        patch.object(
            up,
            "_get_site_reserve_meta",
            return_value={"LinkToMarineReserve": "Test MR"},
        ),
    ):
        project.find.return_value = MagicMock(id=99999)
        subject_set.where.return_value = []
        yield {"set": subject_set, "subjects": saved}


def _base(tmp_path, name, **extra):
    img = tmp_path / name
    img.write_bytes(b"jpeg")
    return {
        config.drop_id_column: DROP_ID,
        config.csv_sampling_start_column: 120.0,
        config.csv_clip_max_time_column: 300.333,
        config.selection_reason_column: "MaxN (Parapercis colias)",
        "FrameIndex": 18010,
        "ModelName": "species_20260603",
        **extra,
    }, img


def _crop_manifest(tmp_path, n=2, missing=0):
    rows = []
    for i in range(n + missing):
        row, img = _base(tmp_path, f"crop_{i}.jpg")
        if i >= n:
            img.unlink()  # on the manifest, gone from disk
        rows.append(
            {
                **row,
                "CropPath": str(img),
                config.csv_scientific_name_column: "Parapercis colias",
                config.csv_max_interval_column: 2,
                config.csv_confidence_agreement_column: 0.81,
                "DetectionKey": f"{DROP_ID}|18010|{i}",
                "BBox": "140,140,260,220",
                "LabelShown": "Blue cod",
            }
        )
    # A not-cropped box: no image, must be left out of the upload.
    rows.append({**rows[0], "CropPath": None, "DetectionKey": f"{DROP_ID}|18010|9"})
    return pd.DataFrame(rows)


def test_crops_go_to_crops_set_with_clip_metadata_plus_box_keys(tmp_path, panoptes):
    up.upload_validation_crops_to_zooniverse(_crop_manifest(tmp_path))
    created = panoptes["set"].return_value
    assert created.display_name == f"crops_{DROP_ID}"
    subjects = panoptes["subjects"]
    assert len(subjects) == 2  # the not-cropped row is skipped
    meta = subjects[0].metadata
    # Same base keys clips carry...
    for key in (
        "DropID",
        "#VideoFilename",
        "#siteName",
        "#SubjectType",
        "#SelectionReason",
        "#TargetSpecies",
        "#MaxInterval",
        "#ConfidenceAgreement",
        "#SamplingStart",
        "UplAbsSeconds",
        "LinkToMarineReserve",
    ):
        assert key in meta, key
    assert meta["#SubjectType"] == "crop"
    assert meta["UplAbsSeconds"] == 300.333
    # ...plus what joins an answer back to one ML box.
    assert meta["#DetectionKey"] == f"{DROP_ID}|18010|0"
    assert meta["#BBox"] == "140,140,260,220"
    assert meta["#LabelShown"] == "Blue cod"
    assert meta["#ModelName"] == "species_20260603"


def test_frames_go_to_vframes_set_not_old_frames_prefix(tmp_path, panoptes):
    row, img = _base(
        tmp_path,
        "f.jpg",
        **{
            config.selection_reason_column: "Blind (False Negative Check)",
            config.csv_scientific_name_column: None,
            config.csv_max_interval_column: 0,
            config.csv_confidence_agreement_column: None,
            "RenderScale": 1.0,
        },
    )
    up.upload_validation_frames_to_zooniverse(
        pd.DataFrame([{**row, "FramePath": str(img)}])
    )
    assert panoptes["set"].return_value.display_name == f"vframes_{DROP_ID}"
    meta = panoptes["subjects"][0].metadata
    assert meta["#SubjectType"] == "frame"
    assert meta["#RenderScale"] == 1.0
    assert meta["#FrameIndex"] == 18010


def test_blank_target_species_on_blind_frame_does_not_warn(tmp_path, panoptes, caplog):
    row, img = _base(
        tmp_path,
        "f.jpg",
        **{
            config.csv_scientific_name_column: None,
            config.csv_max_interval_column: 0,
            config.csv_confidence_agreement_column: None,
            "RenderScale": 1.0,
        },
    )
    with caplog.at_level(logging.WARNING):
        up.upload_validation_frames_to_zooniverse(
            pd.DataFrame([{**row, "FramePath": str(img)}])
        )
    assert "empty fields" not in caplog.text


def test_missing_files_are_not_counted(tmp_path, panoptes):
    """A re-run must compare against what was actually uploadable."""
    existing = MagicMock()
    existing.subjects = [object(), object()]
    panoptes["set"].where.return_value = [existing]
    # 2 on disk + 1 missing: the set holds 2, so this is "already uploaded".
    up.upload_validation_crops_to_zooniverse(_crop_manifest(tmp_path, n=2, missing=1))
    assert panoptes["subjects"] == []


def test_existing_set_without_identity_metadata_and_wrong_count_raises(
    tmp_path, panoptes
):
    """Subjects from before the identity keys existed cannot be matched, so
    the count rule still applies and a mismatch is left for a human."""
    existing = MagicMock()
    existing.subjects = [object()]
    panoptes["set"].where.return_value = [existing]
    with pytest.raises(RuntimeError, match="inconsistent"):
        up.upload_validation_crops_to_zooniverse(_crop_manifest(tmp_path, n=2))


def test_partial_set_is_resumed_with_only_the_missing_subjects(tmp_path, panoptes):
    """A crash after one of two crops was linked: the next run uploads the
    other one, re-uploads nothing, and creates no second set."""
    manifest = _crop_manifest(tmp_path, n=2)
    up.upload_validation_crops_to_zooniverse(manifest)
    first, second = panoptes["subjects"]
    already_there = MagicMock()
    # Zooniverse hands metadata back as strings; identity matching must cope.
    already_there.metadata = {k: str(v) for k, v in first.metadata.items()}
    existing = MagicMock()
    existing.subjects = [already_there]
    panoptes["set"].where.return_value = [existing]
    del panoptes["subjects"][:]

    up.upload_validation_crops_to_zooniverse(manifest)

    uploaded = panoptes["subjects"]
    assert len(uploaded) == 1
    assert uploaded[0].metadata["#DetectionKey"] == second.metadata["#DetectionKey"]
    existing.add.assert_called()  # linked into the EXISTING set
    existing.save.assert_not_called()


def test_complete_set_uploads_nothing_even_if_counts_differ(tmp_path, panoptes):
    """Every planned subject is already there (plus a stray extra): nothing is
    added, and the stray is reported, not treated as an error."""
    manifest = _crop_manifest(tmp_path, n=2)
    up.upload_validation_crops_to_zooniverse(manifest)
    present = [MagicMock(metadata=dict(s.metadata)) for s in panoptes["subjects"]]
    stray = MagicMock(metadata={**present[0].metadata, "#DetectionKey": "other|1|9"})
    existing = MagicMock()
    existing.subjects = present + [stray]
    panoptes["set"].where.return_value = [existing]
    del panoptes["subjects"][:]

    up.upload_validation_crops_to_zooniverse(manifest)
    assert panoptes["subjects"] == []


def test_subjects_linked_in_chunks(tmp_path, panoptes, monkeypatch):
    monkeypatch.setattr(up, "_LINK_CHUNK", 2)
    up.upload_validation_crops_to_zooniverse(_crop_manifest(tmp_path, n=5))
    calls = panoptes["set"].return_value.add.call_args_list
    assert [len(c.args[0]) for c in calls] == [2, 2, 1]


def test_upload_says_which_workflow_and_never_links_one(tmp_path, panoptes, caplog):
    with caplog.at_level(logging.WARNING):
        up.upload_validation_crops_to_zooniverse(_crop_manifest(tmp_path))
    assert f"Attach subject set 'crops_{DROP_ID}'" in caplog.text
    assert "species-check workflow" in caplog.text
    assert not hasattr(up, "Workflow")
