"""Validation sync-back: version-aware answers, box matching, verdicts, status.

Panoptes is never called: workflow versions come from a fake fetch, and
classifications are built by hand in the shape `classifications_for_set`
returns.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spyfish.config.base import ZooCropsStatus, ZooFramesStatus
from spyfish.config.wrapper import config
from spyfish.zooniverse import validation_sync as vs
from spyfish.zooniverse.parse_classifications import _parse_subject_set_display_name
from spyfish.zooniverse.subject_keys import CROP_PATH_COLUMN, FRAME_PATH_COLUMN

DROP = "KSF_20240124_BUV_KSF_085_06"
CROP_WF = "32407"
FRAME_WF = "32835"
MISS_IOU = 0.3

# ── workflow versions, in the shape Panoptes /workflow_versions returns ──────

CROP_VERSION = {
    "major_number": 5,
    "minor_number": 94,
    "first_task": "T0",
    "tasks": {"T0": {"type": "single", "answers": [{"label": "Yes"}, {"label": "No"}]}},
    "strings": {
        "T0.question": "Is the target fish mostly within the box and the right species?",
        "T0.answers.0.label": "Yes",
        "T0.answers.1.label": "No",
    },
}

# Before 2026-09-20: "Are ALL fish in a box?"  No (index 1) → draw.
FRAME_VERSION_OLD = {
    "major_number": 16,
    "minor_number": 57,
    "first_task": "T2",
    "tasks": {
        "T0": {
            "type": "single",
            "answers": [{"label": "Yes", "next": ""}, {"label": "No", "next": "T1"}],
        },
        "T1": {"type": "drawing", "tools": [{"type": "rectangle"}]},
        "T2": {
            "type": "single",
            "answers": [{"label": "Yes", "next": "T0"}, {"label": "No", "next": "T0"}],
        },
    },
    "strings": {
        "T0.question": "Are ALL fish in an annotation box?",
        "T0.answers.0.label": "Yes",
        "T0.answers.1.label": "No",
        "T2.question": "Are the boxes all on fish and correctly sized?",
        "T2.answers.0.label": "Yes",
        "T2.answers.1.label": "No",
    },
}

# After: "Any fish NOT in a box?"  Yes (index 0) → draw. Same indexes, opposite meaning.
FRAME_VERSION_NEW = {
    **FRAME_VERSION_OLD,
    "major_number": 18,
    "minor_number": 58,
    "tasks": {
        **FRAME_VERSION_OLD["tasks"],
        "T0": {
            "type": "single",
            "answers": [{"label": "Yes", "next": "T1"}, {"label": "No", "next": ""}],
        },
    },
    "strings": {
        **FRAME_VERSION_OLD["strings"],
        "T0.question": "Are there any fish visible that are not in an annotation box?",
    },
}

VERSIONS_BY_WORKFLOW = {
    CROP_WF: [CROP_VERSION],
    FRAME_WF: [FRAME_VERSION_OLD, FRAME_VERSION_NEW],
}


@pytest.fixture
def versions(tmp_path):
    return vs.WorkflowVersions(
        tmp_path / "wf", fetch=lambda wid: VERSIONS_BY_WORKFLOW[wid]
    )


def _cls(cid, meta, annotations, wid, version, user="u1"):
    """One classification, as `classifications_for_set` returns it."""
    return {
        "classification_id": cid,
        "user_id": user,
        "user_ip": None,
        "annotations": annotations,
        "subject_id": f"s{cid}",
        "subject_set_id": "77",
        "workflow_id": wid,
        "subject_metadata": meta,
        "subject_locations": [],
        "classification_metadata": {
            "workflow_version": version,
            "started_at": "2026-10-01T00:00:00Z",
            "finished_at": "2026-10-01T00:00:06Z",
        },
    }


def _single(task, index):
    return {"task": task, "value": index, "taskType": "single"}


def _rect(x1, y1, x2, y2):
    return {
        "toolType": "rectangle",
        "x_center": (x1 + x2) / 2,
        "y_center": (y1 + y2) / 2,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def _drawing(*rects):
    return {"task": "T1", "value": list(rects), "taskType": "drawing"}


# ── crops ────────────────────────────────────────────────────────────────────


def _crop_manifest(keys):
    rows = []
    for i, key in enumerate(keys):
        rows.append(
            {
                config.drop_id_column: DROP,
                "FrameIndex": 1000 + i,
                config.csv_clip_max_time_column: 40.0 + i,
                config.csv_scientific_name_column: "Parapercis colias",
                "LabelShown": "Blue cod",
                config.csv_confidence_agreement_column: 0.8,
                "DetectionKey": key,
                "BBox": "10,10,60,60",
                "ModelName": "species_20260825",
                CROP_PATH_COLUMN: f"/x/{key}.jpg",
                "NotCroppedReason": None,
            }
        )
    # A box that was never uploaded (too small): must not appear in the verdicts.
    rows.append({**rows[0], "DetectionKey": "tiny", CROP_PATH_COLUMN: None})
    return pd.DataFrame(rows)


def _crop_cls(cid, key, index, user):
    return _cls(
        cid, {vs.DETECTION_KEY: key}, [_single("T0", index)], CROP_WF, "5.94", user
    )


def test_crop_verdicts_majority_split_and_no_votes(versions):
    manifest = _crop_manifest(["a", "b", "c"])
    cls = [
        _crop_cls(1, "a", 0, "u1"),
        _crop_cls(2, "a", 0, "u2"),
        _crop_cls(3, "a", 1, "u3"),
        _crop_cls(4, "b", 0, "u1"),
        _crop_cls(5, "b", 1, "u2"),
    ]
    votes = vs.crop_votes(cls, versions)
    assert set(votes["answer"]) == {"Yes", "No"}
    assert votes["question"].str.startswith("Is the target fish").all()

    verdicts = vs.crop_verdicts(votes, manifest).set_index("DetectionKey")
    assert list(verdicts.index) == ["a", "b", "c"]  # "tiny" was never shown
    assert verdicts.loc["a", ["n_votes", "n_yes", "n_no", "verdict"]].tolist() == [
        3,
        2,
        1,
        vs.VERDICT_CORRECT,
    ]
    assert verdicts.loc["b", "verdict"] == vs.VERDICT_SPLIT
    assert verdicts.loc["c", "verdict"] == vs.VERDICT_NO_VOTES
    assert verdicts.loc["a", "LabelShown"] == "Blue cod"  # joined from the manifest


def test_crop_precision_by_label_ignores_split(versions):
    manifest = _crop_manifest(["a", "b", "c"])
    cls = [
        _crop_cls(1, "a", 0, "u1"),
        _crop_cls(2, "b", 1, "u1"),
        _crop_cls(3, "c", 0, "u1"),
        _crop_cls(4, "c", 1, "u2"),
    ]
    verdicts = vs.crop_verdicts(vs.crop_votes(cls, versions), manifest)
    table = vs.crop_precision_by_label(verdicts)
    row = table.loc["Blue cod"]
    assert (row["crops"], row["correct"], row["wrong"], row["split"]) == (3, 1, 1, 1)
    assert row["precision"] == 0.5


# ── frames ───────────────────────────────────────────────────────────────────


def _frame_manifest(frames):
    """frames: {FrameIndex: (RenderScale, [video-pixel bbox, ...])}"""
    return pd.DataFrame(
        [
            {
                config.drop_id_column: DROP,
                "FrameIndex": f,
                config.csv_clip_max_time_column: f / 25.0,
                config.selection_reason_column: "Blind",
                "ModelName": "species_20260825",
                FRAME_PATH_COLUMN: f"/x/{f}.jpg",
                "RenderScale": scale,
                "Boxes": json.dumps(
                    [
                        {"box_index": i, "class": "fish", "confidence": 0.9, "bbox": b}
                        for i, b in enumerate(boxes)
                    ]
                ),
            }
            for f, (scale, boxes) in frames.items()
        ]
    )


def _frame_cls(cid, frame, annotations, version, user="u1"):
    return _cls(
        cid, {vs.FRAME_INDEX_KEY: str(frame)}, annotations, FRAME_WF, version, user
    )


def test_gate_answer_is_read_through_version_routing(versions):
    """Same index, opposite meaning across versions: routing decides, not the label."""
    manifest = _frame_manifest({500: (1.0, [])})
    cls = [
        _frame_cls(1, 500, [_single("T2", 0), _single("T0", 1), _drawing()], "16.57"),
        _frame_cls(2, 500, [_single("T2", 0), _single("T0", 0), _drawing()], "18.58"),
        _frame_cls(3, 500, [_single("T2", 1), _single("T0", 1)], "18.58"),
    ]
    answers, _ = vs.frame_votes(cls, versions, manifest, MISS_IOU)
    by_id = answers.set_index("classification_id")
    assert by_id.loc[1, "fish_missed"] == "Yes"  # old wording, "No" → draw
    assert by_id.loc[2, "fish_missed"] == "Yes"  # new wording, "Yes" → draw
    assert by_id.loc[3, "fish_missed"] == "No"
    assert by_id["boxes_all_correct"].tolist() == ["Yes", "Yes", "No"]


def test_drawn_boxes_map_to_video_pixels_and_match_ml_boxes(versions):
    """Half-size render: a box drawn at (50..100) is the ML box at (100..200)."""
    manifest = _frame_manifest({500: (0.5, [[100, 100, 200, 200]])})
    cls = [
        _frame_cls(
            1,
            500,
            [
                _single("T2", 0),
                _single("T0", 0),
                _drawing(_rect(50, 50, 100, 100), _rect(300, 300, 350, 350)),
            ],
            "18.58",
            "u1",
        ),
        _frame_cls(
            2,
            500,
            [_single("T2", 0), _single("T0", 0), _drawing(_rect(302, 298, 352, 352))],
            "18.58",
            "u2",
        ),
        _frame_cls(3, 500, [_single("T2", 0), _single("T0", 1)], "18.58", "u3"),
    ]
    answers, drawn = vs.frame_votes(cls, versions, manifest, MISS_IOU)
    assert answers.set_index("classification_id")["n_new_fish"].tolist() == [1, 1, 0]
    redraw = drawn[(drawn["classification_id"] == 1) & (~drawn["is_miss"])].iloc[0]
    assert (redraw["x1"], redraw["y1"], redraw["x2"], redraw["y2"]) == (
        100,
        100,
        200,
        200,
    )
    assert redraw["best_iou_with_ml"] == 1.0

    missed = vs.cluster_missed_boxes(drawn, {500: 3}, MISS_IOU)
    assert len(missed) == 1  # u1 and u2 drew the same fish
    assert int(missed.iloc[0]["n_supporters"]) == 2
    assert bool(missed.iloc[0]["consensus"])  # 2 of 3 answers
    assert missed.iloc[0]["x1"] == pytest.approx(602, abs=1)


def test_frame_verdicts_per_frame(versions):
    manifest = _frame_manifest({500: (1.0, [[0, 0, 10, 10]]), 600: (1.0, [])})
    cls = [
        _frame_cls(1, 500, [_single("T2", 0), _single("T0", 1)], "18.58", "u1"),
        _frame_cls(
            2,
            500,
            [_single("T2", 1), _single("T0", 0), _drawing(_rect(50, 50, 90, 90))],
            "18.58",
            "u2",
        ),
        _frame_cls(
            3,
            500,
            [_single("T2", 1), _single("T0", 0), _drawing(_rect(52, 48, 92, 92))],
            "18.58",
            "u3",
        ),
    ]
    answers, drawn = vs.frame_votes(cls, versions, manifest, MISS_IOU)
    missed = vs.cluster_missed_boxes(drawn, {500: 3}, MISS_IOU)
    verdicts = vs.frame_verdicts(answers, missed, manifest).set_index("FrameIndex")
    f500 = verdicts.loc[500]
    assert (f500["n_votes"], f500["n_ml_boxes"]) == (3, 1)
    assert f500["boxes_verdict"] == vs.VERDICT_WRONG
    assert f500["missed_verdict"] == "missed"
    assert f500["n_consensus_missed"] == 1
    f600 = verdicts.loc[600]  # shown, nobody answered
    assert (f600["n_votes"], f600["boxes_verdict"], f600["missed_verdict"]) == (
        0,
        vs.VERDICT_NO_VOTES,
        vs.VERDICT_NO_VOTES,
    )


# ── workflow versions cache ──────────────────────────────────────────────────


def test_versions_refetch_once_when_a_version_is_missing(tmp_path):
    served = [[CROP_VERSION]]
    calls = []

    def fetch(wid):
        calls.append(wid)
        return served[-1]

    versions = vs.WorkflowVersions(tmp_path, fetch=fetch)
    assert versions.get(CROP_WF, "5.94").first_task == "T0"
    assert calls == [CROP_WF]
    served.append([CROP_VERSION, {**CROP_VERSION, "minor_number": 95}])
    assert versions.get(CROP_WF, "5.95").first_task == "T0"
    assert calls == [CROP_WF, CROP_WF]
    with pytest.raises(KeyError):
        versions.get(CROP_WF, "9.1")
    # Second instance reads the cache file, no fetch.
    again = vs.WorkflowVersions(tmp_path, fetch=lambda wid: pytest.fail("fetched"))
    assert again.get(CROP_WF, "5.95").first_task == "T0"


# ── orchestration ────────────────────────────────────────────────────────────


def test_display_name_parser_knows_validation_prefixes():
    assert _parse_subject_set_display_name(f"vframes_{DROP}") == ("vframes", DROP)
    assert _parse_subject_set_display_name(f"crops_{DROP}") == ("crops", DROP)
    assert _parse_subject_set_display_name(f"clips_{DROP}") == ("clips", DROP)
    assert _parse_subject_set_display_name("crops_test_KSF_20240124_BUV") == (
        None,
        None,
    )


def _completion(set_type, pct):
    return pd.DataFrame(
        [
            {
                "project_id": 14054,
                "subject_set_id": "77",
                "subject_set_type": set_type,
                "drop_id": DROP,
                "total": 10,
                "retired": int(10 * pct / 100),
                "pct_retired": pct,
                "workflow_ids": ["32407"],
                "fully_complete": pct >= 100,
            }
        ]
    )


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_project_root", tmp_path)
    return tmp_path


def test_crops_sync_writes_verdicts_and_advances(local_root, versions):
    manifest = _crop_manifest(["a"])
    path = config.get_validation_manifest_path(DROP, "crops")
    path.parent.mkdir(parents=True)
    manifest.to_csv(path, index=False)
    db = MagicMock()
    kind = vs.KINDS[0]
    cls = [_crop_cls(1, "a", 0, "u1"), _crop_cls(2, "a", 0, "u2")]
    with patch.object(vs, "classifications_for_set", return_value=cls) as fetch:
        assert vs._sync_one(
            kind, DROP, _completion("crops", 100.0), versions, db, False
        )
        fetch.assert_called_once_with("77", ["32407"])
    db.advance_status.assert_called_once_with(
        DROP, ZooCropsStatus.COLUMN, ZooCropsStatus.COMPLETE
    )
    verdicts = pd.read_csv(config.get_validation_verdicts_path(DROP, "crops"))
    assert verdicts["verdict"].tolist() == [vs.VERDICT_CORRECT]
    assert config.get_validation_classifications_path(DROP, "crops").exists()

    # Re-run reads the cache, never Panoptes.
    db.reset_mock()
    with patch.object(vs, "classifications_for_set") as fetch:
        assert vs._sync_one(
            kind, DROP, _completion("crops", 100.0), versions, db, False
        )
        fetch.assert_not_called()


def test_unretired_or_missing_set_leaves_status_alone(local_root, versions):
    db = MagicMock()
    kind = vs.KINDS[1]
    with patch.object(vs, "classifications_for_set") as fetch:
        assert not vs._sync_one(
            kind, DROP, _completion("vframes", 60.0), versions, db, False
        )
        assert not vs._sync_one(
            kind, DROP, _completion("crops", 100.0), versions, db, False
        )
        fetch.assert_not_called()
    db.advance_status.assert_not_called()


def test_retired_set_without_classifications_raises(local_root, versions):
    """A retired set that joins to nothing is a data problem, not an absence."""
    manifest = _crop_manifest(["a"])
    path = config.get_validation_manifest_path(DROP, "crops")
    path.parent.mkdir(parents=True)
    manifest.to_csv(path, index=False)
    db = MagicMock()
    with patch.object(vs, "classifications_for_set", return_value=[]):
        with pytest.raises(RuntimeError, match="no classifications"):
            vs._sync_one(
                vs.KINDS[0], DROP, _completion("crops", 100.0), versions, db, False
            )
    db.advance_status.assert_not_called()
    assert not config.get_validation_classifications_path(DROP, "crops").exists()


def test_frames_sync_writes_both_csvs(local_root, versions):
    manifest = _frame_manifest({500: (1.0, [[0, 0, 10, 10]])})
    path = config.get_validation_manifest_path(DROP, "frames")
    path.parent.mkdir(parents=True)
    manifest.to_csv(path, index=False)
    db = MagicMock()
    cls = [
        _frame_cls(
            1,
            500,
            [_single("T2", 0), _single("T0", 0), _drawing(_rect(50, 50, 90, 90))],
            "18.58",
        ),
    ]
    with patch.object(vs, "classifications_for_set", return_value=cls):
        assert vs._sync_one(
            vs.KINDS[1], DROP, _completion("vframes", 100.0), versions, db, False
        )
    db.advance_status.assert_called_once_with(
        DROP, ZooFramesStatus.COLUMN, ZooFramesStatus.COMPLETE
    )
    missed = pd.read_csv(config.get_validation_missed_boxes_path(DROP))
    assert missed[config.drop_id_column].tolist() == [DROP]
    assert bool(missed.iloc[0]["consensus"])
    verdicts = pd.read_csv(config.get_validation_verdicts_path(DROP, "frames"))
    assert verdicts["n_consensus_missed"].tolist() == [1]


def test_answer_polarity_reads_config_labels_and_refuses_unknown_ones():
    """Labels are resolved per workflow version, so a reworded answer must
    raise rather than silently count as no vote (every verdict would empty)."""
    assert vs.answer_polarity(None) is None
    assert vs.answer_polarity("Yes") == vs.YES
    assert vs.answer_polarity("No") == vs.NO
    with pytest.raises(ValueError, match="neither a yes nor a no"):
        vs.answer_polarity("Correct")
