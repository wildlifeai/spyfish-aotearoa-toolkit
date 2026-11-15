"""--db-refresh pass for the Zooniverse validation sections (zoo_frames / zoo_crops)."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spyfish.config.base import ZooCropsStatus, ZooFramesStatus
from spyfish.config.wrapper import config
from spyfish.orchestrator import db_refresh

DROP_A = "KSF_20240124_BUV_KSF_085_01"
DROP_B = "KSF_20240124_BUV_KSF_085_02"


def _listing(*sets, project_id=None):
    """`subject_completion_from_api`'s frame for (type, drop_id, total) sets."""
    project_id = project_id or config.zooniverse_project_id
    return pd.DataFrame(
        [
            {
                "project_id": project_id,
                "subject_set_id": str(i),
                "subject_set_type": set_type,
                "drop_id": drop_id,
                "total": total,
            }
            for i, (set_type, drop_id, total) in enumerate(sets)
        ],
        columns=[
            "project_id",
            "subject_set_id",
            "subject_set_type",
            "drop_id",
            "total",
        ],
    )


def _frames_candidates(*drop_ids):
    return lambda _db, section, _excl: (
        [{"drop_id": d} for d in drop_ids] if section == ZooFramesStatus.COLUMN else []
    )


@patch("spyfish.orchestrator.db_refresh._ok_drops_not_at")
def test_set_on_zooniverse_marks_pending_drop_uploaded(candidates):
    """DB says pending, Zooniverse holds the set: the DB was behind, not the upload."""
    db = MagicMock()
    candidates.side_effect = _frames_candidates(DROP_A, DROP_B)
    listing = MagicMock(return_value=_listing(("vframes", DROP_A, 12)))

    assert db_refresh.refresh_zooniverse_validation(db, listing) == 1
    db.update_section_status.assert_called_once_with(
        DROP_A, ZooFramesStatus.COLUMN, ZooFramesStatus.UPLOADED
    )


@patch("spyfish.orchestrator.db_refresh._ok_drops_not_at")
def test_settled_statuses_are_excluded_and_nothing_is_requested(candidates):
    """Uploaded, complete and (for crops) skipped drops are never asked about,
    and with no candidate left Zooniverse is not contacted at all."""
    candidates.return_value = []
    listing = MagicMock()
    db_refresh.refresh_zooniverse_validation(MagicMock(), listing)
    excludes = {call.args[1]: call.args[2] for call in candidates.call_args_list}
    assert set(excludes[ZooFramesStatus.COLUMN]) == {
        ZooFramesStatus.UPLOADED,
        ZooFramesStatus.COMPLETE,
    }
    assert set(excludes[ZooCropsStatus.COLUMN]) == {
        ZooCropsStatus.UPLOADED,
        ZooCropsStatus.COMPLETE,
        ZooCropsStatus.SKIPPED,
    }
    listing.assert_not_called()


@patch("spyfish.orchestrator.db_refresh._ok_drops_not_at")
def test_empty_set_does_not_count_as_uploaded(candidates):
    candidates.side_effect = lambda _db, section, _excl: (
        [{"drop_id": DROP_A}] if section == ZooCropsStatus.COLUMN else []
    )
    db = MagicMock()
    listing = MagicMock(return_value=_listing(("crops", DROP_A, 0)))
    assert db_refresh.refresh_zooniverse_validation(db, listing) == 0
    db.update_section_status.assert_not_called()


@patch("spyfish.orchestrator.db_refresh._ok_drops_not_at")
def test_only_the_matching_kind_in_the_upload_project_counts(candidates):
    """A clips set, the other kind's set, or a same-named set in another
    source project says nothing about this drop's vframes set."""
    candidates.side_effect = _frames_candidates(DROP_A, DROP_B)
    other_project = next(
        p
        for p in [*config.zooniverse_source_project_ids, -1]
        if p != config.zooniverse_project_id
    )
    listing = pd.concat(
        [
            _listing(("clips", DROP_A, 180), ("crops", DROP_A, 40)),
            _listing(("vframes", DROP_B, 12), project_id=other_project),
        ]
    )
    db = MagicMock()
    assert db_refresh.refresh_zooniverse_validation(db, lambda: listing) == 0
    db.update_section_status.assert_not_called()


@patch("spyfish.orchestrator.db_refresh._ok_drops_not_at")
def test_many_drops_cost_one_login_and_one_listing(candidates):
    """Both kinds, many drops: one connect and one paged listing per source
    project, never a query per drop."""
    drops = [f"KSF_20240124_BUV_KSF_085_{i:02d}" for i in range(1, 31)]
    candidates.side_effect = lambda _db, _section, _excl: [
        {"drop_id": d} for d in drops
    ]

    def _set(name, count):
        ss = MagicMock(display_name=name, id=name)
        ss.raw = {"set_member_subjects_count": count, "links": {}}
        return ss

    on_zooniverse = [_set(f"vframes_{d}", 12) for d in drops[:10]] + [
        _set(f"crops_{d}", 40) for d in drops[:5]
    ]
    upload_project = config.zooniverse_project_id
    db = MagicMock()
    with (
        patch("spyfish.zooniverse.parse_classifications.Panoptes") as panoptes,
        patch("panoptes_client.SubjectSet") as subject_set,
        patch("spyfish.zooniverse.upload.Panoptes") as upload_panoptes,
        patch("spyfish.zooniverse.upload.SubjectSet") as upload_subject_set,
    ):
        subject_set.where.side_effect = lambda project_id: (
            on_zooniverse if project_id == upload_project else []
        )
        assert db_refresh.refresh_zooniverse_validation(db) == 15

    panoptes.connect.assert_called_once()
    assert subject_set.where.call_count == len(config.zooniverse_source_project_ids)
    assert all(
        set(c.kwargs) == {"project_id"} for c in subject_set.where.call_args_list
    )
    # The per-drop path (login + set query + subject paging) is not used.
    upload_panoptes.connect.assert_not_called()
    upload_subject_set.where.assert_not_called()
    for ss in on_zooniverse:
        assert not ss.subjects.called and not ss.mock_calls


def test_both_zooniverse_passes_share_one_listing():
    """The clips pass and the validation pass of one refresh list the sets once."""
    with (
        patch("spyfish.zooniverse.parse_classifications.connect_to_zooniverse") as conn,
        patch(
            "spyfish.zooniverse.parse_classifications.subject_completion_from_api",
            return_value=_listing(),
        ) as completion,
    ):
        load = db_refresh._subject_set_listing()
        conn.assert_not_called()  # lazy: nothing until a pass asks
        assert load() is load()
    conn.assert_called_once()
    completion.assert_called_once()


@patch("spyfish.orchestrator.db_refresh._ok_drops_not_at")
def test_upload_project_missing_from_the_listing_raises(candidates, monkeypatch):
    """A listing that cannot contain the upload project's sets would read as
    "no set anywhere": wrong data, so it raises instead."""
    candidates.side_effect = _frames_candidates(DROP_A)
    monkeypatch.setattr(
        type(config), "zooniverse_source_project_ids", property(lambda self: [-1])
    )
    with pytest.raises(ValueError, match="source_project_ids"):
        db_refresh.refresh_zooniverse_validation(MagicMock(), lambda: _listing())
