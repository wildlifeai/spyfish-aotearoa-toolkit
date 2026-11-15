"""--download: which drops are fetched and how each one is handled."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from spyfish.config.base import IngestStatus, MlStatus, VideoPresence
from spyfish.orchestrator.download import download_videos, select_download_drops

SURVEY = "KSF_20240124_BUV"
READY = f"{SURVEY}_KSF_085_01"
COMPLETE = f"{SURVEY}_KSF_085_02"


def test_survey_mode_takes_every_drop_whatever_its_ml_state():
    """Stages after ML (QA frames, Zooniverse validation, BIIGLE) need the
    videos of ml_complete drops too, so a survey download is not ml_ready only."""
    db = MagicMock()
    db.get_survey_ml_status_summary.return_value = {
        MlStatus.READY: [READY],
        MlStatus.COMPLETE: [COMPLETE],
    }
    assert select_download_drops(db, SURVEY) == [READY, COMPLETE]
    db.get_survey_ml_status_summary.assert_called_once_with(SURVEY)
    db.get_deployments_eligible.assert_not_called()


def test_no_survey_takes_the_next_ml_batch(monkeypatch):
    """Same frame --ml will pick: ml_ready, priority order, top limit_processing."""
    from spyfish.orchestrator import download as mod

    monkeypatch.setattr(type(mod.config), "limit_processing", property(lambda self: 2))
    db = MagicMock()
    db.get_deployments_eligible.return_value = [
        {"drop_id": f"{SURVEY}_KSF_085_0{i}"} for i in (1, 2, 3)
    ]
    assert select_download_drops(db) == [
        f"{SURVEY}_KSF_085_01",
        f"{SURVEY}_KSF_085_02",
    ]
    db.get_deployments_eligible.assert_called_once_with(
        MlStatus.COLUMN, [MlStatus.READY]
    )


@patch("spyfish.orchestrator.download.S3Handler")
def test_download_videos_skips_present_and_fetches_every_other_state(
    mock_s3_class, tmp_path, monkeypatch
):
    """Excluded / errored drops and drops flagged archived are fetched too:
    the flag goes stale after a restore, and a bad drop's video is what you
    need to diagnose it. Only a failed transfer counts as failed."""
    from spyfish.orchestrator import download as mod

    monkeypatch.setattr(type(mod.config), "media_dir", property(lambda self: tmp_path))
    present = f"{SURVEY}_KSF_085_03"
    archived = f"{SURVEY}_KSF_085_04"
    excluded = f"{SURVEY}_KSF_085_05"
    (tmp_path / f"{present}.mp4").write_bytes(b"x")

    def rec(drop_id, **over):
        base = {
            "drop_id": drop_id,
            "ingest_status": IngestStatus.OK,
            "video_path": f"media/{drop_id}.mp4",
            "video_presence": VideoPresence.PRESENT,
        }
        base.update(over)
        return base

    db = MagicMock()
    db.get_deployments_by_ids.return_value = {
        READY: rec(READY),
        present: rec(present),
        archived: rec(archived, video_presence=VideoPresence.ARCHIVED),
        excluded: rec(excluded, ingest_status=IngestStatus.EXCLUDED),
    }
    s3 = mock_s3_class.return_value
    # The archived flag is stale: the restore has finished, S3 serves it.
    s3.get_object_storage.side_effect = lambda key: {
        "storage_class": "DEEP_ARCHIVE" if archived in key else "STANDARD",
        "restore": None,
    }
    s3.is_restored.return_value = True
    s3.download_object_from_s3.side_effect = lambda key, filename: (
        Path(filename).write_bytes(b"x") or True
    )

    result = download_videos([READY, present, archived, excluded], db=db)

    assert result == {
        "downloaded": [READY, archived, excluded],
        "present": [present],
        "no_video": [],
        "failed": [],
    }
    assert [
        c.kwargs["key"]
        for c in mock_s3_class.return_value.download_object_from_s3.call_args_list
    ] == [f"media/{d}.mp4" for d in (READY, archived, excluded)]


@patch("spyfish.orchestrator.download.S3Handler")
def test_no_video_is_reported_apart_from_failures(mock_s3_class, tmp_path, monkeypatch):
    """No video_path (NULL or the text "nan" a blank SharePoint cell becomes)
    or no object at the key (404) is "no video", not a failed run. Still in
    DEEP_ARCHIVE or a transfer that breaks is a failure. No transfer is tried
    for any of the no-video drops."""
    from spyfish.orchestrator import download as mod

    monkeypatch.setattr(type(mod.config), "media_dir", property(lambda self: tmp_path))
    null_path = f"{SURVEY}_KSF_085_06"
    nan_path = f"{SURVEY}_KSF_085_07"
    not_in_s3 = f"{SURVEY}_KSF_085_08"
    frozen = f"{SURVEY}_KSF_085_09"
    db = MagicMock()
    db.get_deployments_by_ids.return_value = {
        READY: {"ingest_status": IngestStatus.OK, "video_path": f"media/{READY}.mp4"},
        null_path: {"ingest_status": IngestStatus.OK, "video_path": None},
        nan_path: {"ingest_status": IngestStatus.OK, "video_path": "nan"},
        not_in_s3: {
            "ingest_status": IngestStatus.OK,
            "video_path": f"media/{not_in_s3}.mp4",
            "video_presence": VideoPresence.ABSENT,
        },
        frozen: {"ingest_status": IngestStatus.OK, "video_path": f"media/{frozen}.mp4"},
    }
    s3 = mock_s3_class.return_value
    s3.get_object_storage.side_effect = lambda key: (
        None
        if not_in_s3 in key
        else {
            "storage_class": "DEEP_ARCHIVE" if frozen in key else "STANDARD",
            "restore": None,
        }
    )
    s3.is_restored.return_value = False
    s3.download_object_from_s3.return_value = False

    result = download_videos([READY, null_path, nan_path, not_in_s3, frozen], db=db)

    assert result == {
        "downloaded": [],
        "present": [],
        "no_video": [null_path, nan_path, not_in_s3],
        "failed": [frozen, READY],
    }
    s3.download_object_from_s3.assert_called_once_with(
        key=f"media/{READY}.mp4", filename=str(tmp_path / f"{READY}.mp4")
    )


@patch("spyfish.orchestrator.download.S3Handler")
def test_dry_run_checks_s3_but_transfers_nothing(mock_s3_class, tmp_path, monkeypatch):
    """A dry run still HEADs each video (read-only), so it reports what is
    missing for real; it transfers nothing and creates no media dir."""
    from spyfish.orchestrator import download as mod

    monkeypatch.setattr(
        type(mod.config), "media_dir", property(lambda self: tmp_path / "new")
    )
    missing = f"{SURVEY}_KSF_085_02"
    db = MagicMock()
    db.get_deployments_by_ids.return_value = {
        d: {"ingest_status": IngestStatus.OK, "video_path": f"media/{d}.mp4"}
        for d in (READY, missing)
    }
    s3 = mock_s3_class.return_value
    s3.get_object_storage.side_effect = lambda key: (
        None if missing in key else {"storage_class": "STANDARD", "restore": None}
    )
    result = download_videos([READY, missing], db=db, dry_run=True)
    assert result["downloaded"] == [READY]
    assert result["no_video"] == [missing]
    s3.download_object_from_s3.assert_not_called()
    assert not (tmp_path / "new").exists()


@patch("spyfish.orchestrator.download.S3Handler")
def test_every_check_runs_before_the_first_transfer(
    mock_s3_class, tmp_path, monkeypatch
):
    """A missing video that sorts after a 9 GB transfer must be reported
    before that transfer starts, not hours later."""
    from spyfish.orchestrator import download as mod

    monkeypatch.setattr(type(mod.config), "media_dir", property(lambda self: tmp_path))
    later = f"{SURVEY}_KSF_085_09"
    db = MagicMock()
    db.get_deployments_by_ids.return_value = {
        d: {"ingest_status": IngestStatus.OK, "video_path": f"media/{d}.mp4"}
        for d in (READY, later)
    }
    calls = []
    s3 = mock_s3_class.return_value
    s3.get_object_storage.side_effect = lambda key: calls.append(("head", key)) or (
        None if later in key else {"storage_class": "STANDARD", "restore": None}
    )
    s3.download_object_from_s3.side_effect = lambda key, filename: (
        calls.append(("get", key)) or Path(filename).write_bytes(b"x") or True
    )
    download_videos([READY, later], db=db)
    assert [c[0] for c in calls] == ["head", "head", "get"]
