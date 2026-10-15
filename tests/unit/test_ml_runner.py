from unittest.mock import patch

from spyfish.config.base import InvalidTransitionError, MlStatus
from spyfish.orchestrator.ml_runner import MLRunner


def _make_records(n: int) -> list[dict]:
    return [
        {
            "drop_id": f"KSF_20240124_BUV_KSF_085_{i+1:02d}",
            "video_path": f"path/{i+1}.mp4",
            "sampling_start": 0,
            "sampling_end": 100,
            "video_presence": "present",
        }
        for i in range(n)
    ]


@patch("spyfish.orchestrator.ml_runner.os.makedirs")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_get_inference_targets(mock_s3_class, mock_db_class, mock_makedirs):
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value

    # Mock database to return one item ready for ML
    mock_db.get_deployments_eligible.return_value = [
        {
            "drop_id": "KSF_20240124_BUV_KSF_085_01",
            "video_path": "path/1.mp4",
            "sampling_start": 120,
            "sampling_end": 100,
            "video_presence": "present",
        }
    ]

    # Mock S3 download to succeed
    mock_s3.download_object_from_s3.return_value = True

    runner = MLRunner()
    targets = runner.get_inference_targets()

    assert len(targets) == 1
    assert targets[0]["drop_id"] == "KSF_20240124_BUV_KSF_085_01"


@patch("spyfish.orchestrator.ml_runner.os.makedirs")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_get_inference_targets_caps_at_limit(
    mock_s3_class, mock_db_class, mock_exists, mock_makedirs
):
    """With 5 eligible drops and limit=3, only 3 downloads should be attempted."""
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value
    mock_db.get_deployments_eligible.return_value = _make_records(5)
    mock_s3.download_object_from_s3.return_value = True
    # db_path check → True; local video existence → False (so we download)
    mock_exists.side_effect = lambda p: str(p).endswith(".db")

    runner = MLRunner()
    runner.limit = 3
    targets = runner.get_inference_targets()

    assert len(targets) == 3
    # First 3 by priority order
    assert [t["drop_id"] for t in targets] == [
        "KSF_20240124_BUV_KSF_085_01",
        "KSF_20240124_BUV_KSF_085_02",
        "KSF_20240124_BUV_KSF_085_03",
    ]
    assert mock_s3.download_object_from_s3.call_count == 3


@patch("spyfish.orchestrator.ml_runner.os.makedirs")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_get_inference_targets_never_backfills_past_the_limit(
    mock_s3_class, mock_db_class, mock_exists, mock_makedirs
):
    """A failed drop shrinks the batch; it is never replaced from further
    down the queue.

    This reverses an earlier choice (backfill until `limit` succeed). That
    walk had no bound: on 2026-10-14 a 9-drop targeted run tried to download
    every one of 800+ ml_ready drops because three targets had no video, and
    only a mis-set bucket (every download 404'd) kept it from pulling the
    whole backlog at egress rates. The batch is the top `limit` by priority;
    what fails in it is logged, not papered over."""
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value
    mock_db.get_deployments_eligible.return_value = _make_records(5)
    # First 2 downloads fail, the rest would succeed if asked
    mock_s3.download_object_from_s3.side_effect = [False, False, True, True, True]
    mock_exists.side_effect = lambda p: str(p).endswith(".db")

    runner = MLRunner()
    runner.limit = 3
    targets = runner.get_inference_targets()

    # Only drop 03 survives; 04 and 05 are outside the top 3 and never touched
    assert [t["drop_id"] for t in targets] == ["KSF_20240124_BUV_KSF_085_03"]
    assert mock_s3.download_object_from_s3.call_count == 3


@patch("spyfish.orchestrator.ml_runner.os.makedirs")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_get_inference_targets_all_downloads_fail(
    mock_s3_class, mock_db_class, mock_exists, mock_makedirs
):
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value
    mock_db.get_deployments_eligible.return_value = _make_records(3)
    mock_s3.download_object_from_s3.return_value = False
    mock_exists.side_effect = lambda p: str(p).endswith(".db")

    runner = MLRunner()
    runner.limit = 3
    targets = runner.get_inference_targets()

    assert targets == []


@patch("spyfish.orchestrator.ml_runner.os.makedirs")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_force_survey_resets_stranded_running_drops(
    mock_s3_class, mock_db_class, mock_exists, mock_makedirs
):
    """--ml --survey --force recovers drops stranded in ml_running by a killed job.

    ml_running has no other exit: nothing but a completing inference run
    advances it, so a SLURM timeout or Ctrl-C parks drops there permanently.
    """
    mock_db = mock_db_class.return_value
    mock_s3 = mock_s3_class.return_value
    mock_exists.side_effect = lambda p: str(p).endswith(".db")
    mock_s3.download_object_from_s3.return_value = True

    survey_id = "KSF_20240124_BUV"
    mock_db.get_survey_ml_status_summary.return_value = {
        MlStatus.RUNNING: [f"{survey_id}_KSF_085_01", f"{survey_id}_KSF_085_02"],
        MlStatus.COMPLETE: [f"{survey_id}_KSF_085_03"],
        MlStatus.PENDING: [f"{survey_id}_KSF_085_04"],
    }
    mock_db.get_deployments_eligible.return_value = []

    runner = MLRunner()
    runner.get_inference_targets(survey_id=survey_id, force=True)

    reset = {
        call.args[0]
        for call in mock_db.update_section_status.call_args_list
        if call.args[1:] == (MlStatus.COLUMN, MlStatus.READY)
    }
    assert reset == {
        f"{survey_id}_KSF_085_01",
        f"{survey_id}_KSF_085_02",
        f"{survey_id}_KSF_085_03",
    }
    # ml_pending drops still need --check-arrivals, --force must not touch them.
    assert f"{survey_id}_KSF_085_04" not in reset


@patch("spyfish.orchestrator.ml_runner.process_one_drop")
@patch("spyfish.orchestrator.ml_runner.run_inference_main")
@patch("spyfish.orchestrator.ml_runner.AnnotationDatabaseManager")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_run_inference_loop_skips_drops_claimed_by_another_job(
    mock_s3_class,
    mock_db_class,
    mock_exists,
    mock_ann_db_class,
    mock_inference,
    mock_process_one_drop,
):
    """Losing the claim race on one drop must not abort the whole batch.

    Targets are selected minutes-to-hours before they are claimed (the video
    download sits in between), so a concurrent ML job can take some of them
    first, leaving them in ml_running. Claiming is per-drop: the stolen drop is
    skipped, the rest still run.
    """
    mock_db = mock_db_class.return_value
    mock_exists.return_value = True  # model weights + db present
    mock_db.get_deployment.return_value = {"ml_status": MlStatus.COMPLETE}

    targets = _make_records(3)
    stolen_id = targets[1]["drop_id"]
    for t in targets:
        t["VideoURL"] = f"/media/{t['drop_id']}.mp4"

    def claim(drop_id, section, to_status):
        if to_status == MlStatus.RUNNING and drop_id == stolen_id:
            raise InvalidTransitionError(
                f"{drop_id}: invalid MlStatus transition "
                f"{MlStatus.RUNNING!r} → {MlStatus.RUNNING!r}"
            )

    mock_db.advance_status.side_effect = claim

    runner = MLRunner()
    successes = runner.run_inference_loop(targets)

    kept = [targets[0]["drop_id"], targets[2]["drop_id"]]
    assert runner.claimed_drop_ids == kept
    assert successes == kept
    # The stolen drop belongs to the other job: never inferred, never re-statused.
    inferred = [call.args[0]["drop_id"] for call in mock_inference.call_args_list]
    assert inferred == kept
    assert stolen_id not in {
        call.args[0] for call in mock_db.update_section_status.call_args_list
    }


@patch("spyfish.orchestrator.ml_runner.run_inference_main")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_run_inference_loop_bails_when_no_drop_can_be_claimed(
    mock_s3_class, mock_db_class, mock_exists, mock_inference
):
    """Every target already claimed elsewhere → no inference, empty result."""
    mock_db = mock_db_class.return_value
    mock_exists.return_value = True
    mock_db.advance_status.side_effect = InvalidTransitionError("already running")

    targets = _make_records(2)
    for t in targets:
        t["VideoURL"] = f"/media/{t['drop_id']}.mp4"

    runner = MLRunner()

    assert runner.run_inference_loop(targets) == []
    assert runner.claimed_drop_ids == []
    mock_inference.assert_not_called()


@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_finalize_batch_results_only_touches_claimed_drops(
    mock_s3_class, mock_db_class
):
    """The ml_error safety net must never reach a drop another job owns."""
    mock_db = mock_db_class.return_value
    mock_db.get_deployment.return_value = {"ml_status": MlStatus.RUNNING}

    runner = MLRunner()
    runner.claimed_drop_ids = ["KSF_20240124_BUV_KSF_085_01"]
    runner.finalize_batch_results(successful_drops=[])

    errored = {
        call.args[0]
        for call in mock_db.update_section_status.call_args_list
        if call.args[1:] == (MlStatus.COLUMN, MlStatus.ERROR)
    }
    assert errored == {"KSF_20240124_BUV_KSF_085_01"}


@patch("spyfish.orchestrator.ml_runner.process_one_drop")
@patch("spyfish.orchestrator.ml_runner.extract_slate_frame")
@patch("spyfish.orchestrator.ml_runner.run_inference_main")
@patch("spyfish.orchestrator.ml_runner.AnnotationDatabaseManager")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_run_inference_loop_extracts_slate_before_ml_and_survives_its_failure(
    mock_s3_class,
    mock_db_class,
    mock_exists,
    mock_ann_db_class,
    mock_inference,
    mock_slate,
    mock_process_one_drop,
):
    """The slate JPEG is taken from the local video before inference; a slate
    failure is logged and the drop still runs ML and completes."""
    from pathlib import Path

    mock_db = mock_db_class.return_value
    mock_exists.return_value = True
    mock_db.get_deployment.return_value = {"ml_status": MlStatus.COMPLETE}

    targets = _make_records(2)
    for t in targets:
        t["VideoURL"] = f"/media/{t['drop_id']}.mp4"
    failing = targets[0]["drop_id"]

    order = []
    mock_slate.side_effect = lambda drop_id, path, max_seconds=None: (
        order.append(("slate", drop_id)),
        (
            (_ for _ in ()).throw(RuntimeError("cannot open"))
            if drop_id == failing
            else None
        ),
    )
    mock_inference.side_effect = lambda args: order.append(("ml", args["drop_id"]))

    successes = MLRunner().run_inference_loop(targets)

    assert successes == [t["drop_id"] for t in targets]
    assert order == [
        ("slate", targets[0]["drop_id"]),
        ("ml", targets[0]["drop_id"]),
        ("slate", targets[1]["drop_id"]),
        ("ml", targets[1]["drop_id"]),
    ]
    first = mock_slate.call_args_list[0]
    assert first.args == (failing, Path(f"/media/{failing}.mp4"))
    # the scan is capped at sampling_start: the rig is on the bottom by then
    assert first.kwargs == {"max_seconds": targets[0]["sampling_start"]}
