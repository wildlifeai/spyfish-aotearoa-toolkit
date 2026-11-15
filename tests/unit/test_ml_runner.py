import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from spyfish.config.base import InvalidTransitionError, MlStatus
from spyfish.config.wrapper import config
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


@pytest.fixture
def download_runner(tmp_path, monkeypatch):
    """An MLRunner whose DB and S3 are mocks and whose media dir is empty.

    Returns ``make(records, limit)`` -> ``(runner, s3)``. Downloads succeed
    (and write the file) unless the test overrides ``s3.download_object_from_s3``.
    """
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: tmp_path))

    def make(records, limit=None):
        with (
            patch("spyfish.orchestrator.ml_runner.DatabaseManager") as db_class,
            patch("spyfish.orchestrator.ml_runner.S3Handler") as s3_class,
        ):
            runner = MLRunner()
        db, s3 = db_class.return_value, s3_class.return_value
        db.get_deployments_eligible.return_value = records
        db.get_deployments_by_ids.return_value = {r["drop_id"]: r for r in records}
        s3.get_object_storage.return_value = {"storage_class": "STANDARD"}
        s3.download_object_from_s3.side_effect = lambda key, filename: (
            Path(filename).write_bytes(b"x") or True
        )
        if limit is not None:
            runner.limit = limit
        return runner, s3

    return make


def test_get_inference_targets(download_runner, tmp_path):
    runner, _ = download_runner(_make_records(1))
    targets = runner.get_inference_targets()

    assert len(targets) == 1
    assert targets[0]["drop_id"] == "KSF_20240124_BUV_KSF_085_01"
    assert targets[0]["VideoURL"] == str(tmp_path / "1.mp4")


def test_get_inference_targets_caps_at_limit(download_runner):
    """With 5 eligible drops and limit=3, only 3 downloads should be attempted."""
    runner, s3 = download_runner(_make_records(5), limit=3)
    targets = runner.get_inference_targets()

    # First 3 by priority order
    assert [t["drop_id"] for t in targets] == [
        "KSF_20240124_BUV_KSF_085_01",
        "KSF_20240124_BUV_KSF_085_02",
        "KSF_20240124_BUV_KSF_085_03",
    ]
    assert s3.download_object_from_s3.call_count == 3


def test_get_inference_targets_never_backfills_past_the_limit(download_runner):
    """A failed drop shrinks the batch; it is never replaced from further
    down the queue.

    This reverses an earlier choice (backfill until `limit` succeed). That
    walk had no bound: on 2026-10-14 a 9-drop targeted run tried to download
    every one of 800+ ml_ready drops because three targets had no video, and
    only a mis-set bucket (every download 404'd) kept it from pulling the
    whole backlog at egress rates. The batch is the top `limit` by priority;
    what fails in it is logged, not papered over."""
    runner, s3 = download_runner(_make_records(5), limit=3)
    # First 2 downloads fail, the rest would succeed if asked
    s3.download_object_from_s3.side_effect = lambda key, filename: (
        False
        if key in ("path/1.mp4", "path/2.mp4")
        else Path(filename).write_bytes(b"x") or True
    )
    targets = runner.get_inference_targets()

    # Only drop 03 survives; 04 and 05 are outside the top 3 and never touched
    assert [t["drop_id"] for t in targets] == ["KSF_20240124_BUV_KSF_085_03"]
    assert s3.download_object_from_s3.call_count == 3


def test_get_inference_targets_all_downloads_fail(download_runner):
    runner, s3 = download_runner(_make_records(3), limit=3)
    s3.download_object_from_s3.side_effect = None
    s3.download_object_from_s3.return_value = False

    assert runner.get_inference_targets() == []


def test_get_inference_targets_survives_a_raising_download(download_runner):
    """A transfer that raises costs that drop, not the batch."""
    runner, s3 = download_runner(_make_records(2), limit=2)

    def flaky(key, filename):
        if key == "path/1.mp4":
            raise RuntimeError("connection reset")
        return Path(filename).write_bytes(b"x") or True

    s3.download_object_from_s3.side_effect = flaky
    targets = runner.get_inference_targets()

    assert [t["drop_id"] for t in targets] == ["KSF_20240124_BUV_KSF_085_02"]


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


@patch("spyfish.orchestrator.ml_runner.process_one_drop")
@patch("spyfish.orchestrator.ml_runner.extract_slate_frame")
@patch("spyfish.orchestrator.ml_runner.run_inference_main")
@patch("spyfish.orchestrator.ml_runner.AnnotationDatabaseManager")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_inference_only_stops_at_ml_inferred(
    mock_s3_class,
    mock_db_class,
    mock_exists,
    mock_ann_db_class,
    mock_inference,
    mock_slate,
    mock_process_one_drop,
):
    """--ml-infer (GPU job): raw CSV written, drop parked at ml_inferred, no
    post-processing and no annotations DB touched. --ml-post finishes it."""
    mock_db = mock_db_class.return_value
    mock_exists.return_value = True

    targets = _make_records(2)
    for t in targets:
        t["VideoURL"] = f"/media/{t['drop_id']}.mp4"

    successes = MLRunner().run_inference_loop(targets, post_process=False)

    assert successes == [t["drop_id"] for t in targets]
    transitions = [
        (c.args[0], c.args[2]) for c in mock_db.advance_status.call_args_list
    ]
    for t in targets:
        assert transitions.count((t["drop_id"], MlStatus.RUNNING)) == 1
        assert transitions.count((t["drop_id"], MlStatus.INFERRED)) == 1
        assert (t["drop_id"], MlStatus.COMPLETE) not in transitions
    mock_process_one_drop.assert_not_called()
    mock_db.sync_annotation_counts.assert_not_called()
    mock_ann_db_class.assert_not_called()


@patch("spyfish.orchestrator.ml_runner.process_one_drop")
@patch("spyfish.orchestrator.ml_runner.extract_slate_frame")
@patch("spyfish.orchestrator.ml_runner.run_inference_main")
@patch("spyfish.orchestrator.ml_runner.AnnotationDatabaseManager")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_full_ml_passes_through_inferred_to_complete(
    mock_s3_class,
    mock_db_class,
    mock_exists,
    mock_ann_db_class,
    mock_inference,
    mock_slate,
    mock_process_one_drop,
):
    """A plain --ml still ends at ml_complete, via the ml_inferred checkpoint,
    with post-processing run in the same loop."""
    mock_db = mock_db_class.return_value
    mock_exists.return_value = True
    # sync_annotation_counts did not advance this drop (zero detections)
    mock_db.get_deployment.return_value = {"ml_status": MlStatus.INFERRED}

    targets = _make_records(1)
    targets[0]["VideoURL"] = f"/media/{targets[0]['drop_id']}.mp4"
    drop = targets[0]["drop_id"]

    assert MLRunner().run_inference_loop(targets) == [drop]

    transitions = [c.args[2] for c in mock_db.advance_status.call_args_list]
    assert transitions == [MlStatus.RUNNING, MlStatus.INFERRED, MlStatus.COMPLETE]
    mock_process_one_drop.assert_called_once()
    mock_db.sync_annotation_counts.assert_called_once_with([drop])


@patch("spyfish.orchestrator.ml_runner.process_one_drop")
@patch("spyfish.orchestrator.ml_runner.AnnotationDatabaseManager")
@patch("spyfish.orchestrator.ml_runner.os.path.exists")
@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_run_post_processing_finishes_inferred_drops_of_one_survey(
    mock_s3_class, mock_db_class, mock_exists, mock_ann_db_class, mock_process_one_drop
):
    """--ml-post (CPU job): every ml_inferred drop of the survey is
    post-processed and advanced to ml_complete; the eligibility query is
    scoped to the survey,
    and one failing drop stays at ml_inferred (its raw CSV is on disk, the
    next --ml-post retries it) with the error recorded, without stopping the
    rest. ml_error would force a GPU re-inference for a CPU-side failure."""
    mock_db = mock_db_class.return_value
    mock_exists.return_value = True
    survey = "KSF_20240124_BUV"
    mine = [f"{survey}_KSF_085_01", f"{survey}_KSF_085_02"]
    mock_db.get_deployments_eligible.return_value = [{"drop_id": d} for d in mine]
    # Every drop is still at ml_inferred when the runner checks it.
    mock_db.get_deployment.return_value = {"ml_status": MlStatus.INFERRED}

    failing = mine[1]

    def post(drop_id, **kwargs):
        if drop_id == failing:
            raise ValueError("degenerate inference")

    mock_process_one_drop.side_effect = post

    done = MLRunner().run_post_processing(survey_id=survey)

    assert done == [mine[0]]
    mock_db.get_deployments_eligible.assert_called_once_with(
        MlStatus.COLUMN, [MlStatus.INFERRED], survey_id=survey
    )
    processed = [c.kwargs["drop_id"] for c in mock_process_one_drop.call_args_list]
    assert processed == mine
    transitions = [
        (c.args[0], c.args[2]) for c in mock_db.advance_status.call_args_list
    ]
    assert (mine[0], MlStatus.COMPLETE) in transitions
    assert (failing, MlStatus.ERROR) not in transitions  # stays ml_inferred
    assert mock_db.add_validation_error.call_args.kwargs["column_name"] == "ml_post"
    assert mock_db.add_validation_error.call_args.kwargs["drop_id"] == failing


@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_post_processing_uses_the_model_that_wrote_the_raw_csv(
    mock_s3_class, mock_db_class, tmp_path, monkeypatch
):
    """--ml-post finishes the raw CSV on disk, not the configured model: a
    --retrain can promote a newer model between the GPU job and the CPU job.
    Of several raw CSVs the one written last wins, whatever its training
    stamp (a rollback runs an older model after a newer one); the zooniverse
    and biigle raw CSVs share the suffix and are never mistaken for a model."""
    monkeypatch.setattr(config, "_project_root", tmp_path)
    config.pipeline_model_dir.mkdir(parents=True)
    # A real checkpoint, not an empty file: the runner now reads train_args.imgsz
    # from the weights at init (spyfish/ml/model_meta.py).
    torch.save(
        {"train_args": {"imgsz": 640}},
        config.pipeline_model_dir / "species_20260701.pt",
    )
    drop = "KSF_20240124_BUV_KSF_085_01"
    ann_dir = config.get_drop_annotations_dir(drop)
    ann_dir.mkdir(parents=True)
    runner = MLRunner()
    assert Path(runner.model).stem == "species_20260701"

    assert runner._inferred_model_name(drop) == "species_20260701"  # nothing on disk

    # Written oldest first, one hour apart: the newer-stamped species_20260801
    # ran earlier, species_20260601 is the run that just finished.
    names = ("species_20260801", "species_20260601", "zooniverse", "biigle_expert")
    for i, name in enumerate(names):
        path = ann_dir / f"{drop}_{name}_raw.csv"
        path.write_text("x\n")
        os.utime(path, (1_700_000_000 + 3600 * i,) * 2)
    assert runner._inferred_model_name(drop) == "species_20260601"


def test_archived_drop_with_local_video_is_used(download_runner, tmp_path):
    """`archived` is an S3 fact. A video already on disk (fetched by hand after
    a restore, or before the archive) must be processed without anyone editing
    video_presence first; an unrestored archive only blocks a DOWNLOAD."""
    records = _make_records(2)
    records[0]["video_presence"] = "archived"  # on disk
    records[1]["video_presence"] = "archived"  # not on disk
    (tmp_path / "1.mp4").write_bytes(b"x")
    runner, s3 = download_runner(records, limit=2)
    s3.get_object_storage.return_value = {"storage_class": "DEEP_ARCHIVE"}
    s3.is_restored.return_value = False

    targets = runner.get_inference_targets()

    assert [t["drop_id"] for t in targets] == ["KSF_20240124_BUV_KSF_085_01"]
    s3.download_object_from_s3.assert_not_called()


@patch("spyfish.orchestrator.ml_runner.DatabaseManager")
@patch("spyfish.orchestrator.ml_runner.S3Handler")
def test_degenerate_inference_goes_to_ml_error_even_from_inferred(
    mock_s3_class, mock_db_class
):
    """Most --ml-post failures stay at ml_inferred for a retry, but a saturated
    raw CSV fails identically every time: it needs a person, so ml_error."""
    from spyfish.ml.process_ml_annotations import DegenerateInferenceError

    mock_db = mock_db_class.return_value
    mock_db.get_deployment.return_value = {"ml_status": MlStatus.INFERRED}
    drop = "KSF_20240124_BUV_KSF_085_01"
    runner = MLRunner()

    runner._mark_ml_error(drop, ValueError("disk"), column_name="ml_post")
    mock_db.advance_status.assert_not_called()

    runner._mark_ml_error(drop, DegenerateInferenceError("900/frame"), "ml_post")
    mock_db.advance_status.assert_called_once_with(
        drop, MlStatus.COLUMN, MlStatus.ERROR
    )
