"""The flags that keep CPU-only work off GPU jobs (2026-11-11):

- ``preannotate_frames.py --upload-only``: upload what an earlier run wrote,
  with no detector and no classifier loaded.
- ``run_pipeline.py --sync-only``: the end-of-run S3 sync and nothing else.
- ``scripts/slurm/submit_retrain.sh``: prep (CPU), train ``--no-upload`` (GPU),
  ``--sync-only`` (CPU), each after the one before.

No network, no model, no SLURM: S3, BIIGLE and sbatch are all stand-ins.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

import run_pipeline
from uoa.scripts.two_stage import preannotate_frames as script

REPO = Path(__file__).resolve().parents[2]
FRAME_NAMES = [
    "ESK_B_20220119_PM_M_v01__t000010.00.jpg",
    "ESK_B_20220119_PM_M_v02__t000020.00.jpg",
]
UPLOAD_ARGS = [
    "--volume-name",
    "UoA 20220119 hand-count",
    "--s3-prefix",
    "uoa/process_files/biigle_uploads/20220119_hand_count",
    "--disk-id",
    "98",
    "--project-id",
    "4510",
]


def _coco(names=FRAME_NAMES, boxes=True) -> dict:
    return {
        "images": [
            {"id": i, "file_name": n, "width": 8, "height": 8}
            for i, n in enumerate(names, start=1)
        ],
        "annotations": (
            [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}]
            if boxes
            else []
        ),
        "categories": [{"id": 1, "name": "Chrysophrys auratus"}],
    }


@pytest.fixture
def run_dir(tmp_path):
    """A frames dir and, next to it, the COCO a finished GPU run leaves."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for name in FRAME_NAMES:
        Image.new("RGB", (8, 8)).save(frames_dir / name)
    (tmp_path / script.COCO_NAME).write_text(json.dumps(_coco()))
    return tmp_path


def _run(monkeypatch, *argv) -> None:
    monkeypatch.setattr(sys, "argv", ["preannotate_frames", *map(str, argv)])
    script.main()


def _no_models(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("--upload-only must not run the detector / classifier")

    monkeypatch.setattr(script, "run_preannotation", boom)


# ── preannotate_frames --upload-only ─────────────────────────────────────────


def test_upload_only_uploads_the_outputs_on_disk_without_a_model(run_dir, monkeypatch):
    _no_models(monkeypatch)
    upload = MagicMock()
    monkeypatch.setattr(script, "upload_preannotations", upload)

    _run(monkeypatch, "--frames-dir", run_dir / "frames", "--upload-only", *UPLOAD_ARGS)

    upload.assert_called_once()
    frames, coco, args = upload.call_args.args
    assert [p.name for p in frames] == FRAME_NAMES
    assert coco == _coco()
    assert (args.volume_name, args.disk_id, args.project_id) == (
        "UoA 20220119 hand-count",
        98,
        4510,
    )


def test_upload_and_upload_only_make_the_same_upload_call(run_dir, monkeypatch):
    """--upload hands the upload what the run just produced; --upload-only hands
    it the same thing read back from disk."""
    calls = []
    monkeypatch.setattr(
        script,
        "upload_preannotations",
        lambda frames, coco, args: calls.append(([p.name for p in frames], coco)),
    )
    monkeypatch.setattr(script, "run_preannotation", lambda args, frames, out: _coco())
    _run(monkeypatch, "--frames-dir", run_dir / "frames", "--upload", *UPLOAD_ARGS)
    _no_models(monkeypatch)
    _run(monkeypatch, "--frames-dir", run_dir / "frames", "--upload-only", *UPLOAD_ARGS)

    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_plain_run_does_not_upload(run_dir, monkeypatch):
    upload = MagicMock()
    monkeypatch.setattr(script, "upload_preannotations", upload)
    monkeypatch.setattr(script, "run_preannotation", lambda args, frames, out: _coco())
    _run(monkeypatch, "--frames-dir", run_dir / "frames")
    upload.assert_not_called()


def test_upload_only_reaches_s3_and_biigle_like_upload(run_dir, monkeypatch):
    """The shared upload itself: every frame to S3, one volume, one COCO call."""
    from spyfish.biigle import biigle_handler, upload_frames
    from spyfish.storage import s3_handler

    s3 = MagicMock()
    s3.upload_file_to_s3.return_value = True
    handler = MagicMock()
    handler.build_s3_url.return_value = "disk-98://prefix/"
    handler.create_volume_from_s3_files.return_value = {"id": 31000}
    coco_upload = MagicMock()
    s3_cls = MagicMock(return_value=s3)
    monkeypatch.setattr(s3_handler, "S3Handler", s3_cls)
    monkeypatch.setattr(biigle_handler, "BiigleHandler", lambda: handler)
    monkeypatch.setattr(upload_frames, "upload_coco_annotations_to_biigle", coco_upload)
    _no_models(monkeypatch)

    _run(
        monkeypatch,
        "--frames-dir",
        run_dir / "frames",
        "--upload-only",
        "--s3-bucket",
        "marine-buv-kalindi",
        *UPLOAD_ARGS,
    )

    s3_cls.assert_called_once_with(bucket="marine-buv-kalindi")
    prefix = "uoa/process_files/biigle_uploads/20220119_hand_count/"
    assert [c.kwargs["key"] for c in s3.upload_file_to_s3.call_args_list] == [
        prefix + n for n in FRAME_NAMES
    ]
    handler.build_s3_url.assert_called_once_with(prefix, disk_id=98)
    handler.create_volume_from_s3_files.assert_called_once_with(
        volume_name="UoA 20220119 hand-count",
        s3_url="disk-98://prefix/",
        files=FRAME_NAMES,
        project_id=4510,
        media_type="image",
    )
    coco_upload.assert_called_once_with(volume_id=31000, coco_data=_coco())


def test_failed_frame_upload_creates_no_volume(run_dir, monkeypatch):
    from spyfish.biigle import biigle_handler
    from spyfish.storage import s3_handler

    s3 = MagicMock()
    s3.upload_file_to_s3.return_value = False
    handler_cls = MagicMock()
    monkeypatch.setattr(s3_handler, "S3Handler", lambda bucket: s3)
    monkeypatch.setattr(biigle_handler, "BiigleHandler", handler_cls)
    _no_models(monkeypatch)

    with pytest.raises(RuntimeError, match="S3 upload failed"):
        _run(
            monkeypatch,
            "--frames-dir",
            run_dir / "frames",
            "--upload-only",
            *UPLOAD_ARGS,
        )
    handler_cls.assert_not_called()


def test_upload_only_without_a_coco_refuses(run_dir, monkeypatch):
    (run_dir / script.COCO_NAME).rename(run_dir / "moved_away.json")
    upload = MagicMock()
    monkeypatch.setattr(script, "upload_preannotations", upload)
    _no_models(monkeypatch)

    with pytest.raises(SystemExit, match="coco_annotations.json not found"):
        _run(
            monkeypatch,
            "--frames-dir",
            run_dir / "frames",
            "--upload-only",
            *UPLOAD_ARGS,
        )
    upload.assert_not_called()


@pytest.mark.parametrize("change", ["frame_gone", "frame_added", "no_frames_dir"])
def test_upload_only_refuses_when_frames_and_coco_disagree(
    run_dir, monkeypatch, change
):
    frames_dir = run_dir / "frames"
    if change == "frame_gone":
        (frames_dir / FRAME_NAMES[0]).rename(run_dir / "elsewhere.jpg")
        expected = "do not match"
    elif change == "frame_added":
        Image.new("RGB", (8, 8)).save(frames_dir / "ESK_B_20220119_PM_M_v03__t1.jpg")
        expected = "do not match"
    else:
        frames_dir.rename(run_dir / "frames_moved")
        expected = "frames dir not found"
    upload = MagicMock()
    monkeypatch.setattr(script, "upload_preannotations", upload)
    _no_models(monkeypatch)

    with pytest.raises(SystemExit, match=expected):
        _run(monkeypatch, "--frames-dir", frames_dir, "--upload-only", *UPLOAD_ARGS)
    upload.assert_not_called()


def test_upload_only_reads_the_out_dir_the_run_used(run_dir, monkeypatch):
    out_dir = run_dir / "elsewhere"
    out_dir.mkdir()
    (run_dir / script.COCO_NAME).rename(out_dir / script.COCO_NAME)
    upload = MagicMock()
    monkeypatch.setattr(script, "upload_preannotations", upload)
    _no_models(monkeypatch)

    _run(
        monkeypatch,
        "--frames-dir",
        run_dir / "frames",
        "--out-dir",
        out_dir,
        "--upload-only",
        *UPLOAD_ARGS,
    )
    upload.assert_called_once()


@pytest.mark.parametrize(
    "argv",
    [
        ["--upload-only"],  # no volume name / prefix
        ["--upload-only", "--upload", *UPLOAD_ARGS],
    ],
)
def test_upload_only_argument_errors(run_dir, monkeypatch, argv, capsys):
    upload = MagicMock()
    monkeypatch.setattr(script, "upload_preannotations", upload)
    _no_models(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "--frames-dir", run_dir / "frames", *argv)
    assert exc.value.code == 2
    capsys.readouterr()
    upload.assert_not_called()


def test_upload_only_imports_no_torch_and_no_ultralytics(run_dir):
    """A fresh interpreter, as on the CPU job: the whole path, upload stubbed at
    the S3 / BIIGLE classes, must not pull in the model libraries."""
    driver = r"""
import sys
from unittest.mock import MagicMock

from spyfish.biigle import biigle_handler, upload_frames
from spyfish.storage import s3_handler

s3_handler.S3Handler = MagicMock()
s3_handler.S3Handler.return_value.upload_file_to_s3.return_value = True
biigle_handler.BiigleHandler = MagicMock()
biigle_handler.BiigleHandler.return_value.create_volume_from_s3_files.return_value = {"id": 1}
upload_frames.upload_coco_annotations_to_biigle = MagicMock()

from uoa.scripts.two_stage import preannotate_frames as script

sys.argv = ["preannotate_frames", "--frames-dir", sys.argv[1], "--upload-only",
            "--volume-name", "v", "--s3-prefix", "p", "--disk-id", "98",
            "--project-id", "4510"]
script.main()
assert upload_frames.upload_coco_annotations_to_biigle.call_count == 1
loaded = [m for m in ("torch", "ultralytics", "open_clip") if m in sys.modules]
print("LOADED", loaded)
"""
    env = {k: v for k, v in os.environ.items() if k != "TWO_STAGE_SOURCE"}
    run = subprocess.run(
        [sys.executable, "-c", driver, str(run_dir / "frames")],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    assert "LOADED []" in run.stdout


def test_upload_after_a_real_run_matches_upload_only(run_dir):
    """The GPU path end to end with a stub detector (child interpreter: the
    two-stage modules bind TWO_STAGE_SOURCE at import): what --upload sends is
    what --upload-only then reads back from the files that run wrote."""
    driver = r"""
import json, sys
from unittest.mock import MagicMock

import torch
import ultralytics


class Box:
    xyxy = torch.tensor([[1.0, 1.0, 5.0, 6.0]])
    cls = torch.tensor([0])
    conf = torch.tensor([0.9])


class Result:
    boxes = [Box()]


class FakeYOLO:
    names = {0: "fish"}
    built = 0

    def __init__(self, path):
        FakeYOLO.built += 1

    def predict(self, source, **kwargs):
        return [Result() for _ in source]


ultralytics.YOLO = FakeYOLO

from uoa.scripts.two_stage import preannotate_frames as script

sent = []
script.upload_preannotations = lambda frames, coco, args: sent.append(
    ([p.name for p in frames], coco)
)
common = ["--frames-dir", sys.argv[1], "--volume-name", "v", "--s3-prefix", "p"]
sys.argv = ["preannotate_frames", *common, "--classifier", "none", "--upload"]
script.main()
built_by_upload = FakeYOLO.built
sys.argv = ["preannotate_frames", *common, "--upload-only"]
script.main()
print("RESULT " + json.dumps({
    "built_by_upload": built_by_upload,
    "built_by_upload_only": FakeYOLO.built - built_by_upload,
    "same": sent[0] == sent[1],
    "boxes": len(sent[1][1]["annotations"]),
    "categories": [c["name"] for c in sent[1][1]["categories"]],
}))
"""
    (run_dir / script.COCO_NAME).rename(run_dir / "stale.json")
    run = subprocess.run(
        [sys.executable, "-c", driver, str(run_dir / "frames")],
        cwd=REPO,
        env={**os.environ, "TWO_STAGE_SOURCE": "uoa"},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(ln for ln in run.stdout.splitlines() if ln.startswith("RESULT "))
    assert json.loads(line[len("RESULT ") :]) == {
        "built_by_upload": 1,
        "built_by_upload_only": 0,
        "same": True,
        "boxes": 2,
        "categories": ["fish"],
    }


# ── run_pipeline.py --sync-only ──────────────────────────────────────────────


@pytest.fixture
def pipeline(monkeypatch):
    """run_pipeline.main with the DB, the stage runner and the sync replaced."""
    runner_run = MagicMock()
    sync = MagicMock(return_value=True)
    monkeypatch.setattr(run_pipeline, "DatabaseManager", MagicMock())
    monkeypatch.setattr(run_pipeline.StageRunner, "run", runner_run)
    monkeypatch.setattr(run_pipeline, "sync_pipeline_results", sync)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_pipeline.py", *argv])
        run_pipeline.main()

    run.runner_run = runner_run
    run.sync = sync
    return run


def test_sync_only_syncs_once_and_runs_no_stage(pipeline):
    pipeline("--sync-only")
    pipeline.sync.assert_called_once_with()
    pipeline.runner_run.assert_not_called()


def test_sync_only_exits_non_zero_when_the_sync_fails(pipeline):
    pipeline.sync.return_value = False
    with pytest.raises(SystemExit) as exc:
        pipeline("--sync-only")
    assert exc.value.code == 1
    pipeline.sync.assert_called_once_with()
    pipeline.runner_run.assert_not_called()


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--no-upload"], "cannot be combined with --no-upload"),
        (["--dry-run"], "cannot be combined with --no-upload"),
        (["--retrain", "--species"], "--sync-only runs no stage: drop --retrain"),
        (["--ml-post"], "--sync-only runs no stage: drop --ml-post"),
    ],
)
def test_sync_only_refuses_what_contradicts_it(pipeline, capsys, extra, message):
    with pytest.raises(SystemExit) as exc:
        pipeline("--sync-only", *extra)
    assert exc.value.code == 2
    # argparse wraps its error text, so compare with whitespace collapsed
    assert message in " ".join(capsys.readouterr().err.split())
    pipeline.sync.assert_not_called()
    pipeline.runner_run.assert_not_called()


def test_no_upload_still_skips_the_sync_and_a_plain_run_still_ends_with_it(pipeline):
    pipeline("--retrain", "--species", "--no-upload")
    pipeline.sync.assert_not_called()
    assert pipeline.runner_run.call_count == 1

    pipeline("--retrain", "--species")
    pipeline.sync.assert_called_once_with()
    assert pipeline.runner_run.call_count == 2


# ── scripts/slurm/submit_retrain.sh ──────────────────────────────────────────


def _submit_retrain(tmp_path, *argv, **env):
    """Run the submit script with an `sbatch` on PATH that only records its
    arguments and hands back a job id, so nothing can reach a scheduler."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "sbatch_calls.txt"
    fake = bin_dir / "sbatch"
    fake.write_text(
        "#!/bin/bash\n"
        f'echo "$*" >> "{log}"\n'
        f'echo $((9000 + $(wc -l < "{log}")))\n'
    )
    fake.chmod(0o755)
    run = subprocess.run(
        ["bash", str(REPO / "scripts" / "slurm" / "submit_retrain.sh"), *argv],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", **env},
        capture_output=True,
        text=True,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return run, calls


def _tail(line: str) -> str:
    """`<job file basename> <flags>` of an sbatch line."""
    words = line.split()
    i = next(n for n, w in enumerate(words) if w.endswith(".sl"))
    return " ".join([Path(words[i]).name, *words[i + 1 :]])


def test_submit_retrain_dry_run_prints_three_chained_lines(tmp_path):
    run, calls = _submit_retrain(tmp_path, "--dry-run")
    assert run.returncode == 0, run.stderr
    assert calls == []  # nothing submitted
    lines = [ln for ln in run.stdout.splitlines() if ln.startswith("sbatch ")]
    assert [_tail(ln) for ln in lines] == [
        "cpu.sl --retrain --data-prep",
        "train_job.sl --retrain --species --no-upload",
        "cpu.sl --sync-only",
    ]
    assert "--dependency" not in lines[0]
    assert "--dependency=afterok:<job1>" in lines[1]
    assert "--dependency=afterok:<job2>" in lines[2]
    assert "nothing submitted" in run.stdout


def test_submit_retrain_chains_each_job_on_the_one_before(tmp_path):
    run, calls = _submit_retrain(tmp_path, SYNC_TIME="00:30:00")
    assert run.returncode == 0, run.stderr
    assert [_tail(c) for c in calls] == [
        "cpu.sl --retrain --data-prep",
        "train_job.sl --retrain --species --no-upload",
        "cpu.sl --sync-only",
    ]
    assert "--dependency" not in calls[0]
    assert "--dependency=afterok:9001" in calls[1]
    assert "--dependency=afterok:9002" in calls[2]
    assert "--time=00:30:00" in calls[2]
    # Strictly sequential: one job has no dependency, the others exactly one.
    assert sum("--dependency" in c for c in calls) == 2


def test_submit_retrain_rejects_other_arguments(tmp_path):
    run, calls = _submit_retrain(tmp_path, "--species")
    assert run.returncode == 2
    assert calls == []


@pytest.mark.parametrize(
    "path",
    [
        "scripts/slurm/submit_retrain.sh",
        "uoa/scripts/two_stage/uoa_preannotate.sl",
        "uoa/scripts/two_stage/uoa_preannotate_cpu.sl",
    ],
)
def test_shell_files_parse(path):
    run = subprocess.run(["bash", "-n", str(REPO / path)], capture_output=True)
    assert run.returncode == 0, run.stderr


def test_cpu_upload_job_is_a_uoa_cpu_job_running_upload_only():
    text = (REPO / "uoa/scripts/two_stage/uoa_preannotate_cpu.sl").read_text()
    code = [ln for ln in text.splitlines() if not ln.startswith("# ")]
    assert "#SBATCH --account=uoa04631" in code
    assert not any("gpu" in ln.lower() for ln in code)
    assert "source scripts/slurm/_env.sh" in code
    assert any("--upload-only" in ln for ln in code)
    assert any(ln.startswith("spyfish_stamp_start") for ln in code)
    assert "spyfish_end" in code
