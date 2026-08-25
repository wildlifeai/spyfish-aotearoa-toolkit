"""Tests for freeze_dataset_snapshot, which now runs BEFORE training."""
import yaml

from spyfish.ml.training.train import freeze_dataset_snapshot


def _workspace(tmp_path):
    src = tmp_path / "species"
    (src / "labels" / "train").mkdir(parents=True)
    (src / "labels" / "val").mkdir(parents=True)
    (src / "images" / "train").mkdir(parents=True)
    (src / "images" / "val").mkdir(parents=True)
    (src / "data.yaml").write_text(
        yaml.safe_dump({"nc": 2, "names": ["fish", "Pagrus auratus"]}, sort_keys=False)
    )
    (src / "class_map.json").write_text('{"0": {"scientific_name": "fish"}}')
    stem = "KSF_20240124_BUV_KSF_085_01__frame_10.000s"
    (src / "labels" / "train" / f"{stem}.txt").write_text("0 .5 .5 .1 .1\n")
    (src / "images" / "train" / f"{stem}.jpg").write_text("x")
    (src / "labels" / "val" / f"{stem}v.txt").write_text("1 .5 .5 .1 .1\n")
    (src / "images" / "val" / f"{stem}v.jpg").write_text("x")
    return src


def test_snapshot_written_into_a_run_dir_that_has_no_weights_yet(tmp_path):
    """The snapshot must not depend on best.pt: it is written before training,
    so a run killed mid-epoch still records what it was training on."""
    src = _workspace(tmp_path)
    run_dir = tmp_path / "runs" / "20260823_031610_species"

    snap = freeze_dataset_snapshot(src / "data.yaml", run_dir)

    assert snap == run_dir / "dataset"
    assert yaml.safe_load((snap / "data.yaml").read_text())["names"] == [
        "fish", "Pagrus auratus"
    assert (snap / "class_map.json").exists()
    assert (snap / "labels" / "train").glob("*.txt")
    assert not (run_dir / "weights").exists()


def test_snapshot_captures_the_dataset_at_call_time(tmp_path):
    """A later rebuild of the shared workspace must not alter a frozen snapshot.

    training/species/ is shared and --data-prep deletes it; a snapshot taken
    after training would silently capture the replacement.
    """
    src = _workspace(tmp_path)
    run_dir = tmp_path / "runs" / "20260823_031610_species"
    snap = freeze_dataset_snapshot(src / "data.yaml", run_dir)

    # Simulate a concurrent data prep replacing the workspace mid-training.
    (src / "data.yaml").write_text(
        yaml.safe_dump({"nc": 1, "names": ["fish"]}, sort_keys=False)
    )
    for txt in (src / "labels" / "train").glob("*.txt"):
        txt.unlink()

    assert yaml.safe_load((snap / "data.yaml").read_text())["names"] == [
        "fish",
        "Pagrus auratus",
    ]
    assert list((snap / "labels" / "train").glob("*.txt"))


def test_missing_data_yaml_returns_none_rather_than_failing_the_run(tmp_path):
    assert freeze_dataset_snapshot(tmp_path / "nope.yaml", tmp_path / "run") is None
