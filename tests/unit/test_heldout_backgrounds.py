"""Val and test keep every reviewed empty frame; only train is thinned.

Background frames used to be admitted to the train split alone, subsampled to
`background_ratio`. Val and test therefore held only frames with an animal in
them, so no val/test frame could ever produce a false positive, and a fully
empty drop chosen for test contributed nothing at all (found 2026-10-15 while
building the frozen test set).
"""

from pathlib import Path

from spyfish.ml.training.prepare_training_data import assemble_yolo_dataset

SURVEY = "SLI_20260114_BUV"
TRAIN, VAL, TEST, EMPTY = (
    f"{SURVEY}_SLI_001_01",
    f"{SURVEY}_SLI_003_01",
    f"{SURVEY}_SLI_044_01",
    f"{SURVEY}_SLI_065_01",
)


def _drop(root: Path, staged: Path, drop: str, positives: int, empties: int):
    for i in range(positives + empties):
        stem = f"{drop}__frame_{i}.000s"
        img = root / SURVEY / drop / "training_frames" / f"{stem}.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_text("x")
        lbl = staged / drop / f"{stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("1 .5 .5 .1 .1\n" if i < positives else "")


def _split_labels(out: Path, split: str):
    d = out / "species" / "labels" / split
    files = list(d.glob("*.txt")) if d.is_dir() else []
    return (
        sum(1 for f in files if f.read_text().strip()),
        sum(1 for f in files if not f.read_text().strip()),
    )


def test_val_and_test_keep_all_empty_frames(tmp_path):
    images, staged, out = tmp_path / "data", tmp_path / "staged", tmp_path / "out"
    _drop(images, staged, TRAIN, positives=2, empties=8)
    _drop(images, staged, VAL, positives=2, empties=4)
    _drop(images, staged, TEST, positives=2, empties=3)
    _drop(images, staged, EMPTY, positives=0, empties=10)

    assemble_yolo_dataset(
        train_drops=[TRAIN],
        val_drops=[VAL],
        test_drops=[TEST, EMPTY],
        images_dir=images,
        species_labels_dir=staged,
        output_dir=out,
        class_names=["fish", "Parapercis colias"],
        symlink=False,
    )

    assert _split_labels(out, "val") == (2, 4)
    assert _split_labels(out, "test") == (2, 13), "fully empty test drop dropped out"
    # Train is still thinned to background_ratio: never all 8 of its empties.
    _, train_empties = _split_labels(out, "train")
    assert train_empties < 8
