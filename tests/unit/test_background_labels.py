"""Reviewed-but-empty frames must become YOLO background labels.

A Biigle annotation report lists only frames that HAVE annotations, so a frame
the expert reviewed and left empty produced no .txt at all and never reached
training. The corpus held 1 background frame in 5054 while the frame selector
was deliberately picking "Blind (False Negative Check)" frames for review
(found 2026-08-24).
"""

import pandas as pd

from spyfish.biigle.biigle_to_yolo import _write_background_labels

DROP = "KSF_20240124_BUV_KSF_085_01"


def _drop_dir(tmp_path, universe=None, annotated=()):
    d = tmp_path / DROP
    (d / "annotations").mkdir(parents=True)
    labels = d / "labels"
    labels.mkdir()
    for stem in annotated:
        (labels / f"{stem}.txt").write_text("0 .5 .5 .1 .1\n")
    if universe is not None:
        pd.DataFrame({"filename": universe}).to_csv(
            d / "annotations" / f"{DROP}_biigle_expert_universe.csv", index=False
        )
    return d, labels


def test_unannotated_reviewed_frames_become_empty_labels(tmp_path):
    d, labels = _drop_dir(
        tmp_path,
        universe=[f"{DROP}__frame_{s}.jpg" for s in ("0010.0s", "0020.0s", "0030.0s")],
        annotated=[f"{DROP}__frame_0010.0s"],
    )

    n = _write_background_labels(d, labels)

    assert n == 2
    for s in ("0020.0s", "0030.0s"):
        p = labels / f"{DROP}__frame_{s}.txt"
        assert p.exists() and p.read_text() == "", "background label must be empty"
    # the annotated frame keeps its boxes
    assert (labels / f"{DROP}__frame_0010.0s.txt").read_text().strip()


def test_no_universe_file_writes_nothing(tmp_path):
    """Drops synced before the universe existed must not get guessed negatives."""
    d, labels = _drop_dir(tmp_path, universe=None, annotated=[f"{DROP}__frame_0010.0s"])
    assert _write_background_labels(d, labels) == 0
    assert len(list(labels.glob("*.txt"))) == 1


def test_frames_on_disk_are_not_used_as_the_universe(tmp_path):
    """--biigle-upload re-extracts frames, so frames/ can hold frames NEWER than
    the reviewed volume. Calling those empty would teach the model that
    unreviewed water has no fish."""
    d, labels = _drop_dir(
        tmp_path,
        universe=[f"{DROP}__frame_0010.0s.jpg"],
        annotated=[f"{DROP}__frame_0010.0s"],
    )
    frames = d / "frames"
    frames.mkdir()
    for s in ("0010.0s", "0999.0s"):  # 0999 was never in the volume
        (frames / f"{DROP}__frame_{s}.jpg").write_text("x")

    assert _write_background_labels(d, labels) == 0
    assert not (labels / f"{DROP}__frame_0999.0s.txt").exists()


def test_rerun_is_idempotent(tmp_path):
    d, labels = _drop_dir(
        tmp_path,
        universe=[f"{DROP}__frame_{s}.jpg" for s in ("0010.0s", "0020.0s")],
        annotated=[f"{DROP}__frame_0010.0s"],
    )
    assert _write_background_labels(d, labels) == 1
    assert _write_background_labels(d, labels) == 0
