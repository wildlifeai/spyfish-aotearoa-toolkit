"""A drop's expert review volume and survey training volume are merged, not ranked.

The two volumes sample different frames of the same video: the ML frame
selector writes `__frame_0146.781s`, the training-frame sampler
`__frame_382.000s`. "Expert wins" (2026-08-22) assumed they competed and
dropped every training-volume frame of 39 drops; `biigle_to_yolo` also wiped
the training volume's negatives, so the rebuilt labels/ held none (found
2026-10-15 on SLI_20260114_BUV_SLI_001_01).
"""

import json

import pandas as pd

from spyfish.biigle.biigle_to_yolo import biigle_to_yolo
from spyfish.config.species import species_registry
from spyfish.config.wrapper import config

DROP = "SLI_20260114_BUV_SLI_001_01"
SURVEY = "SLI_20260114_BUV"
BLUE_COD = "Blue cod - Parapercis colias"
WRASSE = "Scarlet wrasse - Pseudolabrus miles"
_POINTS = json.dumps([864.0, 486.0, 1056.0, 486.0, 1056.0, 594.0, 864.0, 594.0])
_ATTRS = json.dumps({"width": 1920, "height": 1080})


def _frame(secs):
    return f"{DROP}__frame_{secs}.jpg"


def _rows(frames, label):
    return pd.DataFrame(
        {
            "filename": frames,
            "label_name": [label] * len(frames),
            "points": [_POINTS] * len(frames),
            "attributes": [_ATTRS] * len(frames),
        }
    )


def _drop(
    tmp_path, expert=None, training=None, expert_reviewed=None, training_reviewed=None
):
    ann = tmp_path / SURVEY / DROP / "annotations"
    ann.mkdir(parents=True)
    for suffix, df in (
        (config.biigle_expert_raw_suffix, expert),
        (config.biigle_training_raw_suffix, training),
    ):
        if df is not None:
            df.to_csv(ann / f"{DROP}{suffix}", index=False)
    for suffix, names in (
        (config.biigle_expert_reviewed_frames_suffix, expert_reviewed),
        (config.biigle_training_reviewed_frames_suffix, training_reviewed),
    ):
        if names is not None:
            pd.DataFrame({"filename": names}).to_csv(
                ann / f"{DROP}{suffix}", index=False
            )
    return ann.parent / "labels"


def _class_ids(labels, secs):
    text = (labels / f"{DROP}__frame_{secs}.txt").read_text()
    return {int(line.split()[0]) for line in text.splitlines()}


def _ids():
    m = species_registry(class_map_path=config.class_map_path).name_to_class_id()
    return m[BLUE_COD], m[WRASSE]


def test_both_sources_are_converted(tmp_path):
    labels = _drop(
        tmp_path,
        expert=_rows([_frame("0146.781s")], WRASSE),
        training=_rows([_frame("382.000s")], BLUE_COD),
    )
    biigle_to_yolo(tmp_path, config.class_map_path)

    cod, wrasse = _ids()
    assert _class_ids(labels, "0146.781s") == {wrasse}
    assert _class_ids(labels, "382.000s") == {cod}, "training-volume frame was dropped"


def test_frame_in_both_volumes_keeps_only_the_expert_boxes(tmp_path, caplog):
    shared = _frame("0500.000s")
    labels = _drop(
        tmp_path,
        expert=_rows([shared], WRASSE),
        training=_rows([shared], BLUE_COD),
    )
    with caplog.at_level("WARNING"):
        biigle_to_yolo(tmp_path, config.class_map_path)

    _, wrasse = _ids()
    assert _class_ids(labels, "0500.000s") == {wrasse}
    assert any(
        "in both the expert and the training volume" in r.message
        for r in caplog.records
    )


def test_negatives_from_both_reviewed_lists_survive_the_rebuild(tmp_path):
    labels = _drop(
        tmp_path,
        expert=_rows([_frame("0146.781s")], WRASSE),
        training=_rows([_frame("382.000s")], BLUE_COD),
        expert_reviewed=[_frame("0146.781s"), _frame("0163.131s")],
        training_reviewed=[_frame("382.000s"), _frame("688.000s"), _frame("958.000s")],
    )
    # A stale label from an earlier run must not survive the wipe.
    labels.mkdir()
    (labels / f"{DROP}__frame_9999.000s.txt").write_text("0 .5 .5 .1 .1\n")

    biigle_to_yolo(tmp_path, config.class_map_path)

    empties = sorted(p.stem for p in labels.glob("*.txt") if not p.read_text().strip())
    assert empties == [
        f"{DROP}__frame_0163.131s",
        f"{DROP}__frame_688.000s",
        f"{DROP}__frame_958.000s",
    ]
    assert not (labels / f"{DROP}__frame_9999.000s.txt").exists()


def test_fully_empty_training_drop_gets_its_negatives(tmp_path):
    """Header-only training CSV: every reviewed frame is a background frame."""
    reviewed = [_frame(f"{s}.000s") for s in (100, 200, 300)]
    labels = _drop(
        tmp_path,
        training=_rows([], BLUE_COD),
        training_reviewed=reviewed,
    )
    biigle_to_yolo(tmp_path, config.class_map_path)

    txts = list(labels.glob("*.txt"))
    assert len(txts) == 3 and all(not p.read_text() for p in txts)
