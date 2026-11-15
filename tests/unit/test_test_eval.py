"""Frozen-test-set records: one model x one test-set version, per class, with
contamination measured rather than footnoted (built 2026-10-16)."""

import json
from pathlib import Path

import pytest
import yaml

from spyfish.config.wrapper import config
from spyfish.ml.training import test_eval as te

DROP_A = "SLI_20260114_BUV_SLI_044_01"
DROP_B = "TUK_20250124_BUV_TUK_002_01"
RUN = "species_20260825_053442"
LIVE = {
    0: "fish",
    1: "Chrysophrys auratus",
    2: "Parapercis colias",
    3: "Helicolenus percoides",
}
OLD_MODEL = {
    0: "fish",
    1: "Parapercis colias",
    2: "Chrysophrys auratus",
}  # no sea perch, ids differ


def _frame(drop, secs):
    return f"{drop}__frame_{secs}.000s.jpg"


def _dataset(tmp_path, labels):
    """An assembled species/ dataset with a test split. labels: {image: [(cid, ...)]}"""
    ds = tmp_path / "species"
    for split in ("images", "labels"):
        (ds / split / "test").mkdir(parents=True)
    (ds / "data.yaml").write_text(yaml.safe_dump({"names": LIVE}))
    for fn, boxes in labels.items():
        (ds / "images" / "test" / fn).write_text("jpg")
        txt = "\n".join(f"{c} .5 .5 .1 .1" for c in boxes)
        (ds / "labels" / "test" / f"{Path(fn).stem}.txt").write_text(
            txt + ("\n" if txt else "")
        )
    return ds


def test_version_ignores_order_and_changes_on_growth():
    v = te.test_set_version([DROP_A, DROP_B])
    assert v == te.test_set_version([DROP_B, DROP_A]) and len(v) == 8
    assert v != te.test_set_version([DROP_A, DROP_B, "BNP_20250118_BUV_BNP_121_01"])


def test_model_name_from_run_dir_and_from_promoted_file(tmp_path):
    best = tmp_path / "runs" / RUN / "weights" / "best.pt"
    assert te.model_name_for(best) == RUN
    assert te.model_name_for(tmp_path / f"{RUN}.pt") == RUN
    with pytest.raises(ValueError):
        te.model_name_for(tmp_path / "runs" / "whatever" / "weights" / "best.pt")


def test_snapshot_found_under_canonical_and_legacy_run_names(tmp_path):
    runs = tmp_path / "runs"
    assert te.find_run_snapshot(RUN, runs) is None
    (runs / "20260825_053442_species" / "dataset").mkdir(parents=True)
    (runs / "20260825_053442_species" / "dataset" / "train.txt").write_text("")
    assert te.find_run_snapshot(RUN, runs).parent.name == "20260825_053442_species"
    (runs / RUN / "dataset").mkdir(parents=True)
    (runs / RUN / "dataset" / "train.txt").write_text("")
    assert te.find_run_snapshot(RUN, runs).parent.name == RUN


def test_seen_images_counts_train_and_val_and_is_none_without_snapshot(tmp_path):
    snap = tmp_path / "dataset"
    snap.mkdir()
    (snap / "train.txt").write_text(_frame(DROP_A, 1) + "\n")
    (snap / "val.txt").write_text(_frame(DROP_A, 2) + "\n")
    images = [_frame(DROP_A, 1), _frame(DROP_A, 2), _frame(DROP_B, 1)]
    seen = te.seen_images(snap, images)
    assert seen == {"train": {_frame(DROP_A, 1)}, "val": {_frame(DROP_A, 2)}}
    assert te.seen_images(None, images) is None


def test_model_view_rewrites_ids_and_sends_unknown_species_to_fish(tmp_path):
    ds = _dataset(tmp_path, {_frame(DROP_A, 1): [1, 3], _frame(DROP_A, 2): []})
    yaml_path, remapped, dropped = te.build_model_view(
        ds, "test", te.split_images(ds, "test"), LIVE, OLD_MODEL, tmp_path / "view"
    )
    lbl = tmp_path / "view" / "labels" / "val"
    # snapper: live id 1 -> old model id 2; sea perch: unknown -> old fish (0)
    assert (lbl / f"{Path(_frame(DROP_A, 1)).stem}.txt").read_text().split("\n")[
        :2
    ] == [
        "2 .5 .5 .1 .1",
        "0 .5 .5 .1 .1",
    ]
    assert (
        lbl / f"{Path(_frame(DROP_A, 2)).stem}.txt"
    ).read_text() == ""  # empty frame kept
    assert (remapped, dropped) == (1, 0)
    assert yaml.safe_load(yaml_path.read_text())["names"] == OLD_MODEL
    assert (tmp_path / "view" / "images" / "val" / _frame(DROP_A, 1)).exists()


def test_boxes_dropped_when_model_has_no_fallback_class(tmp_path):
    ds = _dataset(tmp_path, {_frame(DROP_A, 1): [3]})
    _, remapped, dropped = te.build_model_view(
        ds,
        "test",
        te.split_images(ds, "test"),
        LIVE,
        {0: "Chrysophrys auratus"},
        tmp_path / "v",
    )
    assert (remapped, dropped) == (0, 1)


def _fake_scorer(calls):
    def scorer(weights, data_yaml, imgsz, out_dir):
        n = len(list((Path(data_yaml).parent / "images" / "val").iterdir()))
        calls.append((out_dir.name, n))
        return {
            "mAP50": 0.5,
            "mAP50_95": 0.3,
            "precision": 0.6,
            "recall": 0.4,
            "per_class": [
                {"name": "fish", "mAP50": 0.5, "mAP50_95": 0.3, "P": 0.6, "R": 0.4}
            ],
        }

    return scorer


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(config), "local_training_dir", property(lambda s: tmp_path)
    )
    monkeypatch.setattr(
        type(config), "training_test_drops", property(lambda s: {DROP_A, DROP_B})
    )
    monkeypatch.setattr(type(config), "training_imgsz", property(lambda s: 640))
    # The fake weights are a text file: no train_args to read a size from.
    monkeypatch.setattr(te, "checkpoint_imgsz", lambda p: None)
    fake_yolo = type("Y", (), {"__init__": lambda self, p: None, "names": OLD_MODEL})
    monkeypatch.setitem(
        __import__("sys").modules, "ultralytics", type("M", (), {"YOLO": fake_yolo})
    )
    monkeypatch.setattr(te, "validate_model_path", lambda p: Path(p))
    return tmp_path


def test_record_has_all_and_clean_views_with_contamination(isolated):
    ds = _dataset(
        isolated,
        {_frame(DROP_A, 1): [1], _frame(DROP_A, 2): [], _frame(DROP_B, 1): [2]},
    )
    snap = isolated / "runs" / RUN / "dataset"
    snap.mkdir(parents=True)
    (snap / "train.txt").write_text(_frame(DROP_A, 1) + "\n")
    (snap / "val.txt").write_text("")
    weights = isolated / "runs" / RUN / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_text("pt")

    calls = []
    rec = te.evaluate_on_test(
        weights,
        dataset_dir=ds,
        results_dir=isolated / "results",
        scorer=_fake_scorer(calls),
    )

    assert sorted(calls) == [("all", 3), ("clean", 2)]
    assert rec["contamination"] == {
        "snapshot_found": True,
        "n_in_train": 1,
        "n_in_val": 0,
        "n_clean": 2,
    }
    assert rec["test_set"]["version"] == te.test_set_version([DROP_A, DROP_B])
    assert (
        rec["model_name"] == RUN
        and rec["all"]["n_images"] == 3
        and rec["clean"]["n_images"] == 2
    )
    saved = json.loads(
        (
            te.record_dir_for(RUN, rec["test_set"]["version"], isolated / "results")
            / "record.json"
        ).read_text()
    )
    assert saved["all"]["per_class"][0]["name"] == "fish"
    assert te.load_records(isolated / "results")[0]["model_name"] == RUN


def test_without_snapshot_clean_is_unknown(isolated):
    ds = _dataset(isolated, {_frame(DROP_A, 1): [1]})
    weights = isolated / f"{RUN}.pt"
    weights.write_text("pt")
    rec = te.evaluate_on_test(
        weights,
        dataset_dir=ds,
        results_dir=isolated / "results",
        scorer=_fake_scorer([]),
    )
    assert rec["contamination"]["snapshot_found"] is False and rec["clean"] is None


def test_records_to_frames_and_promotion_decision(monkeypatch):
    monkeypatch.setattr(
        type(config), "retrain_min_improvement_pct", property(lambda s: 2.0)
    )
    base = {
        "test_set": {"version": "abc", "n_drops": 2},
        "contamination": {
            "snapshot_found": True,
            "n_in_train": 0,
            "n_in_val": 0,
            "n_clean": 3,
        },
        "clean": None,
    }
    pc = [{"name": "fish", "mAP50": 0.5, "mAP50_95": 0.3, "P": 0.6, "R": 0.4}]
    prod = {
        **base,
        "model_name": "species_20260603",
        "model_stamp": "20260603000000",
        "all": {
            "n_images": 3,
            "mAP50": 0.50,
            "mAP50_95": 0.3,
            "precision": 0.6,
            "recall": 0.4,
            "per_class": pc,
        },
    }
    new = {
        **prod,
        "model_name": RUN,
        "model_stamp": "20260825053442",
        "all": {**prod["all"], "mAP50": 0.53},
    }
    overall, per_class = te.records_to_frames([prod, new])
    assert list(overall.model_name) == ["species_20260603", RUN] and len(per_class) == 2

    assert te.promotion_decision(new, prod)[0] is True
    assert (
        te.promotion_decision({**new, "all": {**new["all"], "mAP50": 0.51}}, prod)[0]
        is False
    )
    assert te.promotion_decision(new, None)[0] is True
    other = {**prod, "test_set": {"version": "zzz", "n_drops": 3}}
    ok, why = te.promotion_decision(new, other)
    assert ok is False and "versions differ" in why


def test_val_metrics_rounds_the_four_headline_numbers():
    box = type("B", (), {"map50": 0.123456, "map": 0.2, "mp": 0.65432, "mr": 0.400049})
    assert te.val_metrics(type("M", (), {"box": box})) == {
        "mAP50": 0.1235,
        "mAP50_95": 0.2,
        "precision": 0.6543,
        "recall": 0.4,
    }


def test_promotion_margin_is_one_rule(monkeypatch):
    monkeypatch.setattr(
        type(config), "retrain_min_improvement_pct", property(lambda s: 2.0)
    )
    assert te.promotion_margin() == pytest.approx(0.02)
    ok, delta = te.clears_promotion_margin(0.53, 0.50)
    assert ok is True and delta == pytest.approx(0.03)
    assert te.clears_promotion_margin(0.51, 0.50)[0] is False
    # an explicit margin wins over config
    assert te.clears_promotion_margin(0.51, 0.50, margin=0.0)[0] is True


def test_shared_vocabulary_scores_each_model_on_the_common_classes(isolated):
    ds = _dataset(
        isolated,
        {_frame(DROP_A, 1): [1], _frame(DROP_A, 2): [3], _frame(DROP_B, 1): [2]},
    )
    snap = isolated / "runs" / RUN / "dataset"
    snap.mkdir(parents=True)
    (snap / "train.txt").write_text(_frame(DROP_A, 1) + "\n")
    (snap / "val.txt").write_text(_frame(DROP_B, 1) + "\n")
    weights = isolated / "runs" / RUN / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_text("pt")
    loose = isolated / "species_20260603.pt"  # no run, no snapshot
    loose.write_text("pt")
    models = {"new": weights, "old": loose}
    images = te.split_images(ds, "test")

    seen_by, unfilterable = te.trained_on(models, images)
    # train only: the val-listed frame is not counted as trained on
    assert seen_by == {"new": {_frame(DROP_A, 1)}} and unfilterable == ["old"]

    calls = []

    def scorer(weights, data_yaml, imgsz, out_dir, classes=None, verbose=False):
        calls.append((out_dir.name, imgsz, classes))
        return {"mAP50": 0.5, "mAP50_95": 0.3, "precision": 0.6, "recall": 0.4}

    out = te.score_on_shared_vocabulary(
        models, images, isolated / "cmp", dataset_dir=ds, split="test", scorer=scorer
    )
    # the fake YOLO gives both models OLD_MODEL's three classes
    assert out["shared_vocabulary"] == sorted(OLD_MODEL.values())
    assert calls == [("val_new", 640, [0, 1, 2]), ("val_old", 640, [0, 1, 2])]
    assert out["models"]["new"]["nc"] == 3
    # sea perch (live id 3) is not in the model: its one box goes to 'fish'
    assert out["models"]["new"]["gt_boxes_remapped_to_fallback"] == 1
    assert (isolated / "cmp" / "new" / "data.yaml").exists()
