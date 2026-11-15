"""The second-stage classifier's class rule and its decision record
(spyfish/ml/classifier/classes.py, moved out of scripts/wip 2026-11-10).

A species is a class at >= min_drops drops and >= min_crops crops; the rest
train as `other fish`. The thresholds live in config and the choice is written
beside the checkpoint as classes.json.
"""

import json

import pandas as pd
import pytest

from spyfish.config.wrapper import config
from spyfish.ml.classifier import classes as cl
from spyfish.ml.classifier import crops
from spyfish.ml.classifier.metrics import apply_prior, iou, score_predictions


def _meta(rows):
    """rows: [(species, drop, n_crops)] -> crops_meta-shaped frame."""
    out = []
    for species, drop, n in rows:
        out += [
            {"species": species, "drop": drop, "crop_id": f"{len(out) + i:06d}"}
            for i in range(n)
        ]
    return pd.DataFrame(out)


D = [f"KSF_20240124_BUV_KSF_08{i}_01" for i in range(5)]


def test_class_rule_counts_crops_and_drops():
    meta = _meta(
        [("Parapercis colias", D[0], 40), ("Parapercis colias", D[1], 40)]
        + [("Chrysophrys auratus", D[0], 100)]  # one drop: cannot be held out
        + [("Latris lineata", D[0], 10), ("Latris lineata", D[1], 10)]  # too few crops
    )
    target, classes, stats = cl.classes_for(
        meta, source="biigle_csv", min_drops=2, min_crops=30
    )
    assert target == ["Parapercis colias"]
    assert classes == ["Parapercis colias", cl.OTHER]
    assert stats.loc["Chrysophrys auratus", "tier"] == cl.OTHER
    assert list(stats.index) == [
        "Chrysophrys auratus",
        "Parapercis colias",
        "Latris lineata",
    ]  # busiest first


def test_thresholds_default_to_config():
    meta = _meta([("Parapercis colias", d, 10) for d in D])  # 50 crops, 5 drops
    target, _, _ = cl.classes_for(meta, source="biigle_csv")
    assert (target == ["Parapercis colias"]) == (
        50 >= config.classifier_training_min_crops
        and 5 >= config.classifier_training_min_drops
    )


def test_uoa_source_fixes_the_papers_classes_whatever_the_counts(caplog):
    meta = _meta([("Girella tricuspidata", D[0], 1), ("Parapercis colias", D[0], 500)])
    target, classes, stats = cl.classes_for(meta, source="uoa")
    assert target == ["Girella tricuspidata"]
    assert classes[-1] == crops.UOA_OTHER == "fish"
    assert "no crops on disk" in caplog.text  # the other three paper species


def test_class_table_records_the_decision(tmp_path):
    meta = _meta(
        [
            ("Parapercis colias", D[0], 40),
            ("Parapercis colias", D[1], 40),
            ("Latris lineata", D[2], 5),
        ]
    )
    target, classes, stats = cl.classes_for(
        meta, source="biigle_csv", min_drops=2, min_crops=30
    )
    path = cl.write_class_table(
        tmp_path / "run" / cl.CLASS_TABLE_FILENAME,
        target,
        classes,
        stats,
        extra={"recipe": {"epochs": 6}},
    )
    doc = json.loads(path.read_text())
    assert doc["classes"] == classes and doc["target"] == target
    assert doc["min_crops"] == config.classifier_training_min_crops
    assert {r["species"]: r["tier"] for r in doc["species"]} == {
        "Parapercis colias": "class",
        "Latris lineata": cl.OTHER,
    }
    assert doc["species"][0]["common_name"]  # registry supplies the common name
    assert doc["recipe"] == {"epochs": 6}
    assert (
        cl.read_class_table(path) == doc
        and cl.read_class_table(tmp_path / "none.json") is None
    )


def test_classifier_boxes_drop_bait_catchall_and_excluded_species(monkeypatch):
    monkeypatch.setattr(crops, "EXCLUDED_SPECIES", {"Evechinus chloroticus"})
    boxes = pd.DataFrame(
        {"species": ["bait", "fish", "Evechinus chloroticus", "Parapercis colias"]}
    )
    assert list(crops.classifier_boxes(boxes)["species"]) == ["Parapercis colias"]


def test_crop_margin_and_min_box_come_from_inference_config():
    assert crops.CROP_MARGIN == config.classifier_crop_margin
    assert crops.MIN_BOX_PX == config.classifier_min_box_px
    assert crops.BIOCLIP_BACKBONE == config.classifier_backbone


def test_resolve_source_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("TWO_STAGE_SOURCE", "yolo_dataset")
    assert crops.resolve_source() == "yolo_dataset"
    monkeypatch.setenv("TWO_STAGE_SOURCE", "nope")
    with pytest.raises(ValueError):
        crops.resolve_source()
    monkeypatch.delenv("TWO_STAGE_SOURCE")
    assert crops.resolve_source() == config.classifier_training_source


# ── metrics ──────────────────────────────────────────────────────────────


def test_score_predictions_matches_greedily_and_ignores_bait():
    experts = pd.DataFrame(
        {
            "filename": ["a.jpg", "a.jpg"],
            "species": ["Parapercis colias", "Chrysophrys auratus"],
            "x_min": [0, 100],
            "y_min": [0, 100],
            "x_max": [50, 150],
            "y_max": [50, 150],
        }
    )
    pred = pd.DataFrame(
        {
            "filename": ["a.jpg", "a.jpg", "a.jpg"],
            "yolo_class": ["fish", "bait", "fish"],
            "conf": [0.9, 0.9, 0.2],
            "x_min": [2, 100, 300],
            "y_min": [2, 100, 300],
            "x_max": [50, 150, 350],
            "y_max": [50, 150, 350],
        }
    )
    scored, summary = score_predictions(pred, experts)
    assert list(scored["found"]) == [
        True,
        False,
    ]  # the bait box does not count as finding the snapper
    assert summary == {"n_pred": 2, "matched_pred": 1, "precision": 0.5, "recall": 0.5}
    _, at_conf = score_predictions(pred, experts, conf=0.5)
    assert at_conf["n_pred"] == 1
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_apply_prior_leaves_other_and_unknown_reserves_alone():
    classes = ["Chrysophrys auratus", cl.OTHER]
    prior = pd.DataFrame(
        {
            "reserve_code": ["KSF", "KSF"],
            "scientific_name": ["Chrysophrys auratus", "Parapercis colias"],
            "presence_prob": [0.5, 0.01],
        }
    )
    proba = pd.DataFrame([[0.6, 0.4], [0.6, 0.4]]).to_numpy()
    out = apply_prior(proba, classes, pd.Series(["KSF", "UOA"]), prior, alpha=1.0)
    assert out[0].tolist() == pytest.approx(
        [0.3 / 0.7, 0.4 / 0.7]
    )  # snapper halved at KSF, other untouched
    assert out[1].tolist() == pytest.approx([0.6, 0.4])  # UOA is not in the prior


# ── names written before the snapper rename ──────────────────────────────


def test_uoa_classes_are_registry_names():
    """The paper says Pagrus auratus; every label resolves to the registry name
    before it meets UOA_CLASSES, so the list must already be canonical."""
    for name in crops.UOA_CLASSES:
        assert crops.species_name(name) == name, name
    assert "Chrysophrys auratus" in crops.UOA_CLASSES


def test_crop_table_and_predictions_are_canonicalised_on_read(tmp_path, caplog):
    meta = pd.DataFrame(
        {
            "crop_id": ["000001", "000002"],
            "drop": [D[0], D[0]],
            "species": ["Pagrus auratus", "fish"],
        }
    )
    meta.to_csv(tmp_path / "crops_meta.csv", index=False)
    out = crops.read_crops_meta(tmp_path / "crops_meta.csv")
    assert list(out["species"]) == ["Chrysophrys auratus", "fish"]
    assert out["crop_id"].dtype == object  # ids keep their leading zeros
    assert "canonicalised on read" in caplog.text

    pd.DataFrame(
        {"filename": ["a.jpg"], "yolo_class": ["Pagrus auratus"], "conf": [0.9]}
    ).to_csv(tmp_path / "pred.csv", index=False)
    assert list(crops.read_predictions(tmp_path / "pred.csv")["yolo_class"]) == [
        "Chrysophrys auratus"
    ]


def _groups():
    import numpy as np

    rng = np.random.default_rng(0)
    return rng.choice(
        [f"ESK_B_2022011{d}_EM_C" for d in range(8)] + ["MOT_S_20220119_PM_M"] * 4, 240
    )


def test_grouped_folds_without_stills_is_plain_group_kfold():
    """No always-train rows: the same index pairs GroupKFold gave before the helper."""
    import numpy as np
    from sklearn.model_selection import GroupKFold

    g = _groups()
    n = min(cl.N_FOLDS, len(np.unique(g)))
    want = list(GroupKFold(n_splits=n).split(g, groups=g))
    got = cl.grouped_folds(g)
    assert len(got) == len(want)
    for (tr, te), (wtr, wte) in zip(got, want):
        assert (tr == wtr).all() and (te == wte).all()


def test_snapper_stills_train_in_every_fold_and_are_never_scored():
    """The stills are other people's footage: they help training, never the score."""
    import numpy as np

    g = np.concatenate([_groups(), [crops.UOA_STILLS] * 30])
    stills = np.flatnonzero(g == crops.UOA_STILLS)
    folds = cl.grouped_folds(g)
    assert all(np.isin(stills, tr).all() for tr, _ in folds)
    assert not any(np.isin(stills, te).any() for _, te in folds)
    tested = np.sort(np.concatenate([te for _, te in folds]))
    assert (tested == np.flatnonzero(cl.scored_rows(g))).all()
