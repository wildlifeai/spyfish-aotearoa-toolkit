"""The MaxN confidence cutoff: picked per model on val at the F-beta peak,
clamped to the inference floor, stored beside the weights, read back at MaxN
time (built 2026-11-12)."""

import json
import logging

import numpy as np
import pytest

from spyfish.config.wrapper import config
from spyfish.ml import model_meta as mm
from spyfish.ml.training import threshold as th
from spyfish.orchestrator import retrain_runner as rr

RUN = "species_20260825_053442"
NAMES = ["fish", "Chrysophrys auratus", "bait"]
PX = np.linspace(0, 1, 1000)


def _curves(r_mid=(0.5, 0.6, 0.3), p_mid=(0.3, 0.4, 0.8)):
    """Monotone synthetic curves: recall falls and precision rises with the
    cutoff, each class crossing 0.5 at its own confidence."""
    k = 12.0
    r = np.stack([1 / (1 + np.exp(k * (PX - c))) for c in r_mid])
    p = np.stack([1 / (1 + np.exp(-k * (PX - c))) for c in p_mid])
    return p, r


# ── the arithmetic ───────────────────────────────────────────────────────────


def test_fbeta_is_f1_at_beta_one_and_leans_on_recall_above_it():
    assert th.fbeta(0.5, 0.5, 1.0) == pytest.approx(0.5)
    assert th.fbeta(0.95, 0.50, 1.0) == pytest.approx(0.655, abs=1e-3)
    # Same P and R: F2 is pulled toward the (lower) recall.
    assert th.fbeta(0.95, 0.50, 2.0) < th.fbeta(0.95, 0.50, 1.0)
    # And it rewards a recall-heavy model that F1 would score the same.
    assert th.fbeta(0.50, 0.95, 2.0) > th.fbeta(0.95, 0.50, 2.0)
    assert th.fbeta(0.0, 0.0, 2.0) == 0.0


def test_f2_picks_a_lower_cutoff_than_f1():
    p, r = _curves()
    f2 = th.select_threshold(PX, p, r, NAMES, beta=2.0, floor=0.05, exclude=["bait"])
    f1 = th.select_threshold(PX, p, r, NAMES, beta=1.0, floor=0.05, exclude=["bait"])
    assert f2["threshold"] < f1["threshold"]
    assert f2["f1_threshold"] == pytest.approx(f1["threshold"], abs=0.002)
    assert not f2["floored"]
    assert f2["unfloored_threshold"] == f2["threshold"]
    # At its own cutoff F2 is at least what F2 is at the F1 cutoff.
    assert f2["mean_fbeta"] >= f2["mean_fbeta_at_f1_threshold"]


def test_floor_clamps_and_says_so():
    p, r = _curves()
    free = th.select_threshold(PX, p, r, NAMES, beta=2.0, floor=0.05, exclude=["bait"])
    floored = th.select_threshold(
        PX, p, r, NAMES, beta=2.0, floor=0.60, exclude=["bait"]
    )
    assert free["threshold"] < 0.60
    assert floored["threshold"] == 0.60 and floored["floored"]
    assert floored["unfloored_threshold"] == free["threshold"]


def test_excluded_class_is_reported_but_not_in_the_mean():
    p, r = _curves()
    # Bait's curves crossing far from the others would drag the mean if counted.
    with_bait = th.select_threshold(PX, p, r, NAMES, beta=2.0, floor=0.05)
    without = th.select_threshold(
        PX, p, r, NAMES, beta=2.0, floor=0.05, exclude=["bait"]
    )
    assert with_bait["threshold"] != without["threshold"]
    assert without["excluded_classes"] == ["bait"]
    rows = {row["name"]: row for row in without["per_class"]}
    assert set(rows) == set(NAMES)
    assert rows["bait"]["counted"] is False and rows["fish"]["counted"] is True
    for row in without["per_class"]:
        assert {"P", "R", "fbeta", "own_best_threshold", "own_best_fbeta"} <= set(row)
        assert row["own_best_fbeta"] >= row["fbeta"] - 1e-6


def test_select_rejects_bad_shapes_and_values():
    p, r = _curves()
    with pytest.raises(ValueError):
        th.select_threshold(PX, p[:2], r, NAMES, beta=2.0, floor=0.25)
    with pytest.raises(ValueError):
        th.select_threshold(PX, p, r, NAMES, beta=0.0, floor=0.25)
    with pytest.raises(ValueError):
        th.select_threshold(PX, p, r, NAMES, beta=2.0, floor=0.0)


def test_metrics_at_reads_one_cutoff():
    p, r = _curves()
    at = th.metrics_at(PX, p, r, NAMES, conf=0.40, beta=2.0, exclude=["bait"])
    j = int(np.searchsorted(PX, 0.40))
    rows = {row["name"]: row for row in at["per_class"]}
    assert rows["fish"]["P"] == pytest.approx(p[0, j], abs=1e-4)
    assert rows["fish"]["R"] == pytest.approx(r[0, j], abs=1e-4)
    assert at["recall"] == pytest.approx(
        np.mean([rows[n]["R"] for n in NAMES[:2]]), abs=1e-4
    )


def test_selection_table_prints_every_class_and_the_cutoff():
    p, r = _curves()
    sel = th.select_threshold(PX, p, r, NAMES, beta=2.0, floor=0.60, exclude=["bait"])
    text = th.selection_table(sel)
    for n in NAMES:
        assert n in text
    assert "floored from" in text and "plain F1 would pick" in text


# ── the sidecar ──────────────────────────────────────────────────────────────


def test_sidecar_sits_beside_the_weights_under_their_stem(tmp_path):
    best = tmp_path / "runs" / RUN / "weights" / "best.pt"
    assert mm.threshold_sidecar_path(best) == best.parent / "best.thresholds.json"
    promoted = tmp_path / f"{RUN}.pt"
    assert mm.threshold_sidecar_path(promoted) == tmp_path / f"{RUN}.thresholds.json"


def test_sidecar_roundtrip_and_missing(tmp_path):
    weights = tmp_path / f"{RUN}.pt"
    weights.write_bytes(b"pt")
    assert mm.read_threshold_sidecar(weights) is None
    path = mm.write_threshold_sidecar(weights, {"threshold": 0.31, "beta": 2.0})
    assert path.exists()
    assert mm.read_threshold_sidecar(weights)["threshold"] == 0.31
    with pytest.raises(ValueError):
        mm.write_threshold_sidecar(weights, {"beta": 2.0})
    path.write_text(json.dumps({"beta": 2.0}))
    with pytest.raises(ValueError):
        mm.read_threshold_sidecar(weights)


# ── resolving at MaxN time ──────────────────────────────────────────────────


def _set_threshold_setting(value):
    config._yaml_config["ml_inference"]["maxn_confidence_threshold"] = value


def test_resolve_reads_sidecar_when_config_says_model(pipeline_env, tmp_path):
    weights = tmp_path / f"{RUN}.pt"
    weights.write_bytes(b"pt")
    _set_threshold_setting("model")
    with pytest.raises(ValueError, match="thresholds.json"):
        mm.resolve_maxn_threshold(weights)
    mm.write_threshold_sidecar(
        weights, {"threshold": 0.31, "beta": 2.0, "selected_on": "val"}
    )
    assert mm.resolve_maxn_threshold(weights) == pytest.approx(0.31)


def test_number_in_config_overrides_sidecar_with_a_warning(
    pipeline_env, tmp_path, caplog
):
    weights = tmp_path / f"{RUN}.pt"
    weights.write_bytes(b"pt")
    mm.write_threshold_sidecar(weights, {"threshold": 0.31, "beta": 2.0})
    _set_threshold_setting(0.40)
    with caplog.at_level(logging.WARNING):
        assert mm.resolve_maxn_threshold(weights) == pytest.approx(0.40)
    assert "overrides" in caplog.text and "0.310" in caplog.text
    # Same value: no warning.
    caplog.clear()
    _set_threshold_setting(0.31)
    with caplog.at_level(logging.WARNING):
        mm.resolve_maxn_threshold(weights)
    assert "overrides" not in caplog.text


def test_cutoff_below_the_inference_floor_is_refused(pipeline_env, tmp_path):
    weights = tmp_path / f"{RUN}.pt"
    weights.write_bytes(b"pt")
    _set_threshold_setting("model")
    # The test config's confidence_threshold is 0.25; a sidecar written when
    # the floor was lower must not be honoured silently.
    mm.write_threshold_sidecar(weights, {"threshold": 0.10, "beta": 2.0})
    with pytest.raises(ValueError, match="below ml_inference.confidence_threshold"):
        mm.resolve_maxn_threshold(weights)


def test_config_property_validates_the_setting(pipeline_env):
    _set_threshold_setting("model")
    assert config.maxn_confidence_threshold == "model"
    _set_threshold_setting(0.4)
    assert config.maxn_confidence_threshold == 0.4
    for bad in ("auto", 0, 1.5, True):
        _set_threshold_setting(bad)
        with pytest.raises(ValueError):
            config.maxn_confidence_threshold
    assert config.threshold_beta == 2.0


def test_model_path_for_name_finds_production_then_archive(pipeline_env):
    config.pipeline_model_dir.mkdir(parents=True, exist_ok=True)
    config.archived_models_dir.mkdir(parents=True, exist_ok=True)
    archived = config.archived_models_dir / "species_20260603_120000.pt"
    archived.write_bytes(b"pt")
    live = config.pipeline_model_dir / f"{RUN}.pt"
    live.write_bytes(b"pt")
    assert config.model_path_for_name(RUN) == live
    assert config.model_path_for_name("species_20260603_120000") == archived
    with pytest.raises(FileNotFoundError):
        config.model_path_for_name("species_20200101_000000")


# ── promotion carries it ───────────────────────────────────────────────────


def test_promotion_renames_the_sidecar_with_the_weights(
    pipeline_env, tmp_path, monkeypatch
):
    best = tmp_path / "runs" / RUN / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"new")
    mm.write_threshold_sidecar(best, {"threshold": 0.31, "beta": 2.0})
    # An outgoing production model with its own sidecar goes to the archive whole.
    config.pipeline_model_dir.mkdir(parents=True, exist_ok=True)
    old = config.pipeline_model_dir / "species_20260603_120000.pt"
    old.write_bytes(b"old")
    mm.write_threshold_sidecar(old, {"threshold": 0.44, "beta": 2.0})

    dest = rr._promote_model_locally(str(best), "species")

    assert dest == config.pipeline_model_dir / f"{RUN}.pt"
    assert json.loads(mm.threshold_sidecar_path(dest).read_text())["threshold"] == 0.31
    assert not old.exists() and not mm.threshold_sidecar_path(old).exists()
    archived = config.archived_models_dir / old.name
    assert archived.exists()
    assert (
        json.loads(mm.threshold_sidecar_path(archived).read_text())["threshold"] == 0.44
    )
    _set_threshold_setting("model")
    assert mm.resolve_maxn_threshold(config.model_path_for_name(RUN)) == pytest.approx(
        0.31
    )


def test_promotion_without_a_sidecar_warns(pipeline_env, tmp_path, caplog):
    best = tmp_path / "runs" / RUN / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"new")
    with caplog.at_level(logging.WARNING):
        dest = rr._promote_model_locally(str(best), "species")
    assert dest.exists() and not mm.threshold_sidecar_path(dest).exists()
    assert "no MaxN cutoff" in caplog.text
