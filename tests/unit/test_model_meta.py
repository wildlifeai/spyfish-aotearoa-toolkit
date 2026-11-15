"""The inference image size comes from the checkpoint; config only overrides.

Before 2026-11-10 ``ml_inference.imgsz`` was a bare number with no link to the
loaded weights, so a model trained at 1280 and promoted into pipeline_model/
would have run at 640 with no error. These tests pin the new contract.
"""

import logging

import pytest
import torch

from spyfish.config.wrapper import config
from spyfish.ml import model_meta
from spyfish.ml.model_meta import (
    checkpoint_imgsz,
    inference_batch_size,
    resolve_imgsz,
)


@pytest.fixture
def ckpt_1280(tmp_path):
    p = tmp_path / "species_20261110_000000.pt"
    torch.save({"train_args": {"imgsz": 1280, "batch": 8}, "model": None}, p)
    return p


@pytest.fixture
def ckpt_no_args(tmp_path):
    p = tmp_path / "species_20260101_000000.pt"
    torch.save({"model": None}, p)
    return p


def _set_imgsz(monkeypatch, value):
    monkeypatch.setattr(type(config), "imgsz", property(lambda self: value))


def test_checkpoint_imgsz_reads_train_args(ckpt_1280, ckpt_no_args):
    assert checkpoint_imgsz(ckpt_1280) == 1280
    assert checkpoint_imgsz(ckpt_no_args) is None


def test_checkpoint_imgsz_accepts_hw_pair(tmp_path):
    p = tmp_path / "species_20261110_000001.pt"
    torch.save({"train_args": {"imgsz": [736, 1280]}}, p)
    assert checkpoint_imgsz(p) == 1280


def test_model_setting_uses_the_checkpoint(ckpt_1280, monkeypatch, caplog):
    _set_imgsz(monkeypatch, "model")
    with caplog.at_level(logging.INFO):
        assert resolve_imgsz(ckpt_1280) == 1280
    assert "from species_20261110_000000.pt" in caplog.text


def test_model_setting_with_silent_checkpoint_raises(ckpt_no_args, monkeypatch):
    _set_imgsz(monkeypatch, "model")
    with pytest.raises(ValueError, match="records no train_args.imgsz"):
        resolve_imgsz(ckpt_no_args)


def test_integer_override_wins_and_warns_on_mismatch(ckpt_1280, monkeypatch, caplog):
    _set_imgsz(monkeypatch, 640)
    with caplog.at_level(logging.WARNING):
        assert resolve_imgsz(ckpt_1280) == 640
    assert "trained at 1280" in caplog.text


def test_integer_override_matching_checkpoint_is_quiet(ckpt_1280, monkeypatch, caplog):
    _set_imgsz(monkeypatch, 1280)
    with caplog.at_level(logging.WARNING):
        assert resolve_imgsz(ckpt_1280) == 1280
    assert "overrides" not in caplog.text


def test_loaded_model_object_is_used_instead_of_rereading(monkeypatch, tmp_path):
    """A YOLO object already in hand carries ``.ckpt``; the file is not re-read."""
    _set_imgsz(monkeypatch, "model")

    class FakeYolo:
        ckpt = {"train_args": {"imgsz": 960}}

    def boom(_):
        raise AssertionError("checkpoint file must not be re-read")

    monkeypatch.setattr(model_meta, "checkpoint_imgsz", boom)
    assert resolve_imgsz(tmp_path / "absent.pt", model=FakeYolo()) == 960


@pytest.mark.parametrize(
    "imgsz, expected",
    [(640, 32), (320, 32), (1280, 8), (960, 14), (2560, 2), (10000, 1)],
)
def test_batch_scales_by_squared_ratio_never_below_one(monkeypatch, imgsz, expected):
    monkeypatch.setattr(type(config), "ml_batch_size", property(lambda s: 32))
    monkeypatch.setattr(type(config), "ml_batch_size_imgsz", property(lambda s: 640))
    assert inference_batch_size(imgsz) == expected


def test_config_imgsz_accepts_model_or_positive_int(monkeypatch):
    base = dict(config.ml_inference)
    for value, expected in (("model", "model"), (" Model ", "model"), (1280, 1280)):
        monkeypatch.setattr(
            type(config),
            "ml_inference",
            property(lambda s, v=value: {**base, "imgsz": v}),
        )
        assert config.imgsz == expected


@pytest.mark.parametrize("bad", ["auto", 0, -640, True])
def test_config_imgsz_rejects_everything_else(monkeypatch, bad):
    base = dict(config.ml_inference)
    monkeypatch.setattr(
        type(config), "ml_inference", property(lambda s: {**base, "imgsz": bad})
    )
    with pytest.raises(ValueError, match="must be 'model' or a positive integer"):
        config.imgsz


def test_model_name_from_run_dir():
    from spyfish.ml.model_meta import model_name_from_run_dir

    best = "/x/runs/species_20260825_053442/weights/best.pt"
    assert model_name_from_run_dir(best) == "species_20260825_053442"
    legacy = "/x/runs/20260825_053442_species/weights/best.pt"
    # a pre-standard run dir is only a name when the caller asks for the swap
    assert model_name_from_run_dir(legacy) is None
    assert (
        model_name_from_run_dir(legacy, swap_legacy=True) == "species_20260825_053442"
    )
    assert model_name_from_run_dir("/x/runs/whatever/weights/best.pt") is None
    assert model_name_from_run_dir("best.pt", swap_legacy=True) is None
