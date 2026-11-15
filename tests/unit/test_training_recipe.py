"""The training recipe (augmentation + stability) lives in config.yaml and reaches YOLO.

Until 2026-11-10 it was two dicts inside train.py. The WIP detector job, written
from config.yaml, never saw them and trained on Ultralytics defaults with AMP on,
so its first result could not be compared with the species model. These tests pin
the single source: every trainer reads ``training.augmentation`` and
``training.stability`` and passes them verbatim.
"""

import ultralytics

from spyfish.config.wrapper import config
from spyfish.ml.training import train as train_mod

EXPECTED_AUG = {
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "fliplr",
    "mosaic",
    "mixup",
    "bgr",
}
EXPECTED_STAB = {
    "warmup_epochs",
    "warmup_bias_lr",
    "nbs",
    "amp",
    "box",
    "cache",
    "workers",
}


def test_recipe_is_read_from_config():
    aug, stab = config.training_augmentation, config.training_stability
    assert EXPECTED_AUG <= set(aug)
    assert EXPECTED_STAB <= set(stab)
    assert (
        stab["amp"] is False
    ), "fp16 training NaN-ed on underwater data; must stay off"
    assert stab["cache"] == "disk", "RAM cache OOM-kills under SLURM"


def test_recipe_properties_return_copies():
    config.training_augmentation["hsv_h"] = 99
    assert config.training_augmentation["hsv_h"] != 99


def test_train_model_passes_the_recipe_verbatim(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_project_root", tmp_path)
    base = config.models_root_dir / "base_model" / "cfd-yolov12x-1.00.pt"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"")
    project_dir = tmp_path / "runs"

    captured = {}

    class FakeYOLO:
        def __init__(self, path):
            captured["weights"] = path

        def train(self, **kwargs):
            captured["kwargs"] = kwargs
            w = project_dir / kwargs["name"] / "weights"
            w.mkdir(parents=True)
            (w / "best.pt").write_bytes(b"")

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

    best = train_mod.train_model(
        data_yaml=str(tmp_path / "data.yaml"),
        base_model_path=str(base),
        project_dir=project_dir,
        run_name="species_20261110_000000",
        epochs=1,
        patience=1,
        imgsz=640,
    )
    assert best.name == "best.pt"
    kw = captured["kwargs"]
    for k, v in {**config.training_augmentation, **config.training_stability}.items():
        assert kw[k] == v, f"{k} not passed verbatim"
    assert kw["amp"] is False


def test_no_recipe_constants_left_in_train_py():
    """A second copy is how the two trainers drifted apart in the first place."""
    assert not hasattr(train_mod, "WATER_AUG_PARAMS")
    assert not hasattr(train_mod, "STABILITY_PARAMS")


def test_cli_recipe_string_for_the_detector_job():
    """What train_detector.sl appends to `yolo detect train`: key=value pairs
    Ultralytics' CLI parses, booleans spelled as Python prints them."""
    recipe = " ".join(
        f"{k}={v}"
        for k, v in {
            **config.training_augmentation,
            **config.training_stability,
        }.items()
    )
    assert "amp=False" in recipe
    assert "cache=disk" in recipe
    assert "hsv_h=0.05" in recipe
    assert "box=5.0" in recipe
