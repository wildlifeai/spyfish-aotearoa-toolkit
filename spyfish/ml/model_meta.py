"""What a checkpoint says about itself, and the inference settings that follow from it.

A YOLO checkpoint records the image size it was trained at (``train_args.imgsz``).
Inference has to use that size: a model trained at 1280 and run at 640 sees a
picture a quarter of the size it learned on, and the gain from training at 1280
vanishes without any error. Until 2026-11-10 ``ml_inference.imgsz`` was a bare
number in config with no link to the loaded weights, so promoting a model trained
at another size silently ran it at the wrong one. Now the model is the source of
truth and the config value is an explicit override for experiments.

The MaxN confidence cutoff follows the same rule since 2026-11-12. It is picked
per model on the val split (``spyfish.ml.training.threshold``) and written beside
the weights as ``<model>.thresholds.json``; ``ml_inference.maxn_confidence_threshold``
is ``model`` (read the sidecar) or a number that overrides it with a warning.

``torch`` is imported lazily so the dashboard can import this module without it.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Union

from spyfish.config.wrapper import config

MODEL_IMGSZ = "model"  # the config value meaning "read it from the checkpoint"
MODEL_THRESHOLD = "model"  # the config value meaning "read it from the sidecar"
THRESHOLD_SIDECAR_SUFFIX = ".thresholds.json"


def checkpoint_train_args(model_path: Union[str, Path]) -> dict:
    """The ``train_args`` dict saved in a YOLO checkpoint, ``{}`` when it has none.

    Loads the file once on CPU. A checkpoint that is not a dict (a bare
    ``state_dict``) or has no ``train_args`` yields ``{}``, which callers treat
    as "unknown"; a file that does not load at all raises, like inference would.
    """
    import torch  # heavy, only when a model is actually inspected

    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return {}
    args = ckpt.get("train_args") or {}
    return dict(args) if isinstance(args, dict) else {}


def _normalise_imgsz(imgsz) -> Optional[int]:
    """One int from a ``train_args.imgsz`` value, None when there is none."""
    if imgsz is None:
        return None
    # Ultralytics may store a single int or a [h, w] pair.
    if isinstance(imgsz, (list, tuple)):
        imgsz = max(imgsz)
    return int(imgsz)


def checkpoint_imgsz(model_path: Union[str, Path]) -> Optional[int]:
    """Image size the checkpoint was trained at, or None when it does not say."""
    return _normalise_imgsz(checkpoint_train_args(model_path).get("imgsz"))


# Run dirs from before the model name standard: `{stamp}_{type}`.
_LEGACY_RUN_DIR = re.compile(r"^(20\d{6}(?:_\d{6})?)_([a-z][a-z0-9_]*)$")


def model_name_from_run_dir(
    model_path: Union[str, Path], swap_legacy: bool = False
) -> Optional[str]:
    """The model name a run's weights file gets from its run dir, or None.

    `train.py` writes ``runs/{type}_{stamp}/weights/best.pt``, so the run dir
    (two levels above the file) is the canonical model name. None when that
    dir does not follow the standard (`validation_patterns.model_name`): a
    hand-built path, or a file that is not inside a run.

    ``swap_legacy`` also accepts a run dir from before the standard,
    ``{stamp}_{type}``, and returns it type first with its training stamp
    kept. Promotion wants that (the file is renamed anyway); a test record
    does not, because its name has to match what the annotations DB holds.
    """
    run_dir = Path(model_path).parent.parent.name
    if swap_legacy:
        legacy = _LEGACY_RUN_DIR.match(run_dir)
        if legacy:
            run_dir = f"{legacy.group(2)}_{legacy.group(1)}"
    if config.model_name_stamp(run_dir) is None:
        return None
    return run_dir


def resolve_imgsz(model_path: Union[str, Path], model=None) -> int:
    """Inference image size for ``model_path``: the checkpoint's own, unless config overrides.

    ``ml_inference.imgsz`` is either ``model`` (use the size in the checkpoint)
    or an integer. An integer wins but logs a WARNING when it differs from what
    the checkpoint was trained at, because that mismatch is exactly the mistake
    this function exists to catch. ``model`` with a checkpoint that records no
    size raises: guessing 640 is how the mistake used to happen silently.

    ``model`` (a loaded Ultralytics YOLO object) is accepted so callers that
    already hold one skip the second read of the weights file.
    """
    setting = config.imgsz
    if model is not None and isinstance(getattr(model, "ckpt", None), dict):
        args = model.ckpt.get("train_args") or {}
        trained = _normalise_imgsz(args.get("imgsz"))
    else:
        trained = checkpoint_imgsz(model_path)

    name = Path(str(model_path)).name
    if setting == MODEL_IMGSZ:
        if trained is None:
            raise ValueError(
                f"ml_inference.imgsz is '{MODEL_IMGSZ}' but {name} records no "
                "train_args.imgsz. Set ml_inference.imgsz to the size this model was "
                "trained at, or re-export the checkpoint with its training arguments."
            )
        logging.info(f"Inference imgsz {trained} (from {name})")
        return trained

    chosen = int(setting)
    if trained is not None and trained != chosen:
        logging.warning(
            f"ml_inference.imgsz={chosen} overrides {name}, which was trained at "
            f"{trained}. Boxes will be worse than the model's own metrics suggest; "
            f"set ml_inference.imgsz to '{MODEL_IMGSZ}' unless this is a deliberate experiment."
        )
    else:
        logging.info(f"Inference imgsz {chosen} (ml_inference.imgsz)")
    return chosen


def inference_batch_size(imgsz: int) -> int:
    """Frames per predict() call at ``imgsz``, scaled from the configured size.

    ``ml_inference.batch_size`` is sized for ``ml_inference.batch_size_imgsz``
    (640). GPU memory per frame grows with the square of the side, so the batch
    shrinks by that ratio for a bigger model and never drops below 1. A smaller
    model does not grow the batch: the configured number is a ceiling.
    """
    base = config.ml_batch_size
    ref = config.ml_batch_size_imgsz
    if imgsz <= ref:
        return base
    scaled = max(1, int(base * (ref / imgsz) ** 2))
    logging.info(
        f"Inference batch_size {scaled} (config {base} at imgsz {ref}, model runs at {imgsz})"
    )
    return scaled


# ---------------------------------------------------------------------------
# MaxN confidence cutoff: a property of the checkpoint, kept beside it
# ---------------------------------------------------------------------------


def threshold_sidecar_path(model_path: Union[str, Path]) -> Path:
    """``<weights stem>.thresholds.json`` next to the weights file.

    A run's ``weights/best.pt`` gets ``weights/best.thresholds.json``; the
    promoted copy ``species_<stamp>.pt`` gets ``species_<stamp>.thresholds.json``,
    so promotion renames the sidecar along with the weights.
    """
    p = Path(model_path)
    return p.with_name(p.stem + THRESHOLD_SIDECAR_SUFFIX)


def write_threshold_sidecar(model_path: Union[str, Path], selection: dict) -> Path:
    """Write the cutoff selection (``spyfish.ml.training.threshold``) beside the weights."""
    if "threshold" not in selection:
        raise ValueError("a threshold sidecar needs a 'threshold' key.")
    path = threshold_sidecar_path(model_path)
    path.write_text(json.dumps(selection, indent=2))
    return path


def read_threshold_sidecar(model_path: Union[str, Path]) -> Optional[dict]:
    """The cutoff selection written beside the weights, or None when there is none."""
    path = threshold_sidecar_path(model_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "threshold" not in data:
        raise ValueError(
            f"{path} has no 'threshold' key; re-run the threshold selection."
        )
    return data


def resolve_maxn_threshold(model_path: Union[str, Path]) -> float:
    """The MaxN confidence cutoff to run ``model_path`` at.

    ``ml_inference.maxn_confidence_threshold`` is either ``model`` (the value
    in the checkpoint's sidecar, picked on val at the F-beta peak) or a number.
    A number wins but logs a WARNING when the sidecar says otherwise, because
    a hand-set cutoff is exactly what this exists to replace. ``model`` with
    no sidecar raises and names the command that writes one: a guessed cutoff
    is how 0.40 sat in config for six months with its rationale lost.

    Whatever the source, the cutoff must not be below
    ``ml_inference.confidence_threshold``: the raw CSV holds no detections
    under that, so a lower value would be clipped without anyone noticing.
    """
    setting = config.maxn_confidence_threshold
    floor = config.confidence_threshold
    name = Path(str(model_path)).name
    sidecar = read_threshold_sidecar(model_path)
    from_model = float(sidecar["threshold"]) if sidecar else None

    if setting == MODEL_THRESHOLD:
        if from_model is None:
            raise ValueError(
                f"ml_inference.maxn_confidence_threshold is '{MODEL_THRESHOLD}' but "
                f"{threshold_sidecar_path(model_path).name} does not exist beside {name}. "
                f"Write it with `python -m spyfish.ml.training.threshold --model "
                f"{model_path}` (one val pass), or set a number in config."
            )
        chosen = from_model
        logging.info(
            f"MaxN cutoff {chosen:.3f} (from {threshold_sidecar_path(model_path).name}, "
            f"F{sidecar.get('beta', 1.0):g} on {sidecar.get('selected_on', 'val')})"
        )
    else:
        chosen = float(setting)
        if from_model is not None and abs(chosen - from_model) > 1e-9:
            logging.warning(
                f"ml_inference.maxn_confidence_threshold={chosen} overrides {name}, whose "
                f"sidecar picked {from_model:.3f}. MaxN counts will not match the "
                f"model's own metrics; set it to '{MODEL_THRESHOLD}' unless this is a "
                f"deliberate experiment."
            )
        else:
            logging.info(
                f"MaxN cutoff {chosen:.3f} (ml_inference.maxn_confidence_threshold)"
            )

    if chosen < floor:
        raise ValueError(
            f"MaxN cutoff {chosen:.3f} for {name} is below ml_inference.confidence_threshold "
            f"({floor}): the raw CSV holds no detections under that, so the cutoff would be "
            f"silently clipped. Re-run the threshold selection or lower confidence_threshold "
            f"and re-run inference."
        )
    return chosen
