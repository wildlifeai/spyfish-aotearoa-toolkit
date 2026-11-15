"""Pick a model's MaxN confidence cutoff from its validation P/R curves.

A detector's precision and recall both move with the confidence cutoff, so
"the precision of the model" means nothing without the cutoff next to it.
Ultralytics reports P and R at the cutoff where the MEAN F1 over all classes
peaks. Two things are wrong with using that as the pipeline's cutoff:

* F1 weighs a miss and a false alarm the same. In MaxN a missed fish in the
  peak frame lowers the count with nothing downstream to recover it, while a
  spurious box still has to survive the persistence filter and the review
  stages. Recall is worth more here, so the cutoff is picked at the peak of
  F-beta with ``training.threshold_beta`` (2.0: a miss costs about twice a
  false alarm). Beta 1.0 gives Ultralytics' own choice back.
* The cutoff used to be a hand-set number in config
  (``ml_inference.maxn_confidence_threshold``), the same for every model. It
  is a property of one trained checkpoint, like its class list, so it is
  written beside the weights (``<model>.thresholds.json``, see
  ``spyfish.ml.model_meta``) and travels with them on promotion.

Three rules the selection keeps:

1. **Picked on val, reported on test.** The frozen test set is for comparing
   models; a cutoff tuned on it would make the test score optimistic.
   ``test_eval`` reports each class's P/R/F-beta AT the chosen cutoff.
2. **Stored per model.** The next model gets its own; config holds no number
   unless someone is deliberately overriding (and then it warns).
3. **Never below ``ml_inference.confidence_threshold``.** The raw CSV only
   holds detections above that value, so a lower cutoff would be silently
   clipped. The selection clamps to it and says so (``floored``).

Classes in ``ml_inference.maxn_exclude_classes`` (bait) never count toward
MaxN, so they are left out of the mean the cutoff is picked on. Every class
still gets its own row, including the cutoff that would be best for it
alone, so the cost of one global number is visible per species.

CLI (one val pass, writes the sidecar):
    python -m spyfish.ml.training.threshold --model production
    python -m spyfish.ml.training.threshold --model runs/species_20260825_053442/weights/best.pt
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from spyfish.config.wrapper import config
from spyfish.ml.model_meta import (
    checkpoint_imgsz,
    threshold_sidecar_path,
    write_threshold_sidecar,
)
from spyfish.ml.training.yolo_labels import DATA_YAML_FILENAME
from spyfish.utils import validate_model_path

EPS = 1e-16


# ---------------------------------------------------------------------------
# Pure helpers (no ultralytics)
# ---------------------------------------------------------------------------


def fbeta(p, r, beta: float):
    """F-beta of precision ``p`` and recall ``r``, elementwise.

    ``beta`` is how many times more recall matters than precision: 1.0 is F1,
    2.0 counts a miss as roughly twice a false alarm. Zero where both are zero.
    """
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r + EPS)


def _smooth(y: np.ndarray, f: float = 0.1) -> np.ndarray:
    """Box filter over a fraction ``f`` of the curve, as Ultralytics does before
    its own argmax, so one spiky bin does not pick the cutoff."""
    nf = round(len(y) * f * 2) // 2 + 1  # odd width
    pad = np.ones(nf // 2)
    yp = np.concatenate([pad * y[0], y, pad * y[-1]])
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def _index_at(px: np.ndarray, conf: float) -> int:
    """Index of the first curve point at or above ``conf`` (the last point when
    ``conf`` is past the end)."""
    idx = int(np.searchsorted(px, conf, side="left"))
    return min(idx, len(px) - 1)


def _scored_rows(
    px: np.ndarray,
    p_curve: np.ndarray,
    r_curve: np.ndarray,
    class_names: Sequence[str],
    conf: float,
    beta: float,
) -> List[dict]:
    j = _index_at(px, conf)
    fb = fbeta(p_curve, r_curve, beta)
    return [
        {
            "name": name,
            "P": round(float(p_curve[i, j]), 4),
            "R": round(float(r_curve[i, j]), 4),
            "fbeta": round(float(fb[i, j]), 4),
        }
        for i, name in enumerate(class_names)
    ]


def metrics_at(
    px,
    p_curve,
    r_curve,
    class_names: Sequence[str],
    conf: float,
    beta: float,
    exclude: Iterable[str] = (),
) -> dict:
    """P, R and F-beta of every class at one cutoff, plus their means over the
    classes not in ``exclude``. The view of a model AT the cutoff it runs at."""
    px = np.asarray(px, dtype=float)
    p_curve = np.asarray(p_curve, dtype=float)
    r_curve = np.asarray(r_curve, dtype=float)
    rows = _scored_rows(px, p_curve, r_curve, class_names, conf, beta)
    counted = [row for row in rows if row["name"] not in set(exclude)] or rows
    return {
        "precision": round(float(np.mean([row["P"] for row in counted])), 4),
        "recall": round(float(np.mean([row["R"] for row in counted])), 4),
        "fbeta": round(float(np.mean([row["fbeta"] for row in counted])), 4),
        "per_class": rows,
    }


def select_threshold(
    px,
    p_curve,
    r_curve,
    class_names: Sequence[str],
    beta: float,
    floor: float,
    exclude: Iterable[str] = (),
) -> dict:
    """The confidence cutoff at the peak of mean F-beta, clamped to ``floor``.

    ``px`` is the confidence axis (N,), ``p_curve`` and ``r_curve`` are (C, N)
    with one row per entry of ``class_names`` (the order of
    ``metrics.box.ap_class_index``). Classes in ``exclude`` are reported but
    left out of the mean. Returns a JSON-ready dict: the cutoff, whether the
    floor forced it, what plain F1 would have picked, and a per-class table
    at the cutoff with each class's own best cutoff beside it.
    """
    px = np.asarray(px, dtype=float)
    p_curve = np.asarray(p_curve, dtype=float)
    r_curve = np.asarray(r_curve, dtype=float)
    if p_curve.shape != r_curve.shape or p_curve.shape[0] != len(class_names):
        raise ValueError(
            f"curves are {p_curve.shape} and {r_curve.shape} for "
            f"{len(class_names)} classes; expected (classes, points) for both."
        )
    if beta <= 0:
        raise ValueError(f"beta must be positive; got {beta}.")
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1]; got {floor}.")

    excluded = set(exclude)
    counted = [i for i, n in enumerate(class_names) if n not in excluded] or list(
        range(len(class_names))
    )
    fb = fbeta(p_curve, r_curve, beta)
    mean_fb = _smooth(fb[counted].mean(0))
    i_best = int(mean_fb.argmax())
    unfloored = float(px[i_best])
    threshold = max(unfloored, float(floor))
    floored = threshold > unfloored

    mean_f1 = _smooth(fbeta(p_curve, r_curve, 1.0)[counted].mean(0))
    f1_choice = float(px[int(mean_f1.argmax())])

    rows = _scored_rows(px, p_curve, r_curve, class_names, threshold, beta)
    for i, row in enumerate(rows):
        own = _smooth(fb[i])
        row["own_best_threshold"] = round(float(px[int(own.argmax())]), 4)
        row["own_best_fbeta"] = round(float(own.max()), 4)
        row["counted"] = i in counted
    j = _index_at(px, threshold)
    return {
        "threshold": round(threshold, 4),
        "beta": float(beta),
        "floor": float(floor),
        "floored": floored,
        "unfloored_threshold": round(unfloored, 4),
        "f1_threshold": round(f1_choice, 4),
        "mean_fbeta": round(float(fb[counted][:, j].mean()), 4),
        "mean_fbeta_at_f1_threshold": round(
            float(fb[counted][:, _index_at(px, f1_choice)].mean()), 4
        ),
        "excluded_classes": sorted(excluded & set(class_names)),
        "per_class": rows,
    }


def selection_table(selection: dict) -> str:
    """Per-class P / R / F-beta at the cutoff, with each class's own best, for the log."""
    t, b = selection["threshold"], selection["beta"]
    lines = [
        f"{'class':<28}{'P':>8}{'R':>8}{f'F{b:g}':>8}{'own best':>10}{'at':>8}",
    ]
    for row in selection["per_class"]:
        tag = "" if row["counted"] else "  (not in mean)"
        lines.append(
            f"{row['name']:<28}{row['P']:>8.3f}{row['R']:>8.3f}{row['fbeta']:>8.3f}"
            f"{row['own_best_fbeta']:>10.3f}{row['own_best_threshold']:>8.2f}{tag}"
        )
    lines.append(
        f"cutoff {t:.3f} (F{b:g} peak"
        + (
            f", floored from {selection['unfloored_threshold']:.3f}"
            if selection["floored"]
            else ""
        )
        + f"; plain F1 would pick {selection['f1_threshold']:.3f})"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The one function that runs YOLO
# ---------------------------------------------------------------------------


def pick_threshold_on_val(
    model_path,
    data_yaml,
    imgsz: Optional[int] = None,
    out_dir: Optional[Path] = None,
    beta: Optional[float] = None,
    floor: Optional[float] = None,
) -> dict:
    """One ``model.val()`` on the VAL split, pick the cutoff, write the sidecar.

    Val, never test: the frozen test set stays an honest report of the model
    at the cutoff this chose. Returns the selection (also written beside the
    weights by `write_threshold_sidecar`).
    """
    from ultralytics import YOLO

    model_path = Path(validate_model_path(model_path))
    imgsz = imgsz or checkpoint_imgsz(model_path) or config.training_imgsz
    beta = config.threshold_beta if beta is None else float(beta)
    floor = config.confidence_threshold if floor is None else float(floor)
    out_dir = out_dir or (model_path.parent / "threshold_val")

    model = YOLO(str(model_path))
    m = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        plots=False,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        verbose=False,
    )
    names = [model.names[int(ci)] for ci in m.box.ap_class_index]
    selection = select_threshold(
        m.box.px,
        m.box.p_curve,
        m.box.r_curve,
        names,
        beta=beta,
        floor=floor,
        exclude=config.maxn_exclude_classes,
    )
    selection.update(
        {
            "model_path": str(model_path),
            "selected_on": "val",
            "data_yaml": str(data_yaml),
            "imgsz": int(imgsz),
            "selected_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    sidecar = write_threshold_sidecar(model_path, selection)
    logging.info(
        f"MaxN cutoff for {model_path.name}: {selection['threshold']:.3f} "
        f"(F{beta:g} on val, floor {floor}) → {sidecar}\n" + selection_table(selection)
    )
    return selection


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_model(arg: str) -> Path:
    if arg == "production":
        return config.pipeline_model_path
    p = Path(arg)
    return p if p.is_absolute() else config.local_training_dir / p


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model",
        required=True,
        help="weights path (absolute, or relative to local_training_dir), or 'production'",
    )
    ap.add_argument(
        "--data",
        help="data.yaml with a val split (default: the assembled training dataset's)",
    )
    ap.add_argument(
        "--beta",
        type=float,
        help="override training.threshold_beta for this run (recorded in the sidecar)",
    )
    args = ap.parse_args()
    data_yaml = (
        Path(args.data)
        if args.data
        else config.training_dataset_dir / DATA_YAML_FILENAME
    )
    if not data_yaml.exists():
        ap.error(f"{data_yaml} not found; run a data prep or pass --data.")
    model_path = _resolve_model(args.model)
    pick_threshold_on_val(model_path, data_yaml, beta=args.beta)
    print(f"\nwrote {threshold_sidecar_path(model_path)}")


if __name__ == "__main__":
    main()
