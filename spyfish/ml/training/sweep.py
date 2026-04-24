"""
sweep.py — Train multiple model variants and compare metrics side-by-side.

Intended for the model-selection phase, before any variant has been promoted
to production. Each entry in SWEEP_RUNS becomes one training run; all runs
share the same dataset so their test-set metrics are directly comparable.

Edit SWEEP_RUNS below to add/remove experiments. Anything in `extra_params`
overrides the built-in WATER_AUG_PARAMS / STABILITY_PARAMS from train.py.

Usage:
    python -m spyfish.ml.training.sweep --data /path/to/data.yaml
"""

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from spyfish.config.wrapper import config
from spyfish.ml.training.evaluate import evaluate_model
from spyfish.ml.training.train import (
    _clear_yolo_cache,
    train_model,
    validate_dataset,
)

# Each run: name (used for output dir) + extra_params passed to model.train().
# imgsz and batch can be overridden per-run via extra_params.
SWEEP_RUNS: List[Dict] = [
    {"name": "baseline", "extra_params": {}},
    {
        "name": "adamw",
        "extra_params": {"optimizer": "AdamW", "lr0": 0.001, "dropout": 0.1},
    },
    {"name": "highres", "extra_params": {"imgsz": 1280, "batch": 4}},
    {
        "name": "adamw_highres",
        "extra_params": {
            "optimizer": "AdamW",
            "lr0": 0.001,
            "dropout": 0.1,
            "imgsz": 1280,
            "batch": 4,
        },
    },
]


def run_sweep(data_yaml: str, base_model_path: str, sweep_dir: Path) -> Path:
    """Train each SWEEP_RUNS entry, evaluate on test split, write comparison CSV."""
    validate_dataset(data_yaml)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for run in SWEEP_RUNS:
        extra = dict(run["extra_params"])
        imgsz = extra.pop("imgsz", config.training_imgsz)
        batch = extra.pop("batch", config.training_batch)

        _clear_yolo_cache(config.local_training_dir)
        try:
            best_pt = train_model(
                data_yaml=data_yaml,
                base_model_path=base_model_path,
                project_dir=sweep_dir,
                run_name=run["name"],
                epochs=config.training_epochs,
                patience=config.training_patience,
                imgsz=imgsz,
                batch=batch,
                extra_params=extra,
            )
        except Exception as e:
            logging.error(f"Run '{run['name']}' failed: {e}")
            results.append({"run": run["name"], "error": str(e)})
            continue

        metrics = evaluate_model(
            model_path=str(best_pt),
            data_yaml=data_yaml,
            split="test",
            imgsz=imgsz,
            output_dir=sweep_dir / f"{run['name']}_eval",
        )
        metrics["run"] = run["name"]
        metrics["imgsz"] = imgsz
        metrics["batch"] = batch
        results.append(metrics)

    _write_comparison_csv(results, sweep_dir / "comparison.csv")
    _log_ranked_summary(results)
    return sweep_dir / "comparison.csv"


def _write_comparison_csv(results: List[dict], csv_path: Path) -> None:
    keys = [
        "run",
        "mAP50",
        "mAP50_95",
        "precision",
        "recall",
        "imgsz",
        "batch",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})
    logging.info(f"Sweep comparison → {csv_path}")


def _log_ranked_summary(results: List[dict]) -> None:
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    ok.sort(key=lambda x: x.get("mAP50", 0.0), reverse=True)

    logging.info("\n=== Sweep results (sorted by mAP@0.5) ===")
    for r in ok:
        logging.info(
            f"  {r['run']:<20} "
            f"mAP50={r['mAP50']:.4f}  "
            f"mAP50-95={r['mAP50_95']:.4f}  "
            f"P={r['precision']:.4f}  "
            f"R={r['recall']:.4f}  "
            f"(imgsz={r['imgsz']}, batch={r['batch']})"
        )
    for r in failed:
        logging.info(f"  {r['run']:<20} FAILED: {r['error']}")
    logging.info("==========================================\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sweep training configs and compare.")
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to YOLO data.yaml (binary or species).",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default=None,
        help="Subdir name under local_training_dir/runs. Defaults to sweep_<timestamp>.",
    )
    args = parser.parse_args()

    base_model_path = config.base_model_path
    if not base_model_path or not base_model_path.exists():
        raise FileNotFoundError(f"Base model missing: {base_model_path}")

    sweep_name = args.sweep_name or f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = config.local_training_dir / "runs" / sweep_name

    run_sweep(
        data_yaml=args.data,
        base_model_path=str(base_model_path),
        sweep_dir=sweep_dir,
    )


if __name__ == "__main__":
    main()
