"""
sweep.py — Train multiple model variants and compare metrics side-by-side.

Intended for the model-selection phase, before any variant has been promoted
to production. Each entry in SWEEP_RUNS becomes one training run; all runs
share the same dataset so their test-set metrics are directly comparable.

For binary sweeps, the current production model is also evaluated on the same
dataset and added as a baseline row — that way the comparison shows whether
any variant actually beats what's deployed. Species sweeps skip the baseline
because the production model is a 1-class fish detector.

Edit SWEEP_RUNS below to add/remove experiments. Anything in `extra_params`
overrides the built-in WATER_AUG_PARAMS / STABILITY_PARAMS from train.py.

Usage:
    python -m spyfish.ml.training.sweep                  # binary + species
    python -m spyfish.ml.training.sweep --binary-only
    python -m spyfish.ml.training.sweep --species-only
    python -m spyfish.ml.training.sweep --no-report      # skip Markdown report
"""

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
]


def _evaluate_production_baseline(data_yaml: str, sweep_dir: Path) -> Optional[dict]:
    """Evaluate the current production model on this dataset, if available."""
    try:
        prod_path = config.pipeline_model_path
    except (ValueError, FileNotFoundError):
        logging.info("No production model configured — skipping baseline.")
        return None

    if not prod_path or not Path(prod_path).exists():
        logging.info(f"Production model not found at {prod_path} — skipping baseline.")
        return None

    logging.info(f"Evaluating production baseline: {prod_path}")
    try:
        result = evaluate_model(
            model_path=str(prod_path),
            data_yaml=data_yaml,
            split="test",
            imgsz=config.training_imgsz,
            output_dir=sweep_dir / "production_baseline_eval",
        )
    except Exception as e:
        logging.warning(f"Production baseline evaluation failed: {e}")
        return None

    result["run"] = "production_baseline"
    result["imgsz"] = config.training_imgsz
    result["batch"] = ""
    return result


def run_sweep(
    data_yaml: str,
    base_model_path: str,
    sweep_dir: Path,
    include_baseline: bool = False,
) -> Path:
    """Train each SWEEP_RUNS entry, evaluate, write comparison CSV.

    If include_baseline, evaluate the current production model on this
    dataset first and prepend its row to comparison.csv.
    """
    validate_dataset(data_yaml)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []

    if include_baseline:
        baseline = _evaluate_production_baseline(data_yaml, sweep_dir)
        if baseline:
            results.append(baseline)

    _clear_yolo_cache(config.local_training_dir)
    for run in SWEEP_RUNS:
        extra = dict(run["extra_params"])
        imgsz = extra.pop("imgsz", config.training_imgsz)
        batch = extra.pop("batch", config.training_batch)

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
        "eval_split",
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
        batch_str = f"batch={r['batch']}" if r.get("batch") != "" else "batch=n/a"
        logging.info(
            f"  {r['run']:<22} "
            f"mAP50={r['mAP50']:.4f}  "
            f"mAP50-95={r['mAP50_95']:.4f}  "
            f"P={r['precision']:.4f}  "
            f"R={r['recall']:.4f}  "
            f"(imgsz={r['imgsz']}, {batch_str}, eval_split={r.get('eval_split', '?')})"
        )
    for r in failed:
        logging.info(f"  {r['run']:<22} FAILED: {r['error']}")
    logging.info("==========================================\n")


def run_sweep_pipeline(
    binary_data_yaml: Optional[str] = None,
    species_data_yaml: Optional[str] = None,
    train_binary: bool = True,
    train_species: bool = True,
    build_reports: bool = True,
) -> dict:
    """Full sweep pipeline: per dataset, run sweep + optional Markdown report.

    Mirrors run_training_pipeline()'s shape so it slots into retrain_runner
    in place of the single-train path.
    """
    from spyfish.ml.training.sweep_report import build_report

    base_model_path = config.base_model_path
    if not base_model_path or not base_model_path.exists():
        raise FileNotFoundError(f"Base model missing: {base_model_path}")
    base_model_str = str(base_model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = config.local_training_dir / "runs"
    results: dict = {}

    if train_binary and binary_data_yaml:
        sweep_dir = runs_root / f"sweep_{timestamp}_binary"
        comparison_csv = run_sweep(
            data_yaml=binary_data_yaml,
            base_model_path=base_model_str,
            sweep_dir=sweep_dir,
            include_baseline=True,
        )
        report_path = build_report(sweep_dir) if build_reports else None
        results["binary"] = {
            "sweep_dir": str(sweep_dir),
            "comparison_csv": str(comparison_csv),
            "report": str(report_path) if report_path else None,
        }

    if train_species and species_data_yaml:
        sweep_dir = runs_root / f"sweep_{timestamp}_species"
        comparison_csv = run_sweep(
            data_yaml=species_data_yaml,
            base_model_path=base_model_str,
            sweep_dir=sweep_dir,
            include_baseline=False,
        )
        report_path = build_report(sweep_dir) if build_reports else None
        results["species"] = {
            "sweep_dir": str(sweep_dir),
            "comparison_csv": str(comparison_csv),
            "report": str(report_path) if report_path else None,
        }

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sweep training configs and compare.")
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--species-only", action="store_true")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip Markdown report generation after sweeps complete.",
    )
    args = parser.parse_args()

    train_binary = not args.species_only
    train_species = not args.binary_only

    local_dir = config.local_training_dir
    binary_yaml = local_dir / "binary" / "data.yaml"
    species_yaml = local_dir / "species" / "data.yaml"

    if train_binary and not binary_yaml.exists():
        logging.error(
            f"Binary data.yaml not found: {binary_yaml}\n"
            "  Run `python run_pipeline.py --retrain` (without --sweep) once first, "
            "or run prepare+split+assemble manually."
        )
        train_binary = False

    if train_species and not species_yaml.exists():
        logging.error(
            f"Species data.yaml not found: {species_yaml}\n"
            "  Run `python run_pipeline.py --retrain` (without --sweep) once first, "
            "or run prepare+split+assemble manually."
        )
        train_species = False

    if not (train_binary or train_species):
        raise SystemExit("Nothing to sweep — both datasets are missing.")

    run_sweep_pipeline(
        binary_data_yaml=str(binary_yaml) if train_binary else None,
        species_data_yaml=str(species_yaml) if train_species else None,
        train_binary=train_binary,
        train_species=train_species,
        build_reports=not args.no_report,
    )


if __name__ == "__main__":
    main()
