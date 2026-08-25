"""
evaluate.py. Evaluate trained YOLO models and optionally promote to production.

Workflow:
  1. Run model.val() on the test split → collect mAP@0.5, precision, recall.
  2. Compare new model vs current production model on the same test set.
  3. If improvement ≥ retrain_min_improvement_pct → promote (update config, upload to S3).
  4. Save metrics CSV, confusion matrix image, and training curves to S3.

Usage:
    python -m spyfish.ml.training.evaluate --model /path/to/best.pt --data /path/to/data.yaml --test-txt /path/to/test.txt
    python -m spyfish.ml.training.evaluate --model /path/to/best.pt --data /path/to/data.yaml --promote
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml

from spyfish.config.wrapper import config
from spyfish.utils import validate_model_path


def _resolve_split(data_yaml: str, requested_split: str) -> str:
    """Demote 'test' to 'val' when the dataset has no usable test split.

    Small datasets often can't carve out a viable test split, so
    `assemble_yolo_dataset` writes an empty/missing `test:` entry. YOLO would
    crash; instead we warn (val was used for early stopping, so the metrics
    are optimistic) and proceed.
    """
    if requested_split != "test":
        return requested_split
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    test_path = data.get("test")
    has_test = test_path and Path(test_path).exists() and any(Path(test_path).iterdir())
    if has_test:
        return "test"
    logging.warning(
        f"No test split available in {Path(data_yaml).name}, falling back to val. "
        "Note: val was used for early stopping during training, so these metrics "
        "are optimistic (not a held-out evaluation)."
    )
    return "val"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 640,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run YOLO validation on a dataset split and return key metrics.

    Args:
        model_path: Path to the .pt weights file.
        data_yaml: Path to the data.yaml.
        split: Dataset split to evaluate ('test' or 'val').
        imgsz: Input image size.
        output_dir: Where to save YOLO's output files.

    Returns:
        Dict with mAP50, precision, recall, and the full metrics object.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

    logging.info(f"Evaluating model: {model_path}")
    model_path = str(validate_model_path(model_path))
    eval_split = _resolve_split(data_yaml, split)
    logging.info(f"  data={data_yaml}  split={eval_split}  imgsz={imgsz}")

    model = YOLO(model_path)

    val_kwargs = {
        "data": data_yaml,
        "split": eval_split,
        "imgsz": imgsz,
        "save_json": True,
        "plots": True,
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        val_kwargs["project"] = str(output_dir.parent)
        val_kwargs["name"] = output_dir.name

    metrics = model.val(**val_kwargs)

    result = {
        "model_path": model_path,
        "data_yaml": data_yaml,
        "eval_split": eval_split,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "timestamp": datetime.now().isoformat(),
    }

    logging.info(
        f"Results, mAP@0.5: {result['mAP50']:.4f}  "
        f"precision: {result['precision']:.4f}  "
        f"recall: {result['recall']:.4f}"
    )
    return result


def _class_list_mismatch(model_path: str, data_yaml: str) -> Optional[str]:
    """Describe how a checkpoint's class list differs from a dataset's, or None.

    Compares by id->name, not by count: two 20-class lists in different orders
    are just as unusable as lists of different length, and the alphabetical->
    frozen-roster reorder of 2026-08-22 was exactly that case.
    """
    try:
        from ultralytics import YOLO

        model_names = YOLO(str(model_path)).names
        with open(data_yaml) as f:
            data_names = yaml.safe_load(f).get("names", [])
    except Exception as e:  # unreadable checkpoint or yaml, say so and skip
        return f"could not read class lists ({type(e).__name__}: {e})."

    data_map = dict(enumerate(data_names))
    if model_names == data_map:
        return None
    missing = [(i, n) for i, n in data_map.items() if i not in model_names]
    renamed = [
        (i, model_names[i], n)
        for i, n in data_map.items()
        if i in model_names and model_names[i] != n
    ]
    parts = [f"model has {len(model_names)} classes, dataset has {len(data_map)}."]
    if missing:
        parts.append(
            f"{len(missing)} dataset id(s) unknown to the model, e.g. {missing[:3]}."
        )
    if renamed:
        parts.append(
            f"{len(renamed)} id(s) name a different species, e.g. {renamed[:3]}."
        )
    return " ".join(parts)


def compare_with_production(
    new_metrics: dict,
    production_model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 640,
    output_dir: Optional[Path] = None,
) -> Tuple[dict, bool]:
    """
    Compare new model vs production model on the same test set.

    `output_dir` is used as the production eval's write destination so YOLO
    doesn't fall back to the production checkpoint's embedded `project=`,
    which can point at a stale or cross-project filesystem location.

    Returns:
        (production_metrics, should_promote) where should_promote is True if
        new mAP50 > production mAP50 + min_improvement_pct.
    """
    min_improvement = config.retrain_min_improvement_pct / 100.0

    if not Path(production_model_path).exists():
        logging.warning(
            f"Production model not found at {production_model_path}. "
            f"Assuming this is the first trained model, promoting automatically."
        )
        return {}, True

    # A checkpoint can only be scored against a dataset that means the same
    # thing by each class id. Ultralytics does not check: in ap_per_class it
    # re-keys names with `{i: names[k] for i, k in enumerate(unique_classes)
    # if k in names}`, silently dropping ids the model has no name for, and the
    # per-class plot then indexes off the end - `KeyError: 16`, three frames
    # from anything mentioning classes (run 8598080, 2026-08-23: a 20-class
    # production model against a 50-class dataset). Even without the crash the
    # comparison is meaningless, since id 2 is a different species to each side.
    mismatch = _class_list_mismatch(production_model_path, data_yaml)
    if mismatch:
        logging.warning(
            f"Skipping production comparison: {mismatch} Promotion cannot be "
            f"decided automatically - evaluate both models on a common class "
            f"list before promoting (see scripts comparing checkpoints)."
        )
        return {}, False

    logging.info(f"Evaluating production model for comparison: {production_model_path}")
    prod_metrics = evaluate_model(
        production_model_path,
        data_yaml,
        split=split,
        imgsz=imgsz,
        output_dir=output_dir,
    )

    improvement = new_metrics["mAP50"] - prod_metrics["mAP50"]
    should_promote = improvement >= min_improvement

    logging.info(
        f"\n=== Model comparison ===\n"
        f"  New model mAP@0.5:        {new_metrics['mAP50']:.4f}\n"
        f"  Production model mAP@0.5: {prod_metrics['mAP50']:.4f}\n"
        f"  Improvement:              {improvement:+.4f} (threshold: {min_improvement:+.4f})\n"
        f"  Decision:                 {'✅ PROMOTE' if should_promote else '❌ KEEP EXISTING'}\n"
        f"========================\n"
    )
    return prod_metrics, should_promote


def save_metrics(
    new_metrics: dict,
    prod_metrics: dict,
    output_path: Path,
) -> None:
    """Save metrics comparison as a CSV."""
    rows = []
    for role, m in [("new", new_metrics), ("production", prod_metrics)]:
        if m:
            rows.append(
                {"role": role, **{k: v for k, v in m.items() if k != "timestamp"}}
            )
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved metrics comparison → {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_evaluation_pipeline(
    model_path: str,
    data_yaml: str,
    model_type: str = "species",
    split: str = "test",
) -> dict:
    """
    Full evaluation pipeline: evaluate → compare.

    Args:
        model_path: Path to the new trained model.
        data_yaml: Path to data.yaml.
        model_type: Label used in results-dir and promoted-model naming.
        split: Dataset split to evaluate ('test' or 'val').

    Returns:
        Dict with evaluation results and promotion decision.
    """
    imgsz = config.training_imgsz
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Evaluate new model
    local_results_dir = config.training_results_dir / f"{timestamp}_{model_type}"
    new_metrics = evaluate_model(
        model_path, data_yaml, split=split, imgsz=imgsz, output_dir=local_results_dir
    )

    # Compare with production model
    try:
        production_model_path = str(config.pipeline_model_path)
    except (ValueError, FileNotFoundError):
        production_model_path = ""

    prod_metrics, should_promote = compare_with_production(
        new_metrics,
        production_model_path,
        data_yaml,
        split=split,
        imgsz=imgsz,
        output_dir=local_results_dir / "production_eval",
    )

    # Save metrics CSV
    metrics_csv = local_results_dir / "metrics_comparison.csv"
    save_metrics(new_metrics, prod_metrics, metrics_csv)

    summary = {
        "new_metrics": new_metrics,
        "production_metrics": prod_metrics,
        "should_promote": should_promote,
    }
    logging.info(
        f"Evaluation pipeline complete: {json.dumps({k: v for k, v in new_metrics.items() if isinstance(v, (int, float, str))}, indent=2)}"
    )
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO model and compare with production."
    )
    parser.add_argument(
        "--model", required=True, type=str, help="Path to new model weights (.pt)"
    )
    parser.add_argument(
        "--data", required=True, type=str, help="Path to YOLO data.yaml"
    )
    parser.add_argument("--model-type", type=str, default="species")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = parser.parse_args()

    run_evaluation_pipeline(
        model_path=args.model,
        data_yaml=args.data,
        model_type=args.model_type,
        split=args.split,
    )


if __name__ == "__main__":
    main()
