"""
train.py — Train binary and species YOLO detection models for Spyfish Aotearoa.

Adapted from yolov12_comparison/train_models.py, with:
  - S3 download of base model weights if not cached locally
  - Water/underwater augmentation params
  - Per-model output to process_files/training/{timestamp}/
  - Upload of trained weights to S3 after training

Usage:
    python -m spyfish.ml.training.train --binary-data /path/to/binary/data.yaml --species-data /path/to/species/data.yaml
    python -m spyfish.ml.training.train --binary-only
    python -m spyfish.ml.training.train --species-only
"""

import argparse
import gc
import glob
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from spyfish.config.wrapper import config
from spyfish.utils import validate_model_path

# Water/underwater augmentation params (validated in yolov12_comparison experiments)
WATER_AUG_PARAMS = {
    "hsv_h": 0.05,  # hue variance for blue/green shifts
    "hsv_s": 0.8,  # saturation variance for murky vs clear water
    "hsv_v": 0.6,  # value variance for lighting depth changes
    "degrees": 10.0,  # slight rotation
    "fliplr": 0.5,  # horizontal flips
    "mosaic": 1.0,  # combine 4 images
    "mixup": 0.1,  # slight mixup for overlapping
    "bgr": 0.1,  # 10% chance to swap BGR channels
}

# Stability params to prevent NaN during training (validated in yolov12_comparison)
STABILITY_PARAMS = {
    "warmup_epochs": 5.0,
    "warmup_bias_lr": 0.0001,
    "nbs": 64,
    "amp": False,  # Disable AMP — fp16 causes NaN on some underwater datasets
    "box": 5.0,  # Lower bounding box loss penalty (default 7.5)
}

# Class-imbalance handling lives here, not in prepare_training_data.py.
# Trim/oversample were removed — they were destructive (oversample copies whole
# frames; trim throws away annotations). The right place for class balancing in
# YOLO is the loss/sampler, which preserves all data:
#   - `image_weights=True` (Ultralytics arg) → samples images more often when
#     they contain rare classes. Pass via extra_params in the SWEEP_RUNS or
#     here directly. Verify the running ultralytics version supports it.
#   - Per-class loss weighting: subclass the loss or use a custom dataset.
# Defer enabling either until a real training run shows the model collapses on
# tail species. For our current data the tail is so sparse (1–10 examples for
# several species) that floor-merging into 'fish' is doing most of the work.


def _clear_yolo_cache(training_dir: Path) -> None:
    """
    Clear YOLO .cache files before switching between binary and species datasets.
    YOLO gets confused when swapping datasets with the same directory structure.
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for cache_file in glob.glob(str(training_dir / "**" / "*.cache"), recursive=True):
        try:
            os.remove(cache_file)
            logging.debug(f"Removed cache: {cache_file}")
        except OSError:
            pass


def validate_dataset(data_yaml: str) -> None:
    """
    Check that the train and val splits in a YOLO data.yaml are non-empty before training.

    Raises:
        FileNotFoundError: If the data.yaml file doesn't exist.
        ValueError: If train or val split directories are missing or contain no images.
    """
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    issues = []
    for split in ("train", "val"):
        split_dir = data.get(split)
        if not split_dir:
            issues.append(f"  '{split}' key missing from {yaml_path.name}")
            continue
        split_path = Path(split_dir)
        if not split_path.exists():
            issues.append(f"  '{split}' directory does not exist: {split_path}")
            continue
        n_images = sum(
            1
            for p in split_path.rglob("*")
            if p.suffix.lower() in config.image_extensions
        )
        if n_images == 0:
            issues.append(f"  '{split}' has 0 images in {split_path}")
        else:
            logging.info(f"  {split}: {n_images} images found in {split_path}")

    if issues:
        raise ValueError(
            f"Dataset validation failed for {yaml_path}:\n" + "\n".join(issues) + "\n"
            "Ensure prepare_training_data and split_data completed successfully before training."
        )


def train_model(
    data_yaml: str,
    base_model_path: str,
    project_dir: Path,
    run_name: str,
    epochs: int,
    patience: int,
    imgsz: int,
    workers: int = 8,
    batch: int = 16,
    extra_params: Optional[dict] = None,
) -> Path:
    """
    Train a single YOLO model run.

    Args:
        data_yaml: Path to YOLO data.yaml.
        base_model_path: Base model weights (.pt).
        project_dir: Root directory for YOLO training outputs.
        run_name: Name for this training run (YOLO creates project_dir/run_name/).
        epochs: Max training epochs.
        patience: Early-stopping patience.
        imgsz: Input image size.
        workers: DataLoader worker count.
        extra_params: Additional kwargs passed to model.train().

    Returns:
        Path to the best.pt weights file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

    logging.info(f"\n{'='*60}")
    logging.info(
        f"Training: {run_name}  (data={data_yaml}, imgsz={imgsz}, epochs={epochs})"
    )
    logging.info(f"{'='*60}\n")

    params = {
        "data": data_yaml,
        "epochs": epochs,
        "patience": patience,
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers,
        "project": str(project_dir),
        "name": run_name,
        "optimizer": config.training_optimizer,
        "lr0": config.training_lr0,
        "dropout": config.training_dropout,
        **WATER_AUG_PARAMS,
        **STABILITY_PARAMS,
    }
    if extra_params:
        params.update(extra_params)

    base_model_path = str(validate_model_path(base_model_path))
    model = YOLO(base_model_path)
    model.train(**params)

    best_weights = project_dir / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(
            f"Training completed but best.pt not found at {best_weights}"
        )

    logging.info(f"Training complete — best weights: {best_weights}")
    return best_weights


def run_training_pipeline(
    binary_data_yaml: Optional[str] = None,
    species_data_yaml: Optional[str] = None,
    train_binary: bool = True,
    train_species: bool = True,
) -> dict:
    """
    Full training pipeline: train on local base model.
    """
    local_training_dir = config.local_training_dir
    epochs = config.training_epochs
    patience = config.training_patience
    imgsz = config.training_imgsz
    batch = config.training_batch

    base_model_path = config.base_model_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    # Base model must exist locally
    if not base_model_path or not base_model_path.exists():
        logging.error(
            f"Base model weights not found at {base_model_path}. Automatic download is disabled."
        )
        raise FileNotFoundError(f"Base model missing: {base_model_path}")

    base_model_path_str = str(base_model_path)

    # Binary model
    if train_binary and binary_data_yaml:
        logging.info(f"Validating binary dataset: {binary_data_yaml}")
        validate_dataset(binary_data_yaml)
        _clear_yolo_cache(local_training_dir)
        best_pt = train_model(
            data_yaml=binary_data_yaml,
            base_model_path=base_model_path_str,
            project_dir=local_training_dir / "runs",
            run_name=f"{timestamp}_binary",
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
        )
        results["binary"] = {"local": str(best_pt)}

    # Species model
    if train_species and species_data_yaml:
        logging.info(f"Validating species dataset: {species_data_yaml}")
        validate_dataset(species_data_yaml)
        _clear_yolo_cache(local_training_dir)
        best_pt = train_model(
            data_yaml=species_data_yaml,
            base_model_path=base_model_path_str,
            project_dir=local_training_dir / "runs",
            run_name=f"{timestamp}_species",
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
        )
        results["species"] = {"local": str(best_pt)}

    logging.info(f"\nTraining pipeline complete: {results}")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Train binary and/or species YOLO models."
    )
    parser.add_argument(
        "--binary-data", type=str, default=None, help="Path to binary data.yaml"
    )
    parser.add_argument(
        "--species-data", type=str, default=None, help="Path to species data.yaml"
    )
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--species-only", action="store_true")
    args = parser.parse_args()

    train_binary = not args.species_only
    train_species = not args.binary_only

    if not args.binary_data and train_binary:
        local_dir = config.local_training_dir
        args.binary_data = str(local_dir / "binary" / "data.yaml")

    if not args.species_data and train_species:
        local_dir = config.local_training_dir
        args.species_data = str(local_dir / "species" / "data.yaml")

    run_training_pipeline(
        binary_data_yaml=args.binary_data,
        species_data_yaml=args.species_data,
        train_binary=train_binary,
        train_species=train_species,
    )


if __name__ == "__main__":
    main()
