"""
train.py. Train the species YOLO detection model for Spyfish Aotearoa.

Adapted from yolov12_comparison/train_models.py, with:
  - S3 download of base model weights if not cached locally
  - Water/underwater augmentation params
  - Per-model output to process_files/training/{timestamp}/
  - Upload of trained weights to S3 after training

Usage:
    python -m spyfish.ml.training.train --species-data /path/to/species/data.yaml
"""

import argparse
import gc
import glob
import logging
import os
import shutil
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
    "amp": False,  # Disable AMP, fp16 causes NaN on some underwater datasets
    "box": 5.0,  # Lower bounding box loss penalty (default 7.5)
    # Cache decoded images to DISK, not RAM. cache=True (RAM) OOM-kills under
    # SLURM: Ultralytics sizes the RAM cache from psutil, which reports the whole
    # node's memory (100s of GB) rather than the job's cgroup limit (--mem), so it
    # caches to RAM and blows past the limit. 'disk' is fast across epochs with
    # bounded RAM. (2026-06-03: was True → OOM-killed at --mem=32G.)
    "cache": "disk",
    "workers": 8,  # parallel dataloader processes for epoch 1's disk reads
}

# Class-imbalance handling lives here, not in prepare_training_data.py.
# Trim/oversample were removed, they were destructive (oversample copies whole
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
    Clear YOLO .cache files before training. YOLO gets confused when the
    dataset changed under an unchanged directory structure between runs.
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
        # The dataset snapshot is written into project_dir/run_name before this
        # call, so the dir already exists; without exist_ok Ultralytics would
        # train into "<run_name>2" and leave the snapshot orphaned. Run names
        # are timestamped to the second, so nothing else can collide here.
        "exist_ok": True,
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

    logging.info(f"Training complete, best weights: {best_weights}")
    return best_weights


def freeze_dataset_snapshot(data_yaml: Path, run_dir: Path) -> Optional[Path]:
    """Freeze the exact dataset a model trains on, beside its weights.

    Called BEFORE training starts, not after. Two reasons, and the second is a
    correctness one:

      * A run that dies (SLURM timeout, OOM, node failure) leaves weights or
        partial results with no record of what produced them. Run
        20260821_020313 has no snapshot at all and its val split is now
        unrecoverable, so it cannot be compared against any later checkpoint.
      * ``training/species/`` is a SHARED workspace that ``--retrain
        --data-prep`` deletes and rebuilds. A snapshot taken after a 3-hour
        training run captures whatever is on disk at that moment, which may be
        a completely different dataset. Taken first, it captures what training
        actually opened.

    Writes a self-contained ``dataset/`` snapshot into the run dir (sibling of
    ``weights/``) so ``(model, data)`` is reproducible, you can always answer
    "what was this .pt trained on?". Captures the lean essentials only:
      - ``data.yaml`` (resolved class list)
      - ``class_map.json`` sidecar (if present next to data.yaml)
      - the label ``.txt`` files for each split (tiny; the annotations themselves)
      - ``{split}.txt`` lists of image filenames per split

    Frames are referenced by filename, never copied (they're large and live in
    the deployment tree). Best-effort: logs and returns None on any failure
    rather than failing the training run.
    """
    try:
        data_yaml = Path(data_yaml)
        src_dir = data_yaml.parent  # e.g. .../training/species
        snap = Path(run_dir) / "dataset"
        snap.mkdir(parents=True, exist_ok=True)

        shutil.copy2(data_yaml, snap / "data.yaml")
        sidecar = src_dir / "class_map.json"
        if sidecar.exists():
            shutil.copy2(sidecar, snap / "class_map.json")

        for split in ("train", "val", "test"):
            lbl_src = src_dir / "labels" / split
            img_src = src_dir / "images" / split
            if lbl_src.is_dir():
                dst = snap / "labels" / split
                dst.mkdir(parents=True, exist_ok=True)
                for txt in lbl_src.glob("*.txt"):
                    shutil.copy2(txt, dst / txt.name)
            if img_src.is_dir():
                names = sorted(p.name for p in img_src.iterdir() if p.is_file())
                (snap / f"{split}.txt").write_text("\n".join(names) + "\n")

        logging.info(f"Froze dataset snapshot → {snap}")
        return snap
    except Exception as e:
        logging.warning(f"Could not freeze dataset snapshot: {e}")
        return None


def run_training_pipeline(species_data_yaml: str) -> dict:
    """
    Full training pipeline: train the species model on the local base model.
    """
    local_training_dir = config.local_training_dir

    base_model_path = config.base_model_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Base model must exist locally
    if not base_model_path or not base_model_path.exists():
        logging.error(
            f"Base model weights not found at {base_model_path}. Automatic download is disabled."
        )
        raise FileNotFoundError(f"Base model missing: {base_model_path}")

    logging.info(f"Validating species dataset: {species_data_yaml}")
    validate_dataset(species_data_yaml)
    _clear_yolo_cache(local_training_dir)
    run_dir = local_training_dir / "runs" / f"{timestamp}_species"
    freeze_dataset_snapshot(Path(species_data_yaml), run_dir)
    best_pt = train_model(
        data_yaml=species_data_yaml,
        base_model_path=str(base_model_path),
        project_dir=local_training_dir / "runs",
        run_name=f"{timestamp}_species",
        epochs=config.training_epochs,
        patience=config.training_patience,
        imgsz=config.training_imgsz,
        batch=config.training_batch,
    )
    results = {"species": {"local": str(best_pt)}}

    logging.info(f"\nTraining pipeline complete: {results}")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the species YOLO model.")
    parser.add_argument(
        "--species-data", type=str, default=None, help="Path to species data.yaml"
    )
    args = parser.parse_args()

    species_data = args.species_data or str(
        config.local_training_dir / "species" / "data.yaml"
    )
    run_training_pipeline(species_data_yaml=species_data)


if __name__ == "__main__":
    main()
