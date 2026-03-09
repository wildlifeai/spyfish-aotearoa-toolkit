"""
retrain_runner.py — Orchestrator for the Spyfish retraining pipeline.

This module coordinates the full flow:
3. Balance and prepare training data (prepare_training_data).
4. Split data into train/val/test sets (split_data).
5. Train YOLO models (train).
6. Evaluate and optionally promote to production (evaluate).
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, List

from spyfish.config import config
from spyfish.biigle.sync_annotations import sync_biigle_annotations
from spyfish.biigle.biigle_to_yolo import biigle_to_yolo
from spyfish.ml.training.prepare_training_data import prepare_from_annotations, assemble_yolo_dataset
from spyfish.ml.training.split_data import split_data
from spyfish.ml.training.train import run_training_pipeline
from spyfish.ml.training.evaluate import run_evaluation_pipeline


def _promote_model_locally(model_path: str, model_type: str):
    """Promote a model by copying it to the pipeline_model directory."""
    dest_dir = config.pipeline_model_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"promoted_{model_type}.pt"

    logging.info(f"Promoting {model_type} model locally: {model_path} -> {dest_path}")
    shutil.copy2(model_path, dest_path)
    logging.info(f"✅ {model_type.upper()} model promoted locally.")
    return dest_path


def run_retraining(
    project_id: Optional[int] = None,
    volume_id: Optional[int] = None,
    binary_only: bool = False,
    species_only: bool = False,
    auto_promote: bool = False,
    skip_sync: bool = False,
) -> dict:
    """
    Run the full retraining pipeline.
    """
    logging.info("Starting Retraining Pipeline...")

    # 1. Sync Biigle Annotations (updates annotations DB)
    if not skip_sync:
        logging.info("Step 1: Syncing active Biigle annotations to local DB...")
        sync_biigle_annotations()

    # Configuration for retraining
    training_cfg = config.get_section("training")
    local_training_dir = Path(training_cfg.get("local_training_dir", "process_files/training"))
    # The training scripts look for images. Expert frames are usually in data_quality/{drop_id}/biigle_frames/
    images_dir = config.data_quality_dir

    # 2. Export Rectangle annotations from Biigle (essential for YOLO points)
    # We use either the provided IDs or the default project_id from config.
    pid = project_id or config.biigle_project_id
    labels_dir = local_training_dir / "labels_raw"
    class_map_path = local_training_dir / "class_map.json"

    logging.info(f"Step 2: Exporting local annotations to YOLO format...")
    spot_check_dir = local_training_dir / "spot_checks"
    class_map = biigle_to_yolo(
        data_quality_dir=images_dir,
        labels_dir=labels_dir,
        class_map_path=class_map_path,
        spot_check_dir=spot_check_dir
    )

    if not class_map:
        logging.warning("No labels exported from Biigle. Retraining cannot proceed.")
        return {}

    class_names = sorted(class_map.keys(), key=lambda x: class_map[x])

    # 3. Balance and Prepare
    logging.info("Step 3: Balancing annotations and preparing YOLO layout...")
    balanced_df, species_names = prepare_from_annotations()

    # 4. Split
    logging.info("Step 4: Splitting data into train/val/test...")
    train_drops, val_drops, test_drops = split_data(
        balanced_df=balanced_df,
        images_dir=images_dir,
        output_dir=local_training_dir
    )

    # 5. Assemble YOLO layout
    logging.info("Step 5: Assembling final YOLO dataset layout...")
    species_yaml, binary_yaml = assemble_yolo_dataset(
        train_drops=train_drops,
        val_drops=val_drops,
        test_drops=test_drops,
        images_dir=images_dir,
        species_labels_dir=labels_dir,
        output_dir=local_training_dir,
        class_names=species_names,
        build_binary=not species_only
    )

    # 6. Train
    logging.info("Step 6: Training YOLO models...")
    train_results = run_training_pipeline(
        binary_data_yaml=str(binary_yaml) if binary_yaml else None,
        species_data_yaml=str(species_yaml) if species_yaml else None,
        train_binary=not species_only,
        train_species=not binary_only,
        upload_to_s3=False
    )

    # 7. Evaluate & Promote
    eval_results = {}
    if "binary" in train_results:
        logging.info("Step 7a: Evaluating binary model...")
        eval_results["binary"] = run_evaluation_pipeline(
            model_path=train_results["binary"]["local"],
            data_yaml=str(binary_yaml),
            model_type="binary"
        )
        if auto_promote and eval_results["binary"].get("should_promote"):
            _promote_model_locally(train_results["binary"]["local"], "binary")

    if "species" in train_results:
        logging.info("Step 7b: Evaluating species model...")
        eval_results["species"] = run_evaluation_pipeline(
            model_path=train_results["species"]["local"],
            data_yaml=str(species_yaml),
            model_type="species"
        )
        # TODO might not need auto_promote here
        if auto_promote and eval_results["species"].get("should_promote"):
            _promote_model_locally(train_results["species"]["local"], "species")

    logging.info("Retraining Pipeline COMPLETE.")
    return {
        "training": train_results,
        "evaluation": eval_results
    }
