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

from spyfish.biigle.biigle_to_yolo import biigle_to_yolo, draw_frames_on_images
from spyfish.config.wrapper import config
from spyfish.ml.training.evaluate import run_evaluation_pipeline
from spyfish.ml.training.prepare_training_data import (
    assemble_yolo_dataset,
    prepare_from_annotations,
)
from spyfish.ml.training.split_data import split_data
from spyfish.ml.training.train import run_training_pipeline


def _archive_pipeline_model_dir() -> None:
    """Move any .pt files currently in `pipeline_model_dir` to `archived_models_dir`.

    Called before writing new weights on promotion so the outgoing production
    model is preserved for rollback. Filenames are kept as-is. If a file with
    the same name already exists in the archive (unlikely but possible when
    re-promoting an identical model), the older archived copy is overwritten
    — the currently-deployed model is always more recent and more relevant.
    """
    src_dir = config.pipeline_model_dir
    if not src_dir.exists():
        return
    pt_files = list(src_dir.glob("*.pt"))
    if not pt_files:
        return

    archive_dir = config.archived_models_dir
    archive_dir.mkdir(parents=True, exist_ok=True)

    for pt in pt_files:
        dest = archive_dir / pt.name
        logging.info(f"Archiving previous production model: {pt} -> {dest}")
        shutil.move(str(pt), str(dest))


def _derive_promoted_filename(model_path: str, model_type: str) -> str:
    """Derive the promoted filename from the training run directory.

    `train.py` writes best.pt to `runs/{timestamp}_{model_type}/weights/best.pt`,
    so `best.pt.parent.parent.name` is `{timestamp}_{model_type}` — a unique,
    human-readable identifier that already encodes when the model was trained.

    We reuse that directly as the promoted filename so:
      - every promotion produces a distinct file name
      - archive collisions are effectively impossible
      - you can cross-reference `results.csv` in the training run dir for metrics

    If the path doesn't match the expected structure (e.g. a caller passes a
    hand-constructed path), fall back to `promoted_{model_type}.pt` and log.
    """
    try:
        run_dir_name = Path(model_path).parent.parent.name  # {timestamp}_{model_type}
        if run_dir_name and model_type in run_dir_name:
            return f"{run_dir_name}.pt"
    except Exception:
        pass
    logging.warning(
        f"Could not derive training run name from {model_path!r}; "
        f"using fallback filename 'promoted_{model_type}.pt' — "
        "note this will overwrite any prior fallback-named promotion."
    )
    return f"promoted_{model_type}.pt"


def _promote_model_locally(model_path: str, model_type: str):
    """Promote a model by copying it to the pipeline_model directory.

    The current contents of `pipeline_model_dir` are moved to
    `archived_models_dir` first so the outgoing model stays recoverable.
    """
    _archive_pipeline_model_dir()

    dest_dir = config.pipeline_model_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _derive_promoted_filename(model_path, model_type)

    logging.info(f"Promoting {model_type} model locally: {model_path} -> {dest_path}")
    shutil.copy2(model_path, dest_path)
    logging.info(f"✅ {model_type.upper()} model promoted locally as {dest_path.name}.")
    return dest_path


def run_retraining(
    binary_only: bool = False,
    species_only: bool = False,
    auto_promote: bool = False,
) -> dict:
    """
    Run the full retraining pipeline.
    """
    logging.info("Starting Retraining Pipeline...")

    # Configuration for retraining
    local_training_dir = config.local_training_dir
    # The training scripts look for images. Expert frames are usually in data_quality/{drop_id}/biigle_frames/
    images_dir = config.data_quality_dir

    # 2. Export Rectangle annotations from Biigle (essential for YOLO points)
    labels_dir = local_training_dir / "labels_raw"
    class_map_path = local_training_dir / "class_map.json"

    logging.info("Step 2a: Generating per-drop YOLO labels from Biigle expert CSVs...")
    class_map = biigle_to_yolo(
        data_quality_dir=images_dir,
        class_map_path=class_map_path,
    )

    if not class_map:
        logging.warning("No labels exported from Biigle. Retraining cannot proceed.")
        return {}

    logging.info("Step 2b: Collecting per-drop labels into training staging area...")
    labels_dir.mkdir(parents=True, exist_ok=True)
    collected = 0
    for txt in images_dir.glob("**/annotations/*.txt"):
        shutil.copy2(txt, labels_dir / txt.name)
        collected += 1
    logging.info(f"  Collected {collected} label files → {labels_dir}")

    spot_check_dir = local_training_dir / "spot_checks"
    draw_frames_on_images(images_dir, labels_dir, class_map, spot_check_dir)

    # 3. Balance and Prepare
    logging.info("Step 3: Balancing annotations and preparing YOLO layout...")
    balanced_df, species_names = prepare_from_annotations()

    if balanced_df.empty:
        logging.warning(
            "No data remaining after ceiling/floor balancing — retraining cannot proceed. "
            "Check the ceiling threshold or collect more annotations."
        )
        return {}

    # 3b. Filter to drops that have BOTH labels AND images — skip drops missing either.
    _image_exts = {".jpg", ".jpeg", ".png"}
    _trainable_drops = []
    for drop_id in balanced_df["DropID"].unique():
        has_labels = any(labels_dir.glob(f"{drop_id}*.txt"))
        has_images = any(
            p
            for p in images_dir.rglob(f"{drop_id}*")
            if p.suffix.lower() in _image_exts
        )
        if has_labels and has_images:
            _trainable_drops.append(drop_id)
        else:
            logging.warning(
                f"Skipping drop {drop_id} — "
                f"{'no labels' if not has_labels else ''}"
                f"{' and ' if not has_labels and not has_images else ''}"
                f"{'no images' if not has_images else ''} found. "
                "Extract frames and add Rectangle annotations in Biigle before retraining."
            )

    balanced_df = balanced_df[balanced_df["DropID"].isin(_trainable_drops)]
    if balanced_df.empty:
        logging.error(
            "No drops have both labels and images — retraining cannot proceed.\n"
            f"  Labels dir: {labels_dir}\n"
            f"  Images dir: {images_dir}\n"
            "Ensure frame extraction (step 3) and Biigle Rectangle annotation export have both run."
        )
        return {}
    logging.info(
        f"  {len(_trainable_drops)} drops ready for training: {_trainable_drops}"
    )

    # 4. Split
    logging.info("Step 4: Splitting data into train/val/test...")
    train_drops, val_drops, test_drops = split_data(
        balanced_df=balanced_df, images_dir=images_dir, output_dir=local_training_dir
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
        build_binary=not species_only,
    )

    # 6. Train
    logging.info("Step 6: Training YOLO models...")
    train_results = run_training_pipeline(
        binary_data_yaml=str(binary_yaml) if binary_yaml else None,
        species_data_yaml=str(species_yaml) if species_yaml else None,
        train_binary=not species_only,
        train_species=not binary_only,
    )

    # 7. Evaluate & Promote
    eval_results = {}
    if "binary" in train_results:
        logging.info("Step 7a: Evaluating binary model...")
        eval_results["binary"] = run_evaluation_pipeline(
            model_path=train_results["binary"]["local"],
            data_yaml=str(binary_yaml),
            model_type="binary",
        )
        if auto_promote and eval_results["binary"].get("should_promote"):
            _promote_model_locally(train_results["binary"]["local"], "binary")

    if "species" in train_results:
        logging.info("Step 7b: Evaluating species model...")
        eval_results["species"] = run_evaluation_pipeline(
            model_path=train_results["species"]["local"],
            data_yaml=str(species_yaml),
            model_type="species",
        )
        # TODO: decide whether the species model should use the same auto_promote threshold
        # as the binary model, or require a separate manual promotion review.
        if auto_promote and eval_results["species"].get("should_promote"):
            _promote_model_locally(train_results["species"]["local"], "species")

    logging.info("Retraining Pipeline COMPLETE.")
    return {"training": train_results, "evaluation": eval_results}
