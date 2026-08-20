"""
retrain_runner.py. Orchestrator for the Spyfish retraining pipeline.

This module coordinates the full flow:
3. Balance and prepare training data (prepare_training_data).
4. Split data into train/val/test sets (split_data).
5. Train YOLO models (train).
6. Evaluate and optionally promote to production (evaluate).
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from spyfish.biigle.biigle_to_yolo import biigle_to_yolo, draw_frames_on_images
from spyfish.config.wrapper import config
from spyfish.ml.training.evaluate import run_evaluation_pipeline
from spyfish.ml.training.prepare_training_data import (
    _write_sidecar_class_map,
    apply_post_assembly_floor,
    assemble_yolo_dataset,
    discover_extra_drops,
    flatten_and_remap_labels,
    generate_data_yaml,
    prepare_from_annotations,
    print_assembled_summary,
    print_per_drop_species_inventory,
    _write_sidecar_class_map,
)
from spyfish.ml.training.split_data import balance_val_drops, split_data
from spyfish.ml.training.train import run_training_pipeline


def _archive_pipeline_model_dir() -> None:
    """Move any .pt files currently in `pipeline_model_dir` to `archived_models_dir`.

     Called before writing new weights on promotion so the outgoing production
     model is preserved for rollback. Filenames are kept as-is. If a file with
     the same name already exists in the archive (unlikely but possible when
     re-promoting an identical model), the older archived copy is overwritten
    , the currently-deployed model is always more recent and more relevant.
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
    so `best.pt.parent.parent.name` is `{timestamp}_{model_type}`, a unique,
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
        f"using fallback filename 'promoted_{model_type}.pt', "
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
    data_prep: bool = True,
    binary: bool = True,
    species: bool = True,
    auto_promote: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Run the retraining pipeline. Steps are composable, pass any subset of
    `data_prep`, `binary`, `species` to scope the run.

    Defaults run all three. Skipping `data_prep` reuses the existing data.yaml
    on disk (faster iteration on hyperparameter changes). Skipping `binary`
    or `species` runs only the other model. The optimizer / lr / dropout
    used for training come from `config.yaml` (training section).

    `dry_run` runs the FAST part only, flatten labels, build the class map +
    data.yaml, compute the split, print the summary, then stops before the slow
    assembly (image-index walk + symlinking + floor). Use it to sanity-check the
    split and to produce the maps `scripts/wip/suggest_val_drops.py` reads, so
    you can iterate the val balance cheaply before paying for one full assembly.
    """
    logging.info("Starting Retraining Pipeline...")
    logging.info(
        f"Steps: data_prep={data_prep}, binary={binary}, species={species}, "
        f"auto_promote={auto_promote}"
    )

    # Configuration for retraining
    local_training_dir = config.local_training_dir
    images_dir = config.deployment_data_dir

    class_map_path = local_training_dir / "class_map.json"
    labels_staged_dir = local_training_dir / "labels_staged"
    species_yaml: Optional[Path] = local_training_dir / "species" / "data.yaml"
    binary_yaml: Optional[Path] = local_training_dir / "binary" / "data.yaml"

    if not data_prep:
        # Reuse the existing dataset on disk, skip every walk/extract/flatten step.
        logging.info(
            "Skipping data prep (data_prep=False). Reusing existing data.yaml files."
        )
        if binary and not binary_yaml.exists():
            logging.error(
                f"Binary data.yaml not found: {binary_yaml}\n"
                "  Run with --data-prep first, or include binary in a full retrain."
            )
            return {}
        if species and not species_yaml.exists():
            logging.error(
                f"Species data.yaml not found: {species_yaml}\n"
                "  Run with --data-prep first, or include species in a full retrain."
            )
            return {}
        return _train_and_evaluate(
            binary_yaml=binary_yaml if binary else None,
            species_yaml=species_yaml if species else None,
            auto_promote=auto_promote,
        )

    # 1. Generate per-drop YOLO labels from Biigle raw CSVs (uses class_map.json IDs)
    logging.info("Generating per-drop YOLO labels from Biigle expert CSVs...")
    class_map = biigle_to_yolo(
        deployment_data_dir=images_dir,
        class_map_path=class_map_path,
    )

    if not class_map:
        logging.warning("No labels exported from Biigle. Retraining cannot proceed.")
        return {}

    # 2. Balance & prepare, must run before remap so we know the unified class ordering.
    logging.info("Balancing annotations and computing unified class list...")
    balanced_df, species_names = prepare_from_annotations()

    if balanced_df.empty:
        logging.warning(
            "No data remaining after ceiling/floor balancing, retraining cannot proceed. "
            "Check the ceiling threshold or collect more annotations."
        )
        return {}

    # `training_excluded_drops` is consulted by every label-walking helper below
    # so a single exclusion list propagates through floor decisions, extras
    # discovery, and the staged label tree.
    excluded_drops = config.training_excluded_drops

    # 2b. Discover extras (drops under extra_no_survey_id/ without MaxN data).
    extra_drops, extra_species = discover_extra_drops(
        images_dir, excluded_drops=excluded_drops
    )

    # 2c. Build the unified species list. The post-assembly floor (step 7b)
    # decides which species to merge into 'fish' based on actual train image
    # counts after cap + split, so flatten just keeps every species here.
    all_species = set(species_names) | set(extra_species) | {"fish"}
    species_names = sorted(all_species)

    # 3. Flatten + remap class IDs into the unified ordering. Species that
    #    don't survive the post-assembly floor will be merged into 'fish'
    #    after assembly via apply_post_assembly_floor.
    flatten_and_remap_labels(
        deployment_data_dir=images_dir,
        src_class_map_path=class_map_path,
        unified_names=species_names,
        dst_dir=labels_staged_dir,
        excluded_drops=excluded_drops,
    )

    # 3b. Show per-drop bounding-box counts so the user sees imbalance before splits.
    print_per_drop_species_inventory(labels_staged_dir, species_names)

    # 4. Spot-check visualisation uses the remapped labels, reflects what the model actually sees.
    unified_class_map = {name: idx for idx, name in enumerate(species_names)}
    spot_check_dir = local_training_dir / "spot_checks"
    draw_frames_on_images(
        images_dir, labels_staged_dir, unified_class_map, spot_check_dir
    )

    # 5. Filter to drops that have BOTH labels AND images, skip drops missing either.
    # Both checks resolve directly to the canonical per-drop dirs, no tree walk:
    #   labels: labels_staged_dir/<drop_id>/*.txt   (flatten_and_remap_labels writes here)
    #   images: deployment_data_dir/<survey>/<drop>/frames/*.{jpg,…}  (config.get_frames_dir)
    image_exts = set(config.image_extensions)
    _trainable_drops = []
    for drop_id in balanced_df["DropID"].unique():
        drop_labels_dir = labels_staged_dir / drop_id
        has_labels = drop_labels_dir.is_dir() and any(drop_labels_dir.glob("*.txt"))
        drop_frames_dir = config.get_frames_dir(drop_id)
        has_images = drop_frames_dir.is_dir() and any(
            p for p in drop_frames_dir.iterdir() if p.suffix.lower() in image_exts
        )
        if has_labels and has_images:
            _trainable_drops.append(drop_id)
        else:
            logging.warning(
                f"Skipping drop {drop_id}, "
                f"{'no labels' if not has_labels else ''}"
                f"{' and ' if not has_labels and not has_images else ''}"
                f"{'no images' if not has_images else ''} found. "
                "Extract frames and add Rectangle annotations in Biigle before retraining."
            )

    balanced_df = balanced_df[balanced_df["DropID"].isin(_trainable_drops)]
    if balanced_df.empty:
        logging.error(
            "No drops have both labels and images, retraining cannot proceed.\n"
            f"  Labels dir: {labels_staged_dir}\n"
            f"  Images dir: {images_dir}\n"
            "Ensure frame extraction and Biigle Rectangle annotation export have both run."
        )
        return {}
    logging.info(
        f"  {len(_trainable_drops)} drops ready for training: {_trainable_drops}"
    )

    # 6. Split. Extras participate in the survey-aware split alongside MaxN
    # drops so `force_val_drops` / `force_train_drops` and the per-survey
    # val_pct apply to them too. Canonical-ID extras (BNP_*, SLI_*, TUH_*) group
    # with their real survey; volume_<id> extras group under UNKNOWN_SURVEY.
    # Without forcing, the splitter pulls ~val_pct of each group into val,
    # essential for species diversity in val when extras dominate the dataset.
    logging.info("Splitting data into train/val/test...")
    if config.training_auto_balance_val:
        # Species-balanced val: pick whole drops so each multi-source species
        # hits ~val_balance_pct of its boxes (the suggester, inline). Decodes
<<<<<<< HEAD
        # labels via the in-memory species_names — no class_map.json drift.
=======
        # labels via the in-memory species_names, no class_map.json drift.
            labels_staged_dir=labels_staged_dir,
            candidate_drops=candidate_drops,
            val_pct=config.training_val_balance_pct,
            tolerance=config.training_val_balance_tolerance,
            force_val=config.training_force_val_drops,
            force_train=config.training_force_train_drops,
        )
    else:
        train_drops, val_drops, test_drops = split_data(
            balanced_df=balanced_df,
            images_dir=images_dir,
            output_dir=local_training_dir,
            extra_drop_ids=list(extra_drops),
        )

    if dry_run:
        # Fast path: write just the maps the suggester needs (the 59-class
        # decoder + data.yaml), skip the slow assembly (image-index walk +
        # symlink + floor). Lets you iterate the val balance cheaply, then pay
        # for one full assembly on the split you actually want.
        species_dir = local_training_dir / "species"
        species_dir.mkdir(parents=True, exist_ok=True)
        generate_data_yaml(species_names, species_dir)
        _write_sidecar_class_map(species_names, species_dir, class_map_path)
        logging.info(
<<<<<<< HEAD
            "DRY RUN — wrote class_map.json + data.yaml and stopped before "
=======
            "DRY RUN, wrote class_map.json + data.yaml and stopped before "
>>>>>>> 98206891a2c9cfebafc70d233559d5135a50627f
            "assembly. Run scripts/wip/suggest_val_drops.py to plan val, then "
            "re-run --data-prep (without --dry-run) for the full dataset."
        )
        return {"dry_run_complete": True}

    # 7. Assemble YOLO layout.
    logging.info("Assembling final YOLO dataset layout...")
    species_yaml, binary_yaml = assemble_yolo_dataset(
        train_drops=train_drops,
        val_drops=val_drops,
        test_drops=test_drops,
        images_dir=images_dir,
        species_labels_dir=labels_staged_dir,
        output_dir=local_training_dir,
        class_names=species_names,
        build_binary=binary,
        source_class_map_path=class_map_path,
        extra_drops=set(extra_drops),
    )

    # 7b. Post-assembly floor, re-merge any class whose actual train image count
    # (after cap + split) is below class_floor_min_images. The source-level floor
    # works on pre-cap counts; this catches classes that pass that floor but lose
    # most of their frames to per-drop capping or splitter quirks. Always runs
    # when the species dataset was assembled, the data.yaml on disk reflects
    # what training will see, so the floor needs to run regardless of whether
    # the user requested species training in this same invocation. Re-prints the
    # assembled summary so the user sees the final post-floor composition (the
    # one printed inside assemble_yolo_dataset reflects the pre-floor state).
    if species_yaml:
        species_dir = species_yaml.parent
        merged, post_floor_class_names = apply_post_assembly_floor(
            species_dir=species_dir,
            min_images=config.training_floor_min_images,
        )
        if merged:
            logging.info("Re-printing summary after post-assembly floor merge:")
            print_assembled_summary(
                species_dir=species_dir,
                class_names=post_floor_class_names,
                train_drops=train_drops,
                val_drops=val_drops,
                test_drops=test_drops,
                extra_drops=set(extra_drops),
            )

    if not (binary or species):
        logging.info(
            "Data prep complete; binary=False and species=False so skipping training."
        )
        return {"data_prep_complete": True}

    return _train_and_evaluate(
        binary_yaml=binary_yaml if binary else None,
        species_yaml=species_yaml if species else None,
        auto_promote=auto_promote,
    )


def _train_and_evaluate(
    binary_yaml: Optional[Path],
    species_yaml: Optional[Path],
    auto_promote: bool,
) -> dict:
    """Train the requested models, evaluate each, optionally promote on improvement."""
    logging.info("Training YOLO models...")
    train_results = run_training_pipeline(
        binary_data_yaml=str(binary_yaml) if binary_yaml else None,
        species_data_yaml=str(species_yaml) if species_yaml else None,
        train_binary=binary_yaml is not None,
        train_species=species_yaml is not None,
    )

    eval_results = {}
    if "binary" in train_results:
        logging.info("Evaluating binary model...")
        eval_results["binary"] = run_evaluation_pipeline(
            model_path=train_results["binary"]["local"],
            data_yaml=str(binary_yaml),
            model_type="binary",
        )
        if auto_promote and eval_results["binary"].get("should_promote"):
            _promote_model_locally(train_results["binary"]["local"], "binary")

    if "species" in train_results:
        logging.info("Evaluating species model...")
        eval_results["species"] = run_evaluation_pipeline(
            model_path=train_results["species"]["local"],
            data_yaml=str(species_yaml),
            model_type="species",
        )
        if auto_promote and eval_results["species"].get("should_promote"):
            _promote_model_locally(train_results["species"]["local"], "species")

    logging.info("Retraining Pipeline COMPLETE.")
    return {"training": train_results, "evaluation": eval_results}
