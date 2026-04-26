"""
prepare_training_data.py — Balance and prepare expert annotations for YOLO training.

This module is intentionally split into small, composable functions so each step
can be run and tested independently:

  prepare_from_annotations()   — Load expert annotations, apply trim-dominant + floor.
  make_binary_labels()         — Convert existing multi-class YOLO .txt labels → binary (all → class 0).
  generate_data_yaml()         — Write a YOLO-compatible data.yaml for a given split.
  copy_split_files()           — Copy images + labels into clean train/val/test directory layout.
  compute_species_fractions()  — Utility: per-species fraction of total MaxInterval counts.
  trim_dominant_species()      — Anti-monoculture: trim only the most-dominant species if wildly over fair share.
  apply_floor()                — Remap rare species below threshold → "fish".
  identify_rare_classes()      — Names of species below the oversample threshold.
  oversample_rare_in_train()   — Replicate rare-class frames in train split (no removal).

Typical usage order:
  1. biigle_to_yolo.py         → writes species labels to labels_dir/
  2. prepare_from_annotations()→ balanced df with species list
  3. make_binary_labels()       → writes binary labels to binary_labels_dir/
  4. split_data.py              → assigns drops to train/val/test, writes drop-level .txt files
  5. copy_split_files()         → assembles clean YOLO dataset folders per split
  6. generate_data_yaml()       → writes data.yaml for each of species and binary

Usage (standalone):
    python -m spyfish.ml.training.prepare_training_data --images-dir /path/images --labels-dir /path/labels --output-dir /path/training
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

from spyfish.biigle.class_map import load_class_map, load_class_map_by_id
from spyfish.config.wrapper import config

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_species_fractions(df: pd.DataFrame) -> pd.Series:
    """
    Return per-species fraction of total MaxInterval annotations, sorted descending.

    Args:
        df: Annotation DataFrame with 'ScientificName' and 'MaxInterval' columns.

    Returns:
        pd.Series indexed by ScientificName.
    """
    total = df["MaxInterval"].sum()
    if total == 0:
        return pd.Series(dtype=float)
    return (df.groupby("ScientificName")["MaxInterval"].sum() / total).sort_values(
        ascending=False
    )


def _print_species_summary(df: pd.DataFrame, label: str = "Dataset") -> None:
    """Log a readable per-species summary."""
    fractions = compute_species_fractions(df)
    logging.info(f"\n=== {label} summary ===")
    for species, frac in fractions.items():
        count = int(df[df["ScientificName"] == species]["MaxInterval"].sum())
        logging.info(f"  {species:<40} {frac:.1%}  ({count} intervals)")
    logging.info(f"  Total drops: {df['DropID'].nunique()}  |  Total rows: {len(df)}")
    logging.info("=" * (len(label) + 16) + "\n")


# ---------------------------------------------------------------------------
# Ceiling / floor balancing
# ---------------------------------------------------------------------------


def trim_dominant_species(
    df: pd.DataFrame,
    ceiling_pct: float,
    min_frames_per_drop: Optional[int] = None,
) -> pd.DataFrame:
    """
    Trim ONLY the single most-dominant species, ONLY if it's wildly over fair share.

    Anti-monoculture sanity check, not a balancing pass. Other species — even if
    technically above ceiling_pct — are left alone. Use class oversampling
    (`oversample_rare_in_train`) to give rare species more training signal
    instead of removing dominant-class examples.

    Trigger threshold: `max(ceiling_pct, 2/N)` where N = distinct species count.
      - N=2 species at 50/50: threshold = 100% → never triggers.
      - N=3, ceiling=40%: threshold = 66% → only fires for very dominant species.
      - N=10, ceiling=40%: threshold = 40% → fires whenever any species exceeds ceiling.

    The triggering species is trimmed down to the threshold (not lower).
    Frames are removed from its most-populated drops first; within each drop,
    least-diverse frames go first (monoculture). Per-drop floor is enforced.
    """
    if min_frames_per_drop is None:
        min_frames_per_drop = config.training_min_frames_per_drop

    fractions = compute_species_fractions(df)
    if fractions.empty:
        return df

    n_species = len(fractions)
    threshold = max(ceiling_pct, 2.0 / n_species)
    top_species = fractions.index[0]
    top_frac = float(fractions.iloc[0])

    if top_frac <= threshold:
        logging.info(
            f"No species above trigger threshold ({threshold:.0%} = max of "
            f"ceiling {ceiling_pct:.0%} and 2/N={2 / n_species:.0%} for {n_species} species). "
            f"Top species '{top_species}' at {top_frac:.0%} — leaving dataset untouched."
        )
        return df.reset_index(drop=True)

    logging.info(
        f"Trimming dominant species '{top_species}': {top_frac:.0%} → ≤{threshold:.0%}"
    )
    df = _trim_species_to_threshold(df, top_species, threshold, min_frames_per_drop)
    final_frac = float(compute_species_fractions(df).get(top_species, 0.0))
    logging.info(f"  '{top_species}' now at {final_frac:.0%}")
    return df.reset_index(drop=True)


def _trim_species_to_threshold(
    df: pd.DataFrame,
    species: str,
    threshold: float,
    min_frames_per_drop: int,
) -> pd.DataFrame:
    """Trim `species` frames until its fraction ≤ threshold, respecting per-drop floor.

    Whole frames (DropID, TimeOfMax) are removed atomically. Drops with the most
    species frames are visited first; within a drop, least-diverse frames go first
    so multi-species training signal is preserved.
    """
    species_mask = df["ScientificName"] == species
    total_remaining = float(df["MaxInterval"].sum())
    species_remaining = float(df.loc[species_mask, "MaxInterval"].sum())

    if total_remaining == 0 or species_remaining / total_remaining <= threshold:
        return df

    # Pre-compute per-frame attributes once. Without these, each loop iteration
    # would re-scan df via boolean masks — O(N²) for the trim phase.
    grouped = df.groupby(["DropID", "TimeOfMax"])
    frame_indices = grouped.groups  # (drop, time) → row Index
    frame_intervals = grouped["MaxInterval"].sum().to_dict()
    frame_diversity = grouped["ScientificName"].nunique().to_dict()
    frame_species_intervals = (
        df.loc[species_mask]
        .groupby(["DropID", "TimeOfMax"])["MaxInterval"]
        .sum()
        .to_dict()
    )
    frames_per_drop = df.groupby("DropID")["TimeOfMax"].nunique().to_dict()

    species_frames = df.loc[species_mask, ["DropID", "TimeOfMax"]].drop_duplicates()
    drops_by_count = species_frames["DropID"].value_counts()

    dropped_indices: set = set()
    frames_removed = 0
    for drop_id in drops_by_count.index:
        if species_remaining / total_remaining <= threshold:
            break

        candidates = sorted(
            (frame_diversity[(drop_id, t)], t)
            for t in species_frames.loc[
                species_frames["DropID"] == drop_id, "TimeOfMax"
            ]
        )

        for _, time_of_max in candidates:
            if (
                species_remaining / total_remaining <= threshold
                or frames_per_drop[drop_id] <= min_frames_per_drop
            ):
                break
            key = (drop_id, time_of_max)
            dropped_indices.update(frame_indices[key])
            total_remaining -= frame_intervals[key]
            species_remaining -= frame_species_intervals[key]
            frames_per_drop[drop_id] -= 1
            frames_removed += 1

    if frames_removed:
        logging.info(f"  Removed {frames_removed} frame(s) from '{species}'")
        return df.drop(index=list(dropped_indices))

    if total_remaining > 0 and species_remaining / total_remaining > threshold:
        logging.warning(
            f"  '{species}' still over threshold; limited by "
            f"{min_frames_per_drop}-frame-per-drop floor."
        )
    return df


def identify_rare_classes(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
) -> List[str]:
    """Return species names whose post-balancing fraction is below `threshold`."""
    if threshold is None:
        threshold = config.training_oversample_rare_threshold
    fractions = compute_species_fractions(df)
    return fractions[fractions < threshold].index.tolist()


def oversample_rare_in_train(
    train_images_dir: Path,
    train_labels_dir: Path,
    rare_class_ids: List[int],
    oversample_factor: int,
) -> int:
    """
    Replicate train-split frames containing any rare class.

    For each .txt label file referencing at least one rare class_id, creates
    `oversample_factor` additional (image, label) copies with a `_copyN` suffix
    on the stem. The duplicates are pure file copies — augmentation during
    training (mosaic, HSV, etc.) provides the variation that prevents
    memorization.

    Train split only — never call this on val/test.

    Returns the number of duplicate (image, label) pairs created.
    """
    if not rare_class_ids or oversample_factor < 1:
        return 0

    rare_set = set(rare_class_ids)
    n_dup = 0
    for label_path in sorted(train_labels_dir.glob("*.txt")):
        lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
        has_rare = any(int(ln.split()[0]) in rare_set for ln in lines)
        if not has_rare:
            continue

        stem = label_path.stem
        img_path = None
        for ext in config.image_extensions:
            p = train_images_dir / (stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            logging.debug(f"oversample: no image for label {stem}")
            continue

        for i in range(1, oversample_factor + 1):
            new_stem = f"{stem}_copy{i}"
            new_img = train_images_dir / (new_stem + img_path.suffix)
            new_lbl = train_labels_dir / (new_stem + ".txt")
            if not new_img.exists():
                shutil.copy2(img_path, new_img)
            if not new_lbl.exists():
                shutil.copy2(label_path, new_lbl)
            n_dup += 1

    if n_dup:
        logging.info(
            f"Oversampled rare classes: {n_dup} duplicate (image, label) "
            f"pair(s) added to train split"
        )
    return n_dup


def apply_floor(df: pd.DataFrame, floor_pct: float) -> pd.DataFrame:
    """
    Remap rare species (below floor_pct of total MaxInterval) → 'fish'.
    Applied *once* after the ceiling is stable.

    Rows that share a (DropID, TimeOfMax) with the same merged label are collapsed
    by summing MaxInterval so there are no duplicate frame-species pairs.

    Args:
        df: Expert annotation DataFrame.
        floor_pct: Minimum species fraction; anything below is merged into 'fish'.

    Returns:
        DataFrame with rare species remapped.
    """
    fractions = compute_species_fractions(df)
    rare_species = fractions[fractions < floor_pct].index.tolist()

    if not rare_species:
        logging.info("No species below floor threshold — no remapping needed.")
        return df

    logging.info(
        f"Floor: remapping {len(rare_species)} rare species → 'fish': {rare_species}"
    )
    df = df.copy()
    df.loc[df["ScientificName"].isin(rare_species), "ScientificName"] = "fish"

    # Re-collapse duplicates after merge
    df = df.groupby(
        ["DropID", "ScientificName", "TimeOfMax", "AnnotatedBy"], as_index=False
    )["MaxInterval"].sum()
    return df


# ---------------------------------------------------------------------------
# DB → balanced DataFrame
# ---------------------------------------------------------------------------


def prepare_from_annotations(
    deployment_data_dir: Optional[Path] = None,
    ceiling_pct: Optional[float] = None,
    floor_pct: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load expert annotations from local CSV files, apply trim-dominant + floor balancing.

    Globs for *_biigle_expert_maxn.csv under deployment_data_dir.
    This function is 100% offline and does NOT use the database.

    Args:
        deployment_data_dir: Root directory to search for expert CSVs. Defaults to config.deployment_data_dir.
        ceiling_pct: Anti-monoculture trim threshold. Defaults to config training.class_ceiling_pct.
        floor_pct: Min per-species fraction (rare → 'fish'). Defaults to config training.class_floor_pct.

    Returns:
        (balanced_df, species_class_names)
    """
    ceiling_pct = ceiling_pct or config.training_ceiling_pct
    floor_pct = floor_pct or config.training_floor_pct
    deployment_data_dir = deployment_data_dir or config.deployment_data_dir

    logging.info(f"Loading expert MaxN annotations from {deployment_data_dir}...")
    maxn_glob = f"**/annotations/*{config.biigle_expert_maxn_suffix}"
    all_dfs = []
    for csv_path in deployment_data_dir.glob(maxn_glob):
        logging.debug(f"  Found expert MaxN: {csv_path}")
        all_dfs.append(pd.read_csv(csv_path))

    if not all_dfs:
        raise RuntimeError(
            f"No expert MaxN CSVs found in {deployment_data_dir}. "
            "Run sync_biigle_annotations first."
        )

    df = pd.concat(all_dfs, ignore_index=True)

    # Standardize column naming to match what balancing logic expects
    # (BiigleParser.format_count_annotations_output already does most of this)
    logging.info(
        f"Loaded {len(df)} expert MaxN rows from {df['DropID'].nunique()} drops."
    )

    excluded = config.training_excluded_drops
    if excluded:
        loaded_drops = set(df["DropID"].unique())
        hit = loaded_drops & excluded
        if hit:
            df = df[~df["DropID"].isin(hit)]
            logging.info(
                f"Excluded {len(hit)} drop(s) per {config.training_excluded_drops_file.name}: "
                f"{sorted(hit)}"
            )
        stale = excluded - loaded_drops
        if stale:
            logging.warning(
                f"{len(stale)} drop(s) listed in {config.training_excluded_drops_file.name} "
                f"were not found in loaded annotations (stale entries?): {sorted(stale)}"
            )

    # Trim only the dominant species (anti-monoculture, not full balance)
    df = trim_dominant_species(df, ceiling_pct)

    # Floor: merge rare species into 'fish'
    df = apply_floor(df, floor_pct)

    _print_species_summary(df, label="Balanced dataset")

    species_class_names = sorted(df["ScientificName"].unique().tolist())
    return df, species_class_names


# ---------------------------------------------------------------------------
# Label remapping (unify class IDs across sources)
# ---------------------------------------------------------------------------


def _build_id_remap(
    src_class_map_path: Path,
    unified_names: List[str],
    fallback_species: str = "fish",
) -> Dict[int, Optional[int]]:
    """Build {old_id: new_id} by decoding the source map to species names, then
    looking each up in `unified_names`. Species missing from `unified_names`
    redirect to `fallback_species` if present, otherwise map to None (dropped).
    """
    src_map = load_class_map_by_id(src_class_map_path)
    unified_ids = {name: idx for idx, name in enumerate(unified_names)}
    fallback_id = unified_ids.get(fallback_species)
    return {
        old_id: unified_ids.get(species, fallback_id)
        for old_id, species in src_map.items()
    }


def _rewrite_label_file(
    src_txt: Path, dst_txt: Path, id_remap: Dict[int, Optional[int]]
) -> None:
    """Rewrite a single YOLO .txt through id_remap, dropping lines whose class
    resolves to None."""
    out_lines = []
    for line in src_txt.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        old_id = int(parts[0])
        new_id = id_remap.get(old_id)
        if new_id is None:
            continue
        out_lines.append(f"{new_id} " + " ".join(parts[1:]))
    dst_txt.write_text("\n".join(out_lines))


def discover_extra_drops(deployment_data_dir: Path) -> Tuple[List[str], List[str]]:
    """Find drops with labels + frames but no MaxN CSV (extras from download-volume
    or manually-dropped datasets).

    Returns (drop_ids, extra_species_names). Extras bypass ceiling/floor balancing
    by design; their species are unioned into the unified class list so they get
    their own class IDs rather than falling back to "fish".

    Species detection order:
      1. Read the raw Biigle CSV if present (authoritative label names).
      2. Fall back to decoding YOLO class IDs via a class_map.json sidecar in
         annotations/ — required for Route B (manually-authored labels, no CSV).

    Drops without either are skipped with a warning.
    """
    extras: List[str] = []
    species_set: set[str] = set()

    for labels_dir in deployment_data_dir.glob("**/labels"):
        drop_dir = labels_dir.parent
        drop_id = drop_dir.name
        annotations_dir = drop_dir / "annotations"
        frames_dir = drop_dir / "frames"

        # Skip if MaxN is present — normal pipeline handles this drop.
        maxn_suffix = config.biigle_expert_maxn_suffix
        if annotations_dir.is_dir() and any(annotations_dir.glob(f"*{maxn_suffix}")):
            continue

        label_files = list(labels_dir.glob("*.txt"))
        if not label_files:
            continue
        if not frames_dir.is_dir() or not any(frames_dir.iterdir()):
            continue

        # Two distinct lookup needs:
        #   - raw-CSV path: resolve current label names → use the GLOBAL class_map
        #     (authoritative for current aliases like 'Bait', 'Fish: review required').
        #     Step 1 of retrain (biigle_to_yolo) already rewrote .txt files against
        #     the global, so no historical-ID concern here.
        #   - .txt-only path: decode pre-existing YOLO IDs that were written at some
        #     prior point — prefer a per-drop sidecar if present, else fall back to
        #     the global. Sidecar preserves the original ID→species mapping.
        if not config.class_map_path.exists():
            logging.warning(
                f"Skipping extra drop {drop_id}: global class_map missing at "
                f"{config.class_map_path}."
            )
            continue

        drop_species: set[str] = set()
        raw_suffix = config.biigle_expert_raw_suffix
        raw_csvs = (
            list(annotations_dir.glob(f"*{raw_suffix}"))
            if annotations_dir.is_dir()
            else []
        )
        if raw_csvs:
            name_to_id = load_class_map(config.class_map_path)
            id_to_name = load_class_map_by_id(config.class_map_path)
            # _MANUAL_OVERRIDES guarantees the fish bucket exists in the global map.
            fish_cid = name_to_id["fish"]
            df = pd.read_csv(raw_csvs[0])
            for label in (
                df.get("label_name", pd.Series()).dropna().astype(str).unique()
            ):
                cid = name_to_id.get(label) or name_to_id.get(label.strip())
                if cid is None:
                    logging.warning(
                        f"  {drop_id}: label {label!r} not in class_map — "
                        "routed to 'fish' bucket. Consider adding to _MANUAL_OVERRIDES."
                    )
                    cid = fish_cid
                drop_species.add(id_to_name[cid])
        else:
            sidecar = annotations_dir / "class_map.json"
            decode_path = sidecar if sidecar.exists() else config.class_map_path
            id_to_name = load_class_map_by_id(decode_path)
            for lf in label_files:
                for line in lf.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cid = int(parts[0])
                        except ValueError:
                            continue
                        if cid in id_to_name:
                            drop_species.add(id_to_name[cid])

        extras.append(drop_id)
        species_set.update(drop_species)

    if extras:
        logging.info(
            f"discover_extra_drops: {len(extras)} extras → train split "
            f"(species contributed: {sorted(species_set)})"
        )
    return extras, sorted(species_set)


def flatten_and_remap_labels(
    deployment_data_dir: Path,
    src_class_map_path: Path,
    unified_names: List[str],
    dst_dir: Path,
    fallback_species: str = "fish",
) -> int:
    """Walk per-drop `labels/*.txt`, remap class IDs to the unified ordering
    (index in `unified_names`), write into `dst_dir/<drop_id>/`. Returns file count.

    Per-drop subdir layout decouples the downstream lookup from filename
    prefixes — works for UUID-stemmed labels (e.g. Biigle web-UI uploads)
    that don't naturally start with their drop_id.
    """
    id_remap = _build_id_remap(src_class_map_path, unified_names, fallback_species)
    dst_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for src_txt in deployment_data_dir.glob("**/labels/*.txt"):
        drop_id = src_txt.parent.parent.name
        drop_dst = dst_dir / drop_id
        drop_dst.mkdir(parents=True, exist_ok=True)
        _rewrite_label_file(src_txt, drop_dst / src_txt.name, id_remap)
        n += 1

    logging.info(
        f"flatten_and_remap_labels: {n} files → {dst_dir}/<drop_id>/ "
        f"(unified classes: {len(unified_names)})"
    )
    return n


# ---------------------------------------------------------------------------
# Binary label creation from existing YOLO labels
# ---------------------------------------------------------------------------


def make_binary_labels(
    species_labels_dir: Path,
    binary_labels_dir: Path,
    overwrite: bool = False,
) -> int:
    """
    Convert existing multi-class YOLO .txt label files → binary labels.

    Each annotation's class ID is remapped to 0 (regardless of original class).
    Empty label files (no detections) are preserved as-is.

    This means you can train a binary fish-detector from the same images
    without re-downloading or re-annotating anything.

    Args:
        species_labels_dir: Directory containing multi-class YOLO .txt files.
        binary_labels_dir: Output directory for binary .txt files.
        overwrite: If False, skip files that already exist in binary_labels_dir.

    Returns:
        Number of label files processed.
    """
    binary_labels_dir.mkdir(parents=True, exist_ok=True)

    # species_labels_dir has per-drop subdirs (from flatten_and_remap_labels);
    # mirror that structure in binary_labels_dir so copy_split_files() finds them.
    label_files = list(species_labels_dir.glob("*/*.txt"))
    if not label_files:
        logging.warning(f"No .txt files found in {species_labels_dir}")
        return 0

    processed = 0
    for src_path in label_files:
        drop_id = src_path.parent.name
        dst_drop_dir = binary_labels_dir / drop_id
        dst_drop_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_drop_dir / src_path.name

        if dst_path.exists() and not overwrite:
            continue

        lines = src_path.read_text().strip().splitlines()
        binary_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Replace class_id (index 0) with 0, keep cx cy w h
                binary_lines.append("0 " + " ".join(parts[1:]))
            elif parts:
                logging.debug(f"Unexpected label format in {src_path.name}: {line!r}")

        dst_path.write_text("\n".join(binary_lines))
        processed += 1

    logging.info(
        f"make_binary_labels: wrote {processed} binary label files → {binary_labels_dir}/<drop_id>/"
    )
    return processed


# ---------------------------------------------------------------------------
# Dataset layout helpers
# ---------------------------------------------------------------------------


def _build_image_index(images_dir: Path) -> Dict[str, Path]:
    """Map frame-image stem → path, restricted to canonical `frames/` dirs.

    One walk of `images_dir`; downstream lookups are O(1). Excludes derivative
    dirs (qa_frames, zooniverse_frames, biigle_frames, …) by checking that the
    immediate parent is exactly `frames`.
    """
    exts = {e.lower() for e in config.image_extensions}
    index: Dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and p.parent.name == "frames":
            index.setdefault(p.stem, p)
    return index


def copy_split_files(
    drop_ids: List[str],
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    split_name: str,
    symlink: bool = False,
    image_index: Optional[Dict[str, Path]] = None,
) -> Tuple[int, int]:
    """
    Copy (or symlink) images and their corresponding label files into a
    canonical YOLO dataset layout:

        output_dir/
          images/{split_name}/   ← source JPEGs
          labels/{split_name}/   ← YOLO .txt label files

    Looks up each label's matching image via `image_index` (stem → Path). If
    not provided, the index is built from `images_dir` on entry — pass a
    pre-built index when calling this multiple times against the same
    `images_dir` to avoid redundant tree walks.

    Args:
        drop_ids: List of DropIDs to include in this split.
        images_dir: Source directory containing JPEG frames (root data_quality).
        labels_dir: Source directory containing YOLO .txt label files.
        output_dir: Root output directory.
        split_name: One of 'train', 'val', 'test'.
        symlink: Use symlinks instead of copies.
        image_index: Optional pre-built `{stem: Path}` map; built locally if None.

    Returns:
        (n_images_copied, n_labels_copied)
    """
    img_out = output_dir / "images" / split_name
    lbl_out = output_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    if image_index is None:
        image_index = _build_image_index(images_dir)

    n_images, n_labels = 0, 0

    for drop_id in drop_ids:
        # Labels live in labels_dir/<drop_id>/ after flatten_and_remap_labels.
        drop_labels_dir = labels_dir / drop_id
        drop_labels = (
            list(drop_labels_dir.glob("*.txt")) if drop_labels_dir.is_dir() else []
        )
        if not drop_labels:
            logging.warning(
                f"No label files found for {drop_id} in {labels_dir} — skipping."
            )
            continue

        for lbl_path in drop_labels:
            img_path = image_index.get(lbl_path.stem)
            if img_path is None:
                logging.warning(f"No image found for label {lbl_path.stem} — skipping.")
                continue

            dst_img = img_out / img_path.name
            dst_lbl = lbl_out / lbl_path.name

            if symlink:
                if not dst_img.exists():
                    dst_img.symlink_to(img_path.resolve())
                if not dst_lbl.exists():
                    dst_lbl.symlink_to(lbl_path.resolve())
            else:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)

            n_images += 1
            n_labels += 1

    log_fn = logging.warning if (n_images == 0 and drop_ids) else logging.info
    log_fn(
        f"copy_split_files [{split_name}]: {n_images} images + {n_labels} labels → {output_dir}"
    )
    if n_images == 0 and drop_ids:
        logging.warning(
            f"  No labelled images found for {split_name} split — "
            "check that expert frames exist under images_dir and labels_dir is populated."
        )
    return n_images, n_labels


def generate_data_yaml(
    class_names: List[str],
    output_dir: Path,
    filename: str = "data.yaml",
) -> Path:
    """
    Write a YOLO-compatible data.yaml pointing at the canonical dataset layout
    (images/train, images/val, images/test) relative to output_dir.

    Args:
        class_names: Ordered list of class names (index = YOLO class ID).
        output_dir: Root of the dataset (contains images/ and labels/ subdirs).
        filename: Output YAML filename (default: data.yaml).

    Returns:
        Path to the generated data.yaml.
    """
    data = {
        "nc": len(class_names),
        "names": class_names,
        "train": str(output_dir / "images" / "train"),
        "val": str(output_dir / "images" / "val"),
    }

    test_dir = output_dir / "images" / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        data["test"] = str(test_dir)

    yaml_path = output_dir / filename
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logging.info(f"generate_data_yaml: {len(class_names)} classes → {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Convenience: build a full YOLO dataset from splits + labels in one call
# ---------------------------------------------------------------------------


def _write_sidecar_class_map(
    class_names: List[str],
    output_dir: Path,
    source_class_map_path: Optional[Path] = None,
) -> Path:
    """Write a class_map.json sidecar next to data.yaml, reordered to match class_names.

    Pulls richer metadata (aphia_id, common_name, aliases) from the source class
    map when scientific names match; falls back to minimal entries otherwise.
    """
    source_lookup: Dict[str, dict] = {}
    if source_class_map_path and source_class_map_path.exists():
        registry = json.loads(source_class_map_path.read_text())
        source_lookup = {e["scientific_name"]: e for e in registry.values()}

    sidecar: Dict[str, dict] = {}
    for idx, name in enumerate(class_names):
        src = source_lookup.get(name, {})
        entry = {
            "class_id": idx,
            "aphia_id": src.get("aphia_id"),
            "scientific_name": name,
            "common_name": src.get("common_name", name),
        }
        if "aliases" in src:
            entry["aliases"] = src["aliases"]
        sidecar[str(idx)] = entry

    sidecar_path = output_dir / "class_map.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    logging.info(f"Wrote sidecar class_map → {sidecar_path}")
    return sidecar_path


def _group_drops_by_survey(
    drops: List[str], extra_drops: Set[str]
) -> List[Tuple[str, List[str]]]:
    """Group drops by survey for printout. Extras land in their own '[extras]' bucket.

    Falls back to '[extras]' for any drop whose ID doesn't parse as a valid
    DropID (e.g. 'volume_30505') so unconventional IDs don't crash the print.
    """
    by_survey: Dict[str, List[str]] = {}
    extras_in_split: List[str] = []
    for d in drops:
        if d in extra_drops:
            extras_in_split.append(d)
            continue
        try:
            survey = config.get_survey_id_from_drop(d)
        except Exception:
            extras_in_split.append(d)
            continue
        by_survey.setdefault(survey, []).append(d)
    groups: List[Tuple[str, List[str]]] = sorted(
        (s, sorted(ds)) for s, ds in by_survey.items()
    )
    if extras_in_split:
        groups.append(("[extras]", sorted(extras_in_split)))
    return groups


def print_assembled_summary(
    species_dir: Path,
    class_names: List[str],
    train_drops: List[str],
    val_drops: List[str],
    test_drops: List[str],
    extra_drops: Optional[Set[str]] = None,
) -> None:
    """Comprehensive post-assembly summary: drops per split + bounding-box counts.

    Reflects the FINAL composition the model trains on — includes extras
    (no MaxN row) and oversampled rare-class copies. Two sections:

      1. Drops per split, grouped by survey, with extras tagged separately.
      2. Per-species bounding-box counts read from the on-disk YOLO labels.

    The bounding-box count is a different unit from `print_species_breakdown`
    (which sums MaxInterval) so the totals won't match — bounding boxes are
    raw annotation counts; MaxInterval is a peak-fish-count aggregate.
    """
    extras = set(extra_drops or [])

    # --- Drops per split ---
    logging.info("\n=== Drops per split (assembled training set) ===")
    for split_name, drops in [
        ("TRAIN", train_drops),
        ("VAL", val_drops),
        ("TEST", test_drops),
    ]:
        if not drops:
            logging.info(f"{split_name} (0 drops): empty")
            continue
        groups = _group_drops_by_survey(drops, extras)
        logging.info(f"{split_name} ({len(drops)} drops, {len(groups)} groups):")
        for group_name, ds in groups:
            logging.info(f"  {group_name} ({len(ds)}): {', '.join(ds)}")
    logging.info("=" * 60)

    # --- Per-species bounding-box counts ---
    counts: Dict[str, Dict[str, int]] = {
        name: {"train": 0, "val": 0, "test": 0} for name in class_names
    }
    for split in ("train", "val", "test"):
        labels_dir = species_dir / "labels" / split
        if not labels_dir.is_dir():
            continue
        for txt in labels_dir.glob("*.txt"):
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                if 0 <= cid < len(class_names):
                    counts[class_names[cid]][split] += 1

    df = pd.DataFrame.from_dict(counts, orient="index")[["train", "val", "test"]]
    df["total"] = df.sum(axis=1)
    df = df[df["total"] > 0].sort_values("total", ascending=False)

    if df.empty:
        logging.warning(
            "Assembled YOLO dataset contains no annotations — "
            f"check {species_dir / 'labels'}."
        )
        return

    logging.info(
        "\n=== Per-species bounding-box counts (post-balance, "
        "post-oversample, includes extras) ==="
    )
    logging.info(df.to_string())
    logging.info("=" * 90 + "\n")


def assemble_yolo_dataset(
    train_drops: List[str],
    val_drops: List[str],
    test_drops: List[str],
    images_dir: Path,
    species_labels_dir: Path,
    output_dir: Path,
    class_names: List[str],
    build_binary: bool = True,
    symlink: bool = False,
    source_class_map_path: Optional[Path] = None,
    rare_class_names: Optional[List[str]] = None,
    oversample_factor: Optional[int] = None,
    extra_drops: Optional[Set[str]] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Assemble a complete YOLO dataset (species + optional binary) from pre-split drop lists.

    Creates:
        output_dir/species/{images,labels}/{train,val,test}/   + data.yaml + class_map.json
        output_dir/binary/{images,labels}/{train,val,test}/    + data.yaml  (if build_binary=True)

    Extras (drops discovered via `discover_extra_drops`) should be appended to
    `train_drops` before calling this function — they flow through the normal
    per-drop copy loop since they live under `deployment_data/extra_no_survey_id/`.

    Args:
        train_drops, val_drops, test_drops: Drop ID lists from split_data().
        images_dir: Source directory of JPEG frames.
        species_labels_dir: Directory of multi-class YOLO .txt files (already remapped to `class_names`).
        output_dir: Root output directory.
        class_names: Ordered species class names (unified ID space).
        build_binary: Also build a binary (fish/no-fish) dataset.
        symlink: Use symlinks instead of copying files.
        source_class_map_path: Canonical class_map used as metadata source for the sidecar.

    Returns:
        (species_data_yaml_path, binary_data_yaml_path) — binary path is None if not built.
    """
    species_dir = output_dir / "species"
    binary_dir = output_dir / "binary"

    # Walk images_dir once; reused across all 6 copy_split_files calls below.
    image_index = _build_image_index(images_dir)
    logging.info(
        f"assemble_yolo_dataset: indexed {len(image_index)} frame image(s) under {images_dir}"
    )

    for split_name, drops in [
        ("train", train_drops),
        ("val", val_drops),
        ("test", test_drops),
    ]:
        copy_split_files(
            drops,
            images_dir,
            species_labels_dir,
            species_dir,
            split_name,
            symlink=symlink,
            image_index=image_index,
        )

    # Oversample rare classes in TRAIN split only (species dataset only — binary has 1 class)
    factor = (
        oversample_factor
        if oversample_factor is not None
        else config.training_oversample_factor
    )
    if rare_class_names and factor > 0:
        rare_ids = [class_names.index(n) for n in rare_class_names if n in class_names]
        if rare_ids:
            oversample_rare_in_train(
                species_dir / "images" / "train",
                species_dir / "labels" / "train",
                rare_ids,
                factor,
            )

    species_yaml = generate_data_yaml(class_names, species_dir)
    _write_sidecar_class_map(class_names, species_dir, source_class_map_path)

    binary_yaml = None
    if build_binary:
        binary_labels_dir = output_dir / "binary_labels_staging"
        make_binary_labels(species_labels_dir, binary_labels_dir)
        for split_name, drops in [
            ("train", train_drops),
            ("val", val_drops),
            ("test", test_drops),
        ]:
            copy_split_files(
                drops,
                images_dir,
                binary_labels_dir,
                binary_dir,
                split_name,
                symlink=symlink,
                image_index=image_index,
            )
        binary_yaml = generate_data_yaml(["fish"], binary_dir)

    print_assembled_summary(
        species_dir=species_dir,
        class_names=class_names,
        train_drops=train_drops,
        val_drops=val_drops,
        test_drops=test_drops,
        extra_drops=extra_drops,
    )

    return species_yaml, binary_yaml


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Balance expert annotations and prepare YOLO training data."
    )
    parser.add_argument(
        "--images-dir", required=True, type=Path, help="Source JPEG frames directory"
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        type=Path,
        help="Multi-class YOLO .txt labels directory",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Root output directory"
    )
    parser.add_argument(
        "--ceiling-pct",
        type=float,
        default=None,
        help="Max species fraction (overrides config)",
    )
    parser.add_argument(
        "--floor-pct",
        type=float,
        default=None,
        help="Min species fraction (overrides config)",
    )
    parser.add_argument(
        "--no-binary", action="store_true", help="Skip binary dataset creation"
    )
    parser.add_argument(
        "--symlink", action="store_true", help="Symlink files instead of copying"
    )
    args = parser.parse_args()

    df, species_class_names = prepare_from_annotations(
        ceiling_pct=args.ceiling_pct,
        floor_pct=args.floor_pct,
    )

    # Save balanced CSV for split_data.py
    balanced_csv = args.output_dir / "balanced_annotations.csv"
    balanced_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(balanced_csv, index=False)
    logging.info(f"Saved balanced annotations → {balanced_csv}")
    logging.info(f"Species classes ({len(species_class_names)}): {species_class_names}")
    logging.info(
        "Run split_data.py next to generate train/val/test splits, then assemble_yolo_dataset()."
    )


if __name__ == "__main__":
    main()
