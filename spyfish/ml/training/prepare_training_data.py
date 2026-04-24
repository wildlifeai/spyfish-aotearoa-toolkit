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
from typing import Dict, List, Optional, Tuple

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
    """Trim `species` frames until its fraction is ≤ threshold (or per-drop floor blocks more)."""

    def still_over():
        t = df["MaxInterval"].sum()
        s = df[df["ScientificName"] == species]["MaxInterval"].sum()
        return t > 0 and (s / t) > threshold

    if not still_over():
        return df

    species_frames = df[df["ScientificName"] == species][
        ["DropID", "TimeOfMax"]
    ].drop_duplicates()
    drops_by_count = species_frames["DropID"].value_counts()

    frames_removed = 0
    for drop_id in drops_by_count.index:
        if not still_over():
            break

        # Rank this drop's species-frames by diversity ASC (monoculture first)
        drop_species_frames = species_frames[species_frames["DropID"] == drop_id]
        ranked = []
        for _, fr in drop_species_frames.iterrows():
            mask = (df["DropID"] == drop_id) & (df["TimeOfMax"] == fr["TimeOfMax"])
            ranked.append((int(df[mask]["ScientificName"].nunique()), fr["TimeOfMax"]))
        ranked.sort()

        drop_frame_count = df[df["DropID"] == drop_id]["TimeOfMax"].nunique()

        for _, time_of_max in ranked:
            if not still_over() or drop_frame_count <= min_frames_per_drop:
                break
            mask = (df["DropID"] == drop_id) & (df["TimeOfMax"] == time_of_max)
            df = df.drop(index=df[mask].index)
            drop_frame_count -= 1
            frames_removed += 1

    if frames_removed:
        logging.info(f"  Removed {frames_removed} frame(s) from '{species}'")
    elif still_over():
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

        # Resolve label names through the class_map (sidecar preferred over global)
        # so bait/fish bucket aliases route correctly — e.g. "Bait box" → "bait".
        sidecar = (
            annotations_dir / "class_map.json" if annotations_dir.is_dir() else None
        )
        class_map_path = (
            sidecar if sidecar and sidecar.exists() else config.class_map_path
        )
        if not class_map_path.exists():
            logging.warning(
                f"Skipping extra drop {drop_id}: no class_map (sidecar or global) "
                f"available to resolve labels."
            )
            continue
        name_to_id = load_class_map(class_map_path)
        id_to_name = load_class_map_by_id(class_map_path)

        drop_species: set[str] = set()
        raw_suffix = config.biigle_expert_raw_suffix
        raw_csvs = (
            list(annotations_dir.glob(f"*{raw_suffix}"))
            if annotations_dir.is_dir()
            else []
        )
        if raw_csvs:
            df = pd.read_csv(raw_csvs[0])
            for label in (
                df.get("label_name", pd.Series()).dropna().astype(str).unique()
            ):
                cid = name_to_id.get(label) or name_to_id.get(label.strip())
                if cid is None:
                    logging.warning(
                        f"  {drop_id}: label {label!r} not in class_map — dropped."
                    )
                    continue
                drop_species.add(id_to_name[cid])
        else:
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


def copy_split_files(
    drop_ids: List[str],
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    split_name: str,
    symlink: bool = False,
) -> Tuple[int, int]:
    """
    Copy (or symlink) images and their corresponding label files into a
    canonical YOLO dataset layout:

        output_dir/
          images/{split_name}/   ← source JPEGs
          labels/{split_name}/   ← YOLO .txt label files

    Iterates label files for each drop and searches images_dir recursively for
    the matching image. This is robust to any subdirectory structure under images_dir.

    Args:
        drop_ids: List of DropIDs to include in this split.
        images_dir: Source directory containing JPEG frames (root data_quality).
        labels_dir: Source directory containing YOLO .txt label files.
        output_dir: Root output directory.
        split_name: One of 'train', 'val', 'test'.
        symlink: Use symlinks instead of copies.

    Returns:
        (n_images_copied, n_labels_copied)
    """
    img_out = output_dir / "images" / split_name
    lbl_out = output_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

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
            # Search only inside canonical 'frames/' directories — exclude qa_frames,
            # zooniverse_frames, biigle_frames, etc.
            img_path = None
            for ext in config.image_extensions:
                for p in images_dir.rglob(lbl_path.stem + ext):
                    if p.parent.name == "frames":
                        img_path = p
                        break
                if img_path:
                    break

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
            )
        binary_yaml = generate_data_yaml(["fish"], binary_dir)

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
