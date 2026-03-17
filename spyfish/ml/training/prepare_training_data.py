"""
prepare_training_data.py — Balance and prepare expert annotations for YOLO training.

This module is intentionally split into small, composable functions so each step
can be run and tested independently:

  prepare_from_annotations()   — Load expert annotations from DB, apply ceiling/floor balancing.
  make_binary_labels()         — Convert existing multi-class YOLO .txt labels → binary (all → class 0).
  generate_data_yaml()         — Write a YOLO-compatible data.yaml for a given split.
  copy_split_files()           — Copy images + labels into clean train/val/test directory layout.
  compute_species_fractions()  — Utility: per-species fraction of total MaxInterval counts.
  apply_ceiling()              — Remove least-diverse frames for over-represented species (iterative).
  apply_floor()                — Remap rare species below threshold → "fish".

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
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import yaml

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


def apply_ceiling(
    df: pd.DataFrame,
    ceiling_pct: float,
    max_iterations: int = 3,
) -> pd.DataFrame:
    """
    Iteratively remove least-diverse frames for over-represented species.

    'Least diverse' = frames where the fewest other species are co-present.
    Removal is at the *frame* level (DropID + TimeOfMax) — individual bboxes
    cannot be removed without removing the whole frame.

    Args:
        df: Expert annotation DataFrame with 'ScientificName', 'MaxInterval',
            'DropID', 'TimeOfMax' columns.
        ceiling_pct: Maximum allowed fraction for any species (e.g. 0.40).
        max_iterations: Safety cap to prevent infinite loops.

    Returns:
        Filtered DataFrame with ceiling applied.
    """
    for iteration in range(1, max_iterations + 1):
        fractions = compute_species_fractions(df)
        over_ceiling = fractions[fractions > ceiling_pct]

        if over_ceiling.empty:
            logging.info(f"Ceiling satisfied after {iteration - 1} iteration(s).")
            break

        logging.info(
            f"Ceiling iteration {iteration}: over-represented: {over_ceiling.to_dict()}"
        )

        for species, fraction in over_ceiling.items():
            species_frames = df[df["ScientificName"] == species][
                ["DropID", "TimeOfMax"]
            ].drop_duplicates()

            # Score each frame by the number of distinct species it contains (diversity)
            diversity_scores = []
            for _, frame_row in species_frames.iterrows():
                frame_mask = (df["DropID"] == frame_row["DropID"]) & (
                    df["TimeOfMax"] == frame_row["TimeOfMax"]
                )
                diversity_scores.append(
                    {
                        "DropID": frame_row["DropID"],
                        "TimeOfMax": frame_row["TimeOfMax"],
                        "diversity": int(df[frame_mask]["ScientificName"].nunique()),
                    }
                )

            diversity_df = pd.DataFrame(diversity_scores).sort_values("diversity")

            total = df["MaxInterval"].sum()
            species_total = df[df["ScientificName"] == species]["MaxInterval"].sum()
            excess_intervals = max(0, species_total - ceiling_pct * total)
            if excess_intervals <= 0:
                continue

            # Remove least-diverse frames until enough intervals are shed
            removed = 0
            indices_to_drop = []
            for _, frame_row in diversity_df.iterrows():
                frame_mask = (df["DropID"] == frame_row["DropID"]) & (
                    df["TimeOfMax"] == frame_row["TimeOfMax"]
                )
                frame_interval_sum = int(df[frame_mask]["MaxInterval"].sum())
                indices_to_drop.extend(df[frame_mask].index.tolist())
                removed += frame_interval_sum
                if removed >= excess_intervals:
                    break

            if indices_to_drop:
                df = df.drop(index=indices_to_drop)
                logging.info(
                    f"  Removed {len(indices_to_drop)} rows (~{removed} intervals) "
                    f"from '{species}' (was {fraction:.1%})"
                )
    else:
        logging.warning(
            f"Ceiling not fully resolved after {max_iterations} iterations. "
            "Consider increasing ceiling_max_iterations in config.yaml."
        )

    return df.reset_index(drop=True)


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
    data_quality_dir: Optional[Path] = None,
    ceiling_pct: Optional[float] = None,
    floor_pct: Optional[float] = None,
    ceiling_max_iterations: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load expert annotations from local CSV files, apply ceiling + floor balancing.

    Globs for *_biigle_expert_maxn.csv in process_files/data_quality/.
    This function is 100% offline and does NOT use the database.

    Args:
        data_quality_dir: Root directory to search for expert CSVs. Defaults to config.data_quality_dir.
        ceiling_pct: Max per-species fraction. Defaults to config training.class_ceiling_pct.
        floor_pct: Min per-species fraction. Defaults to config training.class_floor_pct.
        ceiling_max_iterations: Safety cap. Defaults to config training.ceiling_max_iterations.

    Returns:
        (balanced_df, species_class_names)
    """
    training_cfg = config.get_section("training")
    ceiling_pct = ceiling_pct or config.training_ceiling_pct
    floor_pct = floor_pct or config.training_floor_pct
    ceiling_max_iterations = ceiling_max_iterations or config.training_ceiling_max_iterations
    data_quality_dir = data_quality_dir or config.data_quality_dir

    logging.info(f"Loading expert MaxN annotations from {data_quality_dir}...")
    all_dfs = []
    for csv_path in data_quality_dir.glob("**/annotations/*_biigle_expert_maxn.csv"):
        logging.debug(f"  Found expert MaxN: {csv_path}")
        all_dfs.append(pd.read_csv(csv_path))

    if not all_dfs:
        raise RuntimeError(
            f"No expert MaxN CSVs found in {data_quality_dir}. "
            "Run sync_biigle_annotations first."
        )

    df = pd.concat(all_dfs, ignore_index=True)

    # Standardize column naming to match what balancing logic expects
    # (BiigleParser.format_count_annotations_output already does most of this)
    logging.info(
        f"Loaded {len(df)} expert MaxN rows from {df['DropID'].nunique()} drops."
    )

    # Ceiling first
    df = apply_ceiling(df, ceiling_pct, max_iterations=ceiling_max_iterations)

    # Floor after ceiling is stable
    df = apply_floor(df, floor_pct)

    _print_species_summary(df, label="Balanced dataset")

    species_class_names = sorted(df["ScientificName"].unique().tolist())
    return df, species_class_names


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

    label_files = list(species_labels_dir.glob("*.txt"))
    if not label_files:
        logging.warning(f"No .txt files found in {species_labels_dir}")
        return 0

    processed = 0
    for src_path in label_files:
        dst_path = binary_labels_dir / src_path.name

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
        f"make_binary_labels: wrote {processed} binary label files → {binary_labels_dir}"
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

    Only images whose stem matches a label file are included.
    Targets frames specifically in {drop_id}/biigle_frames/ subdirectories.

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
        # Target only the biigle_frames folder for this drop
        drop_biigle_frames = images_dir / drop_id / "biigle_frames"
        if not drop_biigle_frames.exists():
            logging.debug(
                f"biigle_frames folder not found for {drop_id} at {drop_biigle_frames}"
            )
            continue

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in drop_biigle_frames.glob(ext):
                lbl_path = labels_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
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

    logging.info(
        f"copy_split_files [{split_name}]: {n_images} images + {n_labels} labels → {output_dir}"
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
) -> Tuple[Path, Optional[Path]]:
    """
    Assemble a complete YOLO dataset (species + optional binary) from pre-split drop lists.

    Creates:
        output_dir/species/{images,labels}/{train,val,test}/   + data.yaml
        output_dir/binary/{images,labels}/{train,val,test}/    + data.yaml  (if build_binary=True)

    Internally calls:
        copy_split_files()     for each split × dataset type
        make_binary_labels()   to derive binary labels from species labels
        generate_data_yaml()   for each dataset type

    Args:
        train_drops, val_drops, test_drops: Drop ID lists from split_data().
        images_dir: Source directory of JPEG frames.
        species_labels_dir: Directory of multi-class YOLO .txt files.
        output_dir: Root output directory.
        class_names: Ordered species class names.
        build_binary: Also build a binary (fish/no-fish) dataset.
        symlink: Use symlinks instead of copying files.

    Returns:
        (species_data_yaml_path, binary_data_yaml_path) — binary path is None if not built.
    """
    species_dir = output_dir / "species"
    binary_dir = output_dir / "binary"

    # Species dataset
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
    species_yaml = generate_data_yaml(class_names, species_dir)

    # Binary dataset (derived from species labels)
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
