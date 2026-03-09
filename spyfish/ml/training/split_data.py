"""
split_data.py — Survey-distributed, drop-ID-aware train/val/test split.

Rules:
  - Each DropID is assigned to exactly one split (no leakage).
  - Each survey (SurveyID) contributes drops to train + val (and test if large enough).
  - Algorithm: For each survey, take 1 drop for val (if survey has ≥3 drops), 1 for test
    (if ≥5 drops), and the rest for train. Remainder fills train to hit target ratios.
  - Prints a split summary (N images per split per species) for manual review before training.

Usage:
    python -m spyfish.ml.training.split_data --images-dir /path/to/images --output-dir /path/to/training --balanced-csv /path/to/balanced.csv
"""
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from spyfish.config import config


# ---------------------------------------------------------------------------
# Core split algorithm
# ---------------------------------------------------------------------------

def get_survey_id_from_drop(drop_id: str) -> str:
    """Derive SurveyID from DropID (e.g. 'KSF_20240124_BUV' from 'KSF_20240124_BUV_KSF_001_01')."""
    import re
    match = re.match(r'^([A-Z]{3}_\d{8}_BUV)', drop_id)
    return match.group(1) if match else drop_id[:16]


def split_drops_by_survey(
    drop_ids: List[str],
    train_pct: float = 0.80,
    val_pct: float = 0.10,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Assign each DropID to train, val, or test while keeping survey representation.

    Strategy:
      - Group drops by SurveyID.
      - For surveys with ≥5 drops: donate 1 to val, 1 to test, rest to train.
      - For surveys with ≥3 drops: donate 1 to val, rest to train.
      - For surveys with 1-2 drops: all go to train (too small to split).
      - Print the split before returning for manual review.

    Args:
        drop_ids: List of all drop IDs.
        train_pct: Target train fraction (used only for the summary printout).
        val_pct: Target val fraction (used only for the summary printout).
        seed: Random seed for reproducibility.

    Returns:
        (train_drops, val_drops, test_drops)
    """
    rng = random.Random(seed)

    # Group by survey
    survey_to_drops: Dict[str, List[str]] = {}
    for drop_id in drop_ids:
        survey_id = get_survey_id_from_drop(drop_id)
        survey_to_drops.setdefault(survey_id, []).append(drop_id)

    train_drops: List[str] = []
    val_drops: List[str] = []
    test_drops: List[str] = []

    # Shuffle survey order for reproducibility
    surveys = sorted(survey_to_drops.keys())
    rng.shuffle(surveys)

    for survey in surveys:
        drops = survey_to_drops[survey]
        rng.shuffle(drops)

        if len(drops) >= 5:
            val_drops.append(drops[0])
            test_drops.append(drops[1])
            train_drops.extend(drops[2:])
        elif len(drops) >= 3:
            val_drops.append(drops[0])
            train_drops.extend(drops[1:])
        else:
            train_drops.extend(drops)

    total = len(train_drops) + len(val_drops) + len(test_drops)
    logging.info(
        f"\n=== Split summary ===\n"
        f"  Total drops: {total}\n"
        f"  Train:       {len(train_drops)} ({len(train_drops)/total:.0%}) — target {train_pct:.0%}\n"
        f"  Val:         {len(val_drops)}  ({len(val_drops)/total:.0%}) — target {val_pct:.0%}\n"
        f"  Test:        {len(test_drops)} ({len(test_drops)/total:.0%}) — target {1-train_pct-val_pct:.0%}\n"
        f"====================\n"
    )

    return train_drops, val_drops, test_drops


# ---------------------------------------------------------------------------
# File list writing
# ---------------------------------------------------------------------------

def write_split_txt(
    drop_ids: List[str],
    images_dir: Path,
    output_path: Path,
) -> int:
    """
    Write a YOLO-compatible image path .txt file listing all images for a split.

    Returns:
        Number of image paths written.
    """
    image_paths = []
    for drop_id in drop_ids:
        # Each drop may have multiple frames; match all images for that drop
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(sorted(images_dir.glob(f"{drop_id}*{ext[1:]}")))
            image_paths.extend(sorted(images_dir.glob(f"**/{drop_id}*{ext[1:]}")))

    # Deduplicate and sort
    image_paths = sorted(set(image_paths))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in image_paths:
            f.write(str(p) + "\n")

    logging.info(f"Wrote {len(image_paths)} image paths to {output_path}")
    return len(image_paths)


def print_species_breakdown(
    df: pd.DataFrame,
    train_drops: List[str],
    val_drops: List[str],
    test_drops: List[str],
) -> None:
    """Print per-species annotation count per split for manual inspection."""
    split_map = {d: "train" for d in train_drops}
    split_map.update({d: "val" for d in val_drops})
    split_map.update({d: "test" for d in test_drops})

    df = df.copy()
    df["split"] = df["DropID"].map(split_map)

    pivot = df.groupby(["ScientificName", "split"])["MaxInterval"].sum().unstack(fill_value=0)
    for col in ("train", "val", "test"):
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[["train", "val", "test"]]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)

    logging.info("\n=== Per-species split breakdown ===")
    logging.info(pivot.to_string())
    val_min_images = config.get_section("training").get("val_min_images", 20)
    logging.info(
        f"\n  ⚠ Minimum val images recommended: {val_min_images}. "
        "If any species val count is too low, extract more frames from those deployments."
    )
    logging.info("====================================\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def split_data(
    balanced_df: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Run the survey-distributed split and write train/val/test .txt files.

    Args:
        balanced_df: Output of prepare_training_data() — balanced annotations DataFrame.
        images_dir: Directory containing all training JPEG images.
        output_dir: Root output directory.
        seed: Random seed.

    Returns:
        (train_drops, val_drops, test_drops)
    """
    training_cfg = config.get_section("training")
    train_pct = training_cfg.get("train_pct", 0.80)
    val_pct = training_cfg.get("val_pct", 0.10)

    all_drop_ids = balanced_df["DropID"].unique().tolist()
    if len(all_drop_ids) == 0:
        raise ValueError("No drop IDs found in balanced_df — aborting split.")

    train_drops, val_drops, test_drops = split_drops_by_survey(
        all_drop_ids, train_pct=train_pct, val_pct=val_pct, seed=seed
    )

    print_species_breakdown(balanced_df, train_drops, val_drops, test_drops)

    # Write image list .txt files
    for split_name, drops in [("train", train_drops), ("val", val_drops), ("test", test_drops)]:
        txt_path = output_dir / f"{split_name}.txt"
        n = write_split_txt(drops, images_dir, txt_path)
        logging.info(f"  {split_name}: {n} images from {len(drops)} drops → {txt_path}")

    return train_drops, val_drops, test_drops


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Split balanced annotation dataset into train/val/test.")
    parser.add_argument("--balanced-csv", required=True, type=Path, help="CSV from prepare_training_data.py")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.balanced_csv)
    train_drops, val_drops, test_drops = split_data(df, args.images_dir, args.output_dir, seed=args.seed)
    logging.info("Split complete. Review the summary above before starting training.")


if __name__ == "__main__":
    main()
