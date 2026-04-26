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
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from spyfish.config.wrapper import config

# ---------------------------------------------------------------------------
# Core split algorithm
# ---------------------------------------------------------------------------


def split_drops_by_survey(
    drop_ids: List[str],
    train_pct: float = 0.80,
    val_pct: float = 0.10,
    test_pct: float = 0.10,
    force_val_drops: Optional[set] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Assign each DropID to train, val, or test, stratified by survey.

    Per-survey donation rule (n = drops in survey):
      - val_take  = max(1, ceil(n * val_pct))   if n >= 2 and val_pct > 0
      - test_take = max(1, ceil(n * test_pct))  if n >= 3 and test_pct > 0
      - train_take = n - val_take - test_take   (always >= 1; donations are
        clipped if they would empty train)

    The min-1 floor means any survey large enough to qualify contributes at
    least one drop to val (and to test, when enabled), so small surveys
    aren't silently dropped from evaluation. Singleton surveys go entirely
    to train — losing the only drop to val would mean a species seen there
    would never be in train.

    Stratification (not leakage) is the reason for grouping by survey: drops
    within one survey can be at very different locations/times, so they're
    independent samples — leakage is per-DropID, not per-survey. Grouping
    by survey just ensures val sees a representative spread of survey
    conditions instead of (by random chance) all val drops landing in one
    survey.

    Args:
        drop_ids: List of all drop IDs.
        train_pct: Target fraction (summary printout only — derived from
            val_pct/test_pct in the actual algorithm).
        val_pct: Per-survey fraction donated to val.
        test_pct: Per-survey fraction donated to test (0 disables test).
        seed: Random seed for reproducibility.

    Returns:
        (train_drops, val_drops, test_drops)
    """
    rng = random.Random(seed)
    include_test = test_pct > 0
    force_val = set(force_val_drops or set())

    # Pull forced-val drops out of the survey-grouped pool first; they bypass
    # the donation rule. Useful when a rare-species drop sits in a singleton
    # survey that would otherwise never reach val.
    forced_in_data = [d for d in drop_ids if d in force_val]
    missing_forced = force_val - set(forced_in_data)
    if missing_forced:
        logging.warning(
            f"force_val_drops contains {len(missing_forced)} ID(s) not present in "
            f"the dataset (typo? excluded? wrong survey prefix?): {sorted(missing_forced)}"
        )
    if forced_in_data:
        logging.info(
            f"Forcing {len(forced_in_data)} drop(s) into val: {sorted(forced_in_data)}"
        )

    # Group by survey, excluding forced drops
    survey_to_drops: Dict[str, List[str]] = {}
    for drop_id in drop_ids:
        if drop_id in force_val:
            continue
        survey_id = config.get_survey_id_from_drop(drop_id)
        survey_to_drops.setdefault(survey_id, []).append(drop_id)

    train_drops: List[str] = []
    val_drops: List[str] = list(forced_in_data)
    test_drops: List[str] = []

    # Shuffle survey order for reproducibility
    surveys = sorted(survey_to_drops.keys())
    rng.shuffle(surveys)

    for survey in surveys:
        drops = survey_to_drops[survey]
        rng.shuffle(drops)
        n = len(drops)

        val_take = max(1, math.ceil(n * val_pct)) if n >= 2 and val_pct > 0 else 0
        test_take = max(1, math.ceil(n * test_pct)) if include_test and n >= 3 else 0

        # Always leave at least 1 drop in train. Trim test first, then val.
        overflow = (val_take + test_take) - (n - 1)
        if overflow > 0:
            trim_test = min(overflow, test_take)
            test_take -= trim_test
            overflow -= trim_test
            if overflow > 0:
                val_take = max(0, val_take - overflow)

        val_drops.extend(drops[:val_take])
        test_drops.extend(drops[val_take : val_take + test_take])
        train_drops.extend(drops[val_take + test_take :])

    total = len(train_drops) + len(val_drops) + len(test_drops)
    logging.info(
        f"\n=== Split summary ===\n"
        f"  Total drops: {total}\n"
        f"  Train:       {len(train_drops)} ({len(train_drops) / total:.0%}) — target {train_pct:.0%}\n"
        f"  Val:         {len(val_drops)}  ({len(val_drops) / total:.0%}) — target {val_pct:.0%}\n"
        f"  Test:        {len(test_drops)} ({len(test_drops) / total:.0%}) — target {test_pct:.0%}\n"
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
        for ext in config.image_extensions:
            image_paths.extend(sorted(images_dir.glob(f"{drop_id}*{ext}")))
            image_paths.extend(sorted(images_dir.glob(f"**/{drop_id}*{ext}")))

    # Deduplicate and sort
    image_paths = sorted(set(image_paths))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in image_paths:
            f.write(str(p) + "\n")

    logging.info(f"Wrote {len(image_paths)} image paths to {output_path}")
    return len(image_paths)


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
    train_pct = config.training_train_pct
    val_pct = config.training_val_pct
    test_pct = config.training_test_pct

    all_drop_ids = balanced_df["DropID"].unique().tolist()
    if len(all_drop_ids) == 0:
        raise ValueError("No drop IDs found in balanced_df — aborting split.")

    train_drops, val_drops, test_drops = split_drops_by_survey(
        all_drop_ids,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        force_val_drops=config.training_force_val_drops,
        seed=seed,
    )

    # Write image list .txt files
    for split_name, drops in [
        ("train", train_drops),
        ("val", val_drops),
        ("test", test_drops),
    ]:
        txt_path = output_dir / f"{split_name}.txt"
        n = write_split_txt(drops, images_dir, txt_path)
        logging.info(f"  {split_name}: {n} images from {len(drops)} drops → {txt_path}")

    return train_drops, val_drops, test_drops


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Split balanced annotation dataset into train/val/test."
    )
    parser.add_argument(
        "--balanced-csv",
        required=True,
        type=Path,
        help="CSV from prepare_training_data.py",
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.balanced_csv)
    train_drops, val_drops, test_drops = split_data(
        df, args.images_dir, args.output_dir, seed=args.seed
    )
    logging.info("Split complete. Review the summary above before starting training.")


if __name__ == "__main__":
    main()
