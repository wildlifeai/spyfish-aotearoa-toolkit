"""
split_data.py. Survey-distributed, drop-ID-aware train/val/test split.

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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    force_train_drops: Optional[set] = None,
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
    to train, losing the only drop to val would mean a species seen there
    would never be in train.

    Stratification (not leakage) is the reason for grouping by survey: drops
    within one survey can be at very different locations/times, so they're
    independent samples, leakage is per-DropID, not per-survey. Grouping
    by survey just ensures val sees a representative spread of survey
    conditions instead of (by random chance) all val drops landing in one
    survey.

    Args:
        drop_ids: List of all drop IDs.
        train_pct: Target fraction (summary printout only, derived from
            val_pct/test_pct in the actual algorithm).
        val_pct: Per-survey fraction donated to val.
        test_pct: Per-survey fraction donated to test (0 disables test).
        seed: Random seed for reproducibility.

    Returns:
        (train_drops, val_drops, test_drops)
    """
    rng = random.Random(seed)
    include_test = test_pct > 0
    force_train = set(force_train_drops or set())

    # Pull forced drops out of the survey-grouped pool first; they bypass the
    # donation rule (old-label volumes resolved from force_train_biigle_volumes).
    forced_train_in_data = [d for d in drop_ids if d in force_train]
    missing_train = force_train - set(forced_train_in_data)
    if missing_train:
        logging.warning(
            f"force_train drops contain {len(missing_train)} ID(s) not present in "
            f"the dataset (excluded? wrong survey prefix?): {sorted(missing_train)}"
        )
    if forced_train_in_data:
        logging.info(
            f"Forcing {len(forced_train_in_data)} drop(s) into train: {sorted(forced_train_in_data)}"
        )

    # Group by survey, excluding forced drops (they're already assigned)
    survey_to_drops: Dict[str, List[str]] = {}
    for drop_id in drop_ids:
        if drop_id in force_train:
            continue
        survey_id = config.get_survey_id_from_drop(drop_id)
        survey_to_drops.setdefault(survey_id, []).append(drop_id)

    train_drops: List[str] = list(forced_train_in_data)
    val_drops: List[str] = []
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
        f"  Train:       {len(train_drops)} ({len(train_drops) / total:.0%}), target {train_pct:.0%}\n"
        f"  Val:         {len(val_drops)}  ({len(val_drops) / total:.0%}), target {val_pct:.0%}\n"
        f"  Test:        {len(test_drops)} ({len(test_drops) / total:.0%}), target {test_pct:.0%}\n"
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
# Species-balanced val selection (the suggester, folded into the pipeline)
# ---------------------------------------------------------------------------


def _per_drop_species_counts(
    labels_staged_dir: Path, species_names: List[str], drops: Set[str]
) -> Tuple[Dict[str, Counter], Counter]:
    """``({drop: Counter(species -> box_count)}, Counter(species -> frame_count))``.

    Decodes staged-label class IDs straight through ``species_names`` (index ==
    ID, the same unified ordering ``flatten_and_remap_labels`` wrote), no
    class_map.json / data.yaml needed, so it can't drift out of sync the way the
    standalone suggester could.

    Frames are counted in the same pass because the class floor is a FRAME
    threshold while the val target is a BOX one, and the balancer needs both:
    lifting a species' val target is wasted effort if the floor is going to
    merge it into 'fish' after assembly anyway.
    """
    id_to_name = {i: n for i, n in enumerate(species_names)}
    out: Dict[str, Counter] = {}
    frames: Counter = Counter()
    for drop in drops:
        d = labels_staged_dir / drop
        if not d.is_dir():
            continue
        c: Counter = Counter()
        for txt in d.glob("*.txt"):
            in_frame = set()
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                name = id_to_name.get(cid)
                if name:
                    c[name] += 1
                    in_frame.add(name)
            for name in in_frame:
                frames[name] += 1
        if c:
            out[drop] = c
    return out, frames


def balance_val_drops(
    labels_staged_dir: Path,
    species_names: List[str],
    candidate_drops: List[str],
    val_pct: float,
    tolerance: float = 0.05,
    force_train: Optional[Set[str]] = None,
    overshoot_weight: float = 0.0,
    max_share: float = 1.0,
    min_boxes: int = 0,
    floor_min_frames: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    """Greedy species-balanced val selection (the in-pipeline suggester).

     Picks whole drops into val so each multi-source species reaches ~``val_pct``
     of its boxes, stopping when even the worst-covered species is within
     ``tolerance`` of its target. Single-source species (one drop) are train-only
    , you can't validate a species the model can't also train on. ``force_train``
     drops (old-label volumes, resolved from force_train_biigle_volumes) are
     never picked for val. Returns (train, val, []), no test split.

    ``overshoot_weight`` penalises each box a candidate drop carries beyond any
    species' remaining deficit — those boxes leave train for no val benefit, so
    with a positive weight the greedy prefers the smallest drop that covers a
    need over a box-heavy one. 0 scores by coverage gain alone (overshoot-blind).

    ``max_share`` is a HARD feasibility cap: a candidate drop is refused when
    adding it would push any multi-source species past this share of its boxes
    in val. Drop-concentrated species then resolve train-heavy (small or zero
    val, logged as under-covered) instead of 77-97% val. 1.0 disables.

    ``min_boxes`` is an absolute floor under each species' target, because
    ``val_pct`` alone is scale-invariant and so gets rare species badly wrong:
    8% of blue cod's 22112 boxes is a fine sample, 8% of Batoidea's 71 is 6.
    The floor is CLAMPED to what ``max_share`` permits, so a species that cannot
    reach it keeps the smaller feasible target. Without that clamp the greedy
    chases an unreachable target and keeps pulling drops into val until no
    feasible drop is left, emptying train to satisfy one rare class.

    ``floor_min_frames`` (``class_floor_min_images``) stops the floor being
    spent on species that will not survive to be classes. ``apply_post_assembly_floor``
    merges anything under that many train frames into 'fish' AFTER assembly, so
    lifting the val target for, say, Notolabrus fucicola (24 frames) only pulls
    drops out of train for a class that is about to be deleted. 0 disables the
    check and lifts every species.
    """
    force_train = set(force_train or set())
    per_drop, frames_per_species = _per_drop_species_counts(
        labels_staged_dir, species_names, set(candidate_drops)
    )

    total: Counter = Counter()
    drops_per_species: Counter = Counter()
    for c in per_drop.values():
        total.update(c)
        for s in c:
            drops_per_species[s] += 1
    single_source = {s for s, n in drops_per_species.items() if n <= 1}
    # Floor first, then clamp to the most val boxes max_share will ever allow,
    # so every target in this dict is actually reachable.
    def _target(name: str, n: int) -> int:
        wanted = round(n * val_pct)
        # Only lift species that will still be a class after the post-assembly
        # floor; the rest are merged into 'fish' and the val budget is wasted.
        if min_boxes and frames_per_species[name] >= floor_min_frames:
            wanted = max(min_boxes, wanted)
        reachable = int(n * max_share) if max_share < 1.0 else n
        return max(1, min(wanted, reachable))

    target = Counter(
        {s: _target(s, n) for s, n in total.items() if s not in single_source}
    )
    if min_boxes:
        lifted = {
            s: target[s]
            for s, n in total.items()
            if s in target and target[s] > max(1, round(n * val_pct))
        }
        skipped = sorted(s for s in target if frames_per_species[s] < floor_min_frames)
        if skipped:
            logging.info(
                f"balance_val_drops: floor NOT lifted for {len(skipped)} species "
                f"under {floor_min_frames} frames (they merge into 'fish' at the "
                f"post-assembly floor): {skipped}"
            )
        if lifted:
            moves = {
                s: f"{round(total[s] * val_pct)}->{t}"
                for s, t in sorted(lifted.items())
            }
            logging.info(
                f"balance_val_drops: min_boxes={min_boxes} lifted the val target "
                f"for {len(lifted)} rare species: {moves}"
            )

    current: Counter = Counter()
    val: List[str] = []
    remaining = {d: c for d, c in per_drop.items() if d not in force_train}
    while remaining and target:
        worst = max((target[s] - current[s]) / target[s] for s in target)
        if worst <= tolerance:
            break
        best, best_score = None, None
        for d, counts in remaining.items():
            # Feasibility first: refuse any drop that would push a targeted
            # species past max_share of its boxes in val. Concentrated species
            # then stay train-heavy rather than val-heavy.
            if max_share < 1.0 and any(
                (current[s] + n) / total[s] > max_share
                for s, n in counts.items()
                if s in target
            ):
                continue
            gain = sum(
                min(n, target[s] - current[s])
                for s, n in counts.items()
                if target.get(s, 0) - current[s] > 0
            )
            if gain == 0:
                continue
            # Boxes beyond any remaining deficit leave train for no val benefit.
            overshoot = sum(counts.values()) - gain
            score = gain - overshoot_weight * overshoot
            if best_score is None or score > best_score:
                best, best_score = d, score
        if best is None:
            uncovered = sorted(s for s in target if current[s] < target[s])
            if uncovered:
                logging.info(
                    f"balance_val_drops: {len(uncovered)} species left "
                    f"under-covered in val (max_share={max_share:.0%} refused "
                    f"their drops; they stay train-heavy): {uncovered}"
                )
            break
        val.append(best)
        current.update(remaining.pop(best))

    val_set = set(val)
    train = [d for d in candidate_drops if d not in val_set]
    logging.info(
        f"balance_val_drops: {len(val)} val / {len(train)} train drop(s); "
        f"val_pct={val_pct:.0%}, tolerance={tolerance:.0%}, "
        f"{len(single_source)} single-source species kept train-only"
    )
    return train, val, []


def force_train_drops_from_volumes(volume_ids: Set[int]) -> Set[str]:
    """Resolve force_train_biigle_volumes to drop ids, offline.

    Reads the ``volume_id`` column that download_training_volume_labels stamps
    into each drop's ``_biigle_training_raw.csv``. CSVs written before that
    column existed never match — re-download the volume with --force to stamp
    them. Returns the drop ids whose training labels came from a listed volume.
    """
    if not volume_ids:
        return set()
    forced: Set[str] = set()
    suffix = config.biigle_training_raw_suffix
    for csv_path in config.deployment_data_dir.glob(f"**/annotations/*{suffix}"):
        try:
            header = pd.read_csv(csv_path, nrows=0).columns
            if "volume_id" not in header:
                continue
            vols = pd.read_csv(csv_path, usecols=["volume_id"])["volume_id"]
            if vols.empty:
                continue
            if int(vols.iloc[0]) in volume_ids:
                forced.add(csv_path.name[: -len(suffix)])
        except Exception as e:
            logging.warning(f"Could not read volume_id from {csv_path.name}: {e}")
    logging.info(
        f"force_train_biigle_volumes {sorted(volume_ids)} resolved to "
        f"{len(forced)} drop(s)"
    )
    return forced


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def split_data(
    balanced_df: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
    seed: int = 42,
    extra_drop_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Run the survey-distributed split and write train/val/test .txt files.

    Args:
        balanced_df: Output of prepare_training_data(), balanced annotations DataFrame.
        images_dir: Directory containing all training JPEG images.
        output_dir: Root output directory.
        seed: Random seed.
        extra_drop_ids: Extras (drops with labels but no MaxN). When provided,
            they participate in the survey-aware split alongside MaxN drops,
            so forced-train volumes + the per-survey val_pct apply to extras
            too. Volume_<id> extras group under UNKNOWN_SURVEY; canonical-ID
            extras group with their real survey.

    Returns:
        (train_drops, val_drops, test_drops)
    """
    train_pct = config.training_train_pct
    val_pct = config.training_val_pct
    test_pct = config.training_test_pct

    maxn_drops = balanced_df["DropID"].unique().tolist()
    if len(maxn_drops) == 0 and not extra_drop_ids:
        raise ValueError("No drop IDs found to split, aborting.")
    all_drop_ids = sorted(set(maxn_drops) | set(extra_drop_ids or []))

    train_drops, val_drops, test_drops = split_drops_by_survey(
        all_drop_ids,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        force_train_drops=force_train_drops_from_volumes(
            config.training_force_train_biigle_volumes
        ),
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
