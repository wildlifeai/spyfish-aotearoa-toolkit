"""Compare species checkpoints on images NONE of them trained on.

For the frozen test set prefer `python -m spyfish.ml.training.test_eval`, which
keeps a per-model, per-class record with contamination counts. This script
remains for a quick side-by-side on the shared vocabulary of several models; it
is a command line over the same functions (`score_on_shared_vocabulary`).

Checkpoints from different data preps have incompatible class-id schemes (17 /
20 / 50 classes, two alphabetical, the newest in frozen roster order). Pointing
`model.val` at another model's data.yaml is what produced `KeyError: 16` in the
pipeline: ids mean different species in each scheme, and Ultralytics silently
drops names it cannot resolve.

So each model is scored on identical IMAGES with GT rewritten into its own id
space, restricted to the vocabulary they all share:

  * GT species a model does not name are remapped to 'fish' (what that model's
    own class floor would have done with them anyway).
  * `classes=` restricts each model's PREDICTIONS to the shared ids, so a model
    is not charged false positives for knowing species the others never saw.

CONTAMINATION FILTER (2026-08-25). The live `training/species/` workspace is
rebuilt by every data prep, so its val split is NOT the one any given model was
validated against: parts of it were in that model's TRAINING set. Scoring a
model on images it trained on flatters it, and the flattery is uneven, so the
ranking tracks training exposure rather than skill. Measured on the 20260822 /
20260823 pair: 171 of 640 val images (26.7%) had been in the older model's train
split, and the older model then scored HIGHER on the newer model's val than on
its own, the signature of exactly that.

This script therefore excludes any image appearing in ANY model's frozen
train.txt snapshot, leaving only images none of them has seen. A model with no
snapshot (20260821 kept none, the feature postdates it) cannot be filtered and
is reported as UNFILTERABLE rather than quietly compared.

Usage (from project root). A model is a run name under the training runs dir,
a weights path (absolute, or relative to local_training_dir), or 'production':

    python scripts/compare_models_shared_vocab.py species_20260823_031610 species_20260825_053442
    python scripts/compare_models_shared_vocab.py species_20260825_053442 production --split test

Comparing exactly TWO is usually the better question ("did this run beat the
last one?"), and it keeps far more images: the clean set excludes every image
in ANY compared model's train split, so each extra model shrinks it.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spyfish.biigle.biigle_to_yolo import drop_id_from_frame_filename  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.training.test_eval import (  # noqa: E402
    score_on_shared_vocabulary,
    split_images,
    trained_on,
)
from spyfish.utils import validate_model_path  # noqa: E402


def resolve_model(arg: str) -> Path:
    """Weights file for one command-line model argument."""
    if arg == "production":
        return config.pipeline_model_path
    if arg.endswith(".pt"):
        p = Path(arg)
        return p if p.is_absolute() else config.local_training_dir / p
    # A run name. Bare stamps and the pre-standard `{stamp}_species` run dirs
    # are still accepted, since those are the runs this script was written for.
    for run in (arg, f"species_{arg}", f"{arg}_species"):
        best = config.training_runs_dir / run / "weights" / "best.pt"
        if best.exists():
            return best
    return config.training_runs_dir / arg / "weights" / "best.pt"


def tag_for(arg: str, weights: Path) -> str:
    """Short name for a model in the tables and the output dir names."""
    if not arg.endswith(".pt"):
        return arg
    return weights.parent.parent.name if weights.stem == "best" else weights.stem


def n_deployments(images) -> int:
    return len({drop_id_from_frame_filename(f) for f in images})


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(
        description="Compare species checkpoints on images none of them trained on."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="run names (e.g. species_20260825_053442), weights paths, or 'production'",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="which split of the live species/ dataset to score on. 'test' is the "
        "frozen test set (training.test_drops_file); it exists once a data prep "
        "has assembled it",
    )
    parser.add_argument(
        "--include-seen",
        action="store_true",
        help="score on every image of the split, including ones a model trained on "
        "(inflated, but shows the full picture); default scores unseen images only",
    )
    parser.add_argument(
        "--allow-unfilterable",
        action="store_true",
        help="proceed even if a model has no snapshot (its score may be inflated)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "model_compare",
        help="where the per-model views, Ultralytics output and comparison.json/"
        ".csv are written (default: ~/model_compare)",
    )
    args = parser.parse_args()

    dataset_dir = config.training_dataset_dir
    split_all = split_images(dataset_dir, args.split)
    if not split_all:
        raise SystemExit(
            f"No {args.split} split at {dataset_dir / 'images' / args.split}. Run a "
            f"data prep (python run_pipeline.py --retrain --data-prep) first."
        )

    resolved = [(arg, resolve_model(arg)) for arg in args.models]
    models = {tag_for(arg, p): p for arg, p in resolved}
    missing = [tag for tag, p in models.items() if not p.exists()]
    if missing:
        raise SystemExit(f"No weights for: {missing}")
    models = {tag: Path(validate_model_path(p)) for tag, p in models.items()}

    # ── Build the contamination-free evaluation set ──────────────────────────
    seen_by, unfilterable = trained_on(models, split_all)
    contaminated = set().union(*seen_by.values()) if seen_by else set()
    clean = [f for f in split_all if args.include_seen or f not in contaminated]

    print("=" * 78)
    print("EVALUATION SET")
    print("=" * 78)
    print(
        f"  current {args.split} split          : {len(split_all):>5} images from "
        f"{n_deployments(split_all):>4} deployments"
    )
    for tag in models:
        if tag in seen_by:
            print(
                f"  seen in {tag} train : {len(seen_by[tag]):>5} images from "
                f"{n_deployments(seen_by[tag]):>4} deployments  "
                f"({'INCLUDED, scores inflated' if args.include_seen else 'excluded'})"
            )
        else:
            print(f"  {tag}: NO SNAPSHOT, cannot filter, results UNFILTERABLE")
    print(
        f"  -> clean evaluation set     : {len(clean):>5} images from "
        f"{n_deployments(clean):>4} deployments"
    )
    if unfilterable:
        print(
            f"  WARNING: {unfilterable} lack snapshots; their scores may include "
            f"images they trained on."
        )
    if unfilterable and not (args.allow_unfilterable or args.include_seen):
        raise SystemExit(
            f"{unfilterable} have no dataset snapshot, so images they trained on "
            f"cannot be excluded and their scores would be inflated. Drop them, or "
            f"pass --allow-unfilterable if you accept that."
        )
    if not clean:
        raise SystemExit("No uncontaminated images left, nothing to compare.")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    scored = score_on_shared_vocabulary(
        models, clean, out, dataset_dir=dataset_dir, split=args.split
    )
    shared = scored["shared_vocabulary"]

    results = {
        "_eval_images": len(clean),
        "_eval_deployments": n_deployments(clean),
        "_unfilterable": unfilterable,
        "_shared_vocabulary": shared,
    }
    for tag, r in scored["models"].items():
        results[tag] = {
            "nc": r["nc"],
            "imgsz": r["imgsz"],
            "map50": r["mAP50"],
            "map50_95": r["mAP50_95"],
            "precision": r["precision"],
            "recall": r["recall"],
            "per_class": r["per_class"],
        }

    (out / "comparison.json").write_text(json.dumps(results, indent=2))
    with (out / "comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "nc", "mAP50", "mAP50_95", "precision", "recall"])
        for tag in models:
            r = results[tag]
            w.writerow(
                [tag, r["nc"], r["map50"], r["map50_95"], r["precision"], r["recall"]]
            )

    print("\n" + "=" * 78)
    print(
        f"SCORED ON {results['_eval_images']} images from "
        f"{results['_eval_deployments']} deployments of the {args.split} split, "
        + (
            "INCLUDING images models trained on"
            if args.include_seen
            else "seen by no model"
        )
    )
    print("=" * 78)
    print(f"{'model':<28}{'nc':>4}{'mAP50':>9}{'mAP50-95':>10}{'P':>8}{'R':>8}")
    for tag in models:
        r = results[tag]
        print(
            f"{tag:<28}{r['nc']:>4}{r['map50']:>9.4f}{r['map50_95']:>10.4f}"
            f"{r['precision']:>8.4f}{r['recall']:>8.4f}"
        )
    print(f"\nWROTE {out / 'comparison.json'}")


if __name__ == "__main__":
    main()
