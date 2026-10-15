"""Measure what the 0.25 confidence clip is hiding, without disturbing anything.

Every raw CSV in the project has a minimum confidence of exactly 0.2500 and the
values pile up against that boundary (0.2500, 0.2502, 0.2503, ...), which is the
signature of a hard clip rather than a natural floor. So the question "should
confidence_threshold be lower?" cannot be answered from existing data: the
detections that would decide it were discarded at inference and never written.

This re-runs inference on ONE drop at a lower threshold, into a SEPARATE CSV,
and reports what appears in the hidden band.

Why not just change config.yaml: `MLRunner` writes to
`{drop_id}_{model_name}_raw.csv`, the production artifact that MaxN, frame
selection and the BIIGLE boxes all derive from. Lowering the config value and
re-running `--ml` would overwrite that file with experimental output and change
the threshold for every subsequent run. This script writes to
`{drop_id}_{model_name}_conf{NNN}_raw.csv` instead and leaves config alone.

The headline number is NOT how many extra detections appear. Most will be second
boxes on animals already detected above 0.25, which are worth nothing. What
matters is how many are spatially UNMATCHED, i.e. plausible animals the current
threshold misses entirely. Those are the only ones that would add recall.

Usage (from project root, on NeSI where the videos live):
    python scripts/confidence_threshold_experiment.py --drop-id KSF_20240124_BUV_KSF_085_01
    python scripts/confidence_threshold_experiment.py --drop-id ... --confidence 0.10
    python scripts/confidence_threshold_experiment.py --drop-id ... --analyse-only
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.ml.run_inference import get_video_fps, run_yolo_inference
from spyfish.utils import validate_model_path

# Two boxes on the same animal overlap heavily; a genuinely new animal does not.
# 0.3 is deliberately loose: the aim is to be CONSERVATIVE about calling a
# detection "new", so a low-confidence box that merely overlaps an existing one
# is treated as a duplicate rather than counted as extra recall.
IOU_MATCH = 0.3


def _iou(a, b) -> float:
    """IoU of two (x, y, w, h) centre-format boxes, as written in the raw CSV."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix, iy = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def analyse(low_csv: Path, baseline_conf: float) -> None:
    """Report what lives below `baseline_conf`, and how much of it is new."""
    df = pd.read_csv(low_csv)
    if df.empty:
        logging.info("Low-confidence CSV is empty, nothing to analyse.")
        return

    hidden = df[df.confidence < baseline_conf]
    visible = df[df.confidence >= baseline_conf]

    logging.info(f"\n{'=' * 68}")
    logging.info(f"{len(df):,} detections total at the experimental threshold")
    logging.info(
        f"  {len(visible):,} at or above {baseline_conf} (what the pipeline sees today)"
    )
    logging.info(f"  {len(hidden):,} BELOW {baseline_conf} (currently discarded)")
    if hidden.empty:
        logging.info("Nothing hidden, the current threshold loses no detections.")
        return

    # Per timestamp, does each hidden box overlap a visible one? Grouping by
    # time keeps the comparison to boxes in the same video frame.
    vis_by_t: dict = {}
    for t, g in visible.groupby("time_seconds"):
        vis_by_t[t] = g[["x", "y", "w", "h"]].to_numpy()

    unmatched_rows = []
    for _, r in hidden.iterrows():
        boxes = vis_by_t.get(r["time_seconds"])
        box = (r["x"], r["y"], r["w"], r["h"])
        if boxes is None or max((_iou(box, b) for b in boxes), default=0.0) < IOU_MATCH:
            unmatched_rows.append(r)

    unmatched = pd.DataFrame(unmatched_rows)
    logging.info(f"\n{'-' * 68}")
    logging.info("Of the hidden detections:")
    dup = len(hidden) - len(unmatched)
    logging.info(
        f"  {dup:,} ({100 * dup / len(hidden):.0f}%) overlap a box already found "
        "above the threshold - duplicates, no recall gained"
    )
    logging.info(
        f"  {len(unmatched):,} ({100 * len(unmatched) / len(hidden):.0f}%) are "
        "UNMATCHED - candidate animals the current threshold misses"
    )

    if not unmatched.empty:
        logging.info("\nUnmatched detections by class:")
        for cls, n in unmatched["class"].value_counts().items():
            sub = unmatched[unmatched["class"] == cls]
            logging.info(
                f"  {cls:<28} {n:>5}   median conf {sub.confidence.median():.3f}"
            )
        n_frames = unmatched["time_seconds"].nunique()
        logging.info(
            f"\nSpread over {n_frames:,} distinct timestamps "
            f"({len(unmatched) / n_frames:.1f} per affected frame)."
        )

    logging.info(f"\n{'=' * 68}")
    logging.info(
        "NEXT STEP: the unmatched count is a CANDIDATE count, not a fish count.\n"
        "Draw them with spyfish/ml/draw_frames.py, or upload a sample to the\n"
        "BIIGLE playground project (4081), and have someone judge how many are\n"
        "real animals. That true-positive rate is what decides the threshold."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop-id", required=True, help="Deployment to re-run.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.10,
        help="Experimental threshold (default 0.10).",
    )
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="Skip inference, just re-analyse an existing experimental CSV.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.validate_drop_id(args.drop_id)

    baseline = float(config.confidence_threshold)
    if args.confidence >= baseline:
        raise SystemExit(
            f"--confidence must be BELOW the current threshold ({baseline}) "
            "or there is nothing hidden to reveal."
        )

    model_path = validate_model_path(config.pipeline_model_path)
    model_name = Path(model_path).stem
    tag = f"conf{int(round(args.confidence * 1000)):03d}"
    out_csv = (
        config.get_drop_annotations_dir(args.drop_id)
        / f"{args.drop_id}_{model_name}_{tag}_raw.csv"
    )

    if not args.analyse_only:
        deployment = DatabaseManager().get_deployment(args.drop_id)
        if deployment is None:
            raise SystemExit(f"{args.drop_id} is not in the database.")

        video_path = str(config.get_video_path(args.drop_id))
        if not Path(video_path).exists():
            raise SystemExit(
                f"Video not found: {video_path}. Run this where the videos are "
                "(NeSI), or download the drop first."
            )

        true_fps = get_video_fps(video_path)
        vid_stride = max(1, round(true_fps / config.ml_fps))
        logging.info(
            f"{args.drop_id}: re-running inference at conf={args.confidence} "
            f"(pipeline uses {baseline}); model {model_name}"
        )
        logging.info(f"Writing to {out_csv} (production raw CSV untouched)")

        run_yolo_inference(
            video_path,
            str(model_path),
            args.confidence,
            int(config.imgsz),
            str(out_csv),
            true_fps,
            vid_stride,
            args.drop_id,
            float(deployment["sampling_start"]),
            float(deployment["sampling_end"]),
        )

    if not out_csv.exists():
        raise SystemExit(f"No experimental CSV at {out_csv}.")
    analyse(out_csv, baseline)


if __name__ == "__main__":
    main()
