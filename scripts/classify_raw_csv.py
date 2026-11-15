"""Add the second-stage classifier's columns to a raw CSV that was written without them.

`--ml` now names every YOLO box with BioCLIP as it runs. Drops inferred before
that (or with ``ml_inference.classifier.enabled: false``) have the 8-column raw
CSV. This reads such a CSV, seeks each detection's frame in the drop's video,
crops the box exactly as inference does (``SpeciesClassifier.name_boxes``) and
rewrites the CSV in the 12-column form: ``class`` = the classifier's name (or
the configured fallback), ``confidence`` untouched, plus
``yolo_class / yolo_confidence / clf_class / clf_confidence``. No YOLO re-run.

    python scripts/classify_raw_csv.py HOR_20251204_BUV_HOR_003_01
    python scripts/classify_raw_csv.py --survey HOR_20251204_BUV
    python scripts/classify_raw_csv.py KSF_20240124_BUV_KSF_085_03 --model cfd_binary_water_20260301 \\
        --video media/KSF_20240124_BUV_KSF_085_03.mp4 --checkpoint /path/to/bioclip_x.pt

The original is kept beside it as ``*_raw.pre_classifier.csv`` (never
overwritten once it exists). A CSV that already carries the classifier columns
is skipped unless ``--force``. MaxN and everything downstream read the raw CSV,
so after this a ``--ml-post`` (or ``--db-refresh``) is what moves the new names
into the MaxN CSV and the annotations DB; this script changes the raw CSV only.
Needs the video on local disk, like every stage that touches pixels.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spyfish.config.base import MlStatus  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.database.manager import DatabaseManager  # noqa: E402
from spyfish.extraction.extract_frames import (  # noqa: E402
    VideoOpenError,
    iter_raw_frames,
)
from spyfish.ml.classifier.species_classifier import SpeciesClassifier  # noqa: E402
from spyfish.ml.run_inference import CLASSIFIER_COLUMNS, RAW_COLUMNS  # noqa: E402

# Frames held while boxes collect into one classifier batch. Each is a full
# frame (~25 MB at 4K), so this bounds memory, not the batch size.
MAX_BUFFERED_FRAMES = 32
# A frame up to this many frames ahead is reached by decoding forward instead
# of seeking. A seek lands on the previous keyframe and decodes forward anyway,
# so for the short gaps between sampled frames reading on is never slower.
MAX_READ_AHEAD = 120


def classify_csv(
    raw_csv: Path, video: Path, classifier: SpeciesClassifier, drop_id: str
) -> pd.DataFrame:
    """The raw CSV's rows with the classifier's columns filled in.

    Frames are read in order (forward decode for short gaps, seek for long
    ones) and boxes from several frames go to the classifier together, so a
    GPU sees full batches instead of the 1-3 boxes one frame holds.
    """
    df = pd.read_csv(raw_csv)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{raw_csv.name}: not a raw CSV, missing {missing}")
    out = df.copy()
    out["yolo_class"] = df["class"]
    out["yolo_confidence"] = df["confidence"]
    out["clf_class"] = None
    out["clf_confidence"] = None
    n_frames = df["frame"].nunique()

    frames, boxes, row_ids = [], [], []

    def flush() -> None:
        named = classifier.name_boxes(frames, boxes)
        for idx, naming in zip(row_ids, named):
            if naming is None:
                continue
            out.at[idx, "class"] = classifier.final_label(
                naming, df.at[idx, "class"], float(df.at[idx, "confidence"])
            )
            out.at[idx, "clf_class"] = naming.raw_label
            out.at[idx, "clf_confidence"] = round(naming.prob, 4)
        frames.clear()
        boxes.clear()
        row_ids.clear()

    done = 0
    try:
        for frame_idx, frame, rows in iter_raw_frames(video, df, MAX_READ_AHEAD):
            if frame is None:
                logging.warning(
                    f"{drop_id}: frame {frame_idx} unreadable, YOLO names kept"
                )
                continue
            fi = len(frames)
            frames.append(frame)
            for idx, r in rows.iterrows():
                boxes.append((fi, r["x"], r["y"], r["w"], r["h"]))
                row_ids.append(idx)
            if (
                len(boxes) >= classifier.batch_size
                or len(frames) >= MAX_BUFFERED_FRAMES
            ):
                flush()
            done += 1
            if done % 200 == 0:
                logging.info(f"{drop_id}: {done}/{n_frames} frames")
        if boxes:
            flush()
    except VideoOpenError:
        raise RuntimeError(f"{drop_id}: cannot open {video}") from None
    return out[RAW_COLUMNS + CLASSIFIER_COLUMNS]


def drop_ids_from_args(args) -> list[str]:
    """The drops a ``drop_ids`` / ``--survey`` command line names: with
    ``--survey``, the survey's drops whose raw CSV exists (ML inferred or
    complete). Shared with ``compare_two_stage_names.py``."""
    if args.survey:
        prefix = f"{args.survey}_"
        rows = DatabaseManager().get_deployments_eligible(
            MlStatus.COLUMN, [MlStatus.COMPLETE, MlStatus.INFERRED]
        )
        return [r["drop_id"] for r in rows if r["drop_id"].startswith(prefix)]
    return args.drop_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drop_ids", nargs="*")
    parser.add_argument("--survey", metavar="SURVEY_ID")
    parser.add_argument(
        "--model", help="raw CSV model name; default: the pipeline model"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="classifier weights; default: config"
    )
    parser.add_argument(
        "--video", type=Path, help="video path override (one drop only)"
    )
    parser.add_argument(
        "--force", action="store_true", help="redo CSVs that already have the columns"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write *_raw.classified.csv beside the original instead",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    drop_ids = [config.validate_drop_id(d) for d in drop_ids_from_args(args)]
    if not drop_ids:
        sys.exit("no drops given / found")
    if args.video and len(drop_ids) != 1:
        sys.exit("--video applies to exactly one drop")
    model_name = args.model or config.pipeline_model_name
    classifier = SpeciesClassifier.from_config(args.checkpoint)

    failed = []
    for drop_id in drop_ids:
        raw_csv = config.get_raw_csv_path(drop_id, model_name)
        if not raw_csv.exists():
            logging.warning(f"{drop_id}: no raw CSV for {model_name}, skipped")
            failed.append(drop_id)
            continue
        if not args.force and set(CLASSIFIER_COLUMNS) <= set(
            pd.read_csv(raw_csv, nrows=0).columns
        ):
            logging.info(
                f"{drop_id}: already has classifier columns, skipped (--force to redo)"
            )
            continue
        video = args.video or config.get_video_path(drop_id)
        if not video.exists():
            logging.warning(f"{drop_id}: video not on local disk ({video}), skipped")
            failed.append(drop_id)
            continue
        try:
            out = classify_csv(raw_csv, video, classifier, drop_id)
        except Exception as e:  # noqa: BLE001, one drop must not stop the rest
            logging.error(f"{drop_id}: failed ({e})")
            failed.append(drop_id)
            continue
        if args.dry_run:
            target = raw_csv.with_name(
                raw_csv.name.replace("_raw.csv", "_raw.classified.csv")
            )
        else:
            backup = raw_csv.with_name(
                raw_csv.name.replace("_raw.csv", "_raw.pre_classifier.csv")
            )
            if not backup.exists():
                shutil.copy2(raw_csv, backup)
            target = raw_csv
        out.to_csv(target, index=False)
        named = out["clf_class"].notna()
        agree = (
            (out.loc[named, "class"] == out.loc[named, "yolo_class"]).mean()
            if named.any()
            else float("nan")
        )
        logging.info(
            f"{drop_id}: {len(out)} boxes, {int(named.sum())} classified, written `class` "
            f"agrees with YOLO on {agree:.1%} → {target.name}"
        )
    if failed:
        logging.warning(f"{len(failed)} drop(s) skipped or failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
