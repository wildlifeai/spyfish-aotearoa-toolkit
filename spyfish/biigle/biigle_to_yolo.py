"""
biigle_to_yolo.py — Convert local Biigle expert CSV exports → YOLO label .txt files.

This tool is strictly offline; it consumes CSVs previously exported by sync_biigle_annotations
into process_files/deployment_data/{drop_id}/annotations/.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from spyfish.biigle.class_map import load_class_map
from spyfish.config.wrapper import config

# ---------------------------------------------------------------------------
# Core conversion helpers
# ---------------------------------------------------------------------------


def biigle_rect_to_yolo(
    points: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Convert a Biigle Rectangle annotation to normalised YOLO format.

    Biigle stores rectangles as 8 flat coordinates (4 corners, duplicated):
        [x1, y1, x2, y1, x2, y2, x1, y2]  (top-left → clockwise)

    Returns:
        (cx, cy, w, h) each normalised to [0, 1] by image dimensions.
    """
    x1, y1, x2 = points[0], points[1], points[2]
    y2 = points[5]

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h

    cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def convert_annotations_to_yolo(
    df: pd.DataFrame,
    class_map: Dict[str, int],
    labels_dir: Path,
    default_img_size: Tuple[int, int] = (1920, 1080),
) -> Dict[str, int]:
    """
    Write one YOLO .txt label file per image.

    Args:
        df: Annotation DataFrame with columns: filename, label_name, points (JSON list).
        class_map: label_name → YOLO class_id.
        labels_dir: Output directory for .txt files.
        default_img_size: Fallback (width, height) when image dimensions are unavailable.

    Returns:
        {filename: annotation_count} summary.
    """
    incoming_labels = set(df["label_name"].dropna().astype(str).unique())
    unknown = sorted(incoming_labels - set(class_map.keys()))
    fish_class_id = class_map.get("fish")  # generic-fish fallback bucket
    if unknown:
        affected = int(df["label_name"].isin(unknown).sum())
        if fish_class_id is None:
            raise ValueError(
                f"class_map is missing {len(unknown)} label(s) and has no "
                f"'fish' fallback bucket — {affected} of {len(df)} rows "
                f"({affected / len(df):.1%}) would be dropped silently.\n"
                f"  Missing labels: {unknown}\n"
                f"  Fix: reseed class_map.json with "
                f"`python -m spyfish.biigle.class_map`, or add a fish bucket."
            )
        logging.warning(
            f"{len(unknown)} label(s) not in class_map — routing {affected} of "
            f"{len(df)} rows ({affected / len(df):.1%}) to the 'fish' bucket "
            f"(class_id {fish_class_id}). Unknown labels: {unknown}"
        )

    labels_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, int] = {}

    for filename, group in df.groupby("filename"):
        img_w, img_h = default_img_size
        lines = []

        for _, row in group.iterrows():
            class_id = class_map.get(row["label_name"], fish_class_id)

            points_raw = row.get("points", row.get("shape_points", "[]"))
            if isinstance(points_raw, str):
                points = json.loads(points_raw)
            else:
                points = list(points_raw)

            if len(points) < 6:
                continue

            cx, cy, w, h = biigle_rect_to_yolo(points, img_w, img_h)
            if w <= 0 or h <= 0:
                logging.warning(f"Zero-size bbox for {filename}, skipping")
                continue

            lines.append(f"{class_id} {cx} {cy} {w} {h}")

        stem = Path(str(filename)).stem
        txt_path = labels_dir / f"{stem}.txt"
        txt_path.write_text("\n".join(lines))
        summary[str(filename)] = len(lines)

    logging.info(f"Wrote {len(summary)} label files to {labels_dir}")
    return summary


# ---------------------------------------------------------------------------
# Visual spot-check
# ---------------------------------------------------------------------------


def draw_frames_on_images(
    images_dir: Path,
    labels_dir: Path,
    class_map: Dict[str, int],
    output_dir: Path,
    n_samples: int = 5,
) -> None:
    """
    Draw bounding boxes on a random sample of static JPEG images for visual review.
    """
    try:
        import random

        import cv2
    except ImportError:
        logging.warning("opencv-python not installed — skipping spot-check drawing.")
        return

    id_to_name = {v: k for k, v in class_map.items()}

    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        logging.warning("No label files found for spot-check.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = random.sample(label_files, min(n_samples, len(label_files)))

    for label_path in samples:
        img_path = None
        for ext in config.image_extensions:
            # Search in all subdirectories of images_dir (e.g. deployment_data/survey_id/drop_id/frames/)
            for p in images_dir.rglob(label_path.stem + ext):
                img_path = p
                break
            if img_path:
                break

        if img_path is None:
            logging.debug(f"No image found for {label_path.stem}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    id_to_name.get(cls, str(cls)),
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        logging.info(f"Spot-check image saved: {out_path}")

    logging.info(f"Spot-check complete — {len(samples)} images saved to {output_dir}")


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def biigle_to_yolo(
    deployment_data_dir: Path,
    class_map_path: Path,
) -> Dict[str, int]:
    """
    Finds all expert CSVs in deployment_data and converts them to YOLO .txt files.

    Labels are written into each drop's labels/ folder (sibling of frames/).
    prepare_training_data then walks these to assemble the unified training dataset.
    """
    logging.info(f"Searching for expert CSVs in {deployment_data_dir}...")
    csv_paths = []
    all_dfs = []

    # Strictly use the per-drop expert raw CSVs (skip frozen video-era exports)
    raw_glob = f"**/annotations/*{config.biigle_expert_raw_suffix}"
    for csv_path in sorted(deployment_data_dir.glob(raw_glob)):
        if csv_path.name.startswith("legacy_video_"):
            continue
        logging.debug(f"  Found expert CSV: {csv_path}")
        csv_paths.append(csv_path)
        all_dfs.append(pd.read_csv(csv_path))

    if not all_dfs:
        logging.warning("No expert CSV files found. Retraining cannot proceed.")
        return {}

    class_map = load_class_map(class_map_path)
    logging.info(
        f"Loaded class map with {len(class_map)} label keys from {class_map_path}"
    )

    for csv_path, drop_df in zip(csv_paths, all_dfs):
        labels_dir = csv_path.parent.parent / "labels"
        convert_annotations_to_yolo(drop_df, class_map, labels_dir)
        logging.info(f"  Wrote labels for {csv_path.parent.parent.name} → {labels_dir}")

    return class_map


def download_extra_volume_labels(
    volume_id: int,
    deployment_data_dir: Path,
    class_map_path: Optional[Path] = None,
    report_type: Optional[int] = None,
) -> Dict[str, int]:
    """
    Download annotations from any Biigle volume into a drop-shaped bundle
    under `deployment_data_dir/extra_no_survey_id/volume_<id>/`:

        volume_<id>/
          frames/                                   <- user populates with JPEGs
          annotations/
            volume_<id>_biigle_expert_raw.csv       <- raw Biigle export
            class_map.json                          <- sidecar (audit-only)
          labels/
            <image_stem>.txt                         <- YOLO labels

    This matches the normal deployment-data shape so downstream pipeline steps
    (`biigle_to_yolo`, `flatten_and_remap_labels`) pick it up via their usual
    globs — no parallel code path needed.
    """
    import shutil

    from spyfish.biigle.biigle_parser import BiigleParser

    if report_type is None:
        report_type = config.annotation_report_type_images

    drop_name = f"volume_{volume_id}"
    source_dir = deployment_data_dir / "extra_no_survey_id" / drop_name
    annotations_dir = source_dir / "annotations"
    labels_dir = source_dir / "labels"
    frames_dir = source_dir / "frames"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    parser = BiigleParser()
    logging.info(f"Downloading annotations for volume {volume_id}...")
    df = parser.download_volume_annotations(volume_id, type_id=report_type)

    if df.empty:
        logging.warning(f"No annotations found for volume {volume_id}.")
        return {}

    raw_csv_path = annotations_dir / f"{drop_name}{config.biigle_expert_raw_suffix}"
    df.to_csv(raw_csv_path, index=False)
    logging.info(f"Saved raw CSV ({len(df)} rows) → {raw_csv_path}")

    # Freeze the class_map used at download time (audit trail for humans).
    resolved_class_map_path = class_map_path or config.class_map_path
    sidecar_class_map = annotations_dir / "class_map.json"
    shutil.copy2(resolved_class_map_path, sidecar_class_map)
    logging.info(f"Froze class_map sidecar → {sidecar_class_map}")

    class_map = load_class_map(resolved_class_map_path)
    summary = convert_annotations_to_yolo(df, class_map, labels_dir)
    logging.info(f"Wrote {len(summary)} YOLO label files → {labels_dir}")
    logging.info(
        f"Bundle ready at {source_dir} — populate {frames_dir}/ with JPEGs "
        "before running the training pipeline."
    )
    return class_map


def main():
    parser = argparse.ArgumentParser(
        description="Convert local Biigle expert CSVs to YOLO labels."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Existing: convert local CSVs
    convert_cmd = subparsers.add_parser(
        "convert", help="Convert local expert CSVs to YOLO labels"
    )
    convert_cmd.add_argument(
        "--data-dir", required=True, type=Path, help="Root deployment_data directory"
    )
    convert_cmd.add_argument(
        "--class-map",
        required=True,
        type=Path,
        help="Path to class_map.json (seed via `python -m spyfish.biigle.class_map`)",
    )

    # New: download from arbitrary volume
    download_cmd = subparsers.add_parser(
        "download-volume", help="Download labels from any Biigle volume"
    )
    download_cmd.add_argument(
        "--volume-id", required=True, type=int, help="Biigle volume ID"
    )
    download_cmd.add_argument(
        "--deployment-data-dir",
        type=Path,
        default=None,
        help=(
            "Root deployment_data directory (default: config.deployment_data_dir). "
            "A drop-shaped bundle is created at "
            "<deployment-data-dir>/extra_no_survey_id/volume_<id>/ containing "
            "frames/, annotations/ (raw CSV + class_map sidecar), and labels/."
        ),
    )
    download_cmd.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Path to class_map.json (defaults to config.class_map_path)",
    )
    download_cmd.add_argument(
        "--report-type",
        type=int,
        default=None,
        help="Biigle report type ID (default: image annotations)",
    )

    args = parser.parse_args()

    if args.command == "convert":
        biigle_to_yolo(args.data_dir, args.class_map)
    elif args.command == "download-volume":
        deployment_data_dir = args.deployment_data_dir or config.deployment_data_dir
        download_extra_volume_labels(
            args.volume_id, deployment_data_dir, args.class_map, args.report_type
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
