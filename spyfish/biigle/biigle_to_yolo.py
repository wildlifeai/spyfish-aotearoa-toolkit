"""
biigle_to_yolo.py — Convert local Biigle expert CSV exports → YOLO label .txt files.

This tool is strictly offline; it consumes CSVs previously exported by sync_biigle_annotations
into process_files/data_quality/{drop_id}/annotations/.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

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


def build_class_map(
    df: pd.DataFrame, class_map_path: Optional[Path] = None
) -> Dict[str, int]:
    """
    Build a stable label_name → YOLO class_id mapping.

    Stability rule: sort by Biigle label_id ascending (labels are created in order;
    new labels append, so this mapping is stable across runs unless labels are deleted).

    Args:
        df: Annotation DataFrame containing 'label_id' and 'label_name' columns.
        class_map_path: If provided, load existing map from this JSON file (and extend if new labels appear).

    Returns:
        Dict mapping label_name → integer class ID (0-indexed).
    """
    existing_map: Dict[str, int] = {}
    if class_map_path and class_map_path.exists():
        with open(class_map_path) as f:
            existing_map = {v["name"]: v["class_id"] for v in json.load(f).values()}
        logging.info(
            f"Loaded existing class map with {len(existing_map)} labels from {class_map_path}"
        )

    # Stable sort: by the first (lowest) Biigle label_id seen for each label name
    label_info = (
        df[["label_id", "label_name"]]
        .drop_duplicates("label_name")
        .sort_values("label_id")
    )

    for _, row in label_info.iterrows():
        name = row["label_name"]
        if name not in existing_map:
            existing_map[name] = len(existing_map)

    return existing_map


def save_class_map(class_map: Dict[str, int], path: Path) -> None:
    """Persist class_map as {class_id: {name, class_id}} JSON for human readability."""
    serialisable = {
        str(cid): {"name": name, "class_id": cid} for name, cid in class_map.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logging.info(f"Saved class map ({len(class_map)} classes) → {path}")


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
    labels_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, int] = {}

    for filename, group in df.groupby("filename"):
        img_w, img_h = default_img_size
        lines = []

        for _, row in group.iterrows():
            label_name = row["label_name"]
            class_id = class_map.get(label_name)
            if class_id is None:
                continue

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
        for ext in (".jpg", ".jpeg", ".png"):
            # Search in all subdirectories of images_dir (e.g. data_quality/drop_id/biigle_frames/)
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
    data_quality_dir: Path,
    class_map_path: Path,
) -> Dict[str, int]:
    """
    Finds all expert CSVs in data_quality and converts them to YOLO .txt files.

    Labels are written into each drop's annotations/ folder alongside the source CSV.
    Use biigle_to_yolo_collect() afterwards to copy them into a flat staging directory.
    """
    logging.info(f"Searching for expert CSVs in {data_quality_dir}...")
    csv_paths = []
    all_dfs = []

    # Strictly use the per-drop expert raw CSVs
    for csv_path in sorted(
        data_quality_dir.glob("**/annotations/*_biigle_expert_raw.csv")
    ):
        logging.debug(f"  Found expert CSV: {csv_path}")
        csv_paths.append(csv_path)
        all_dfs.append(pd.read_csv(csv_path))

    if not all_dfs:
        logging.warning("No expert CSV files found. Retraining cannot proceed.")
        return {}

    # Build class map across all drops first so class IDs are consistent
    df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Loaded {len(df)} annotations from {len(all_dfs)} CSVs.")
    class_map = build_class_map(df, class_map_path)
    save_class_map(class_map, class_map_path)

    # Write YOLO .txt labels into each drop's annotations/ folder
    for csv_path, drop_df in zip(csv_paths, all_dfs):
        convert_annotations_to_yolo(drop_df, class_map, csv_path.parent)
        logging.info(
            f"  Wrote labels for {csv_path.parent.parent.name} → {csv_path.parent}"
        )

    return class_map


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Convert local Biigle expert CSVs to YOLO labels."
    )
    parser.add_argument(
        "--data-dir", required=True, type=Path, help="Root data_quality directory"
    )
    parser.add_argument(
        "--class-map",
        required=True,
        type=Path,
        help="Path to write/update the class_map.json",
    )

    args = parser.parse_args()

    biigle_to_yolo(args.data_dir, args.class_map)


if __name__ == "__main__":
    main()
