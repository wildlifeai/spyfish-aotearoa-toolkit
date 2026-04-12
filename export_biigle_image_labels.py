"""
export_biigle_image_labels.py — One-off script to download a Biigle image-volume
annotation report and convert it to YOLO .txt label files.

Use this for datasets that don't follow the dropID pipeline workflow
(e.g. legacy image volumes, external datasets).

Images should be on local disk; this script only downloads annotations
and writes labels alongside a class_map.json.

Output layout:
    <output_dir>/
        raw_annotations.csv      -- raw Biigle export
        class_map.json           -- label_name → YOLO class_id mapping
        labels/
            <image_stem>.txt     -- one YOLO label file per image

Usage:
    python export_biigle_image_labels.py \\
        --volume-id 12345 \\
        [--output-dir process_files/old_labels] \\
        [--no-cache]

Credentials are read from BIIGLE_API_EMAIL and BIIGLE_API_TOKEN in your .env file.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from spyfish.biigle.biigle_parser import BiigleParser
from spyfish.biigle.biigle_to_yolo import (
    build_class_map,
    convert_annotations_to_yolo,
    save_class_map,
)
from spyfish.config.wrapper import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Biigle image-volume annotation report and export YOLO labels."
    )
    parser.add_argument(
        "--volume-id",
        required=True,
        type=int,
        help="Biigle volume ID (the number in the Biigle URL, e.g. https://biigle.de/volumes/12345)",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Biigle account email. Defaults to BIIGLE_API_EMAIL from .env / environment.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Biigle API token. Defaults to BIIGLE_API_TOKEN from .env / environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write outputs into. "
            "Defaults to process_files/old_labels/<volume_id>/"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download even if a cached ZIP exists.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()

    output_dir: Path = args.output_dir or (
        config.data_quality_dir.parent / "old_labels" / str(args.volume_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    class_map_path = config.local_training_dir / "class_map.json"
    raw_csv_path = output_dir / "raw_annotations.csv"

    logging.info(f"Volume ID  : {args.volume_id}")
    logging.info(f"Output dir : {output_dir}")

    # ── 1. Download raw annotations ─────────────────────────────────────────
    parser = BiigleParser(
        email=args.email,
        token=args.token,
    )

    logging.info("Downloading image annotation report from Biigle…")
    df: pd.DataFrame = parser.download_volume_annotations(
        volume_id=args.volume_id,
        type_id=config.annotation_report_type_images,
        use_cache=not args.no_cache,
    )

    if df.empty:
        logging.error("No annotations returned. Check the volume ID and credentials.")
        sys.exit(1)

    logging.info(f"Downloaded {len(df)} annotation rows.")
    df.to_csv(raw_csv_path, index=False)
    logging.info(f"Raw annotations saved → {raw_csv_path}")

    # ── 2. Build class map ──────────────────────────────────────────────────
    class_map = build_class_map(df, class_map_path)
    save_class_map(class_map, class_map_path)
    logging.info(f"Class map ({len(class_map)} classes) saved → {class_map_path}")
    for name, cid in sorted(class_map.items(), key=lambda x: x[1]):
        logging.info(f"  {cid:3d}  {name}")

    # ── 3. Convert to YOLO labels ───────────────────────────────────────────
    summary = convert_annotations_to_yolo(df, class_map, labels_dir)

    total_annotations = sum(summary.values())
    logging.info(
        f"Wrote {len(summary)} label files ({total_annotations} annotations) → {labels_dir}"
    )

    images_missing = [f for f, count in summary.items() if count == 0]
    if images_missing:
        logging.warning(
            f"{len(images_missing)} images had no valid annotations (empty .txt files written)."
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()
