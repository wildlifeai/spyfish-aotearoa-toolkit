"""Drop-level frame sync between S3 and local disk.

Pairs with db_sync.py — same shape, scoped to per-drop image folders.
Use as a retrain preflight on NeSI to materialise image-volume drop
frames locally before training, or to push freshly-extracted local
frames up to S3 outside the Biigle upload flow.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from spyfish.config.wrapper import config
from spyfish.storage.s3_handler import S3Handler

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def download_drop_frames(drop_id: str) -> int:
    """Download all frames for `drop_id` from S3 to its local frames/ dir.

    Skips files already on disk with the same byte size. Returns the
    number of files newly downloaded.
    """
    s3 = S3Handler()
    s3_prefix = config.get_frames_s3_prefix(drop_id)
    local_dir = config.get_frames_dir(drop_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.s3.get_paginator("list_objects_v2")
    candidates: List[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=s3.bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(_IMAGE_EXTS):
                candidates.append((obj["Key"], int(obj["Size"])))

    if not candidates:
        logging.info(f"  {drop_id}: nothing under s3://{s3.bucket}/{s3_prefix}")
        return 0

    new = 0
    for key, size in candidates:
        local_path = local_dir / Path(key).name
        if local_path.exists() and local_path.stat().st_size == size:
            continue
        s3.download_object_from_s3(key, str(local_path))
        new += 1

    logging.info(
        f"  {drop_id}: downloaded {new} new / "
        f"{len(candidates)} total frames → {local_dir}"
    )
    return new


def upload_drop_frames(drop_id: str) -> int:
    """Upload all local frames for `drop_id` to S3.

    Skips keys already present on S3. Returns the number of files
    newly uploaded.
    """
    s3 = S3Handler()
    s3_prefix = config.get_frames_s3_prefix(drop_id)
    local_dir = config.get_frames_dir(drop_id)

    if not local_dir.exists():
        logging.warning(f"  {drop_id}: no local {local_dir} — nothing to upload.")
        return 0

    local_files = [p for p in local_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
    if not local_files:
        logging.info(f"  {drop_id}: no images in {local_dir}")
        return 0

    existing = s3.get_file_paths_set_from_s3(prefix=s3_prefix)
    new = 0
    for p in local_files:
        s3_key = s3_prefix + p.name
        if s3_key in existing:
            continue
        s3.upload_file_to_s3(str(p), key=s3_key, content_type="image/jpeg")
        new += 1

    logging.info(
        f"  {drop_id}: uploaded {new} new / "
        f"{len(local_files)} total frames → s3://{s3.bucket}/{s3_prefix}"
    )
    return new


def download_drops_frames(drop_ids: List[str]) -> Dict[str, int]:
    """Batch download. Returns {drop_id: new_files_count}."""
    return {d: download_drop_frames(d) for d in drop_ids}


def upload_drops_frames(drop_ids: List[str]) -> Dict[str, int]:
    """Batch upload. Returns {drop_id: new_files_count}."""
    return {d: upload_drop_frames(d) for d in drop_ids}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Sync per-drop frame JPEGs between S3 and local disk."
    )
    parser.add_argument(
        "direction",
        choices=("download", "upload"),
        help="Direction of sync.",
    )
    parser.add_argument(
        "drop_ids",
        nargs="+",
        metavar="DROP_ID",
        help="One or more drop IDs to sync.",
    )
    args = parser.parse_args()

    fn = download_drops_frames if args.direction == "download" else upload_drops_frames
    results = fn(args.drop_ids)
    total = sum(results.values())
    logging.info(
        f"\n{args.direction} complete: "
        f"{total} new file(s) across {len(results)} drop(s)."
    )


if __name__ == "__main__":
    main()
