"""
class_map.py — Seed and load the YOLO class map from BUV Species.csv.

The class map assigns a stable integer class_id to each species. IDs are
assigned by sorting BUV Species.csv on AphiaID ascending — AphiaIDs change
only on rare WoRMS taxonomy revisions, so IDs stay stable across re-seeds.

JSON shape on disk (one entry per species, keyed by class_id as string):
    {
      "0": {
        "class_id": 0,
        "aphia_id": 127140,
        "scientific_name": "Pagrus auratus",
        "common_name": "Snapper"
      }, ...
    }

`load_class_map()` returns a flat label_name → class_id dict containing
BOTH the bare scientific name AND the Biigle-style "Common - Scientific"
form, so downstream YOLO conversion matches whichever form appears in the
incoming annotation rows.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

APHIA_ID_COL = "AphiaID"
COMMON_NAME_COL = "CommonName"
SCIENTIFIC_NAME_COL = "ScientificName"


def build_class_map_from_species(species_df: pd.DataFrame) -> Dict[str, dict]:
    """Build the class_map registry dict from a species DataFrame, sorted by AphiaID asc."""
    required = {APHIA_ID_COL, COMMON_NAME_COL, SCIENTIFIC_NAME_COL}
    missing = required - set(species_df.columns)
    if missing:
        raise ValueError(
            f"Species CSV missing required columns: {sorted(missing)}. "
            f"Got: {sorted(species_df.columns)}"
        )

    df = species_df[list(required)].copy()
    df[APHIA_ID_COL] = pd.to_numeric(
        df[APHIA_ID_COL].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    sorted_df = (
        df.dropna(subset=[APHIA_ID_COL, SCIENTIFIC_NAME_COL])
        .astype({APHIA_ID_COL: "int64"})
        .drop_duplicates(subset=[APHIA_ID_COL])
        .sort_values(APHIA_ID_COL)
        .reset_index(drop=True)
    )

    return {
        str(class_id): {
            "class_id": class_id,
            "aphia_id": int(row[APHIA_ID_COL]),
            "scientific_name": str(row[SCIENTIFIC_NAME_COL]),
            "common_name": str(row[COMMON_NAME_COL]),
        }
        for class_id, (_, row) in enumerate(sorted_df.iterrows())
    }


def save_class_map(class_map: Dict[str, dict], path: Path) -> None:
    """Write the registry dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_map, f, indent=2)
    logging.info(f"Saved class map ({len(class_map)} classes) → {path}")


def load_class_map(path: Path) -> Dict[str, int]:
    """Load class_map.json and return a label_name → class_id lookup.

    Returns entries for both the bare scientific name and the Biigle
    "Common - Scientific" form so YOLO conversion matches either.
    """
    with open(path) as f:
        registry = json.load(f)

    lookup: Dict[str, int] = {}
    for entry in registry.values():
        cid = int(entry["class_id"])
        sci = entry["scientific_name"]
        common = entry.get("common_name", "")
        lookup[sci] = cid
        if common:
            lookup[f"{common} - {sci}"] = cid
    return lookup


def reseed_from_s3(out_path: Optional[Path] = None) -> Path:
    """Read BUV Species.csv from S3 and write a fresh class_map.json."""
    from spyfish.config.wrapper import config
    from spyfish.storage.s3_handler import S3Handler

    out_path = out_path or config.class_map_path
    storage = S3Handler(bucket=config.s3_bucket)
    species_df = storage.read_df_from_s3_csv(config.s3_sharepoint_species_csv)
    class_map = build_class_map_from_species(species_df)
    save_class_map(class_map, out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Reseed class_map.json from BUV Species.csv on S3."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to config.class_map_path.",
    )
    args = parser.parse_args()
    reseed_from_s3(args.out)


if __name__ == "__main__":
    main()
