"""
class_map.py — Seed and load the YOLO class map from a Biigle label tree.

The class map assigns a stable integer class_id per label. Species labels
(name contains " - ") are sorted by source_id (Biigle's AphiaID slot) so
IDs stay stable across re-seeds. Non-species labels are bucketed:
  - A label literally named "bait" (case-insensitive) → its own class.
  - Everything else → a shared "fish" class (long-tail bucket).

JSON shape on disk (one entry per class, keyed by class_id as string):
    {
      "0": {
        "class_id": 0,
        "aphia_id": 127140,
        "scientific_name": "Pagrus auratus",
        "common_name": "Snapper"
      },
      ...
      "N": {
        "class_id": N,
        "aphia_id": null,
        "scientific_name": "fish",
        "common_name": "fish",
        "aliases": ["To review", "Unidentified fish"]
      }
    }

`load_class_map()` returns a label_name → class_id lookup with entries for
the bare scientific name, the Biigle "Common - Scientific" form, and any
aliases — so downstream YOLO conversion matches whichever form the
annotation report emits.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional


def save_class_map(class_map: Dict[str, dict], path: Path) -> None:
    """Write the registry dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_map, f, indent=2)
    logging.info(f"Saved class map ({len(class_map)} classes) → {path}")


def load_class_map(path: Path) -> Dict[str, int]:
    """Load class_map.json and return a label_name → class_id lookup.

    Includes the bare scientific name, the Biigle "Common - Scientific"
    form, and any aliases (for bucket classes like bait/fish).
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
        for alias in entry.get("aliases", []):
            lookup[alias] = cid
    return lookup


def load_class_map_by_id(path: Path) -> Dict[int, str]:
    """Load class_map.json and return a class_id → scientific_name lookup.

    Inverse direction of `load_class_map`, for decoding a YOLO .txt row's
    class ID back to a species name when remapping across class-map versions.
    """
    with open(path) as f:
        registry = json.load(f)
    return {
        int(entry["class_id"]): entry["scientific_name"] for entry in registry.values()
    }


def _build_registry_from_labels(labels: List[dict]) -> Dict[str, dict]:
    """Partition Biigle label-tree entries into species / bait / fish and
    assign stable class IDs (species sorted by source_id, then bait, then fish)."""
    species, bait_names, fish_names = [], [], []
    for lbl in labels:
        name = (lbl.get("name") or "").strip()
        if not name:
            continue
        if " - " in name:
            species.append(lbl)
        elif "bait" in name.lower():
            bait_names.append(name)
        else:
            fish_names.append(name)

    species.sort(key=lambda label: int(label.get("source_id") or 0))

    registry: Dict[str, dict] = {}
    for class_id, lbl in enumerate(species):
        common, sci = lbl["name"].split(" - ", 1)
        registry[str(class_id)] = {
            "class_id": class_id,
            "aphia_id": int(lbl["source_id"]) if lbl.get("source_id") else None,
            "scientific_name": sci.strip(),
            "common_name": common.strip(),
        }

    if bait_names:
        cid = len(registry)
        registry[str(cid)] = {
            "class_id": cid,
            "aphia_id": None,
            "scientific_name": "bait",
            "common_name": "bait",
            "aliases": sorted(set(bait_names)),
        }

    if fish_names:
        cid = len(registry)
        registry[str(cid)] = {
            "class_id": cid,
            "aphia_id": None,
            "scientific_name": "fish",
            "common_name": "fish",
            "aliases": sorted(set(fish_names)),
        }

    return registry


# ──────────────────────────────────────────────────────────────────────
#  ⚠ TODO: MANUAL OVERRIDES — REMOVE WHEN UPSTREAM IS FIXED
# ──────────────────────────────────────────────────────────────────────
# Stop-gap entries that aren't yet in Biigle label tree 3511. Without
# them, every reseed_from_label_tree run drops these classes and any
# YOLO label using them becomes unresolvable.
#
# Fix: add these to the Biigle tree, then delete this block + the helper
# below (and the call from reseed_from_label_tree).
#
#   • Kyphosus bigibbus (Grey drummer)             — was in old S3 species list
#   • Kyphosus sp.      (Silver and grey drummers) — was in old S3 species list
#   • fish bucket       — catches workflow labels (e.g. 'Fish - Final',
#                         'To review') that Biigle annotators apply but
#                         which aren't species. Aliases get routed here.
# ──────────────────────────────────────────────────────────────────────
_MANUAL_OVERRIDES: List[dict] = [
    {
        "scientific_name": "Kyphosus bigibbus",
        "common_name": "Grey drummer",
        "aphia_id": 218707,
    },
    {
        "scientific_name": "Kyphosus sp.",
        "common_name": "Silver and grey drummers",
        "aphia_id": 126015,
    },
    {
        "scientific_name": "fish",
        "common_name": "fish",
        "aphia_id": None,
        "aliases": ["Fish - Final", "To review", "Fish: review required"],
    },
    {
        "scientific_name": "bait",
        "common_name": "bait",
        "aphia_id": None,
        "aliases": ["Bait", "Bait box"],
    },
]


def _apply_manual_overrides(registry: Dict[str, dict]) -> Dict[str, dict]:
    """Append/merge `_MANUAL_OVERRIDES` into a registry. See TODO above."""
    by_sci = {e["scientific_name"]: cid for cid, e in registry.items()}
    next_id = max((int(k) for k in registry.keys()), default=-1) + 1

    for ov in _MANUAL_OVERRIDES:
        sci = ov["scientific_name"]
        if sci in by_sci:
            existing = registry[by_sci[sci]]
            merged = sorted(
                set(existing.get("aliases", [])) | set(ov.get("aliases", []))
            )
            if merged:
                existing["aliases"] = merged
            logging.warning(
                f"  manual override: {sci!r} already in registry — merged aliases"
            )
            continue

        entry: Dict[str, object] = {
            "class_id": next_id,
            "aphia_id": ov.get("aphia_id"),
            "scientific_name": sci,
            "common_name": ov["common_name"],
        }
        if "aliases" in ov:
            entry["aliases"] = sorted(set(ov["aliases"]))
        registry[str(next_id)] = entry
        logging.warning(
            f"  manual override: appended class_id {next_id} → "
            f"{ov['common_name']} - {sci} (TODO: add to Biigle label tree)"
        )
        next_id += 1

    return registry


def reseed_from_label_tree(
    tree_id: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Fetch a Biigle label tree and write a fresh class_map.json.

    Labels are partitioned by the filter policy in `_build_registry_from_labels`:
    species (has " - "), bait (name == "bait"), fish (everything else).
    `_apply_manual_overrides` then re-adds project-specific stop-gap entries.
    """
    from spyfish.biigle.biigle_handler import BiigleHandler
    from spyfish.config.wrapper import config

    tree_id = tree_id or config.default_label_tree_id
    out_path = out_path or config.class_map_path

    labels = BiigleHandler().get_label_tree_labels(tree_id)
    registry = _build_registry_from_labels(labels)
    registry = _apply_manual_overrides(registry)

    species_count = sum(
        1 for e in registry.values() if e["scientific_name"] not in ("bait", "fish")
    )
    logging.info(
        f"Reseeded class_map from label tree {tree_id}: {species_count} species"
    )
    for bucket_name in ("bait", "fish"):
        entry = next(
            (e for e in registry.values() if e["scientific_name"] == bucket_name), None
        )
        if entry:
            logging.warning(
                f"  {bucket_name} class ({entry['class_id']}) — "
                f"{len(entry['aliases'])} label(s) collapsed into one YOLO class: "
                f"{entry['aliases']}"
            )

    save_class_map(registry, out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Reseed class_map.json from a Biigle label tree."
    )
    parser.add_argument(
        "--tree-id",
        type=int,
        default=None,
        help="Biigle label tree ID. Defaults to config.default_label_tree_id.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to config.class_map_path.",
    )
    args = parser.parse_args()
    reseed_from_label_tree(tree_id=args.tree_id, out_path=args.out)


if __name__ == "__main__":
    main()
