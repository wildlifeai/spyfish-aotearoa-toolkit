"""
prepare_training_data.py — Prepare expert annotations + assemble YOLO datasets.

Composable functions:

  prepare_from_annotations()         — Load expert MaxN annotations into a DataFrame.
  count_boxes_per_species_in_source()— Pre-scan source labels for per-species box counts.
  identify_floor_species()           — Names of species below class_floor_pct (merged → 'fish').
  flatten_and_remap_labels()         — Walk source labels, remap class IDs to unified ordering.
  discover_extra_drops()             — Find drops with labels but no MaxN data.
  copy_split_files()                 — Copy images + labels into YOLO train/val/test layout.
  make_binary_labels()               — Convert species labels → binary (all → class 0).
  generate_data_yaml()               — Write a YOLO data.yaml for a given split.
  assemble_yolo_dataset()            — Top-level: builds species + binary datasets from drop lists.

Typical orchestration order (see retrain_runner.py):
  1. biigle_to_yolo.py                → writes per-drop label files
  2. prepare_from_annotations()       → MaxN df + species list
  3. discover_extra_drops()           → adds extras' species to the unified list
  4. count_boxes_per_species_in_source() + identify_floor_species()
                                       → trims unified species list (rare → fish)
  5. flatten_and_remap_labels()       → stages labels with unified class IDs
  6. split_data.py                    → assigns drops to train/val/test
  7. assemble_yolo_dataset()          → writes the final YOLO dataset + data.yaml
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd
import yaml

from spyfish.biigle.class_map import load_class_map, load_class_map_by_id
from spyfish.config.wrapper import config

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------


def prepare_from_annotations(
    deployment_data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load expert MaxN annotations and return (df, species_names).

    Globs for *_biigle_expert_maxn.csv under deployment_data_dir. Drops
    listed in `training_excluded_drops_file` are filtered out. No balancing —
    the orchestrator applies the floor by trimming `unified_names` before
    `flatten_and_remap_labels` so floored species fall back to 'fish'.
    """
    deployment_data_dir = deployment_data_dir or config.deployment_data_dir

    logging.info(f"Loading expert MaxN annotations from {deployment_data_dir}...")
    maxn_glob = f"**/annotations/*{config.biigle_expert_maxn_suffix}"
    all_dfs = [
        pd.read_csv(csv_path) for csv_path in deployment_data_dir.glob(maxn_glob)
    ]

    if not all_dfs:
        raise RuntimeError(
            f"No expert MaxN CSVs found in {deployment_data_dir}. "
            "Run sync_biigle_annotations first."
        )

    df = pd.concat(all_dfs, ignore_index=True)
    logging.info(
        f"Loaded {len(df)} expert MaxN rows from {df['DropID'].nunique()} drops."
    )

    excluded = config.training_excluded_drops
    if excluded:
        loaded_drops = set(df["DropID"].unique())
        hit = loaded_drops & excluded
        if hit:
            df = df[~df["DropID"].isin(hit)]
            logging.info(
                f"Excluded {len(hit)} drop(s) per {config.training_excluded_drops_file.name}: "
                f"{sorted(hit)}"
            )
        stale = excluded - loaded_drops
        if stale:
            logging.warning(
                f"{len(stale)} drop(s) listed in {config.training_excluded_drops_file.name} "
                f"were not found in loaded annotations (stale entries?): {sorted(stale)}"
            )

    species_class_names = sorted(df["ScientificName"].unique().tolist())
    return df, species_class_names


# ---------------------------------------------------------------------------
# Label remapping (unify class IDs across sources)
# ---------------------------------------------------------------------------


def _build_id_remap(
    src_class_map_path: Path,
    unified_names: List[str],
    fallback_species: str = "fish",
) -> Dict[int, Optional[int]]:
    """Build {old_id: new_id} by decoding the source map to species names, then
    looking each up in `unified_names`. Species missing from `unified_names`
    redirect to `fallback_species` if present, otherwise map to None (dropped).
    """
    src_map = load_class_map_by_id(src_class_map_path)
    unified_ids = {name: idx for idx, name in enumerate(unified_names)}
    fallback_id = unified_ids.get(fallback_species)
    return {
        old_id: unified_ids.get(species, fallback_id)
        for old_id, species in src_map.items()
    }


def _rewrite_label_file(
    src_txt: Path, dst_txt: Path, id_remap: Dict[int, Optional[int]]
) -> None:
    """Rewrite a single YOLO .txt through id_remap, dropping lines whose class
    resolves to None."""
    out_lines = []
    for line in src_txt.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        old_id = int(parts[0])
        new_id = id_remap.get(old_id)
        if new_id is None:
            continue
        out_lines.append(f"{new_id} " + " ".join(parts[1:]))
    dst_txt.write_text("\n".join(out_lines))


def _iter_class_ids(label_path: Path) -> Iterator[int]:
    """Yield class IDs from a YOLO label file, skipping blank or malformed lines."""
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            yield int(parts[0])
        except ValueError:
            continue


def count_boxes_per_species_in_source(
    deployment_data_dir: Path,
    global_class_map_path: Path,
) -> Dict[str, int]:
    """Walk every <drop>/labels/*.txt and count bounding boxes per species name.

    Class IDs resolve via the per-drop sidecar (annotations/class_map.json) if
    present, **falling back to the global class_map** when the sidecar doesn't
    know the id — sidecars can lag behind the global when species are added
    after the sidecar was first written.
    """
    global_map = (
        load_class_map_by_id(global_class_map_path)
        if global_class_map_path.exists()
        else {}
    )
    counts: Dict[str, int] = {}
    for labels_dir in deployment_data_dir.glob("**/labels"):
        drop_dir = labels_dir.parent
        sidecar = drop_dir / "annotations" / "class_map.json"
        local_map = load_class_map_by_id(sidecar) if sidecar.exists() else {}
        if not local_map and not global_map:
            continue
        for txt in labels_dir.glob("*.txt"):
            for cid in _iter_class_ids(txt):
                name = local_map.get(cid) or global_map.get(cid)
                if name:
                    counts[name] = counts.get(name, 0) + 1
    return counts


def print_species_totals(
    box_counts: Dict[str, int], floor_species: Optional[Set[str]] = None
) -> None:
    """One-time pre-training overview of species composition.

    Shows total bounding-box counts and fractions, sorted descending. Species
    that will be merged into 'fish' by the floor are marked so the user sees
    both the raw distribution and the floor decision in the same view.
    """
    if not box_counts:
        return
    floor_species = floor_species or set()
    total = sum(box_counts.values())
    sorted_items = sorted(box_counts.items(), key=lambda kv: -kv[1])

    logging.info("\n=== Pre-training species totals (bounding boxes) ===")
    for name, n in sorted_items:
        marker = "  → merged into 'fish'" if name in floor_species else ""
        logging.info(f"  {name:<40} {n / total:.1%}  ({n} boxes){marker}")
    logging.info(f"  Total boxes: {total}  |  Species: {len(sorted_items)}")
    if floor_species:
        kept = len(sorted_items) - len(floor_species)
        logging.info(f"  After floor: {kept} species + 'fish' bucket")
    logging.info("=" * 60 + "\n")


def identify_floor_species(
    box_counts: Dict[str, int],
    floor_pct: float,
    fallback_species: str = "fish",
) -> Set[str]:
    """Species whose bounding-box fraction is below `floor_pct`.

    These are merged into `fallback_species` by `flatten_and_remap_labels`
    (via the unknown-species fallback path — the species is simply absent
    from `unified_names`, so its source class IDs land on the fallback).

    Hardcoded exemptions (never floored, even when below threshold):
      - `fallback_species` itself (we'd have nowhere to redirect to)
      - "bait" — domain-critical: the bait box is visible in every frame and
        must stay a separate class so MaxN inference can exclude it from fish
        counts. Merging into fish would inflate every drop's MaxN by ~1.
    """
    never_floor = {fallback_species, "bait"}
    total = sum(box_counts.values())
    if total == 0:
        return set()
    return {
        name
        for name, n in box_counts.items()
        if n / total < floor_pct and name not in never_floor
    }


def discover_extra_drops(deployment_data_dir: Path) -> Tuple[List[str], List[str]]:
    """Find drops with labels + frames but no MaxN CSV (extras from download-volume
    or manually-dropped datasets).

    Returns (drop_ids, extra_species_names). Extras bypass ceiling/floor balancing
    by design; their species are unioned into the unified class list so they get
    their own class IDs rather than falling back to "fish".

    Species detection order:
      1. Read the raw Biigle CSV if present (authoritative label names).
      2. Fall back to decoding YOLO class IDs via a class_map.json sidecar in
         annotations/ — required for Route B (manually-authored labels, no CSV).

    Drops without either are skipped with a warning.
    """
    extras: List[str] = []
    species_set: set[str] = set()

    for labels_dir in deployment_data_dir.glob("**/labels"):
        drop_dir = labels_dir.parent
        drop_id = drop_dir.name
        annotations_dir = drop_dir / "annotations"
        frames_dir = drop_dir / "frames"

        # Skip if MaxN is present — normal pipeline handles this drop.
        maxn_suffix = config.biigle_expert_maxn_suffix
        if annotations_dir.is_dir() and any(annotations_dir.glob(f"*{maxn_suffix}")):
            continue

        label_files = list(labels_dir.glob("*.txt"))
        if not label_files:
            continue
        if not frames_dir.is_dir() or not any(frames_dir.iterdir()):
            continue

        # Two distinct lookup needs:
        #   - raw-CSV path: resolve current label names → use the GLOBAL class_map
        #     (authoritative for current aliases like 'Bait', 'Fish: review required').
        #     Step 1 of retrain (biigle_to_yolo) already rewrote .txt files against
        #     the global, so no historical-ID concern here.
        #   - .txt-only path: decode pre-existing YOLO IDs that were written at some
        #     prior point — prefer a per-drop sidecar if present, else fall back to
        #     the global. Sidecar preserves the original ID→species mapping.
        if not config.class_map_path.exists():
            logging.warning(
                f"Skipping extra drop {drop_id}: global class_map missing at "
                f"{config.class_map_path}."
            )
            continue

        drop_species: set[str] = set()
        raw_suffix = config.biigle_expert_raw_suffix
        raw_csvs = (
            list(annotations_dir.glob(f"*{raw_suffix}"))
            if annotations_dir.is_dir()
            else []
        )
        if raw_csvs:
            name_to_id = load_class_map(config.class_map_path)
            id_to_name = load_class_map_by_id(config.class_map_path)
            # _MANUAL_OVERRIDES guarantees the fish bucket exists in the global map.
            fish_cid = name_to_id["fish"]
            df = pd.read_csv(raw_csvs[0])
            for label in (
                df.get("label_name", pd.Series()).dropna().astype(str).unique()
            ):
                cid = name_to_id.get(label) or name_to_id.get(label.strip())
                if cid is None:
                    logging.warning(
                        f"  {drop_id}: label {label!r} not in class_map — "
                        "routed to 'fish' bucket. Consider adding to _MANUAL_OVERRIDES."
                    )
                    cid = fish_cid
                drop_species.add(id_to_name[cid])
        else:
            sidecar = annotations_dir / "class_map.json"
            decode_path = sidecar if sidecar.exists() else config.class_map_path
            id_to_name = load_class_map_by_id(decode_path)
            for lf in label_files:
                for line in lf.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cid = int(parts[0])
                        except ValueError:
                            continue
                        if cid in id_to_name:
                            drop_species.add(id_to_name[cid])

        extras.append(drop_id)
        species_set.update(drop_species)

    if extras:
        logging.info(
            f"discover_extra_drops: {len(extras)} extras → train split "
            f"(species contributed: {sorted(species_set)})"
        )
    return extras, sorted(species_set)


def flatten_and_remap_labels(
    deployment_data_dir: Path,
    src_class_map_path: Path,
    unified_names: List[str],
    dst_dir: Path,
    fallback_species: str = "fish",
) -> int:
    """Walk per-drop `labels/*.txt`, remap class IDs to the unified ordering
    (index in `unified_names`), write into `dst_dir/<drop_id>/`. Returns file count.

    Per-drop subdir layout decouples the downstream lookup from filename
    prefixes — works for UUID-stemmed labels (e.g. Biigle web-UI uploads)
    that don't naturally start with their drop_id.
    """
    id_remap = _build_id_remap(src_class_map_path, unified_names, fallback_species)
    dst_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for src_txt in deployment_data_dir.glob("**/labels/*.txt"):
        drop_id = src_txt.parent.parent.name
        drop_dst = dst_dir / drop_id
        drop_dst.mkdir(parents=True, exist_ok=True)
        _rewrite_label_file(src_txt, drop_dst / src_txt.name, id_remap)
        n += 1

    logging.info(
        f"flatten_and_remap_labels: {n} files → {dst_dir}/<drop_id>/ "
        f"(unified classes: {len(unified_names)})"
    )
    return n


# ---------------------------------------------------------------------------
# Binary label creation from existing YOLO labels
# ---------------------------------------------------------------------------


def make_binary_labels(
    species_labels_dir: Path,
    binary_labels_dir: Path,
    overwrite: bool = False,
) -> int:
    """
    Convert existing multi-class YOLO .txt label files → binary labels.

    Each annotation's class ID is remapped to 0 (regardless of original class).
    Empty label files (no detections) are preserved as-is.

    This means you can train a binary fish-detector from the same images
    without re-downloading or re-annotating anything.

    Args:
        species_labels_dir: Directory containing multi-class YOLO .txt files.
        binary_labels_dir: Output directory for binary .txt files.
        overwrite: If False, skip files that already exist in binary_labels_dir.

    Returns:
        Number of label files processed.
    """
    binary_labels_dir.mkdir(parents=True, exist_ok=True)

    # species_labels_dir has per-drop subdirs (from flatten_and_remap_labels);
    # mirror that structure in binary_labels_dir so copy_split_files() finds them.
    label_files = list(species_labels_dir.glob("*/*.txt"))
    if not label_files:
        logging.warning(f"No .txt files found in {species_labels_dir}")
        return 0

    processed = 0
    for src_path in label_files:
        drop_id = src_path.parent.name
        dst_drop_dir = binary_labels_dir / drop_id
        dst_drop_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_drop_dir / src_path.name

        if dst_path.exists() and not overwrite:
            continue

        lines = src_path.read_text().strip().splitlines()
        binary_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Replace class_id (index 0) with 0, keep cx cy w h
                binary_lines.append("0 " + " ".join(parts[1:]))
            elif parts:
                logging.debug(f"Unexpected label format in {src_path.name}: {line!r}")

        dst_path.write_text("\n".join(binary_lines))
        processed += 1

    logging.info(
        f"make_binary_labels: wrote {processed} binary label files → {binary_labels_dir}/<drop_id>/"
    )
    return processed


# ---------------------------------------------------------------------------
# Dataset layout helpers
# ---------------------------------------------------------------------------


def _build_image_index(images_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Map drop_id → {frame-image stem → path}, restricted to canonical `frames/` dirs.

    Scoping by drop_id prevents stem collisions across deployments from silently
    pairing a label with the wrong drop's image.

    One walk of `images_dir`; downstream lookups are O(1). Excludes derivative
    dirs (qa_frames, zooniverse_frames, biigle_frames, …) by checking that the
    immediate parent is exactly `frames`.
    """
    exts = {e.lower() for e in config.image_extensions}
    index: Dict[str, Dict[str, Path]] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and p.parent.name == "frames":
            drop_id = p.parent.parent.name
            index.setdefault(drop_id, {}).setdefault(p.stem, p)
    return index


def copy_split_files(
    drop_ids: List[str],
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    split_name: str,
    symlink: bool = False,
    image_index: Optional[Dict[str, Dict[str, Path]]] = None,
) -> Tuple[int, int]:
    """
    Copy (or symlink) images and their corresponding label files into a
    canonical YOLO dataset layout:

        output_dir/
          images/{split_name}/   ← source JPEGs
          labels/{split_name}/   ← YOLO .txt label files

    Looks up each label's matching image via `image_index` (drop_id → stem →
    Path). If not provided, the index is built from `images_dir` on entry —
    pass a pre-built index when calling this multiple times against the same
    `images_dir` to avoid redundant tree walks.

    Args:
        drop_ids: List of DropIDs to include in this split.
        images_dir: Source directory containing JPEG frames (root data_quality).
        labels_dir: Source directory containing YOLO .txt label files.
        output_dir: Root output directory.
        split_name: One of 'train', 'val', 'test'.
        symlink: Use symlinks instead of copies.
        image_index: Optional pre-built `{drop_id: {stem: Path}}` map; built
            locally if None.

    Returns:
        (n_images_copied, n_labels_copied)
    """
    img_out = output_dir / "images" / split_name
    lbl_out = output_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    if image_index is None:
        image_index = _build_image_index(images_dir)

    n_images, n_labels = 0, 0

    for drop_id in drop_ids:
        # Labels live in labels_dir/<drop_id>/ after flatten_and_remap_labels.
        drop_labels_dir = labels_dir / drop_id
        drop_labels = (
            list(drop_labels_dir.glob("*.txt")) if drop_labels_dir.is_dir() else []
        )
        if not drop_labels:
            logging.warning(
                f"No label files found for {drop_id} in {labels_dir} — skipping."
            )
            continue

        for lbl_path in drop_labels:
            img_path = image_index.get(drop_id, {}).get(lbl_path.stem)
            if img_path is None:
                logging.warning(
                    f"No image found for label {drop_id}/{lbl_path.stem} — skipping."
                )
                continue

            dst_img = img_out / img_path.name
            dst_lbl = lbl_out / lbl_path.name

            if symlink:
                if not dst_img.exists():
                    dst_img.symlink_to(img_path.resolve())
                if not dst_lbl.exists():
                    dst_lbl.symlink_to(lbl_path.resolve())
            else:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)

            n_images += 1
            n_labels += 1

    log_fn = logging.warning if (n_images == 0 and drop_ids) else logging.info
    log_fn(
        f"copy_split_files [{split_name}]: {n_images} images + {n_labels} labels → {output_dir}"
    )
    if n_images == 0 and drop_ids:
        logging.warning(
            f"  No labelled images found for {split_name} split — "
            "check that expert frames exist under images_dir and labels_dir is populated."
        )
    return n_images, n_labels


def generate_data_yaml(
    class_names: List[str],
    output_dir: Path,
    filename: str = "data.yaml",
) -> Path:
    """
    Write a YOLO-compatible data.yaml pointing at the canonical dataset layout
    (images/train, images/val, images/test) relative to output_dir.

    Args:
        class_names: Ordered list of class names (index = YOLO class ID).
        output_dir: Root of the dataset (contains images/ and labels/ subdirs).
        filename: Output YAML filename (default: data.yaml).

    Returns:
        Path to the generated data.yaml.
    """
    data = {
        "nc": len(class_names),
        "names": class_names,
        "train": str(output_dir / "images" / "train"),
        "val": str(output_dir / "images" / "val"),
    }

    test_dir = output_dir / "images" / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        data["test"] = str(test_dir)

    yaml_path = output_dir / filename
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logging.info(f"generate_data_yaml: {len(class_names)} classes → {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Convenience: build a full YOLO dataset from splits + labels in one call
# ---------------------------------------------------------------------------


def _write_sidecar_class_map(
    class_names: List[str],
    output_dir: Path,
    source_class_map_path: Optional[Path] = None,
) -> Path:
    """Write a class_map.json sidecar next to data.yaml, reordered to match class_names.

    Pulls richer metadata (aphia_id, common_name, aliases) from the source class
    map when scientific names match; falls back to minimal entries otherwise.
    """
    source_lookup: Dict[str, dict] = {}
    if source_class_map_path and source_class_map_path.exists():
        registry = json.loads(source_class_map_path.read_text())
        source_lookup = {e["scientific_name"]: e for e in registry.values()}

    sidecar: Dict[str, dict] = {}
    for idx, name in enumerate(class_names):
        src = source_lookup.get(name, {})
        entry = {
            "class_id": idx,
            "aphia_id": src.get("aphia_id"),
            "scientific_name": name,
            "common_name": src.get("common_name", name),
        }
        if "aliases" in src:
            entry["aliases"] = src["aliases"]
        sidecar[str(idx)] = entry

    sidecar_path = output_dir / "class_map.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    logging.info(f"Wrote sidecar class_map → {sidecar_path}")
    return sidecar_path


def _clean_yolo_split_dirs(yolo_root: Path) -> None:
    """Wipe images/{train,val,test} and labels/{train,val,test} under `yolo_root`.

    Called before re-assembling so each retraining run is idempotent: stale
    files from drops no longer in the split (e.g. newly excluded via
    training_excluded_drops.txt) don't leak into training.
    """
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            d = yolo_root / kind / split
            if d.exists():
                shutil.rmtree(d)


def print_per_drop_species_inventory(
    labels_staged_dir: Path, class_names: List[str]
) -> None:
    """Per-drop bounding-box counts in each species, one row per drop.

    Reads from labels_staged_dir (post-floor, post-remap state — what the model
    will actually see). Format: ``DropID  Species1=N  Species2=M  ...``.
    """
    if not labels_staged_dir.exists():
        return

    counts: Dict[str, Dict[str, int]] = {}
    for drop_dir in sorted(labels_staged_dir.iterdir()):
        if not drop_dir.is_dir():
            continue
        per_drop: Dict[str, int] = {}
        for txt in drop_dir.glob("*.txt"):
            for cid in _iter_class_ids(txt):
                if 0 <= cid < len(class_names):
                    name = class_names[cid]
                    per_drop[name] = per_drop.get(name, 0) + 1
        if per_drop:
            counts[drop_dir.name] = per_drop

    if not counts:
        return

    logging.info("\n=== Pre-split species inventory (from labels_staged) ===")
    for drop in sorted(counts):
        species_pairs = sorted(counts[drop].items(), key=lambda kv: -kv[1])
        total = sum(n for _, n in species_pairs)
        species_str = "  ".join(f"{name}={n}" for name, n in species_pairs)
        logging.info(f"  {drop}  ({total} boxes)  {species_str}")
    logging.info("=" * 60 + "\n")


def _group_drops_by_survey(
    drops: List[str], extra_drops: Set[str]
) -> List[Tuple[str, List[str]]]:
    """Group drops by survey for printout. Extras land in their own '[extras]' bucket.

    Falls back to '[extras]' for any drop whose ID doesn't parse as a valid
    DropID (e.g. 'volume_30505') so unconventional IDs don't crash the print.
    """
    by_survey: Dict[str, List[str]] = {}
    extras_in_split: List[str] = []
    for d in drops:
        if d in extra_drops:
            extras_in_split.append(d)
            continue
        try:
            survey = config.get_survey_id_from_drop(d)
        except Exception:
            extras_in_split.append(d)
            continue
        by_survey.setdefault(survey, []).append(d)
    groups: List[Tuple[str, List[str]]] = sorted(
        (s, sorted(ds)) for s, ds in by_survey.items()
    )
    if extras_in_split:
        groups.append(("[extras]", sorted(extras_in_split)))
    return groups


def print_assembled_summary(
    species_dir: Path,
    class_names: List[str],
    train_drops: List[str],
    val_drops: List[str],
    test_drops: List[str],
    extra_drops: Optional[Set[str]] = None,
) -> None:
    """Comprehensive post-assembly summary: drops per split + bounding-box counts.

    Reflects the FINAL composition the model trains on — includes extras
    (drops without MaxN data). Two sections:

      1. Drops per split, grouped by survey, with extras tagged separately.
      2. Per-species bounding-box counts read from the on-disk YOLO labels.

    Bounding-box counts here are post-floor (rare species merged into 'fish').
    """
    extras = set(extra_drops or [])

    # --- Drops per split ---
    logging.info("\n=== Drops per split (assembled training set) ===")
    for split_name, drops in [
        ("TRAIN", train_drops),
        ("VAL", val_drops),
        ("TEST", test_drops),
    ]:
        if not drops:
            logging.info(f"{split_name} (0 drops): empty")
            continue
        groups = _group_drops_by_survey(drops, extras)
        logging.info(f"{split_name} ({len(drops)} drops, {len(groups)} groups):")
        for group_name, ds in groups:
            logging.info(f"  {group_name} ({len(ds)}): {', '.join(ds)}")
    logging.info("=" * 60)

    # --- Per-species bounding-box counts ---
    counts: Dict[str, Dict[str, int]] = {
        name: {"train": 0, "val": 0, "test": 0} for name in class_names
    }
    for split in ("train", "val", "test"):
        labels_dir = species_dir / "labels" / split
        if not labels_dir.is_dir():
            continue
        for txt in labels_dir.glob("*.txt"):
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                if 0 <= cid < len(class_names):
                    counts[class_names[cid]][split] += 1

    df = pd.DataFrame.from_dict(counts, orient="index")[["train", "val", "test"]]
    df["total"] = df.sum(axis=1)
    df = df[df["total"] > 0].sort_values("total", ascending=False)

    if df.empty:
        logging.warning(
            "Assembled YOLO dataset contains no annotations — "
            f"check {species_dir / 'labels'}."
        )
        return

    logging.info(
        "\n=== Per-species bounding-box counts (post-balance, includes extras) ==="
    )
    logging.info(df.to_string())
    logging.info("=" * 90 + "\n")


def assemble_yolo_dataset(
    train_drops: List[str],
    val_drops: List[str],
    test_drops: List[str],
    images_dir: Path,
    species_labels_dir: Path,
    output_dir: Path,
    class_names: List[str],
    build_binary: bool = True,
    symlink: bool = False,
    source_class_map_path: Optional[Path] = None,
    extra_drops: Optional[Set[str]] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Assemble a complete YOLO dataset (species + optional binary) from pre-split drop lists.

    Creates:
        output_dir/species/{images,labels}/{train,val,test}/   + data.yaml + class_map.json
        output_dir/binary/{images,labels}/{train,val,test}/    + data.yaml  (if build_binary=True)

    Extras (drops discovered via `discover_extra_drops`) should be appended to
    `train_drops` before calling this function — they flow through the normal
    per-drop copy loop since they live under `deployment_data/extra_no_survey_id/`.

    Args:
        train_drops, val_drops, test_drops: Drop ID lists from split_data().
        images_dir: Source directory of JPEG frames.
        species_labels_dir: Directory of multi-class YOLO .txt files (already remapped to `class_names`).
        output_dir: Root output directory.
        class_names: Ordered species class names (unified ID space).
        build_binary: Also build a binary (fish/no-fish) dataset.
        symlink: Use symlinks instead of copying files.
        source_class_map_path: Canonical class_map used as metadata source for the sidecar.

    Returns:
        (species_data_yaml_path, binary_data_yaml_path) — binary path is None if not built.
    """
    species_dir = output_dir / "species"
    binary_dir = output_dir / "binary"

    # Wipe previous split output so re-runs don't accumulate stale files (e.g.
    # files from drops that have since been removed from the train/val/test
    # lists or excluded via training_excluded_drops.txt).
    _clean_yolo_split_dirs(species_dir)
    if build_binary:
        _clean_yolo_split_dirs(binary_dir)

    # Walk images_dir once; reused across all 6 copy_split_files calls below.
    image_index = _build_image_index(images_dir)
    n_frames = sum(len(stems) for stems in image_index.values())
    logging.info(
        f"assemble_yolo_dataset: indexed {n_frames} frame image(s) "
        f"across {len(image_index)} drop(s) under {images_dir}"
    )

    for split_name, drops in [
        ("train", train_drops),
        ("val", val_drops),
        ("test", test_drops),
    ]:
        copy_split_files(
            drops,
            images_dir,
            species_labels_dir,
            species_dir,
            split_name,
            symlink=symlink,
            image_index=image_index,
        )

    species_yaml = generate_data_yaml(class_names, species_dir)
    _write_sidecar_class_map(class_names, species_dir, source_class_map_path)

    binary_yaml = None
    if build_binary:
        binary_labels_dir = output_dir / "binary_labels_staging"
        make_binary_labels(species_labels_dir, binary_labels_dir)
        for split_name, drops in [
            ("train", train_drops),
            ("val", val_drops),
            ("test", test_drops),
        ]:
            copy_split_files(
                drops,
                images_dir,
                binary_labels_dir,
                binary_dir,
                split_name,
                symlink=symlink,
                image_index=image_index,
            )
        binary_yaml = generate_data_yaml(["fish"], binary_dir)

    print_assembled_summary(
        species_dir=species_dir,
        class_names=class_names,
        train_drops=train_drops,
        val_drops=val_drops,
        test_drops=test_drops,
        extra_drops=extra_drops,
    )

    return species_yaml, binary_yaml
