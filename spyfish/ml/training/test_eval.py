"""Score a model on the frozen test set and keep the record.

One record is one model scored on one VERSION of the test set. The test set
(`training.test_drops_file`) only ever grows, so its version is a hash of the
drop list: records made on an older version stay valid for that version, and
when the list grows every model can be re-scored on the new one. Over time
the records form a model x test-set matrix that the Model Metrics page reads.

Each record carries:

* per-class metrics, in the model's own vocabulary. Older models have other
  class lists, so the test labels are rewritten into each model's id space;
  a test species the model never had counts as its ``fish`` bucket, which is
  what the model's own class floor would have done with it.
* contamination: how many test images were in that model's train or val
  snapshot (``runs/<run>/dataset/{train,val}.txt``). Models trained before
  the test set was frozen saw some of it. The score is reported twice, on
  all test images and on the unseen ("clean") ones, with the counts, so a
  flattered number is a measured number, not a footnote.

Records live in ``training/results/test_evals/<model>__test-<version>/``:
``record.json`` (everything), ``per_class.csv`` (flat, for pandas), and
YOLO's plots for the ``all`` and ``clean`` runs.

CLI:
    python -m spyfish.ml.training.test_eval --model production
    python -m spyfish.ml.training.test_eval --model runs/species_20260825_053442/weights/best.pt
    python -m spyfish.ml.training.test_eval --all-archived      # backfill history
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

from spyfish.biigle.biigle_to_yolo import drop_id_from_frame_filename
from spyfish.config.wrapper import config
from spyfish.ml.model_meta import (
    checkpoint_imgsz,
    model_name_from_run_dir,
    read_threshold_sidecar,
)
from spyfish.ml.training.rollup import ROLLUP_MAP_FILENAME, rollup_map_hash
from spyfish.ml.training.threshold import metrics_at
from spyfish.ml.training.yolo_labels import (
    DATA_YAML_FILENAME,
    read_class_names_by_id,
    remap_label_file,
)
from spyfish.utils import validate_model_path

FALLBACK_CLASS = "fish"
RECORD_FILE = "record.json"
TEST_EVALS_DIRNAME = "test_evals"

# ---------------------------------------------------------------------------
# Pure helpers (no ultralytics)
# ---------------------------------------------------------------------------


def test_set_version(drop_ids) -> str:
    """Short, order-independent fingerprint of a test drop list.

    The same drops in any order give the same version; adding one drop gives
    a new one. Eight hex chars is plenty for a list that changes a few times
    a year.
    """
    joined = "\n".join(sorted(set(drop_ids)))
    return hashlib.sha1(joined.encode()).hexdigest()[:8]


def model_name_for(model_path: Path) -> str:
    """The canonical model name of a weights file.

    A run's ``best.pt`` is named by its run dir (``runs/species_<stamp>/weights/
    best.pt``); a promoted or archived file by its stem. Same rule as
    `_derive_promoted_filename`, so a record and the model it scores share
    one name whichever copy was evaluated.
    """
    model_path = Path(model_path)
    if config.model_name_stamp(model_path.stem) is not None:
        return model_path.stem
    run_dir = model_name_from_run_dir(model_path)
    if run_dir is not None:
        return run_dir
    raise ValueError(
        f"Cannot name {model_path}: neither its stem nor its run dir follows "
        f"{{type}}_{{YYYYMMDD}}[_{{HHMMSS}}]."
    )


def find_run_snapshot(
    model_name: str, runs_dir: Optional[Path] = None
) -> Optional[Path]:
    """The dataset snapshot `train.py` froze for this model's run, or None.

    Runs are ``runs/<model_name>/dataset``; runs before the name standard were
    ``runs/<stamp>_<type>``, so that spelling is tried too.
    """
    runs_dir = runs_dir or config.training_runs_dir
    candidates = [runs_dir / model_name / "dataset"]
    if "_" in model_name:
        kind, _, stamp = model_name.partition("_")
        candidates.append(runs_dir / f"{stamp}_{kind}" / "dataset")
    for snap in candidates:
        if (snap / "train.txt").exists():
            return snap
    return None


def split_images(dataset_dir: Path, split: str) -> List[str]:
    """Image filenames of one split of an assembled YOLO dataset."""
    img_dir = dataset_dir / "images" / split
    if not img_dir.is_dir():
        return []
    exts = set(config.image_extensions)
    return sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in exts)


def seen_images(
    snapshot: Optional[Path], images: List[str]
) -> Optional[Dict[str, Set[str]]]:
    """Which of `images` were in the run's train and val splits.

    Val counts as seen: it drove early stopping and model selection. None when
    there is no snapshot, which the caller must report rather than treat as
    "nothing seen".
    """
    if snapshot is None:
        return None
    out = {}
    wanted = set(images)
    for split in ("train", "val"):
        txt = snapshot / f"{split}.txt"
        listed = set(txt.read_text().split()) if txt.exists() else set()
        out[split] = wanted & listed
    return out


def live_class_names(dataset_dir: Path) -> Dict[int, str]:
    """id -> name of the assembled dataset, whose labels we are about to decode."""
    return read_class_names_by_id(dataset_dir / DATA_YAML_FILENAME, required=True)


def build_model_view(
    dataset_dir: Path,
    split: str,
    images: List[str],
    live_names: Dict[int, str],
    model_names: Dict[int, str],
    out_dir: Path,
) -> Tuple[Path, int, int]:
    """A copy of `images` whose labels are written in the model's own ids.

    Returns (data.yaml path, boxes remapped to the fallback class, boxes
    dropped because the model has no fallback class either). Images are
    hard-linked, not symlinked: Ultralytics resolves the image path before
    swapping ``/images/`` for ``/labels/``, so a symlinked dir scores against
    the ORIGINAL labels (seen 2026-08-23, two models at mAP 0.002).
    """
    name_to_id = {v: k for k, v in model_names.items()}
    fallback_id = name_to_id.get(FALLBACK_CLASS)
    img_out, lbl_out = out_dir / "images" / "val", out_dir / "labels" / "val"
    shutil.rmtree(out_dir, ignore_errors=True)
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    # Live id -> the model's id for the same species. A species the model does
    # not name goes to its fallback class, or to None (box dropped) when the
    # model has no fallback either.
    id_map = {
        live_id: name_to_id.get(species, fallback_id)
        for live_id, species in live_names.items()
    }
    unknown_ids = {
        live_id for live_id, species in live_names.items() if species not in name_to_id
    }

    remapped = dropped = 0
    for fn in images:
        src_img = dataset_dir / "images" / split / fn
        try:
            os.link(src_img, img_out / fn)
        except OSError:
            shutil.copy2(src_img, img_out / fn)
        stem = Path(fn).stem
        src_lbl = dataset_dir / "labels" / split / f"{stem}.txt"
        dst_lbl = lbl_out / f"{stem}.txt"
        if not src_lbl.exists():
            dst_lbl.write_text("")
            continue
        # An id the live class list cannot name is a corrupt dataset: raise.
        boxes = remap_label_file(
            src_lbl, id_map, dst=dst_lbl, missing="raise", trailing_newline=True
        )
        n_unknown = sum(n for cid, n in boxes.items() if cid in unknown_ids)
        if fallback_id is None:
            dropped += n_unknown
        else:
            remapped += n_unknown

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": "images/val",
                "val": "images/val",
                "nc": len(model_names),
                "names": {int(k): v for k, v in model_names.items()},
            },
            sort_keys=False,
        )
    )
    return yaml_path, remapped, dropped


def record_dir_for(
    model_name: str, version: str, results_dir: Optional[Path] = None
) -> Path:
    results_dir = results_dir or config.training_results_dir
    return results_dir / TEST_EVALS_DIRNAME / f"{model_name}__test-{version}"


def write_record(record: dict, record_dir: Path) -> Path:
    """record.json plus a flat per_class.csv (one row per class per view)."""
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / RECORD_FILE).write_text(json.dumps(record, indent=2))
    rows = []
    for view in ("all", "clean"):
        block = record.get(view)
        if not block:
            continue
        for pc in block["per_class"]:
            rows.append({"model_name": record["model_name"], "view": view, **pc})
    with (record_dir / "per_class.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["model_name", "view", "name", "mAP50", "mAP50_95", "P", "R"]
        )
        w.writeheader()
        w.writerows(rows)
    return record_dir / RECORD_FILE


def load_records(results_dir: Optional[Path] = None) -> List[dict]:
    """Every record under test_evals/, oldest model first."""
    root = (results_dir or config.training_results_dir) / TEST_EVALS_DIRNAME
    if not root.is_dir():
        return []
    records = [json.loads(p.read_text()) for p in sorted(root.glob(f"*/{RECORD_FILE}"))]
    return sorted(records, key=lambda r: (r["model_stamp"], r["test_set"]["version"]))


def records_to_frames(records: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(overall, per_class) tables for charts: one row per record x view (x class)."""
    overall, per_class = [], []
    for r in records:
        base = {
            "model_name": r["model_name"],
            "model_stamp": r["model_stamp"],
            "test_version": r["test_set"]["version"],
            "test_drops": r["test_set"]["n_drops"],
            "snapshot_found": r["contamination"]["snapshot_found"],
            "n_seen": r["contamination"]["n_in_train"] + r["contamination"]["n_in_val"],
        }
        for view in ("all", "clean"):
            block = r.get(view)
            if not block:
                continue
            overall.append(
                {
                    **base,
                    "view": view,
                    "n_images": block["n_images"],
                    **{
                        k: block[k]
                        for k in ("mAP50", "mAP50_95", "precision", "recall")
                    },
                }
            )
            per_class += [{**base, "view": view, **pc} for pc in block["per_class"]]
    return pd.DataFrame(overall), pd.DataFrame(per_class)


def promotion_margin() -> float:
    """mAP50 a new model must gain over production to be promoted, as a
    fraction (`training.retrain_min_improvement_pct` is in percentage points)."""
    return config.retrain_min_improvement_pct / 100.0


def clears_promotion_margin(
    new_map50: float, production_map50: float, margin: Optional[float] = None
) -> Tuple[bool, float]:
    """(promote?, mAP50 gain). The one promotion rule: the gain has to reach
    the margin. Shared by the frozen-test decision and the val comparison."""
    if margin is None:
        margin = promotion_margin()
    delta = new_map50 - production_map50
    return delta >= margin, delta


def promotion_decision(new: dict, production: Optional[dict]) -> Tuple[bool, str]:
    """Promote when the new model beats production on the same test images.

    Both are scored on ALL test images (the clean subsets differ per model, so
    they are not comparable), each in its own vocabulary. The margin is
    `retrain_min_improvement_pct` of mAP50, like the val-based gate before it.
    """
    if production is None:
        return True, "no production record: first model, promote"
    if new["test_set"]["version"] != production["test_set"]["version"]:
        return False, (
            f"test set versions differ ({new['test_set']['version']} vs "
            f"{production['test_set']['version']}); re-score production first"
        )
    margin = promotion_margin()
    verdict, delta = clears_promotion_margin(
        new["all"]["mAP50"], production["all"]["mAP50"], margin
    )
    seen = (
        production["contamination"]["n_in_train"]
        + production["contamination"]["n_in_val"]
    )
    note = (
        f"new mAP50 {new['all']['mAP50']:.4f} vs production {production['all']['mAP50']:.4f} "
        f"({delta:+.4f}, threshold {margin:+.4f})"
    )
    if seen:
        note += f"; production trained on {seen} of these test images, its score is inflated"
    return verdict, note


def per_class_table(records: List[dict], view: str = "all") -> str:
    """Side-by-side per-class mAP50 of several records, for the log."""
    names = sorted({pc["name"] for r in records for pc in r[view]["per_class"]})
    cols = [r["model_name"] for r in records]
    lines = [f"{'class':<28}" + "".join(f"{c:>24}" for c in cols)]
    for n in names:
        row = f"{n:<28}"
        for r in records:
            hit = next((pc for pc in r[view]["per_class"] if pc["name"] == n), None)
            row += f"{hit['mAP50']:>24.3f}" if hit else f"{'-':>24}"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The one function that runs YOLO
# ---------------------------------------------------------------------------


def val_metrics(metrics) -> dict:
    """The four headline numbers of an Ultralytics ``model.val()`` result,
    rounded the way every record and CSV stores them."""
    return {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def checkpoint_class_names(model_path: Path) -> Dict[int, str]:
    """id -> name as the checkpoint itself records them."""
    from ultralytics import YOLO

    return {int(k): v for k, v in YOLO(str(model_path)).names.items()}


def score(
    weights: Path,
    data_yaml: Path,
    imgsz: int,
    out_dir: Path,
    classes: Optional[List[int]] = None,
    verbose: bool = False,
) -> dict:
    """model.val() on a prepared view; overall and per-class numbers.

    ``classes`` restricts the model's PREDICTIONS (and so the classes scored)
    to those ids, for a comparison on a vocabulary several models share; None
    scores every class the model has.

    ``at_threshold`` is the same scoring read AT the MaxN cutoff the model
    runs at (its ``.thresholds.json`` sidecar, picked on val): per-class P, R
    and F-beta at that one confidence, the view the pipeline actually
    experiences. None when the model has no sidecar.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    val_kwargs = {
        "data": str(data_yaml),
        "split": "val",
        "imgsz": imgsz,
        "plots": True,
        "save_json": True,
        "project": str(out_dir.parent),
        "name": out_dir.name,
        "exist_ok": True,
        "verbose": verbose,
    }
    if classes is not None:
        val_kwargs["classes"] = list(classes)
    m = model.val(**val_kwargs)
    names = model.names
    scored_names = [names[int(ci)] for ci in m.box.ap_class_index]
    sidecar = read_threshold_sidecar(weights)
    at_threshold = None
    if sidecar is not None:
        at_threshold = {
            "threshold": float(sidecar["threshold"]),
            "beta": float(sidecar.get("beta", 1.0)),
            **metrics_at(
                m.box.px,
                m.box.p_curve,
                m.box.r_curve,
                scored_names,
                float(sidecar["threshold"]),
                float(sidecar.get("beta", 1.0)),
                exclude=config.maxn_exclude_classes,
            ),
        }
    return {
        **val_metrics(m),
        "per_class": [
            {
                "name": name,
                "mAP50": round(float(m.box.ap50[i]), 4),
                "mAP50_95": round(float(m.box.ap[i]), 4),
                "P": round(float(m.box.p[i]), 4),
                "R": round(float(m.box.r[i]), 4),
            }
            for i, name in enumerate(scored_names)
        ],
        "at_threshold": at_threshold,
    }


def evaluate_on_test(
    model_path,
    dataset_dir: Optional[Path] = None,
    split: str = "test",
    imgsz: Optional[int] = None,
    results_dir: Optional[Path] = None,
    scorer=score,
) -> Optional[dict]:
    """Score one model on the assembled test split and write its record.

    Returns the record, or None when the split is empty (no data prep yet).
    `scorer` is injectable so the record logic is testable without a GPU.
    """
    model_path = Path(validate_model_path(model_path))
    dataset_dir = dataset_dir or config.training_dataset_dir
    # Each model at the size IT was trained at: scoring production at the
    # candidate's training.imgsz would lower its mAP and promote a model that
    # is no better. training.imgsz only stands in for a checkpoint that
    # records no size.
    imgsz = imgsz or checkpoint_imgsz(model_path) or config.training_imgsz
    images = split_images(dataset_dir, split)
    if not images:
        logging.warning(
            f"No {split} images under {dataset_dir}; run a data prep first."
        )
        return None

    model_name = model_name_for(model_path)
    drops = sorted({drop_id_from_frame_filename(fn) for fn in images})
    version = test_set_version(config.training_test_drops or drops)
    record_dir = record_dir_for(model_name, version, results_dir)

    model_names = checkpoint_class_names(model_path)
    live_names = live_class_names(dataset_dir)

    snapshot = find_run_snapshot(model_name)
    seen = seen_images(snapshot, images)
    seen_all = (seen["train"] | seen["val"]) if seen else set()
    contamination = {
        "snapshot_found": seen is not None,
        "n_in_train": len(seen["train"]) if seen else 0,
        "n_in_val": len(seen["val"]) if seen else 0,
        "n_clean": len(images) - len(seen_all),
    }

    views: Dict[str, Optional[dict]] = {"all": None, "clean": None}
    remapped = dropped = 0
    for view, subset in (
        ("all", images),
        ("clean", [f for f in images if f not in seen_all]),
    ):
        if view == "clean" and (seen is None or not seen_all):
            continue  # unknowable, or identical to "all"
        if not subset:
            continue
        work = record_dir / f"_view_{view}"
        yaml_path, n_remapped, n_dropped = build_model_view(
            dataset_dir, split, subset, live_names, model_names, work
        )
        logging.info(
            f"{model_name} [{view}]: {len(subset)} images, {n_remapped} GT box(es) "
            f"remapped to {FALLBACK_CLASS!r}, {n_dropped} dropped"
        )
        if view == "all":
            # The record's counts describe the full test set, not the last
            # view scored.
            remapped, dropped = n_remapped, n_dropped
        views[view] = {
            "n_images": len(subset),
            **scorer(model_path, yaml_path, imgsz, record_dir / view),
        }
        shutil.rmtree(work, ignore_errors=True)
    if views["clean"] is None and seen is not None:
        views["clean"] = views["all"]  # nothing seen: the clean score IS the full score

    record = {
        "model_name": model_name,
        "model_stamp": config.model_name_stamp(model_name),
        "model_path": str(model_path),
        "model_classes": [model_names[k] for k in sorted(model_names)],
        "split": split,
        "imgsz": imgsz,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "test_set": {
            "version": version,
            "n_drops": len(drops),
            "drop_ids": drops,
            "n_images": len(images),
        },
        "contamination": contamination,
        # Same hash on both sides: the model was trained on the class vocabulary
        # the test labels use. Different: per-class rows need reading with the
        # two rollup_map.json files side by side.
        "rollup_map_hash": {
            "test_set": rollup_map_hash(dataset_dir / ROLLUP_MAP_FILENAME),
            "model": (
                rollup_map_hash(snapshot / ROLLUP_MAP_FILENAME) if snapshot else None
            ),
        },
        "gt_boxes_remapped_to_fallback": remapped,
        "gt_boxes_dropped": dropped,
        "all": views["all"],
        "clean": views["clean"],
    }
    write_record(record, record_dir)
    logging.info(
        f"Test record {record_dir.name}: all mAP50={record['all']['mAP50']:.4f}"
        + (
            f", clean mAP50={record['clean']['mAP50']:.4f} on {record['clean']['n_images']} images"
            if record["clean"]
            else ", clean: unknown (no run snapshot)"
        )
        + f"; {contamination['n_in_train']} in train, {contamination['n_in_val']} in val"
    )
    return record


# ---------------------------------------------------------------------------
# Several models side by side on the vocabulary they share
# ---------------------------------------------------------------------------


def run_snapshot_for(model_path: Path) -> Optional[Path]:
    """The frozen dataset snapshot of the run that produced `model_path`, or None.

    A run's own ``best.pt`` has it two levels up (``<run>/dataset``), whatever
    the run dir is called; a promoted or archived file is found by model name.
    """
    model_path = Path(model_path)
    beside = model_path.parent.parent / "dataset"
    if (beside / "train.txt").exists():
        return beside
    try:
        return find_run_snapshot(model_name_for(model_path))
    except ValueError:
        return None


def trained_on(
    models: Dict[str, Path], images: List[str]
) -> Tuple[Dict[str, Set[str]], List[str]]:
    """Per model tag, which of `images` were in its TRAIN split, plus the tags
    that kept no snapshot and so cannot be checked.

    Train only, unlike a test record's contamination count: the usual split
    compared here is the live val split, which a model trained on the current
    data prep validated against in full.
    """
    seen_by: Dict[str, Set[str]] = {}
    unfilterable: List[str] = []
    for tag, path in models.items():
        seen = seen_images(run_snapshot_for(path), images)
        if seen is None:
            unfilterable.append(tag)
        else:
            seen_by[tag] = seen["train"]
    return seen_by, unfilterable


def shared_vocabulary(names_by_model: Dict[str, Dict[int, str]]) -> List[str]:
    """Class names every one of the models has, sorted."""
    return sorted(set.intersection(*[set(n.values()) for n in names_by_model.values()]))


def score_on_shared_vocabulary(
    models: Dict[str, Path],
    images: List[str],
    out_dir: Path,
    dataset_dir: Optional[Path] = None,
    split: str = "val",
    imgsz: Optional[int] = None,
    scorer=score,
) -> dict:
    """Score each model on the same `images`, on the classes they all share.

    Each model gets the images with ground truth rewritten into its own ids
    (`build_model_view`: a species it does not name becomes its fallback
    class), and its predictions are restricted to the shared class ids, so a
    model is not charged false positives for knowing species the others never
    saw. Each runs at the size it was trained at, like `evaluate_on_test`.

    Returns ``{"shared_vocabulary": [...], "models": {tag: {nc, imgsz,
    gt_boxes_remapped_to_fallback, gt_boxes_dropped, mAP50, mAP50_95,
    precision, recall, per_class}}}``. Views and Ultralytics output go under
    `out_dir` (``<tag>/`` and ``val_<tag>/``).
    """
    dataset_dir = dataset_dir or config.training_dataset_dir
    names_by_model = {tag: checkpoint_class_names(p) for tag, p in models.items()}
    shared = shared_vocabulary(names_by_model)
    logging.info(f"shared vocabulary: {len(shared)} classes\n{shared}")
    live_names = live_class_names(dataset_dir)

    scored: Dict[str, dict] = {}
    for tag, weights in models.items():
        names = names_by_model[tag]
        name_to_id = {v: k for k, v in names.items()}
        keep_ids = sorted(name_to_id[s] for s in shared)
        yaml_path, remapped, dropped = build_model_view(
            dataset_dir, split, images, live_names, names, out_dir / tag
        )
        model_imgsz = imgsz or checkpoint_imgsz(weights) or config.training_imgsz
        logging.info(
            f"=== {tag}: nc={len(names)}, {remapped} GT box(es) remapped to "
            f"{FALLBACK_CLASS!r}, {dropped} dropped, scoring {len(keep_ids)} shared "
            f"classes on {len(images)} images at imgsz {model_imgsz}"
        )
        scored[tag] = {
            "nc": len(names),
            "imgsz": model_imgsz,
            "gt_boxes_remapped_to_fallback": remapped,
            "gt_boxes_dropped": dropped,
            **scorer(
                weights,
                yaml_path,
                model_imgsz,
                out_dir / f"val_{tag}",
                classes=keep_ids,
                verbose=True,
            ),
        }
    return {"shared_vocabulary": shared, "models": scored}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_models(args) -> List[Path]:
    paths: List[Path] = []
    for m in args.model or []:
        if m == "production":
            paths.append(config.pipeline_model_path)
        else:
            p = Path(m)
            paths.append(p if p.is_absolute() else config.local_training_dir / p)
    if args.all_archived:
        for d in (config.archived_models_dir, config.pipeline_model_dir):
            if d.is_dir():
                paths += sorted(
                    p for p in d.glob("*.pt") if config.model_name_stamp(p.stem)
                )
    return paths


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model",
        action="append",
        help="weights path (absolute, or relative to local_training_dir), or "
        "'production'. Repeatable.",
    )
    ap.add_argument(
        "--all-archived",
        action="store_true",
        help="every canonical .pt in archived_models/ and pipeline_model/",
    )
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()
    models = _resolve_models(args)
    if not models:
        ap.error("give --model or --all-archived")
    records = [r for r in (evaluate_on_test(m, split=args.split) for m in models) if r]
    if len(records) > 1:
        print("\nper-class mAP50 (all test images):\n" + per_class_table(records))


if __name__ == "__main__":
    main()
