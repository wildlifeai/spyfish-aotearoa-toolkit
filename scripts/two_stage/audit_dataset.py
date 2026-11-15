"""Audit the assembled YOLO dataset: split hygiene, frame sources, box sizes, recall by source.

Answers "are the train/val/test frames the right ones, and do they portray the
situation?" with four tables:

1. Splits: drops and frames per split, overlap between splits, the test split
   against the frozen list, frames by SOURCE (``frames/`` = review frames the
   model's own detections picked; ``training_frames/`` = blind, evenly spaced),
   and background frames (no boxes).
2. Boxes per species per split and per source.
3. Box size per species: median long side in NETWORK pixels at imgsz 640 and
   the share under 32 and 16, the bands where detectors go blind.
4. For each predictions file: recall by split and source, and on the test split
   per species and source. Review frames were chosen because the model saw
   something, so recall there is an upper bound; the blind frames are the
   honest number.

Needs no pixels: labels, the image symlinks (their targets name the source) and
the CSVs. So it runs on a mirror of the NeSI dataset made with ``rsync -a``
(links copied as links)::

    python scripts/two_stage/audit_dataset.py \\
        --root process_files/nesi_mirror/species \\
        --out process_files/nesi_mirror/wip_two_stage \\
        --predictions yolo_predictions.csv detector_predictions.csv \\
        --list-missed "Notolabrus celidotus"

Frame sizes (labels are normalised, predictions are pixels) are recovered from
``crops_meta.csv``, whose pixel boxes are the same labels; a frame without a
crop inherits its drop's size, else 1920x1080 with a warning. ``--list-missed``
writes the NeSI paths of test frames holding a missed box of that species, for
``rsync --files-from``.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.crops import (  # noqa: E402
    OUT_DIR,
    read_crops_meta,
    read_predictions,
)
from spyfish.ml.classifier.metrics import best_iou, score_predictions  # noqa: E402
from spyfish.ml.training.yolo_labels import (  # noqa: E402
    DATA_YAML_FILENAME,
    SPLITS,
    read_class_names,
    read_label_rows,
)

IMGSZ = 640
SMALL, TINY = 32, 16  # network pixels: COCO "small" line, and the stride-8 floor
DEFAULT_SIZE = (1920, 1080)
KNOWN_SIZES = [
    (1920, 1080),
    (3840, 2160),
    (2704, 1520),
    (1440, 1080),
    (1280, 960),
    (1280, 720),
    (2560, 1440),
    (720, 480),
]
SOURCE_NAMES = {"frames": "review", "training_frames": "blind"}


def load_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(frames, boxes)`` from labels and link targets; no image is opened."""
    names = read_class_names(root / DATA_YAML_FILENAME, required=True)
    frames, boxes = [], []
    for split in SPLITS:
        for img in sorted((root / "images" / split).glob("*.jpg")):
            target = Path(os.readlink(img)) if img.is_symlink() else img
            source = SOURCE_NAMES.get(target.parent.name, target.parent.name)
            if target.parent.name in SOURCE_NAMES:
                drop = target.parents[1].name
            elif "__" in img.stem:
                drop = img.stem.split("__")[0]
            else:
                drop = "?"
            label = root / "labels" / split / f"{img.stem}.txt"
            # every non-blank row: one that is not a box fails the unpack below
            lines = read_label_rows(label, boxes_only=False) if label.exists() else []
            frames.append((split, img.name, str(target), drop, source, len(lines)))
            for cid, cx, cy, w, h in lines:
                boxes.append(
                    (
                        split,
                        img.name,
                        drop,
                        source,
                        names[int(cid)],
                        float(cx),
                        float(cy),
                        float(w),
                        float(h),
                    )
                )
    frames = pd.DataFrame(
        frames, columns=["split", "filename", "target", "drop", "source", "n_boxes"]
    )
    boxes = pd.DataFrame(
        boxes,
        columns=[
            "split",
            "filename",
            "drop",
            "source",
            "species",
            "cx",
            "cy",
            "w",
            "h",
        ],
    )
    return frames, boxes


def frame_sizes(boxes: pd.DataFrame, crops: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Pixel size per frame, from a crop's pixel box over the same label's normalised one.

    Every (crop, label) pair of one species on one frame proposes a (W, H);
    the true pairing proposes the true size every time and the wrong pairings
    scatter, so the most common proposal wins.
    """
    sizes: dict[str, tuple[int, int]] = {}
    lab = {k: g for k, g in boxes.groupby(["filename", "species"])}
    for (fname, sp), g in crops.groupby(["filename", "species"]):
        labelled = lab.get((fname, sp))
        if labelled is None:
            continue
        votes: Counter = Counter()
        for c in g.itertuples():
            for r in labelled.itertuples():
                if r.w <= 0 or r.h <= 0:
                    continue
                votes[
                    (round((c.x_max - c.x_min) / r.w), round((c.y_max - c.y_min) / r.h))
                ] += 1
        if votes:
            w, h = votes.most_common(1)[0][0]
            # a wrong crop/label pairing proposes nonsense like 3414x517: only
            # accept a proposal within 3% of a real camera frame size
            for kw, kh in KNOWN_SIZES:
                if abs(w - kw) <= 0.03 * kw and abs(h - kh) <= 0.03 * kh:
                    sizes[fname] = (kw, kh)
                    break
    return sizes


def md(df: pd.DataFrame, **kw) -> str:
    return df.round(3).to_markdown(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=config.training_dataset_dir)
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="folder holding the CSVs; the report goes there too",
    )
    ap.add_argument(
        "--predictions",
        nargs="*",
        default=[artifacts.YOLO_PREDICTIONS],
        metavar="CSV",
    )
    ap.add_argument(
        "--list-missed",
        nargs="*",
        default=[],
        metavar="SPECIES",
        help="write NeSI paths of test frames with a missed box of these species",
    )
    args = ap.parse_args()

    common = species_registry().common_name_map()
    frames, boxes = load_dataset(args.root)
    lines = [f"# Dataset audit: `{args.root}`", ""]

    # 1. splits ---------------------------------------------------------------
    per_split = frames.groupby("split").agg(
        drops=("drop", "nunique"),
        frames=("filename", "size"),
        background=("n_boxes", lambda s: int((s == 0).sum())),
    )
    per_split["boxes"] = boxes.groupby("split").size()
    src = frames.pivot_table(
        index="split", columns="source", values="filename", aggfunc="size", fill_value=0
    )
    drops = {s: set(g["drop"]) for s, g in frames.groupby("split")}
    overlap = {
        f"{a}∩{b}": len(drops.get(a, set()) & drops.get(b, set()))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    frozen = config.training_test_drops
    test_drops = drops.get("test", set())
    lines += [
        "## 1. Splits",
        "",
        md(per_split.join(src, how="left").fillna(0).astype(int)),
        "",
        f"Drop overlap between splits: {overlap}. "
        + ("**Clean.**" if not any(overlap.values()) else "**LEAK.**"),
        "",
    ]
    if frozen:
        lines += [
            f"Frozen test list: {len(frozen)} drops; in the test split: {len(test_drops)}. "
            f"Frozen but not in test (no frames on disk?): {sorted(frozen - test_drops) or 'none'}. "
            f"In test but not frozen: {sorted(test_drops - frozen) or 'none'}.",
            "",
        ]
    lines += [
        "Source: `review` frames were picked by the pipeline from the model's own detections "
        "(MaxN peaks, low confidence, size bands, random detections, 3 blind per drop, start); "
        "`blind` frames are the evenly spaced training extraction, model-independent.",
        "",
    ]

    # 2. boxes per species ----------------------------------------------------
    sp_split = boxes.pivot_table(
        index="species",
        columns="split",
        values="filename",
        aggfunc="size",
        fill_value=0,
    )
    sp_split["total"] = sp_split.sum(axis=1)
    sp_src = boxes.pivot_table(
        index="species",
        columns="source",
        values="filename",
        aggfunc="size",
        fill_value=0,
    )
    tab = sp_split.join(sp_src, rsuffix="_src").sort_values("total", ascending=False)
    tab["common"] = [common.get(s, "") for s in tab.index]
    lines += ["## 2. Boxes per species, by split and by source", "", md(tab), ""]

    # 3. sizes ----------------------------------------------------------------
    # Real sizes when the frames are readable (on NeSI; PIL reads only the
    # header). On a mirror of links, estimate from crops_meta.csv instead,
    # which is wrong for frames with no crop in a volume of mixed sizes.
    from PIL import Image

    sizes = {}
    for r in frames.itertuples():
        if Path(r.target).exists():
            with Image.open(r.target) as im:
                sizes[r.filename] = im.size
    if len(sizes) < len(frames):
        crops_csv = args.out / artifacts.CROPS_META
        est = (
            frame_sizes(boxes, read_crops_meta(crops_csv)) if crops_csv.exists() else {}
        )
        est.update(sizes)
        lines += [
            f"**Frame sizes estimated for {len(frames) - len(sizes)} of {len(frames)} frames** "
            "(images not readable here): run on NeSI for exact numbers.",
            "",
        ]
        sizes = est
    drop_of = frames.set_index("filename")["drop"].to_dict()
    votes: dict[str, Counter] = {}
    for f, wh in sizes.items():
        votes.setdefault(drop_of.get(f, "?"), Counter())[wh] += 1
    by_drop = {d: c.most_common(1)[0][0] for d, c in votes.items()}
    n_default = 0
    ws, hs = [], []
    for r in boxes.itertuples():
        wh = sizes.get(r.filename) or by_drop.get(r.drop)
        if wh is None:
            wh, n_default = DEFAULT_SIZE, n_default + 1
        ws.append(wh[0])
        hs.append(wh[1])
    boxes["frame_w"], boxes["frame_h"] = ws, hs
    wh = np.column_stack([ws, hs])
    scale = IMGSZ / boxes["frame_w"]
    boxes["net_px"] = (
        np.maximum(boxes["w"] * boxes["frame_w"], boxes["h"] * boxes["frame_h"]) * scale
    )
    boxes["long_px"] = np.maximum(
        boxes["w"] * boxes["frame_w"], boxes["h"] * boxes["frame_h"]
    )
    size_tab = boxes.groupby("species").agg(
        boxes=("net_px", "size"),
        median_frame_px=("long_px", "median"),
        median_net_px=("net_px", "median"),
        under32=("net_px", lambda s: (s < SMALL).mean()),
        under16=("net_px", lambda s: (s < TINY).mean()),
    )
    size_tab = size_tab[size_tab["boxes"] >= 20].sort_values("median_net_px")
    size_tab["common"] = [common.get(s, "") for s in size_tab.index]
    test_sizes = (
        boxes[boxes["split"] == "test"]
        .groupby("species")["net_px"]
        .agg(
            test_boxes="size",
            test_median_net_px="median",
            test_under32=lambda s: (s < SMALL).mean(),
        )
    )
    lines += [
        f"## 3. Box size at imgsz {IMGSZ} (frame sizes {dict(Counter(map(tuple, wh.tolist())).most_common(4))}; "
        f"{n_default} boxes fell back to {DEFAULT_SIZE})",
        "",
        "`net_px`: long side of the box as the network sees it. Under 32 is the small-object band, under 16 is below the finest stride.",
        "",
        md(size_tab.join(test_sizes, how="left").fillna(0)),
        "",
    ]

    # 4. recall by source -------------------------------------------------------
    experts = boxes.assign(
        x_min=(boxes["cx"] - boxes["w"] / 2) * boxes["frame_w"],
        y_min=(boxes["cy"] - boxes["h"] / 2) * boxes["frame_h"],
        x_max=(boxes["cx"] + boxes["w"] / 2) * boxes["frame_w"],
        y_max=(boxes["cy"] + boxes["h"] / 2) * boxes["frame_h"],
    )
    for name in args.predictions:
        csv = args.out / name
        if not csv.exists():
            lines += [f"## 4. `{name}`: not found in {args.out}", ""]
            continue
        pred = read_predictions(csv)
        scored, summary = score_predictions(pred, experts)
        by = scored.pivot_table(
            index="split", columns="source", values="found", aggfunc=["size", "mean"]
        )
        by.columns = [f"{a}_{b}" for a, b in by.columns]
        test = scored[scored["split"] == "test"]
        per_sp = test.pivot_table(
            index="species",
            columns="source",
            values="found",
            aggfunc=["size", "mean"],
            fill_value=0,
        )
        per_sp.columns = [f"{a}_{b}" for a, b in per_sp.columns]
        per_sp["test_boxes"] = test.groupby("species").size()
        per_sp["recall_all"] = test.groupby("species")["found"].mean()
        per_sp = per_sp.sort_values("test_boxes", ascending=False)
        per_sp["common"] = [common.get(s, "") for s in per_sp.index]
        lines += [
            f"## 4. Recall of expert boxes: `{name}` (IoU ≥ 0.5, any class, bait boxes excluded)",
            "",
            f"All splits: {summary['n_pred']} detector boxes on annotated frames, precision {summary['precision']:.3f}, recall {summary['recall']:.3f}.",
            "",
            "### By split and frame source (size = expert boxes, mean = recall)",
            "",
            md(by),
            "",
            "### Test split, per species and source",
            "",
            md(per_sp[per_sp["test_boxes"] >= 3]),
            "",
        ]
        for sp in args.list_missed:
            missed = test[(test["species"] == sp) & ~test["found"]]
            targets = (
                frames.set_index("filename")["target"]
                .reindex(missed["filename"].unique())
                .dropna()
            )
            slug = sp.replace(" ", "_").lower()
            out = args.out / f"missed_{slug}_{Path(name).stem}.txt"
            out.write_text("\n".join(targets) + "\n")
            lines += [
                f"Missed `{sp}` boxes on test: {len(missed)} on {len(targets)} frames; NeSI paths in `{out.name}` (rsync --files-from).",
                "",
            ]

    # 5. model-only boxes per drop ---------------------------------------------
    eb = {
        f: g[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
        for f, g in experts.groupby("filename")
    }
    meta = frames.set_index("filename")
    for name in args.predictions:
        csv = args.out / name
        if not csv.exists():
            continue
        pred = read_predictions(csv)
        pred = pred[pred["filename"].isin(eb) & (pred["yolo_class"] != "bait")]
        only = [
            best_iou((r.x_min, r.y_min, r.x_max, r.y_max), eb[r.filename]) < 0.3
            for r in pred.itertuples()
        ]
        pred = pred.assign(
            model_only=only,
            drop=pred["filename"].map(meta["drop"]),
            split=pred["filename"].map(meta["split"]),
        )
        n_exp = experts.groupby("drop").size()
        per = pred.groupby("drop").agg(
            split=("split", "first"),
            model_boxes=("model_only", "size"),
            model_only=("model_only", "sum"),
            model_only_conf_over_0_5=(
                "conf",
                lambda c: int((c[pred.loc[c.index, "model_only"]] > 0.5).sum()),
            ),
        )
        per["expert_boxes"] = n_exp.reindex(per.index).fillna(0).astype(int)
        per["model_only_per_100_expert"] = (
            100 * per["model_only"] / per["expert_boxes"].clip(lower=1)
        ).round(1)
        by_split = per.groupby("split")[
            ["expert_boxes", "model_only", "model_only_conf_over_0_5"]
        ].sum()
        by_split["per_100_expert"] = (
            100 * by_split["model_only"] / by_split["expert_boxes"]
        ).round(1)
        lines += [
            f"## 5. Model-only boxes (IoU < 0.3 with every expert box): `{name}`",
            "",
            "Each is junk, a second box on one fish, or a fish nobody boxed; the contact sheets tell which.",
            "",
            md(by_split),
            "",
            "### Drops with the most confident model-only boxes",
            "",
            md(per.sort_values("model_only_conf_over_0_5", ascending=False).head(20)),
            "",
        ]
        pred[pred["model_only"]].sort_values("conf", ascending=False).to_csv(
            args.out / f"model_only_{Path(name).stem}.csv", index=False
        )

    report = args.out / "audit_report.md"
    report.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
