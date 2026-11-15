"""`not a fish` hard negatives: detector boxes that no expert box explains.

Plan §5.3. On frames the experts annotated, every current-YOLO box whose best
IoU with any expert box is below training.classifier.hard_negatives.iou_negative is a candidate: kelp, sediment,
shadow, the bait bag, the frame, or a fish the expert skipped. The last case is
why a human looks before any of this trains anything, so the script:

1. collects the candidates: `--source yolo` (default) is yolo_predictions.csv vs
   the expert boxes, nobody has looked at those boxes; `--source upload` is the
   boxes the pipeline sent to BIIGLE (`{drop}_coco_annotations_for_biigle.json`)
   that are gone from the expert CSV, so an expert saw each one and deleted it;
   `--source both` unions them, keeping the reviewed copy of a box both have,
2. crops and embeds them with BioCLIP 2 like the positives,
3. clusters the embeddings (k-means) so the review is per cluster, not per box,
4. draws one contact sheet per cluster, boxes sorted by YOLO confidence,
5. runs the drop-grouped probe once more with `not a fish` as a class, to show
   what the extra class costs or buys before anyone has reviewed it.

Writes to <two_stage_dir>/negatives/:
    negatives_meta.csv, negatives_embeddings.npy, cluster_summary.csv,
    sheets/cluster_XX.jpg, negatives_report.md
Review step: look at the sheets, note cluster ids that are mostly fish; those
boxes are not negatives. Nothing here writes to the training set.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.classes import (  # noqa: E402
    N_FOLDS,
    OTHER,
    classes_for,
    fit_probe,
)
from spyfish.ml.classifier.crops import (  # noqa: E402
    NOT_CLASSES,
    NOT_FISH,
    OUT_DIR,
    RAW_SUFFIX,
    SEED,
    embed_crops,
    find_expert_csvs,
    frames_on_disk,
    load_bioclip,
    load_boxes,
    pick_device,
    read_crops_meta,
    read_predictions,
    reserve_of,
)
from spyfish.ml.classifier.metrics import best_iou, per_class_table  # noqa: E402
from spyfish.ml.classifier.sheets import load_font  # noqa: E402

_HN = config.classifier_hard_negatives  # training.classifier.hard_negatives
IOU_NEG = (
    _HN.iou_negative
)  # below this overlap with every expert box, a detector box is a negative candidate
MAX_PER_DROP = (
    _HN.max_per_drop
)  # keep the candidate set the same order of size as the positives
N_CLUSTERS = _HN.n_clusters
SHEET_COLS, SHEET_ROWS, TILE = 10, 6, 160
DEVICE = pick_device()


COLUMNS = [
    "filename",
    "drop",
    "reserve",
    "yolo_class",
    "conf",
    "best_iou",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]


def yolo_candidates() -> pd.DataFrame:
    """Current-model boxes on expert frames that no expert box explains.

    Nobody has looked at these boxes, so a candidate is junk or a fish the
    expert skipped; the cluster review decides which.
    """
    yolo = read_predictions(OUT_DIR / artifacts.YOLO_PREDICTIONS)
    experts = (
        load_boxes()
    )  # includes bait boxes: a YOLO box on the bait bag is explained, not a negative
    frames = experts.drop_duplicates("filename").set_index("filename")[
        ["drop", "reserve"]
    ]
    rows = []
    for fname, g in yolo.groupby("filename"):
        if fname not in frames.index:
            continue
        eb = experts.loc[
            experts["filename"] == fname, ["x_min", "y_min", "x_max", "y_max"]
        ].to_numpy()
        for r in g.itertuples():
            box = (r.x_min, r.y_min, r.x_max, r.y_max)
            best = best_iou(box, eb)
            if best < IOU_NEG:
                rows.append(
                    (
                        fname,
                        frames.loc[fname, "drop"],
                        frames.loc[fname, "reserve"],
                        r.yolo_class,
                        r.conf,
                        best,
                        *box,
                    )
                )
    return pd.DataFrame(rows, columns=COLUMNS).assign(source="yolo")


def uploaded_candidates() -> pd.DataFrame:
    """Boxes the pipeline sent to BIIGLE that the expert deleted.

    `{drop}_coco_annotations_for_biigle.json` is the exact box set uploaded for
    review and the expert raw CSV is what the volume held when it was marked
    done. An uploaded box with IoU < IOU_NEG to every surviving box on its frame
    was looked at and removed (or redrawn, which the IoU match absorbs), which
    makes it a stronger negative than an unreviewed detector box. Frames the
    expert emptied count too. Uploaded bait boxes do not: a deleted bait box
    says nothing about fish. Only frames on disk contribute, since the crop
    comes from the frame.
    """
    experts = load_boxes()
    frames = frames_on_disk()
    by_frame = {
        f: g[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
        for f, g in experts.groupby("filename")
    }
    skip = NOT_CLASSES - {config.catchall_class}  # bait and friends, never `fish`
    rows, n_drops = [], 0
    for csv in find_expert_csvs():
        drop = csv.name[: -len(RAW_SUFFIX)]
        try:
            coco_path = config.get_coco_annotations_path(drop)
        except ValueError:
            continue  # legacy drop id: never went through the upload path
        # the dated upload record is what the expert actually saw; the working
        # COCO can be rewritten by a later re-extraction. Earliest record wins.
        records = sorted(
            coco_path.parent.glob(f"{drop}{config.biigle_upload_record_glob}")
        )
        if records:
            coco = json.loads(records[0].read_text())["coco"]
        elif coco_path.exists():
            coco = json.loads(coco_path.read_text())
        else:
            continue
        images = {im["id"]: im["file_name"] for im in coco["images"]}
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        n_drops += 1
        for a in coco["annotations"]:
            fname = images.get(a["image_id"])
            name = cats.get(a["category_id"], "")
            if fname not in frames or name in skip:
                continue
            x, y, w, h = a["bbox"]
            box = (x, y, x + w, y + h)
            best = best_iou(box, by_frame.get(fname, ()))
            if best < IOU_NEG:
                rows.append(
                    (
                        fname,
                        drop,
                        reserve_of(drop),
                        name,
                        a.get("score", float("nan")),
                        best,
                        *box,
                    )
                )
    logging.info(
        f"{n_drops} drops have both an uploaded COCO and an expert CSV; "
        f"{len(rows)} uploaded boxes are gone from the expert CSV"
    )
    return pd.DataFrame(rows, columns=COLUMNS).assign(source="upload")


def reviewed_candidates() -> pd.DataFrame:
    """Model-only boxes sent to BIIGLE as "To review" that the reviewer deleted.

    ``review_outcome.py`` writes ``review_outcome.csv`` from the upload
    manifests and the re-pulled BIIGLE CSVs; its ``deleted`` rows are boxes a
    person looked at and rejected: junk, or a second box on a fish already
    boxed. Only frames on disk contribute, since the crop comes from the frame.
    """
    path = OUT_DIR / artifacts.REVIEW_OUTCOME
    if not path.exists():
        raise SystemExit(
            f"{path} not found: run review_outcome.py after re-pulling the reviewed volumes"
        )
    out = pd.read_csv(path)
    out = out[out["outcome"] == "deleted"]
    frames = frames_on_disk()
    out = out[out["filename"].isin(frames)]
    cls = (
        out["model_class"]
        if "model_class" in out.columns
        else pd.Series("fish", index=out.index)
    )
    df = pd.DataFrame(
        {
            "filename": out["filename"],
            "drop": out["drop"],
            "reserve": out["drop"].map(reserve_of),
            "yolo_class": cls,
            "conf": out["conf"],
            "best_iou": 0.0,
            "x_min": out["x_min"],
            "y_min": out["y_min"],
            "x_max": out["x_max"],
            "y_max": out["y_max"],
        }
    )
    logging.info(f"{len(df)} reviewed-and-deleted boxes from {path.name}")
    return df[COLUMNS].assign(source="review")


def candidates(source: str) -> pd.DataFrame:
    if source == "yolo":
        df = yolo_candidates()
    elif source == "upload":
        df = uploaded_candidates()
    elif source == "review":
        df = reviewed_candidates()
    else:
        up, yo = uploaded_candidates(), yolo_candidates()
        # the current model re-fires on much of the junk the expert already
        # deleted: keep the reviewed copy, drop the unreviewed one
        dup = []
        for fname, g in yo.groupby("filename"):
            ub = up.loc[
                up["filename"] == fname, ["x_min", "y_min", "x_max", "y_max"]
            ].to_numpy()
            for r in g.itertuples():
                box = (r.x_min, r.y_min, r.x_max, r.y_max)
                if best_iou(box, ub) >= 0.5:
                    dup.append(r.Index)
        df = pd.concat([up, yo.drop(index=dup)], ignore_index=True)
    if df.empty:
        raise SystemExit(f"no `{NOT_FISH}` candidates from source {source}")
    df = df[(df["x_max"] - df["x_min"] >= 16) & (df["y_max"] - df["y_min"] >= 16)]
    rng = np.random.default_rng(SEED)
    keep = [
        (
            idx
            if len(idx) <= MAX_PER_DROP
            else rng.choice(idx, MAX_PER_DROP, replace=False)
        )
        for _, idx in df.groupby("drop").indices.items()
    ]
    df = df.iloc[np.concatenate(keep)].sort_values("filename").reset_index(drop=True)
    df["neg_id"] = [f"n{i:05d}" for i in range(len(df))]
    return df


def embed(df: pd.DataFrame, neg_dir: Path) -> np.ndarray:
    model, preprocess, _ = load_bioclip(DEVICE)
    crops_dir = neg_dir / artifacts.CROPS_DIR
    crops_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def save_thumbnail(crop, r) -> None:
        thumb = crop.copy()
        thumb.thumbnail((224, 224))
        thumb.save(crops_dir / f"{r.neg_id}.jpg", quality=85)

    out = embed_crops(
        df, frames_on_disk(), model, preprocess, DEVICE, save_thumbnail=save_thumbnail
    )
    print(f"embedded {len(df)} negatives in {time.time() - t0:.0f}s")
    return out


def contact_sheets(df: pd.DataFrame, neg_dir: Path) -> None:
    sheets = neg_dir / "sheets"
    sheets.mkdir(exist_ok=True)
    font = load_font(13)
    for c, g in df.groupby("cluster"):
        g = g.sort_values("conf", ascending=False).head(SHEET_COLS * SHEET_ROWS)
        sheet = Image.new(
            "RGB", (SHEET_COLS * TILE, SHEET_ROWS * TILE + 28), (24, 28, 28)
        )
        d = ImageDraw.Draw(sheet)
        d.text(
            (8, 6),
            f"cluster {c:02d}: {len(df[df['cluster'] == c])} boxes, top {len(g)} by YOLO confidence, "
            f"YOLO says {df[df['cluster'] == c]['yolo_class'].value_counts().index[0]} most often",
            fill=(255, 255, 255),
            font=font,
        )
        for k, r in enumerate(g.itertuples()):
            tile = Image.open(
                neg_dir / artifacts.CROPS_DIR / f"{r.neg_id}.jpg"
            ).convert("RGB")
            tile.thumbnail((TILE - 4, TILE - 4))
            x, y = (k % SHEET_COLS) * TILE, 28 + (k // SHEET_COLS) * TILE
            sheet.paste(tile, (x + 2, y + 2))
            d.rectangle(
                [x + 2, y + TILE - 18, x + TILE - 2, y + TILE - 2], fill=(0, 0, 0)
            )
            d.text(
                (x + 5, y + TILE - 17),
                f"{r.yolo_class[:14]} {r.conf:.2f} {r.drop[:3]}",
                fill=(230, 230, 230),
                font=font,
            )
        sheet.save(sheets / f"cluster_{c:02d}.jpg", quality=80)


def probe_with_negatives(
    df: pd.DataFrame, neg_X: np.ndarray, common: dict
) -> list[str]:
    meta = read_crops_meta()
    pos_X = np.load(OUT_DIR / artifacts.EMBEDDINGS)
    target, _classes, _stats = classes_for(meta)
    classes = target + [OTHER, NOT_FISH]
    y = np.concatenate(
        [
            meta["species"].where(meta["species"].isin(target), OTHER).to_numpy(),
            np.full(len(df), NOT_FISH),
        ]
    )
    X = np.vstack([pos_X, neg_X])
    groups = np.concatenate([meta["drop"].to_numpy(), df["drop"].to_numpy()])
    oof = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        oof[te] = fit_probe(X[tr], y[tr]).predict(X[te])
    tab = per_class_table(y, oof, classes)
    is_pos = np.arange(len(y)) < len(meta)
    species_acc = (oof[is_pos] == y[is_pos]).mean()
    fish_called_junk = (oof[is_pos] == NOT_FISH).mean()
    junk_called_fish = (oof[~is_pos] != NOT_FISH).mean()
    return [
        f"## Probe with `{NOT_FISH}` as a class (drop-grouped OOF, before any review)",
        "",
        f"Species accuracy on expert crops {species_acc:.3f} (was 0.626 without the class). "
        f"Expert fish called `{NOT_FISH}`: {fish_called_junk:.1%}. Candidates called a fish class: {junk_called_fish:.1%}, "
        f"which is an upper bound on how many candidates are really fish the expert skipped, plus the probe's own errors.",
        "",
        tab.assign(common=[common.get(c, "") for c in tab.index])
        .round(3)
        .to_markdown(),
        "",
    ]


SOURCE_TEXT = {
    "yolo": "current-model boxes on expert-annotated frames that no expert box explains (unreviewed)",
    "upload": "boxes sent to BIIGLE for review that the expert deleted (each one seen by a person)",
    "both": "the union: every box sent to BIIGLE that the expert deleted, plus unreviewed current-model boxes not already among them",
    "review": "model-only boxes sent to BIIGLE as 'To review' that the reviewer deleted (review_outcome.csv)",
}


def negatives_dir(source: str) -> Path:
    """Where a source's candidates go: ``negatives`` for ``yolo``, else
    ``negatives_<source>`` (review -> negatives_review)."""
    return OUT_DIR / ("negatives" if source == "yolo" else f"negatives_{source}")


def main(source: str, neg_dir: Optional[Path] = None) -> None:
    neg_dir = neg_dir or negatives_dir(source)
    neg_dir.mkdir(parents=True, exist_ok=True)
    df = candidates(source)
    print(
        f"{len(df)} candidates on {df['filename'].nunique()} frames, {df['drop'].nunique()} drops"
    )
    X = embed(df, neg_dir)
    n_clusters = min(N_CLUSTERS, len(df))
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=SEED).fit(X)
    df["cluster"] = km.labels_
    df.to_csv(neg_dir / artifacts.NEGATIVES_META, index=False)
    np.save(neg_dir / artifacts.NEGATIVES_EMBEDDINGS, X)
    contact_sheets(df, neg_dir)

    common = species_registry().common_name_map()
    by_cluster = (
        df.groupby("cluster")
        .agg(
            boxes=("neg_id", "size"),
            drops=("drop", "nunique"),
            mean_conf=("conf", "mean"),
            yolo_top=("yolo_class", lambda s: s.value_counts().index[0]),
            yolo_top_share=(
                "yolo_class",
                lambda s: s.value_counts(normalize=True).iloc[0],
            ),
            mean_side_px=("x_max", lambda s: 0),
        )  # placeholder, filled below
        .round(3)
    )
    df["side"] = np.sqrt((df["x_max"] - df["x_min"]) * (df["y_max"] - df["y_min"]))
    by_cluster["mean_side_px"] = df.groupby("cluster")["side"].mean().round(0)
    by_cluster.to_csv(neg_dir / "cluster_summary.csv")

    lines = [
        f"# `{NOT_FISH}` candidates, source `{source}`: {SOURCE_TEXT[source]}",
        "",
        f"{len(df)} candidates (IoU < {IOU_NEG} with every expert box, at most {MAX_PER_DROP} per drop) on "
        f"{df['filename'].nunique()} expert-annotated frames across {df['drop'].nunique()} drops. "
        f"Detector median confidence on them is {df['conf'].median():.2f}. Clustered into {n_clusters} groups for review; "
        f"one contact sheet per cluster under {neg_dir.name}/sheets/.",
        "",
        "## By source",
        "",
        df["source"].value_counts().rename("boxes").to_frame().to_markdown(),
        "",
        "## By YOLO class",
        "",
        df["yolo_class"].value_counts().rename("boxes").to_frame().to_markdown(),
        "",
        "## By cluster",
        "",
        by_cluster.to_markdown(),
        "",
    ]
    lines += probe_with_negatives(df, X, common)
    (neg_dir / "negatives_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        choices=sorted(SOURCE_TEXT),
        default="yolo",
        help="yolo: current-model boxes nobody reviewed; upload: COCO boxes BIIGLE experts deleted; review: To review boxes deleted (review_outcome.csv); both",
    )
    main(ap.parse_args().source)
