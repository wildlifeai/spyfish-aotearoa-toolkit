"""Turn the static-box review into a `not a fish` training set.

Reads the verdict line the static-box review page copies (claude.ai/artifact/
FTrRBDYmBFo6uLbLjzpe5q) and the finder's CSVs under ``<two_stage_dir>/static_boxes/``,
cuts every crop of every track marked ``junk`` from its video (minus the crops
clicked as fish), and writes the folder ``finetune_bioclip.py --negatives`` reads:
``negatives_meta.csv``, ``negatives_embeddings.npy`` and ``crops/``.

    python scripts/two_stage/static_negatives.py --verdicts verdicts.txt --test-reserves KOK
    FINETUNE_ARGS="--negatives negatives_static --out finetune_static" sbatch scripts/slurm/two_stage.sl finetune

Verdict line: ``; ``-separated groups, ``KEY junk:0,2 fish:1 fishcrops:6.15``, where
``6.15`` is row 6 with sheet crops 1 and 5 being fish. KEY is a survey
(``SLI_20240124``: ``static_tracks_SLI_20240124_BUV_newest.csv`` / ``static_crops_...``),
a first-page reserve code (SLI, AHE, KOK, ORA; see ``LEGACY_KEYS``), or ``HOR_b1`` / ``HOR_b2`` (the first HOR run: b1 is the
unsuffixed ``static_tracks.csv``, b2 was overwritten and is skipped). Row i is row
i of the tracks CSV, and a track's sheet crops are its crops in frame order, so
**do not rerun the finder between marking and running this**: a rerun reorders
the rows. ``canister`` and ``fish`` rows are never used; unmarked rows neither.

``--test-reserves``: the marked drops are none of them frozen test drops, so
without this every crop trains and nothing measures the class. Crops from these
reserves get ``split = test`` (``add_negatives`` keeps a ``split`` column).
Needs the videos on disk and BioCLIP 2 in ``paths.sub_dirs.bioclip``: run on NeSI.
"""

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.extraction.extract_frames import (  # noqa: E402
    read_frame_from_cap,
    read_video_rotation,
)
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.crops import (  # noqa: E402
    crop_with_margin,
    embed_batch,
    load_bioclip,
)
from spyfish.ml.classifier.species_classifier import pick_device  # noqa: E402

OUT_NAME = "negatives_static"


def parse_verdicts(line: str) -> dict[str, dict]:
    """``{key: {"junk": [rows], "fishcrops": {row: {cols 1-based}}}}``."""
    out = {}
    for part in filter(None, (p.strip() for p in line.strip().split(";"))):
        key, *fields = part.split()
        entry = {"junk": [], "fishcrops": {}}
        for f in fields:
            kind, _, values = f.partition(":")
            if kind == "junk":
                entry["junk"] = [int(v) for v in values.split(",") if v]
            elif kind == "fishcrops":
                for v in values.split(","):
                    row, _, cols = v.partition(".")
                    entry["fishcrops"][int(row)] = {int(c) for c in cols}
        out[key] = entry
    return out


# The first review page (2026-11-12) keyed each reserve by its 3-letter code,
# one survey each. Later pages key by survey (``SLI_20240124``), since a reserve
# can have several surveys; these old keys keep meaning the survey they meant.
LEGACY_KEYS = {
    "SLI": "SLI_20260114",
    "AHE": "AHE_20250513",
    "KOK": "KOK_20240219",
    "ORA": "ORA_20260306",
}


def _csv_pair(folder: Path, key: str) -> tuple[Path, Path] | None:
    if key == "HOR_b1":
        pair = folder / "static_tracks.csv", folder / "static_crops.csv"
    elif re.fullmatch(r"[A-Z]{3}(_\d{8})?", key):
        survey = LEGACY_KEYS.get(key, key)
        tracks = folder / f"static_tracks_{survey}_BUV_newest.csv"
        pair = tracks, folder / tracks.name.replace("static_tracks_", "static_crops_")
    else:
        return None
    return pair if all(p.exists() for p in pair) else None


def junk_crops(verdicts: dict, folder: Path) -> pd.DataFrame:
    """One row per crop to use: the junk tracks' sheet crops minus fish clicks."""
    parts = []
    for key, v in verdicts.items():
        pair = _csv_pair(folder, key)
        if pair is None:
            logging.warning(f"{key}: no tracks/crops CSV pair in {folder}, skipped")
            continue
        tracks, crops = (pd.read_csv(p) for p in pair)
        bad = [r for r in v["junk"] if r >= len(tracks)]
        if bad:
            raise ValueError(
                f"{key}: rows {bad} past the {len(tracks)} tracks in {pair[0].name}: "
                "was the finder rerun after marking?"
            )
        for r in v["junk"]:
            tid = tracks.loc[r, "track"]
            mine = (
                crops[crops["track"] == tid].sort_values("frame").reset_index(drop=True)
            )
            skip = {c - 1 for c in v["fishcrops"].get(r, set())}
            mine = mine[~mine.index.isin(skip)].assign(review_key=key, review_row=r)
            parts.append(mine)
    if not parts:
        sys.exit("no junk crops found")
    return pd.concat(parts, ignore_index=True)


def with_corners(df: pd.DataFrame) -> pd.DataFrame:
    """Raw-CSV centre boxes (x, y, w, h) as corner columns."""
    return df.assign(
        x_min=df["x"] - df["w"] / 2,
        y_min=df["y"] - df["h"] / 2,
        x_max=df["x"] + df["w"] / 2,
        y_max=df["y"] + df["h"] / 2,
    )


def cut_and_embed(df: pd.DataFrame, out_dir: Path | None = None) -> np.ndarray:
    """Cut each crop (``x_min..y_max`` on ``frame`` of ``DropID``) from its video
    exactly as training crops are cut and embed it with BioCLIP 2, one row per
    row of ``df``. With ``out_dir``, also save the 224 px thumbnail the
    fine-tune trains on, as ``crops/{neg_id}.jpg``."""
    device = pick_device()
    model, preprocess, _ = load_bioclip(device)
    crops_dir = out_dir / artifacts.CROPS_DIR if out_dir is not None else None
    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)
    emb = [None] * len(df)
    tensors, idx = [], []

    def flush():
        if tensors:
            e = embed_batch(model, tensors, device)
            for k, i in enumerate(idx):
                emb[i] = e[k]
            tensors.clear()
            idx.clear()

    n_drops = df["DropID"].nunique()
    for k, (drop_id, g) in enumerate(df.groupby("DropID"), 1):
        logging.info(f"{drop_id}: {len(g)} crops ({k}/{n_drops} drops)")
        cap = cv2.VideoCapture(
            str(config.get_video_path(config.validate_drop_id(drop_id)))
        )
        if not cap.isOpened():
            raise RuntimeError(f"{drop_id}: video not on local disk")
        rotation = read_video_rotation(cap)
        for i, r in g.sort_values("frame").iterrows():
            frame = read_frame_from_cap(cap, r.time_seconds, int(r.frame), rotation)
            if frame is None:
                raise RuntimeError(f"{drop_id}: frame {r.frame} unreadable")
            img = Image.fromarray(frame[:, :, ::-1])
            crop = crop_with_margin(img, r.x_min, r.y_min, r.x_max, r.y_max)
            if crops_dir is not None:
                thumb = crop.copy()
                thumb.thumbnail((224, 224))
                thumb.save(crops_dir / f"{r.neg_id}.jpg", quality=85)
            tensors.append(preprocess(crop))
            idx.append(i)
            if len(tensors) >= 64:
                flush()
        cap.release()
    flush()
    return np.stack(emb).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--verdicts",
        type=Path,
        required=True,
        help="text file holding the copied verdict line",
    )
    ap.add_argument(
        "--test-reserves",
        nargs="*",
        default=[],
        help="reserve codes whose crops are held out as test",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    folder = config.two_stage_dir / "static_boxes"
    df = with_corners(junk_crops(parse_verdicts(args.verdicts.read_text()), folder))
    df["reserve"] = df["DropID"].str[:3]
    df["drop"] = df["DropID"]
    df["filename"] = df["DropID"] + "_f" + df["frame"].astype(int).astype(str)
    df["source"] = "static"
    df["neg_id"] = [f"s{i:05d}" for i in range(len(df))]
    # Frozen test drops stay test whatever --test-reserves says: a crop from one
    # in training would leak the test set (BNP, KOK, SLI, TUH, TUK have some).
    frozen = set(config.training_test_drops)
    test = df["reserve"].isin(args.test_reserves) | df["drop"].isin(frozen)
    df["split"] = np.where(test, "test", "train")
    print(
        f"{int(df['drop'].isin(frozen).sum())} crops come from frozen test drops (kept test)"
    )

    out_dir = config.two_stage_dir / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = cut_and_embed(df, out_dir)
    cols = [
        "drop",
        "filename",
        "reserve",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "neg_id",
        "source",
        "split",
        "track",
        "review_key",
        "review_row",
        "frame",
        "time_seconds",
        "confidence",
    ]
    df[cols].to_csv(out_dir / artifacts.NEGATIVES_META, index=False)
    np.save(out_dir / artifacts.NEGATIVES_EMBEDDINGS, emb)
    shutil.copy(args.verdicts, out_dir / "verdicts.txt")  # provenance: what was marked
    by = df.groupby(["reserve", "split"]).size().unstack(fill_value=0)
    print(by.to_string())
    print(
        f"\n{len(df)} `not a fish` crops from {df['drop'].nunique()} drops → {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
