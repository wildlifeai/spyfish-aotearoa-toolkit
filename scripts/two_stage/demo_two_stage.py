"""Run the two-stage pipeline on a few frames and draw what each stage said.

For each demo frame:
  1. detector  = the current YOLO model's boxes on that frame (run_yolo_on_frames.py)
  2. classifier = BioCLIP 2 crop embedding -> logistic probe trained on every OTHER drop
  3. site prior = presence_prob at the frame's reserve, multiplied in
  4. fallback   = if the top species probability is below P_MIN, the box is `fish`

Each drawn box shows the expert's label, the YOLO label and the two-stage label.
Green: two-stage agrees with the expert; red: it does not; grey: no expert box there.
Writes PNGs to <two_stage_dir>/demo/ and a summary CSV.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from spyfish.config.species import species_registry  # noqa: E402
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.classes import (  # noqa: E402
    OTHER,
    classes_for,
    fit_probe,
    probe_probabilities,
)
from spyfish.ml.classifier.crops import (  # noqa: E402
    OUT_DIR,
    crop_with_margin,
    embed_batch,
    frames_on_disk,
    load_bioclip,
    pick_device,
    read_crops_meta,
    read_predictions,
)
from spyfish.ml.classifier.metrics import (  # noqa: E402
    apply_prior,
    iou,
    load_site_prior,
)
from spyfish.ml.classifier.sheets import load_font  # noqa: E402

N_FRAMES = 8
P_MIN = 0.5
DEVICE = pick_device()


def pick_frames(meta: pd.DataFrame, yolo: pd.DataFrame) -> list[str]:
    """Frames with several species or a non-dominant species, spread over drops."""
    per_frame = meta.groupby("filename").agg(
        n=("crop_id", "size"), species=("species", "nunique"), drop=("drop", "first")
    )
    per_frame = per_frame[per_frame.index.isin(yolo["filename"])]
    rich = per_frame[(per_frame["species"] >= 2) | (per_frame["n"] >= 3)].sort_values(
        ["species", "n"], ascending=False
    )
    picked, drops = [], set()
    for fname, r in rich.iterrows():
        if r["drop"] in drops:
            continue
        picked.append(fname)
        drops.add(r["drop"])
        if len(picked) >= N_FRAMES:
            break
    return picked


def main() -> None:
    demo_dir = OUT_DIR / artifacts.DEMO_DIR
    demo_dir.mkdir(exist_ok=True)
    meta = read_crops_meta()
    X = np.load(OUT_DIR / artifacts.EMBEDDINGS)
    yolo = read_predictions(OUT_DIR / artifacts.YOLO_PREDICTIONS)
    yolo = yolo[yolo["yolo_class"] != "bait"]
    prior = load_site_prior()
    common = species_registry().common_name_map()
    frames = frames_on_disk()

    target, classes, _stats = classes_for(meta)
    y_all = meta["species"].where(meta["species"].isin(target), OTHER).to_numpy()

    model, preprocess, _ = load_bioclip(DEVICE)
    font = load_font(22)
    small = load_font(18)

    def nice(name):
        return common.get(name, name) if name not in (OTHER, "fish") else name

    rows = []
    for fname in pick_frames(meta, yolo):
        drop = meta.loc[meta["filename"] == fname, "drop"].iloc[0]
        reserve = meta.loc[meta["filename"] == fname, "reserve"].iloc[0]
        train = (meta["drop"] != drop).to_numpy()
        probe = fit_probe(X[train], y_all[train])

        img = Image.open(frames[fname]).convert("RGB")
        dets = yolo[yolo["filename"] == fname].reset_index(drop=True)
        experts = meta[meta["filename"] == fname]
        tensors = [
            preprocess(crop_with_margin(img, r.x_min, r.y_min, r.x_max, r.y_max))
            for r in dets.itertuples()
        ]
        if tensors:
            full = probe_probabilities(
                probe, embed_batch(model, tensors, DEVICE), classes
            )
            full = apply_prior(
                full, classes, pd.Series([reserve] * len(dets)), prior, 1.0
            )
        draw = ImageDraw.Draw(img)
        for r in experts.itertuples():
            draw.rectangle(
                [r.x_min, r.y_min, r.x_max, r.y_max], outline=(255, 255, 255), width=2
            )
            draw.text(
                (r.x_min + 4, r.y_max - 24),
                f"expert: {nice(r.species)}",
                fill=(255, 255, 255),
                font=small,
            )
        for i, r in enumerate(dets.itertuples()):
            k = int(full[i].argmax())
            prob = float(full[i, k])
            label = classes[k] if prob >= P_MIN else "fish"
            best = max(
                (
                    (
                        iou(
                            (r.x_min, r.y_min, r.x_max, r.y_max),
                            (e_.x_min, e_.y_min, e_.x_max, e_.y_max),
                        ),
                        e_.species,
                    )
                    for e_ in experts.itertuples()
                ),
                default=(0, None),
            )
            expert_species = best[1] if best[0] >= 0.5 else None
            expert_class = (
                (expert_species if expert_species in target else OTHER)
                if expert_species
                else None
            )
            colour = (
                (120, 120, 120)
                if expert_class is None
                else ((40, 200, 90) if label == expert_class else (230, 60, 60))
            )
            draw.rectangle(
                [r.x_min, r.y_min, r.x_max, r.y_max], outline=colour, width=4
            )
            text = f"YOLO: {nice(r.yolo_class)} {r.conf:.2f}  |  2-stage: {nice(label)} {prob:.2f}"
            tw = draw.textlength(text, font=font)
            draw.rectangle(
                [r.x_min, r.y_min - 28, r.x_min + tw + 8, r.y_min], fill=colour
            )
            draw.text((r.x_min + 4, r.y_min - 26), text, fill=(0, 0, 0), font=font)
            rows.append(
                {
                    "filename": fname,
                    "drop": drop,
                    "reserve": reserve,
                    "yolo": r.yolo_class,
                    "yolo_conf": round(r.conf, 3),
                    "two_stage": label,
                    "two_stage_p": round(prob, 3),
                    "expert": expert_species,
                    "yolo_ok": (
                        None
                        if expert_class is None
                        else (r.yolo_class if r.yolo_class in target else OTHER)
                        == expert_class
                    ),
                    "two_stage_ok": (
                        None if expert_class is None else label == expert_class
                    ),
                }
            )
        draw.text(
            (12, 12),
            f"{drop}  ·  probe trained on the other {meta.loc[train, 'drop'].nunique()} drops  ·  prior: {reserve}",
            fill=(255, 255, 0),
            font=font,
        )
        out = demo_dir / f"{Path(fname).stem}.jpg"
        img.resize((1440, 810)).save(out, quality=80)
        print(
            "wrote", out.name, f"({len(dets)} detections, {len(experts)} expert boxes)"
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(demo_dir / artifacts.DEMO_BOXES, index=False)
    scored = summary.dropna(subset=["expert"])
    print(
        f"\nboxes with an expert match: {len(scored)}; YOLO right {scored['yolo_ok'].mean():.2f}, two-stage right {scored['two_stage_ok'].mean():.2f}"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
