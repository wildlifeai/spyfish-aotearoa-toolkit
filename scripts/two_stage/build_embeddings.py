"""Crop every expert box on disk and embed the crops with BioCLIP 2.

Plan §4.1–4.2. Writes to the two-stage output folder (paths.sub_dirs.two_stage):
    crops_meta.csv        one row per crop (drop, reserve, species, box, crop file)
    embeddings.npy        L2-normalised image embeddings, same row order
    text_embeddings.npy   one normalised text embedding per species in `species.txt`
    crops/                224px thumbnails for contact sheets and the demo

Crops come from the full-resolution 1920×1080 frames with a 10% margin.
Boxes per (drop, species) are capped so blue cod does not swamp the probe
(no cap in the UoA source mode: every box there counts).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.crops import (  # noqa: E402
    MAX_PER_DROP_SPECIES,
    OUT_DIR,
    SEED,
    classifier_boxes,
    embed_crops,
    load_bioclip,
    load_boxes,
    pick_device,
)

BATCH = 64
DEVICE = pick_device()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crops_dir = OUT_DIR / artifacts.CROPS_DIR
    crops_dir.mkdir(exist_ok=True)

    boxes = classifier_boxes(load_boxes())  # no bait, no catch-all, no excluded species
    rng = np.random.default_rng(SEED)
    keep = []
    for _, g in boxes.groupby(["drop", "species"]):
        idx = g.index.to_numpy()
        if MAX_PER_DROP_SPECIES is not None and len(idx) > MAX_PER_DROP_SPECIES:
            idx = rng.choice(idx, MAX_PER_DROP_SPECIES, replace=False)
        keep.append(idx)
    boxes = (
        boxes.loc[np.concatenate(keep)]
        .sort_values(["frame_path"])
        .reset_index(drop=True)
    )
    boxes["crop_id"] = [f"{i:06d}" for i in range(len(boxes))]
    boxes["crop_file"] = boxes["crop_id"].map(
        lambda c: f"{artifacts.CROPS_DIR}/{c}.jpg"
    )
    print(
        f"{len(boxes)} crops from {boxes['filename'].nunique()} frames, "
        f"{boxes['drop'].nunique()} drops, {boxes['species'].nunique()} species"
    )

    model, preprocess, tokenizer = load_bioclip(DEVICE)

    t0 = time.time()

    def save_thumbnail(crop, r) -> None:
        # 224 px on the long side, small crops scaled up
        crop.resize(
            (224, int(224 * crop.height / max(crop.width, 1)))
            if crop.width >= crop.height
            else (int(224 * crop.width / max(crop.height, 1)), 224)
        ).save(crops_dir / f"{r.crop_id}.jpg", quality=85)

    def progress(done: int, total: int) -> None:
        if done % (BATCH * 10) < BATCH:
            print(f"  {done}/{total} crops, {time.time() - t0:.0f}s", flush=True)

    embeddings = embed_crops(
        boxes,
        "frame_path",
        model,
        preprocess,
        DEVICE,
        save_thumbnail=save_thumbnail,
        batch_size=BATCH,
        progress=progress,
    )

    species = sorted(boxes["species"].unique())
    prompts = [f"a photo of {s}." for s in species]
    with torch.no_grad():
        t = model.encode_text(tokenizer(prompts).to(DEVICE))
    text = torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy()

    boxes.drop(columns=["frame_path"]).to_csv(
        OUT_DIR / artifacts.CROPS_META, index=False
    )
    np.save(OUT_DIR / artifacts.EMBEDDINGS, embeddings)
    np.save(OUT_DIR / artifacts.TEXT_EMBEDDINGS, text)
    (OUT_DIR / artifacts.SPECIES_LIST).write_text("\n".join(species) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
