"""One JPG of the boxes YOLO and BioCLIP disagree on, to see which is right.

Reads the CSV ``scripts/compare_two_stage_names.py --export`` writes, cuts each
box from the drop's video with the same margin the classifier was given, and
lays the crops out grouped by disagreement (YOLO's name → what was written),
each captioned with both names and confidences in common names.

    python scripts/compare_two_stage_names.py HOR_20251204_BUV_HOR_003_01 \\
        --model species_20260825_053442 --export hor003_flagged.csv
    python scripts/flagged_crops_sheet.py hor003_flagged.csv hor003_flagged.jpg

Needs the videos on local disk (NeSI). ``--max-per-pair`` caps each group.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.extraction.extract_frames import (  # noqa: E402
    VideoOpenError,
    iter_raw_frames,
)
from spyfish.ml.classifier.sheets import load_font  # noqa: E402
from spyfish.ml.classifier.species_classifier import crop_box  # noqa: E402

TILE = 240  # crop area, px square
CAPTION = 44  # two caption lines under each crop
HEADER = 34  # group heading row
COLS = 6
BG, FG, MUTED, YOLO_C, CLF_C = "#101615", "#e6ecea", "#9aaba8", "#f2b04b", "#5fc3dc"


def _crops(df: pd.DataFrame) -> list[Image.Image | None]:
    """One crop per row, in row order; None where the frame cannot be read."""
    out: dict[int, Image.Image | None] = {}
    for drop_id, rows in df.groupby("DropID"):
        video = config.get_video_path(config.validate_drop_id(drop_id))
        try:
            for _frame_idx, frame, fr_rows in iter_raw_frames(video, rows):
                for idx, r in fr_rows.iterrows():
                    out[idx] = (
                        None
                        if frame is None
                        else crop_box(
                            frame, r.x, r.y, r.w, r.h, config.classifier_crop_margin
                        )
                    )
        except VideoOpenError:
            sys.exit(f"{drop_id}: cannot open {video}")
    return [out.get(i) for i in df.index]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("flagged_csv", type=Path)
    ap.add_argument("out_jpg", type=Path)
    ap.add_argument("--max-per-pair", type=int, default=24)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    registry = species_registry()
    names = registry.common_name_map()
    common = lambda n: names.get(registry.resolve_name(n)[0], n)  # noqa: E731

    df = pd.read_csv(args.flagged_csv)
    df["pair"] = df["yolo_class"].map(common) + " -> " + df["class"].map(common)
    df = (
        df.sort_values(["pair", "clf_confidence"], ascending=[True, False])
        .groupby("pair", sort=False)
        .head(args.max_per_pair)
    )
    # Biggest disagreement first.
    order = df["pair"].value_counts().index
    df = df.set_index("pair").loc[order].reset_index()
    crops = _crops(df)

    # the built-in font on every machine, so the sheet looks the same everywhere
    small, big = load_font(14, candidates=()), load_font(18, candidates=())
    blocks = []  # (heading, [(crop, caption lines)])
    for pair, rows in df.groupby("pair", sort=False):
        tiles = [
            (
                crops[i],
                (
                    f"YOLO {common(r.yolo_class)} {r.yolo_confidence:.2f}",
                    f"BioCLIP {common(r.clf_class)} {r.clf_confidence:.2f}",
                ),
            )
            for i, r in rows.iterrows()
        ]
        blocks.append((f"{pair}   ({len(tiles)})", tiles))

    cell_h = TILE + CAPTION
    height = sum(HEADER + -(-len(t) // COLS) * cell_h for _, t in blocks) + 8
    sheet = Image.new("RGB", (COLS * TILE, height), BG)
    draw = ImageDraw.Draw(sheet)
    y = 0
    for heading, tiles in blocks:
        draw.text((10, y + 8), heading, fill=FG, font=big)
        y += HEADER
        for k, (crop, (yolo_line, clf_line)) in enumerate(tiles):
            x0, y0 = (k % COLS) * TILE, y + (k // COLS) * cell_h
            if crop is not None:
                crop.thumbnail((TILE - 8, TILE - 8))
                sheet.paste(
                    crop,
                    (x0 + (TILE - crop.width) // 2, y0 + (TILE - crop.height) // 2),
                )
            else:
                draw.text(
                    (x0 + 10, y0 + TILE // 2), "unreadable", fill=MUTED, font=small
                )
            draw.text((x0 + 6, y0 + TILE + 2), yolo_line, fill=YOLO_C, font=small)
            draw.text((x0 + 6, y0 + TILE + 22), clf_line, fill=CLF_C, font=small)
        y += -(-len(tiles) // COLS) * cell_h
    sheet.save(args.out_jpg, quality=88)
    logging.info(f"{len(df)} crops in {len(blocks)} groups → {args.out_jpg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
