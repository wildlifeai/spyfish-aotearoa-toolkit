"""Find ML boxes that never move: sea cucumbers, sea stars, sticks, rocks.

A fish can rest, but not in one spot for minutes on end; a stick on the sand
can. HOR, 2026-11-12: YOLO called such objects blue cod and BioCLIP renamed
them scarlet wrasse, hagfish and rock lobster at 0.86-0.94, because it has no
`not a fish` answer. They are the cleanest `not a fish` negatives there are:
no human pass needed to say they are not fish, and every drop the pipeline has
run on has some, at every reserve.

Tracking: boxes are linked frame to frame into tracks. A box joins the track
whose MEAN centre is nearest, within half a box diagonal or ``link_distance_px``
(whichever is larger) and of similar size, so a fixed object rocking with the
camera in the swell stays one track while a fish that swims off leaves it. A track survives ``max_gap_seconds`` without a detection: long, because YOLO
sees a faint stick on ~1% of frames, so continuity is no signal. A track is
static when it spans ``min_span_seconds`` with ``min_boxes`` boxes and the SPOT
is tight: every box the drop has around the track's centre sits within
``max_spread_px`` (median distance from their median) and agrees in size within
``max_size_spread``. The same pixels, the same size, minute after minute. The
crowd at the bait never is, though linking can carve tight-looking tracks out
of it, which is why the spot is judged and not the track. Knobs: ``training.classifier.static_boxes``.

    python scripts/two_stage/static_boxes.py HOR_20251204_BUV_HOR_033_01 HOR_20251204_BUV_HOR_017_01
    python scripts/two_stage/static_boxes.py --survey HOR_20251204_BUV --sheet hor_static.jpg
    python scripts/two_stage/static_boxes.py --survey HOR_20251204_BUV --model species_20260825_053442

Writes ``static_tracks_{survey}_{model}.csv`` (one row per static track) and
``static_crops_{survey}_{model}.csv`` (``crops_per_track`` boxes per track, spread over its
span, in the raw CSV's columns plus ``DropID`` and ``track``) under
``<two_stage_dir>/static_boxes/``. ``--sheet`` draws each track's crops in a
row, so a resting fish caught by the rule shows up before anything trains.
Needs the videos for ``--sheet`` only.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spyfish.config.base import MlStatus  # noqa: E402
from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.database.manager import DatabaseManager  # noqa: E402
from spyfish.extraction.extract_frames import (  # noqa: E402
    read_frame_from_cap,
    read_video_rotation,
)
from spyfish.ml.classifier.species_classifier import crop_box  # noqa: E402
from spyfish.ml.static_boxes import (  # noqa: E402,F401
    fish_score,
    link_tracks,
    static_tracks,
)

TILE = 200
CAPTION = 40


def draw_sheet(tracks: pd.DataFrame, crops: pd.DataFrame, out: Path) -> None:
    """One row per track: its crops left to right in time, caption on top."""
    names = species_registry().common_name_map()
    registry = species_registry()
    common = lambda n: names.get(registry.resolve_name(n)[0], n)  # noqa: E731
    per_row = config.classifier_static_boxes.crops_per_track
    font = ImageFont.load_default(size=14)
    head = 26
    sheet = Image.new(
        "RGB", (per_row * TILE, len(tracks) * (TILE + head + 8)), "#101615"
    )
    draw = ImageDraw.Draw(sheet)
    for drop_id, drop_crops in crops.groupby("DropID"):
        cap = cv2.VideoCapture(
            str(config.get_video_path(config.validate_drop_id(drop_id)))
        )
        if not cap.isOpened():
            logging.warning(f"{drop_id}: no local video, its tracks are left blank")
            continue
        rotation = read_video_rotation(cap)
        for _, r in drop_crops.sort_values("frame").iterrows():
            row_i = tracks.index[tracks["track"] == r["track"]][0]
            col = list(
                drop_crops[drop_crops["track"] == r["track"]].sort_values("frame").index
            ).index(r.name)
            frame = read_frame_from_cap(
                cap, r["time_seconds"], int(r["frame"]), rotation
            )
            crop = (
                None
                if frame is None
                else crop_box(frame, r.x, r.y, r.w, r.h, config.classifier_crop_margin)
            )
            if crop is not None:
                crop.thumbnail((TILE - 6, TILE - 6))
                y0 = row_i * (TILE + head + 8) + head
                sheet.paste(
                    crop,
                    (
                        col * TILE + (TILE - crop.width) // 2,
                        y0 + (TILE - crop.height) // 2,
                    ),
                )
        cap.release()
    for row_i, t in tracks.iterrows():
        y0 = row_i * (TILE + head + 8)
        draw.text(
            (6, y0 + 5),
            f"{t.DropID.split('_BUV_')[1]}  {t.span_min} min, {t.boxes} boxes, spread {t.spread_px:.0f}px  "
            f"YOLO {common(t.yolo_top)} {t.mean_conf:.2f} / BioCLIP {common(t.bioclip_top)}"
            + (f" {t.bioclip_conf:.2f}" if pd.notna(t.bioclip_conf) else ""),
            fill="#e6ecea",
            font=font,
        )
    sheet.save(out, quality=85)


def _drop_ids(args) -> list[str]:
    if args.survey:
        rows = DatabaseManager().get_deployments_eligible(
            MlStatus.COLUMN, [MlStatus.COMPLETE, MlStatus.INFERRED]
        )
        return [
            r["drop_id"] for r in rows if r["drop_id"].startswith(f"{args.survey}_")
        ]
    return args.drop_ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("drop_ids", nargs="*")
    ap.add_argument("--survey", metavar="SURVEY_ID")
    ap.add_argument("--model", help="raw CSV model name; default: the pipeline model")
    ap.add_argument(
        "--sheet", type=Path, help="also draw the tracks' crops into this JPG"
    )
    ap.add_argument(
        "--newest-csv",
        action="store_true",
        help="per drop, the raw CSV of the newest model it has (surveys run under older models)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model_name = args.model or Path(config.pipeline_model_path).stem
    drop_ids = [config.validate_drop_id(d) for d in _drop_ids(args)]
    if not drop_ids:
        sys.exit("no drops given / found")
    all_tracks, all_crops = [], []
    for drop_id in drop_ids:
        if args.newest_csv:
            found = config.list_ml_raw_csvs(drop_id)
            if not found:
                logging.warning(f"{drop_id}: no ML raw CSV at all, skipped")
                continue
            newest = max(found, key=lambda m: config.model_name_stamp(m))
            raw = found[newest]
            logging.info(f"{drop_id}: using {newest}")
        else:
            raw = config.get_raw_csv_path(drop_id, model_name)
            if not raw.exists():
                logging.warning(f"{drop_id}: no raw CSV for {model_name}, skipped")
                continue
        df = pd.read_csv(raw)
        df = df[df["confidence"] >= config.confidence_threshold]
        tracks, crops = static_tracks(df, drop_id)
        logging.info(f"{drop_id}: {len(tracks)} static track(s) among {len(df)} boxes")
        all_tracks.append(tracks)
        all_crops.append(crops)
    tracks = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    if not tracks.empty:
        # Most static first: the sheet reads from "surely junk" down to the
        # tracks that may be fish holding station at the bait.
        tracks = tracks.sort_values(["spread_px", "size_spread"]).reset_index(drop=True)
    crops = pd.concat(all_crops) if all_crops else pd.DataFrame()
    out_dir = config.two_stage_dir / "static_boxes"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Named by what was scanned, so a second run (other batch, other model)
    # does not overwrite the first.
    scanned = args.survey or (drop_ids[0] if len(drop_ids) == 1 else "drops")
    tag = f"{scanned}_{'newest' if args.newest_csv else model_name}"
    tracks.to_csv(out_dir / f"static_tracks_{tag}.csv", index=False)
    crops.to_csv(out_dir / f"static_crops_{tag}.csv", index=False)
    if tracks.empty:
        print("no static tracks")
        return 0
    pd.set_option("display.width", 200)
    print(tracks.drop(columns=["track"]).to_string(index=False))
    print(f"\n{len(tracks)} static tracks, {len(crops)} crops → {out_dir}")
    if args.sheet:
        draw_sheet(tracks, crops, args.sheet)
        print(f"sheet → {args.sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
