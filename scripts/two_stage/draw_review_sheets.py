"""Draw expert boxes (green) and model-only boxes (red) on pulled frames, plus contact sheets.

Runs on the Mac against the NeSI mirror: labels from ``--root`` (the mirrored
species dataset), model-only boxes from ``--csv`` (audit output), frames from
``--pulled`` (whatever ``sample_review_frames.py`` + rsync brought over). Red
boxes at or above --min-conf are solid, fainter ones dashed-thin.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from scripts.two_stage.audit_dataset import load_dataset  # noqa: E402
from spyfish.config.species import species_registry  # noqa: E402
from spyfish.ml.classifier.sheets import load_font  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", type=Path, default=Path("process_files/nesi_mirror/species")
    )
    ap.add_argument(
        "--pulled", type=Path, default=Path("process_files/nesi_mirror/pulled")
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="default: the model-only CSV found under --pulled",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("process_files/nesi_mirror/review/sample")
    )
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--exclude", nargs="*", default=["volume_30505"])
    ap.add_argument(
        "--drops",
        nargs="*",
        default=[],
        help="only frames of these drops; one row of 3 per drop",
    )
    args = ap.parse_args()

    csv = args.csv or next(args.pulled.rglob("model_only_detector_predictions.csv"))
    mo = pd.read_csv(csv)
    common = species_registry().common_name_map()
    _, boxes = load_dataset(args.root)
    paths = {
        p.name: p
        for p in args.pulled.rglob("*.jpg")
        if not any(e in str(p) for e in args.exclude)
    }
    paths = {f: p for f, p in paths.items() if f in set(mo["filename"])}
    if args.drops:
        keep = set(mo.loc[mo["drop"].isin(args.drops), "filename"])
        paths = {f: p for f, p in paths.items() if f in keep}
    args.out.mkdir(parents=True, exist_ok=True)
    big, small = load_font(20), load_font(15)
    thumbs = []
    drop_of = mo.drop_duplicates("filename").set_index("filename")["drop"]
    order = {d: i for i, d in enumerate(args.drops)}
    for f, p in sorted(
        paths.items(), key=lambda kv: (order.get(drop_of[kv[0]], 99), kv[0])
    ):
        im = Image.open(p).convert("RGB")
        W, H = im.size
        d, lw = ImageDraw.Draw(im), max(2, W // 450)
        e = boxes[boxes["filename"] == f]
        for r in e.itertuples():
            x0, y0, x1, y1 = (
                (r.cx - r.w / 2) * W,
                (r.cy - r.h / 2) * H,
                (r.cx + r.w / 2) * W,
                (r.cy + r.h / 2) * H,
            )
            d.rectangle([x0, y0, x1, y1], outline=(40, 220, 90), width=lw)
            d.text(
                (x0 + 3, y0 + 2),
                common.get(r.species, r.species),
                fill=(40, 220, 90),
                font=big,
            )
        m = mo[mo["filename"] == f]
        n_conf = 0
        for r in m.itertuples():
            strong = r.conf >= args.min_conf
            n_conf += strong
            d.rectangle(
                [r.x_min, r.y_min, r.x_max, r.y_max],
                outline=(240, 50, 50) if strong else (240, 160, 60),
                width=lw if strong else 1,
            )
            d.text(
                (r.x_min + 3, r.y_max - 24),
                f"{r.conf:.2f}",
                fill=(240, 50, 50),
                font=big,
            )
        drop = m["drop"].iloc[0]
        im.save(args.out / f"{drop}__{f}", quality=88)
        t = im.copy()
        t.thumbnail((640, 480))
        thumbs.append((t, drop, len(e), n_conf))
    cols, rows, cw, ch = (3, 3, 640, 500) if args.drops else (4, 3, 640, 500)
    for page in range(0, len(thumbs), cols * rows):
        sheet = Image.new("RGB", (cols * cw, rows * ch), (25, 25, 28))
        sd = ImageDraw.Draw(sheet)
        for i, (t, drop, ne, nr) in enumerate(thumbs[page : page + cols * rows]):
            x, y = (i % cols) * cw, (i // cols) * ch
            sheet.paste(t, (x, y))
            sd.text(
                (x + 4, y + t.size[1] + 4),
                f"{drop}  green {ne} expert, red {nr} model-only >= {args.min_conf}",
                fill=(230, 230, 230),
                font=small,
            )
        sheet.save(args.out / f"_sheet_{page // (cols * rows) + 1}.jpg", quality=85)
    print(f"{len(thumbs)} frames, {(len(thumbs) + 11) // 12} sheets -> {args.out}")


if __name__ == "__main__":
    main()
