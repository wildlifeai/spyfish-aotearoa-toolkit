"""Build the static-box review page (claude.ai/artifact/FTrRBDYmBFo6uLbLjzpe5q).

Reads the finder's sheets + track CSVs pulled from NeSI into
process_files/nesi_mirror/static_boxes/ and writes static_box_review.html next to
this file, from template.html (the page's CSS + JS). Marks live in the viewer's
browser, keyed "<survey or first-page reserve code>:<row>", so a rebuild keeps them
as long as the finder was not rerun on a marked survey. New surveys get rows scoring
below 0 pre-marked junk. Publish the HTML to the same artifact URL.

    python scripts/two_stage/static_review/build_page.py
"""

import base64
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spyfish.config.species import species_registry

HERE = Path(__file__).parent
REPO = HERE.parents[2]
SB = REPO / "process_files/nesi_mirror/static_boxes"
reg = species_registry()
names = reg.common_name_map()
common = lambda n: names.get(reg.resolve_name(n)[0], n) if isinstance(n, str) else ""
ROW = 234
# first page keyed these surveys by reserve code; keep those keys so marks persist
LEGACY = {
    "SLI_20260114": "SLI",
    "AHE_20250513": "AHE",
    "KOK_20240219": "KOK",
    "ORA_20260306": "ORA",
}


def rows_of(fn):
    im = Image.open(fn).convert("RGB")
    out = []
    for i in range(im.height // ROW):
        r = im.crop((0, i * ROW + 26, 1000, (i + 1) * ROW)).resize(
            (860, int((ROW - 26) * 0.86))
        )
        b = io.BytesIO()
        r.save(b, "JPEG", quality=72)
        out.append(base64.b64encode(b.getvalue()).decode())
    return out


tables = {}
for f in sorted(SB.glob("static_tracks_*_BUV_newest.csv")):
    tag = f.name[len("static_tracks_") : -len("_BUV_newest.csv")]
    try:
        t = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        continue
    if len(t) and (SB / f"static_{tag}.jpg").exists():
        tables[tag] = t
# score scale fixed on the four first-reviewed surveys, so scores stay comparable
base = pd.concat([tables[s] for s in LEGACY])
rate = lambda t: -np.log(t.boxes / t.span_min)
mu_c, sd_c, mu_r, sd_r = (
    base.mean_conf.mean(),
    base.mean_conf.std(),
    rate(base).mean(),
    rate(base).std(),
)
score = lambda t: (t.mean_conf - mu_c) / sd_c + (rate(t) - mu_r) / sd_r

groups, SC = [], {}
hor = [
    (
        "HOR_b2",
        "hor_static_b2.jpg",
        [
            ("065_01", 27.1, 84, 6.1, "Blue cod", 0.43, "Rock lobster"),
            ("017_01", 9.8, 22, 9.1, "Blue cod", 0.56, "Blue cod"),
            ("022_02", 16.6, 269, 11.4, "Blue cod", 0.36, "Blue cod"),
            ("017_01", 21.7, 66, 19.5, "Blue cod", 0.39, "Hagfishes"),
            ("033_01", 22.3, 1310, 24.7, "Blue cod", 0.60, "Blue cod"),
        ],
        ["junk", "fish", "junk", "junk", "junk"],
    ),
    (
        "HOR_b1",
        "hor_static_b1.jpg",
        [
            ("003_01", 29.7, 276, 2.9, "Blue cod", 0.40, "Blue cod"),
            ("012_01", 22.7, 276, 7.9, "Blue cod", 0.44, "Scarlet wrasse"),
            ("048_01", 22.6, 168, 13.8, "fish", 0.28, "Blue cod"),
            ("036_01", 28.5, 55, 14.4, "Blue cod", 0.34, "Blue cod"),
            ("036_01", 20.7, 53, 15.4, "Spotty", 0.35, "Blue cod"),
            ("036_01", 29.8, 376, 15.8, "Spotty", 0.38, "Blue cod"),
            ("012_01", 25.9, 111, 19.8, "Blue cod", 0.59, "Blue cod"),
            ("036_01", 27.7, 59, 22.8, "Spotty", 0.34, "Blue cod"),
            ("036_01", 13.3, 286, 23.2, "Blue cod", 0.42, "Blue cod"),
        ],
        ["junk"] * 9,
    ),
]
for key, fn, rows, pre in hor:
    groups.append(
        dict(
            key=key,
            label=f"HOR batch {key[-1]} (earlier finder version)",
            rows=rows_of(SB / fn),
            pre=pre,
            auto=[],
            meta=[
                dict(
                    drop=d,
                    span=s,
                    boxes=n,
                    spread=sp,
                    yolo=y,
                    yconf=yc,
                    clf=cl,
                    cconf=None,
                )
                for d, s, n, sp, y, yc, cl in rows
            ],
        )
    )
order = list(LEGACY) + sorted(t for t in tables if t not in LEGACY)
for tag in order:
    t = tables[tag]
    key = LEGACY.get(tag, tag)
    sc = score(t).round(2).tolist()
    new = tag not in LEGACY
    groups.append(
        dict(
            key=key,
            label=tag.replace("_", " ") + ("" if new else " (first review)"),
            rows=rows_of(SB / f"static_{tag}.jpg"),
            pre=[("junk" if s < 0 else None) for s in sc] if new else [],
            auto=[s < 0 for s in sc] if new else [],
            meta=[
                dict(
                    drop=r.DropID.split("_BUV_")[1],
                    span=r.span_min,
                    boxes=int(r.boxes),
                    spread=r.spread_px,
                    yolo=common(r.yolo_top),
                    yconf=r.mean_conf,
                    clf=common(r.bioclip_top) if isinstance(r.bioclip_top, str) else "",
                    cconf=None if pd.isna(r.bioclip_conf) else r.bioclip_conf,
                )
                for r in t.itertuples()
            ],
        )
    )
    for i, s in enumerate(sc):
        SC[f"{key}:{i}"] = s
D = dict(groups=groups, crop_pre={"HOR_b1:6": [4]})
html = (
    (HERE / "template.html")
    .read_text()
    .replace("__SC__", json.dumps(SC))
    .replace("__D__", json.dumps(D))
)
(HERE / "static_box_review.html").write_text(html)
total = sum(len(g["rows"]) for g in groups)
auto = sum(sum(g["auto"]) for g in groups)
print(
    total,
    "tracks;",
    auto,
    "pre-marked junk by score; groups:",
    [(g["key"], len(g["rows"])) for g in groups],
)
