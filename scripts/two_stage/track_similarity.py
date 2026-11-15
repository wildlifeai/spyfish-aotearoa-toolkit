"""How alike a static track's crops look over time: a stick does not change, a fish does.

For every track in the static-box finder's ``static_crops_*.csv`` (the crops its
sheet shows), embed the crops with BioCLIP 2 and report the mean and the lowest
cosine similarity between them. Writes ``static_similarity_{same tag}.csv`` next
to each crops CSV, keyed by ``track``, so the review page rows (row i = row i of
the tracks CSV) are untouched and the marks stay valid.

    python scripts/two_stage/track_similarity.py               # every crops CSV in the folder
    python scripts/two_stage/track_similarity.py static_crops_SLI_20260114_BUV_newest.csv

Needs the videos on disk and BioCLIP 2 in ``paths.sub_dirs.bioclip``: run on NeSI.
"""

import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from static_negatives import cut_and_embed, with_corners  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402


def similarity(crops: pd.DataFrame) -> pd.DataFrame:
    """One row per track: mean and min pairwise cosine of its crops' embeddings."""
    crops = with_corners(crops.reset_index(drop=True))
    emb = cut_and_embed(crops)  # L2-normalised, so a dot product is the cosine
    rows = []
    for track, g in crops.groupby("track", sort=False):
        e = emb[g.index]
        sims = [float(e[a] @ e[b]) for a, b in itertools.combinations(range(len(e)), 2)]
        rows.append(
            {
                "track": track,
                "crops": len(e),
                "look_alike_mean": round(float(np.mean(sims)), 3) if sims else None,
                "look_alike_min": round(float(np.min(sims)), 3) if sims else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    folder = config.two_stage_dir / "static_boxes"
    files = [folder / a for a in sys.argv[1:]] or sorted(
        folder.glob("static_crops*.csv")
    )
    for f in files:
        out = f.with_name(f.name.replace("static_crops", "static_similarity", 1))
        sim = similarity(pd.read_csv(f))
        sim.to_csv(out, index=False)
        print(
            f"{f.name}: {len(sim)} tracks, look-alike median {sim['look_alike_mean'].median():.2f} → {out.name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
