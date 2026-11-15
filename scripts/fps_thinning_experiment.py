"""Would a lower inference frame rate change MaxN? Test it on existing 3 fps raw CSVs.

Every species-model raw CSV was produced at ``ml_inference.ml_fps`` = 3, so a
lower rate can be simulated with no new inference: keep every 2nd sampled
frame (1.5 fps) or every 3rd (1 fps) and re-run the real ``process_maxn`` on
the thinned CSV. Each thinning has k phases (which of the k frames to keep);
all are run, because the spread between phases is itself the sampling noise.

Two filter settings are run for every rate:

* ``pipeline``: persistence / gap-fill in seconds as configured. The window is
  converted to frames from the CSV's own spacing, so at 1.5 fps a 1.0 s
  window is 2 frames and at 1 fps it is 1 frame, i.e. the filter is OFF.
  This is what the pipeline would actually do if ml_fps were lowered.
* ``nofilter``: persistence 0, gap 0. Isolates the sampling effect.

Outputs (``--out``):
    maxn_long.csv       one row per (model, drop, variant, k, phase, species)
    vs_3fps.csv         per-species agreement of each thinning with 3 fps
    vs_expert.csv       MAE/bias against expert MaxN where the annotations DB
                        has an expert account of the drop
    summary printed to stdout

Usage:
    python scripts/fps_thinning_experiment.py --download   # pull raw CSVs from S3
    python scripts/fps_thinning_experiment.py              # reuse the local copies
    python scripts/fps_thinning_experiment.py --raw-dir /path/with/raw/csvs
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.config.species import species_registry
from spyfish.config.wrapper import config
from spyfish.ml.process_ml_annotations import process_maxn
from spyfish.ml.raw_detections import read_raw_detections


def _species_raw(name: str) -> "tuple[str, str] | None":
    """(drop_id, model) for a species-model raw CSV name or S3 key, else None."""
    parsed = config.parse_ml_raw_csv_name(name)
    return parsed if parsed and parsed[1].startswith("species_") else None


FACTORS = (1, 2, 3)  # 3 fps, 1.5 fps, 1 fps


def variants(k: int) -> dict:
    """Filter settings per thinning factor.

    ``pipeline`` keeps the configured seconds (what lowering ml_fps would do:
    at 1.5 and 1 fps the 1.0 s window rounds to ONE frame, i.e. no filter).
    ``sameframes`` scales the seconds by k so the window stays the same
    number of sampled frames (3 frames = 2.0 s at 1.5 fps, 3.0 s at 1 fps).
    ``nofilter`` isolates the sampling effect.
    """
    return {
        "pipeline": dict(
            persistence_seconds=config.maxn_persistence_seconds,
            gap_fill_seconds=config.maxn_gap_fill_seconds,
        ),
        "sameframes": dict(
            persistence_seconds=config.maxn_persistence_seconds * k,
            gap_fill_seconds=config.maxn_gap_fill_seconds * k,
        ),
        "nofilter": dict(persistence_seconds=0.0, gap_fill_seconds=0.0),
    }


def download_raw_csvs(raw_dir: Path) -> int:
    """Pull every species-model raw CSV under deployment_data on S3."""
    from spyfish.storage.s3_handler import S3Handler

    s3 = S3Handler()
    keys = s3.get_file_paths_set_from_s3(
        f"{config.s3_deployment_data_dir}/", suffixes=(config.ml_raw_suffix,)
    )
    n = 0
    for key in sorted(keys):
        parsed = _species_raw(key)
        if not parsed:
            continue
        drop_id, model = parsed
        target = raw_dir / model / Path(key).name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not s3.download_object_from_s3(key, str(target)):
            raise RuntimeError(f"download failed: {key}")
        n += 1
    return n


def sampling_stride(raw_df: pd.DataFrame) -> int:
    """Video-frame stride of the inference grid: modal gap between detected frames."""
    frames = np.sort(raw_df["frame"].unique())
    if len(frames) < 2:
        return 1
    return int(pd.Series(np.diff(frames)).mode().iloc[0])


def thin(raw_df: pd.DataFrame, k: int, phase: int) -> pd.DataFrame:
    """Keep every k-th sampled frame, starting at `phase`.

    Frames are `first + n * stride` exactly (integer vid_stride), so the grid
    index n is recovered without rounding error and thinning is a true subset.
    """
    if k == 1:
        return raw_df
    stride = sampling_stride(raw_df)
    first = int(raw_df["frame"].min())
    grid_idx = ((raw_df["frame"] - first) // stride).astype(int)
    return raw_df[grid_idx % k == phase]


def drop_maxn(
    raw_df: pd.DataFrame, drop_id: str, model: str, out_csv: Path, **filters
) -> dict:
    """{species: drop-level MaxN} from the real process_maxn on this CSV."""
    df = process_maxn(
        raw_df,
        out_csv,
        drop_id,
        interval_seconds=config.interval_seconds,
        confidence_threshold=config.maxn_confidence_threshold,
        model_name=model,
        exclude_classes=tuple(config.maxn_exclude_classes),
        **filters,
    )
    if df.empty:
        return {}
    per_species = df.groupby(config.csv_scientific_name_column)[
        config.csv_max_interval_column
    ].max()
    return {sp: int(v) for sp, v in per_species.items() if v > 0}


def run_all(raw_dir: Path, work_dir: Path) -> pd.DataFrame:
    rows = []
    files = sorted(raw_dir.rglob("*_raw.csv"))
    for i, path in enumerate(files, 1):
        parsed = _species_raw(path.name)
        if not parsed:
            continue
        drop_id, model = parsed
        raw_df = read_raw_detections(path)
        # Review-frame exports (frames seeked by time carry frame -1) and
        # near-empty runs have no sampling grid to thin.
        raw_df = raw_df[raw_df["frame"] >= 0]
        if raw_df["frame"].nunique() < 3:
            print(f"  skip {path.name}: no sampled grid")
            continue
        stride = sampling_stride(raw_df)
        dt = float(
            np.median(
                np.diff(raw_df.groupby("frame")["time_seconds"].first().sort_index())
            )
        )
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)} {drop_id} {model} stride={stride} dt={dt:.3f}")
        for k in FACTORS:
            for variant, filters in variants(k).items():
                for phase in range(k):
                    thinned = thin(raw_df, k, phase)
                    maxn = drop_maxn(
                        thinned,
                        drop_id,
                        model,
                        work_dir / f"{drop_id}_{model}_{variant}_k{k}p{phase}.csv",
                        **filters,
                    )
                    for sp, v in maxn.items():
                        rows.append(
                            dict(
                                model=model,
                                drop_id=drop_id,
                                variant=variant,
                                k=k,
                                phase=phase,
                                species=sp,
                                maxn=v,
                            )
                        )
    return pd.DataFrame(rows)


def compare_to_3fps(long: pd.DataFrame) -> pd.DataFrame:
    """Per species: how often and how far each thinning departs from 3 fps."""
    base = long[long.k == 1].set_index(["model", "drop_id", "variant", "species"])[
        "maxn"
    ]
    out = []
    for (variant, k, phase), g in long[long.k > 1].groupby(["variant", "k", "phase"]):
        thinned = g.set_index(["model", "drop_id", "variant", "species"])["maxn"]
        b = base[base.index.get_level_values("variant") == variant]
        both = pd.concat([b.rename("base"), thinned.rename("thin")], axis=1).fillna(0)
        both["diff"] = both["thin"] - both["base"]
        for sp, s in both.groupby(level="species"):
            out.append(
                dict(
                    variant=variant,
                    fps=round(3 / k, 2),
                    phase=phase,
                    species=sp,
                    n_drops=len(s),
                    exact_pct=round(100 * (s["diff"] == 0).mean(), 1),
                    lower_pct=round(100 * (s["diff"] < 0).mean(), 1),
                    higher_pct=round(100 * (s["diff"] > 0).mean(), 1),
                    mean_diff=round(s["diff"].mean(), 3),
                    mae=round(s["diff"].abs().mean(), 3),
                    mean_base=round(s["base"].mean(), 2),
                )
            )
    return pd.DataFrame(out)


def expert_maxn(db_path: Path, drop_ids: set) -> pd.DataFrame:
    """Expert drop-level MaxN per species from the annotations DB (absence rows dropped)."""
    con = sqlite3.connect(db_path)
    ph = ",".join("?" * len(drop_ids))
    df = pd.read_sql_query(
        f"""select drop_id, scientific_name as species, max(max_interval) as expert
            from annotations
            where annotated_by='expert' and drop_id in ({ph})
              and scientific_name is not ? and max_interval > 0
            group by drop_id, scientific_name""",
        con,
        params=[*sorted(drop_ids), NULL_DEPLOYMENT],
    )
    con.close()
    return df


def compare_to_expert(long: pd.DataFrame, expert: pd.DataFrame) -> pd.DataFrame:
    """MAE / bias of ML MaxN against expert MaxN for each (variant, fps, phase).

    Species are the union of what either side reported; a side that has no
    row counts 0. Limited to species the ML model can name (its own classes),
    so an expert-only species the model cannot output is not charged to it.
    """
    if expert.empty:
        return pd.DataFrame()
    out = []
    ml_species = set(long["species"])
    expert = expert[expert["species"].isin(ml_species)]
    for (model, variant, k, phase), g in long.groupby(
        ["model", "variant", "k", "phase"]
    ):
        drops = set(g["drop_id"]) & set(expert["drop_id"])
        if not drops:
            continue
        ml = g[g["drop_id"].isin(drops)].set_index(["drop_id", "species"])["maxn"]
        ex = expert[expert["drop_id"].isin(drops)].set_index(["drop_id", "species"])[
            "expert"
        ]
        both = pd.concat([ex, ml.rename("ml")], axis=1).fillna(0)
        both["diff"] = both["ml"] - both["expert"]
        out.append(
            dict(
                model=model,
                variant=variant,
                fps=round(3 / k, 2),
                phase=phase,
                n_drops=len(drops),
                n_pairs=len(both),
                exact_pct=round(100 * (both["diff"] == 0).mean(), 1),
                mae=round(both["diff"].abs().mean(), 3),
                bias=round(both["diff"].mean(), 3),
                mean_expert=round(both["expert"].mean(), 2),
                mean_ml=round(both["ml"].mean(), 2),
            )
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--download", action="store_true", help="pull raw CSVs from S3 first"
    )
    ap.add_argument("--annotations-db", type=Path, default=config.annotations_db_path)
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    if args.download:
        n = download_raw_csvs(args.raw_dir)
        print(f"downloaded {n} new raw CSVs into {args.raw_dir}")

    work = args.out / "maxn_csvs"
    work.mkdir(parents=True, exist_ok=True)
    long = run_all(args.raw_dir, work)
    long.to_csv(args.out / "maxn_long.csv", index=False)
    reg = species_registry()

    vs3 = compare_to_3fps(long)
    vs3["common"] = vs3["species"].map(reg.display_for)
    vs3.to_csv(args.out / "vs_3fps.csv", index=False)

    runs = long[["model", "drop_id"]].drop_duplicates()
    print(f"\n{len(runs)} (model, drop) runs, {runs.drop_id.nunique()} drops")

    print("\n=== Agreement with 3 fps, all species pooled (phase-averaged) ===")
    pooled = (
        vs3.groupby(["variant", "fps"])
        .apply(
            include_groups=False,
            func=lambda g: pd.Series(
                dict(
                    pairs=g.n_drops.sum(),
                    exact_pct=np.average(g.exact_pct, weights=g.n_drops),
                    lower_pct=np.average(g.lower_pct, weights=g.n_drops),
                    higher_pct=np.average(g.higher_pct, weights=g.n_drops),
                    mae=np.average(g.mae, weights=g.n_drops),
                )
            ),
        )
        .round(2)
    )
    print(pooled.to_string())

    print("\n=== Per species (n >= 15 drops), phase-averaged ===")
    per_sp = (
        vs3[vs3.n_drops >= 15]
        .groupby(["common", "variant", "fps"])
        .agg(
            n=("n_drops", "max"),
            exact_pct=("exact_pct", "mean"),
            lower_pct=("lower_pct", "mean"),
            mean_diff=("mean_diff", "mean"),
            mae=("mae", "mean"),
            mean_3fps=("mean_base", "mean"),
        )
        .round(2)
    )
    print(per_sp.to_string())

    expert = expert_maxn(args.annotations_db, set(long["drop_id"]))
    vse = compare_to_expert(long, expert)
    if not vse.empty:
        vse.to_csv(args.out / "vs_expert.csv", index=False)
        print("\n=== Against expert MaxN (drops with an expert account in the DB) ===")
        print(
            vse.sort_values(["model", "variant", "fps", "phase"]).to_string(index=False)
        )
    else:
        print("\nNo expert MaxN in the annotations DB for any of these drops.")


if __name__ == "__main__":
    main()
