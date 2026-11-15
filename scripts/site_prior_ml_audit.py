"""Hold the current ML species records up against the site prior.

The current model emits one hard label per box, so the prior cannot re-weight
it the way it will re-weight the two-stage classifier's probabilities. What it
can do today is say which ML species records sit where experts (and their
neighbours) have hardly ever seen that species. This script:

1. Builds the site prior (same inputs as scripts/build_site_prior.py).
2. Loads ML rows from the annotations DB, newest model per drop, collapsed to
   one record per (drop, species): MaxN, interval count, mean confidence.
3. Joins each record to its reserve's presence_prob and bands it.
4. For drops that also have expert rows, checks whether low-prior ML species
   are the ones experts did not see.

Writes `ml_records_vs_prior.csv` next to the prior outputs. Read-only on both
databases. Sensitive output (species per reserve): stays under process_files/.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from spyfish.config.base import NULL_DEPLOYMENT  # noqa: E402
from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.database.annotation_manager import newest_model_only  # noqa: E402
from spyfish.database.connection import connect_readonly  # noqa: E402
from spyfish.ml.classifier.site_prior import (  # noqa: E402
    RESERVE,
    SPECIES,
    build_site_prior,
    load_expert_rows,
    load_reserve_centroids,
    reserve_code,
)

BANDS = [0, 0.02, 0.05, 0.15, 0.4, 1.01]
BAND_LABELS = ["<0.02", "0.02-0.05", "0.05-0.15", "0.15-0.40", ">=0.40"]


def load_ml_records(annotations_db_path, bucket_names) -> pd.DataFrame:
    with connect_readonly(annotations_db_path) as conn:
        df = pd.read_sql_query(
            "SELECT drop_id, scientific_name, max_interval, confidence_agreement, "
            "external_id, annotated_by FROM annotations WHERE annotated_by = 'ml'",
            conn,
        )
    # Only species models: the binary fish models have nothing to say about species.
    df = df[df["external_id"].str.startswith("species", na=False)]
    df = newest_model_only(df)
    df = df[
        df[SPECIES].notna() & ~df[SPECIES].isin(set(bucket_names) | {NULL_DEPLOYMENT})
    ]
    rec = (
        df.groupby(["drop_id", SPECIES])
        .agg(
            model=("external_id", "first"),
            maxn=("max_interval", "max"),
            n_intervals=("max_interval", "size"),
            mean_conf=("confidence_agreement", "mean"),
        )
        .reset_index()
    )
    rec[RESERVE] = rec["drop_id"].map(reserve_code)
    return rec


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    expert = load_expert_rows(config.annotations_db_path)
    centroids = load_reserve_centroids(config.db_path)
    registry = species_registry()
    common = registry.common_name_map()
    prior = build_site_prior(expert, centroids, common_names=common).long

    ml = load_ml_records(config.annotations_db_path, config.site_prior_bucket_names)
    ml[SPECIES] = ml[SPECIES].map(lambda n: registry.resolve_name(n)[0])
    joined = ml.merge(
        prior[
            [RESERVE, SPECIES, "n_drops", "raw_rate", "neighbour_rate", "presence_prob"]
        ],
        on=[RESERVE, SPECIES],
        how="left",
    )
    missing = joined["presence_prob"].isna()
    if missing.any():
        # species never recorded by any expert anywhere, or a reserve with no centroid
        joined.loc[missing, "presence_prob"] = config.site_prior_floor
    joined["band"] = pd.cut(
        joined["presence_prob"], BANDS, labels=BAND_LABELS, right=False
    )
    joined["common_name"] = joined[SPECIES].map(lambda s: common.get(s, ""))

    # expert view of the same drops: which species did experts record there?
    expert_drops = set(expert["drop_id"])
    expert_pairs = set(zip(expert["drop_id"], expert[SPECIES]))
    joined["drop_has_expert"] = joined["drop_id"].isin(expert_drops)
    joined["expert_saw_it"] = [
        (d, s) in expert_pairs if d in expert_drops else None
        for d, s in zip(joined["drop_id"], joined[SPECIES])
    ]

    out = config.site_prior_dir / "ml_records_vs_prior.csv"
    joined.sort_values([RESERVE, "presence_prob"]).to_csv(out, index=False)
    print(f"wrote {out}\n")

    print("ML (drop, species) records by reserve and prior band")
    by_band = joined.pivot_table(
        index=RESERVE,
        columns="band",
        values="drop_id",
        aggfunc="size",
        fill_value=0,
        observed=False,
    )
    by_band["total"] = by_band.sum(axis=1)
    by_band["drops"] = joined.groupby(RESERVE)["drop_id"].nunique()
    by_band["expert_drops_at_reserve"] = (
        joined.groupby(RESERVE)["n_drops"].first().fillna(0).astype(int)
    )
    print(by_band.to_string(), "\n")

    print("Low-prior ML species (presence_prob < 0.05): what the prior would flag")
    low = joined[joined["presence_prob"] < 0.05]
    tbl = (
        low.groupby([RESERVE, SPECIES, "common_name"])
        .agg(
            records=("drop_id", "size"),
            drops=("drop_id", "nunique"),
            maxn_sum=("maxn", "sum"),
            mean_conf=("mean_conf", "mean"),
            presence_prob=("presence_prob", "first"),
        )
        .reset_index()
        .sort_values([RESERVE, "records"], ascending=[True, False])
    )
    tbl["mean_conf"] = tbl["mean_conf"].round(2)
    print(tbl.to_string(index=False), "\n")

    print(
        "Share of ML MaxN that would move to 'fish'/review if species under a prior cut were not trusted"
    )
    for cut in (0.02, 0.05, 0.15):
        flagged = joined["presence_prob"] < cut
        print(
            f"  cut {cut:<5} records {flagged.sum():>4}/{len(joined)} "
            f"({flagged.mean():.1%}), MaxN sum {joined.loc[flagged, 'maxn'].sum():>5}/{joined['maxn'].sum()} "
            f"({joined.loc[flagged, 'maxn'].sum() / joined['maxn'].sum():.1%})"
        )

    with_expert = joined[joined["drop_has_expert"]]
    if len(with_expert):
        print(
            f"\nDrops with both ML and expert rows: {with_expert['drop_id'].nunique()}"
        )
        print("Did the expert record the ML species on that drop, by prior band?")
        chk = with_expert.pivot_table(
            index="band",
            columns="expert_saw_it",
            values="drop_id",
            aggfunc="size",
            fill_value=0,
            observed=False,
        )
        chk.columns = [f"expert_saw={c}" for c in chk.columns]
        chk["share_expert_saw"] = (
            chk.get("expert_saw=True", 0) / chk.sum(axis=1)
        ).round(2)
        print(chk.to_string())


if __name__ == "__main__":
    main()
