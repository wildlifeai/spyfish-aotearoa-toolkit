"""Build the species-presence prior per marine reserve from expert annotations.

two_stage_classifier_plan.md §6, Phase 5. Reads expert rows (legacy + BIIGLE)
from the annotations DB and reserve centroids (median deployment position) from
the pipeline DB, both read-only, and writes under paths.site_prior:

    species_presence_by_reserve.csv  one row per (reserve, species): raw rate,
                                     neighbour rate, shrunk presence_prob, prior_weight
    reserve_summary.csv              drops, species seen, borrowed share, donors,
                                     neighbour skill (Spearman, leave-self-out)
    rarefaction.csv                  expected species found in n drops per reserve
    neighbour_weights.csv            who each reserve borrows from, and how far away

Settings live in config.yaml under `site_prior`. Everything written is
SENSITIVE (species per reserve): keep it under process_files/, never publish.

    python scripts/build_site_prior.py
    python scripts/build_site_prior.py --out-dir /somewhere/else
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier.site_prior import (  # noqa: E402
    build_site_prior,
    load_expert_rows,
    load_reserve_centroids,
    write_site_prior,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--annotations-db", default=None, help="override config.annotations_db_path"
    )
    parser.add_argument("--pipeline-db", default=None, help="override config.db_path")
    parser.add_argument(
        "--out-dir", default=None, help="override config.site_prior_dir"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    annotations_db = args.annotations_db or config.annotations_db_path
    pipeline_db = args.pipeline_db or config.db_path
    out_dir = args.out_dir or config.site_prior_dir

    rows = load_expert_rows(annotations_db)
    centroids = load_reserve_centroids(pipeline_db)
    logging.info(
        f"Site prior: {len(rows)} expert rows over {rows['drop_id'].nunique()} drops; "
        f"centroids for {len(centroids)} reserves"
    )
    common = species_registry().common_name_map()
    result = build_site_prior(rows, centroids, common_names=common)
    write_site_prior(result, out_dir)

    summary = result.summary
    print("\nReserve summary (expert-reviewed drops):")
    print(summary.to_string(index=False))
    print("\nRarefaction (share of a reserve's observed species found in n drops):")
    wide = result.rarefaction.pivot(
        index=RESERVE_COL, columns="n_drops_sample", values="share_of_observed"
    )
    print(wide.to_string())


RESERVE_COL = "reserve_code"

if __name__ == "__main__":
    main()
