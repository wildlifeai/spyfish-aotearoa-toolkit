"""Where YOLO and the BioCLIP second stage disagree on a drop's boxes.

Reads the drop's raw CSV written with ``ml_inference.classifier`` on (the four
columns ``yolo_class / yolo_confidence / clf_class / clf_confidence``) and prints:

* agreement between the two names, overall and per YOLO class,
* the crosstab (rows YOLO, columns what was written to ``class``),
* the top disagreements,
* the low-confidence cases: boxes under ``p_min`` and what was written for them
  (BioCLIP's name is kept in ``clf_class`` even when the row fell back).

    python scripts/compare_two_stage_names.py HOR_20251204_BUV_HOR_033_01
    python scripts/compare_two_stage_names.py --survey HOR_20251204_BUV
    python scripts/compare_two_stage_names.py HOR_20251204_BUV_HOR_033_01 --min-conf 0.25 --export disagreements.csv

``--export`` writes the disagreeing and low-confidence rows (with frame index and
box) so the crops can be pulled and looked at.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.classify_raw_csv import drop_ids_from_args  # noqa: E402
from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.run_inference import CLASSIFIER_COLUMNS  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", None)


def load(drop_id: str, model_name: str) -> pd.DataFrame | None:
    path = config.get_raw_csv_path(config.validate_drop_id(drop_id), model_name)
    if not path.exists():
        logging.warning(f"{drop_id}: no raw CSV for {model_name} at {path}")
        return None
    df = pd.read_csv(path)
    missing = [c for c in CLASSIFIER_COLUMNS if c not in df.columns]
    if missing:
        logging.warning(
            f"{drop_id}: raw CSV has no classifier columns ({missing}); it was "
            "written with the classifier off."
        )
        return None
    return df.assign(DropID=drop_id)


def _common_name() -> callable:
    """scientific → common name for printing; unknown names and buckets as written."""
    registry = species_registry()
    names = registry.common_name_map()
    # Old checkpoints still carry retired names (Pagrus auratus): follow the
    # synonym first; resolve_name is the silent form of canonical_name.
    return lambda name: names.get(registry.resolve_name(name)[0], name)


NAME_COLUMNS = ["class", "yolo_class", "clf_class"]


def report(df: pd.DataFrame, min_conf: float, p_min: float) -> pd.DataFrame:
    """Print the comparison in common names; return the flagged rows as stored
    (scientific names), for ``--export``."""
    stored = df[df["confidence"] >= min_conf]
    common = _common_name()
    df = stored.assign(
        **{c: stored[c].map(common, na_action="ignore") for c in NAME_COLUMNS}
    )
    named = df[df["clf_class"].notna()]  # boxes the classifier looked at
    agree = named["class"] == named["yolo_class"]
    print(
        f"\n{len(df)} boxes at detection confidence >= {min_conf}; "
        f"{len(named)} classified ({len(df) - len(named)} too small / off-frame kept YOLO's name)"
    )
    print(f"written `class` agrees with YOLO on {agree.mean():.1%} of classified boxes")

    print("\nagreement by YOLO class:")
    by = named.assign(agree=agree).groupby("yolo_class")["agree"].agg(["size", "mean"])
    print(
        by.rename(columns={"size": "boxes", "mean": "agree"})
        .sort_values("boxes", ascending=False)
        .round(3)
        .to_string()
    )

    print("\ncrosstab, rows = YOLO, columns = written `class`:")
    print(pd.crosstab(named["yolo_class"], named["class"]).to_string())

    dis = named[~agree]
    if len(dis):
        print(f"\ntop disagreements ({len(dis)} boxes):")
        top = (
            dis.groupby(["yolo_class", "class"])
            .agg(
                boxes=("confidence", "size"),
                yolo_conf=("yolo_confidence", "mean"),
                clf_conf=("clf_confidence", "mean"),
            )
            .sort_values("boxes", ascending=False)
            .head(15)
        )
        print(top.round(3).to_string())

    low = named[named["clf_confidence"] < p_min]
    if len(low):
        fallback = common(config.classifier_fallback_label)
        generic = int((low["class"] == fallback).sum())
        kept_yolo = int(
            ((low["class"] == low["yolo_class"]) & (low["class"] != fallback)).sum()
        )
        print(
            f"\n{len(low)} boxes under p_min {p_min}: {kept_yolo} kept YOLO's species, "
            f"{generic} written as {fallback!r} (YOLO said {fallback!r} or was not confident), "
            f"{len(low) - kept_yolo - generic} other"
        )
        print("  BioCLIP's own (unconfident) answer on them:")
        print(low["clf_class"].value_counts().head(10).to_string())
    return stored.loc[dis.index.union(low.index)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drop_ids", nargs="*")
    parser.add_argument("--survey", metavar="SURVEY_ID")
    parser.add_argument(
        "--model", help="model name (raw CSV suffix); default: the pipeline model"
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=None,
        help="detection confidence floor; default ml_inference.maxn_confidence_threshold",
    )
    parser.add_argument(
        "--export", metavar="CSV", help="write disagreeing + low-confidence rows here"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model_name = args.model or config.pipeline_model_name
    min_conf = (
        args.min_conf
        if args.min_conf is not None
        else float(config.maxn_confidence_threshold)
    )
    p_min = config.classifier_p_min
    drop_ids = drop_ids_from_args(args)
    if not drop_ids:
        sys.exit("no drops given / found")

    frames = [f for f in (load(d, model_name) for d in drop_ids) if f is not None]
    if not frames:
        sys.exit("no raw CSV with classifier columns found")
    df = pd.concat(frames, ignore_index=True)
    print(f"model {model_name}; drops: {sorted(df['DropID'].unique())}")
    flagged = report(df, min_conf, p_min)
    if args.export:
        cols = [
            "DropID",
            "frame",
            "time_seconds",
            "class",
            "confidence",
            "x",
            "y",
            "w",
            "h",
        ] + CLASSIFIER_COLUMNS
        flagged[cols].to_csv(args.export, index=False)
        print(f"\n{len(flagged)} rows → {args.export}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
