"""What happened to the "To review" boxes: kept as a species, deleted, or still pending.

Reads the upload manifests (``review_upload_*.csv``, written by
``upload_model_only_for_review.py``, one row per box with its BIIGLE
``annotation_id``) and every BIIGLE raw CSV under ``deployment_data``. A
relabel in BIIGLE (annotation view or Largo) keeps the annotation id, so:

* id present, label still "To review"  -> pending
* id present, any other label          -> kept: a fish nobody had boxed
* id gone, frame's CSV re-pulled AFTER the upload -> deleted: not a fish
  (junk, or a second box on a fish already boxed)
* id gone, CSV older than the upload    -> not re-pulled yet; nothing is
  concluded about it

Re-pull first, so the CSVs reflect the review:

    python -m spyfish.biigle.biigle_to_yolo download-volume --volume-id 30505
    python -m spyfish.biigle.biigle_to_yolo download-project --project-id <id> --force

Writes ``review_outcome.csv`` (one row per box) next to the manifests; the
deleted rows are reviewed `not a fish` negatives for the classifier.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.crops import OUT_DIR  # noqa: E402


def current_annotations() -> pd.DataFrame:
    """annotation_id -> label, filename, and the mtime of the CSV it came from."""
    parts = []
    for csv in config.deployment_data_dir.glob("*/*/annotations/*_biigle_*raw.csv"):
        df = pd.read_csv(
            csv, usecols=lambda c: c in {"annotation_id", "label_name", "filename"}
        )
        if "annotation_id" not in df.columns:
            continue
        parts.append(df.assign(csv_mtime=csv.stat().st_mtime))
    if not parts:
        raise SystemExit(f"no BIIGLE raw CSVs under {config.deployment_data_dir}")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--manifests",
        nargs="*",
        default=None,
        help="default: every review_upload_*.csv under wip_two_stage",
    )
    ap.add_argument(
        "--label",
        nargs="+",
        default=["To review", "ML: check"],
        help="labels the boxes were uploaded with (first rounds used To review)",
    )
    args = ap.parse_args()

    paths = (
        [OUT_DIR / m for m in args.manifests]
        if args.manifests
        else sorted(OUT_DIR.glob(artifacts.REVIEW_UPLOAD_GLOB))
    )
    rows = []
    for p in paths:
        m = pd.read_csv(p)
        if "annotation_id" not in m.columns or m["annotation_id"].isna().all():
            print(
                f"{p.name}: no annotation ids (a dry run, or the upload failed); skipped"
            )
            continue
        if "drop" not in m.columns:  # first 30505 manifest predates the drop column
            m["drop"] = p.stem.replace(artifacts.REVIEW_UPLOAD_PREFIX, "")
        rows.append(
            m.dropna(subset=["annotation_id"]).assign(
                manifest=p.name, uploaded=p.stat().st_mtime
            )
        )
    if not rows:
        raise SystemExit("no uploaded manifests found")
    sent = pd.concat(rows, ignore_index=True)
    sent["annotation_id"] = sent["annotation_id"].astype("int64")

    now = current_annotations()
    by_id = now.drop_duplicates("annotation_id").set_index("annotation_id")
    newest = now.groupby("filename")["csv_mtime"].max()
    label = sent["annotation_id"].map(by_id["label_name"])
    refreshed = sent["filename"].map(newest) > sent["uploaded"]
    sent["label_now"] = label
    sent["outcome"] = "not re-pulled"
    sent.loc[label.notna() & label.isin(args.label), "outcome"] = "pending"
    sent.loc[label.notna() & ~label.isin(args.label), "outcome"] = "kept"
    sent.loc[label.isna() & refreshed, "outcome"] = "deleted"

    table = sent.pivot_table(
        index="drop",
        columns="outcome",
        values="annotation_id",
        aggfunc="size",
        fill_value=0,
    )
    decided = table.reindex(columns=["kept", "deleted"], fill_value=0).sum(axis=1)
    table["share_fish"] = (table.get("kept", 0) / decided.where(decided > 0)).round(3)
    pd.set_option("display.width", 200)
    print(table.sort_values("share_fish", ascending=False).to_string())
    kept = sent[sent["outcome"] == "kept"]
    tot = sent["outcome"].value_counts().to_dict()
    n_dec = tot.get("kept", 0) + tot.get("deleted", 0)
    print(f"\n{len(sent)} boxes: {tot}")
    if n_dec:
        print(f"share that were fish, of the decided: {tot.get('kept', 0) / n_dec:.1%}")
    if len(kept):
        print("kept as:", kept["label_now"].value_counts().head(10).to_dict())
    out = OUT_DIR / artifacts.REVIEW_OUTCOME
    sent.drop(columns=["uploaded"]).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
