"""Map the frames of curated BIIGLE volumes to the deployment each came from.

Curated volumes under ``extra_no_survey_id/`` (volume 30505: 1,887 historical
Zooniverse frame subjects) are one "drop" to data prep, so the per-drop frame
cap either skips them or would cut 29 deployments to 50 frames together. This
writes ``training.extra_frame_deployments_file`` so the cap sees each source
deployment as its own drop (``select_frames_per_drop``).

How a frame is traced: its filename is the Zooniverse image UUID, which is the
``subject_location`` of a frame subject in the subjects export; the subject's
metadata gives site, event date, original video file, frame number and time.
The deployment key is ``{SiteID}_{YYYYMMDD}``, which needs no DropID.

``drop_id`` comes from the SharePoint list, else the legacy export files,
through the old-to-new site crosswalk, within ``--window`` days (see
``drop_matches``). The cap does not depend on it; it is there to link the
frames back to videos and metadata.

    python scripts/build_extra_frame_deployments.py volume_30505
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from spyfish.config.wrapper import config  # noqa: E402

UUID = re.compile(r"subject_location/([0-9a-f-]{36})")


def subjects_by_uuid() -> pd.DataFrame:
    """Every frame subject in every subjects export, keyed by image UUID."""
    files = sorted(config.legacy_zooniverse_dir.glob("**/*subjects*.csv"))
    if not files:
        raise SystemExit(
            f"no Zooniverse subjects export under {config.legacy_zooniverse_dir}"
        )
    parts = [
        pd.read_csv(f, usecols=["subject_id", "metadata", "locations"]) for f in files
    ]
    s = pd.concat(parts, ignore_index=True)
    s["uuid"] = s["locations"].str.extract(UUID)[0]
    s = s.dropna(subset=["uuid"]).drop_duplicates("uuid")
    md = s["metadata"].apply(json.loads)
    for col, key in (
        ("site_id", "#SiteID"),
        ("event_date", "#EventDate"),
        ("video_filename", "#VideoFilename"),
        ("frame_number", "#frame_number"),
        ("time_seconds", "#TimeOfMaxSeconds"),
        ("reserve", "!LinkToMarineReserve"),
    ):
        s[col] = md.apply(lambda d, k=key: d.get(k))
    s["event"] = pd.to_datetime(
        s["event_date"], format="mixed", dayfirst=True, errors="coerce"
    )
    logging.info(f"{len(s)} image subjects from {len(files)} export file(s)")
    return s.drop(columns=["metadata", "locations"]).set_index("uuid")


def drop_matches(keys: pd.DataFrame, window: int) -> pd.DataFrame:
    """DropID per source deployment, from the SharePoint list, else the legacy files.

    Zooniverse metadata uses the site codes and dates of its day: Taranaki
    sites as ``TNK_…`` (now ``SLI_…`` for Tapuae, ``PAR_…`` for Parininihi,
    per survey, from ``site_matching_provisional.csv``), and event dates that
    were later corrected (Parininihi 2012: April in Zooniverse, February in
    SharePoint). So a match is the same site, through that crosswalk, within
    ``window`` days, and only when exactly one deployment qualifies:

    1. ``reference/deployments.csv``, the cached SharePoint list (real event dates);
    2. ``deployment_match.csv`` ``from_files``, DropIDs that exist only in the
       legacy exports (not registered in SharePoint), dated by their survey.
    """
    help_dir = config.legacy_help_dir
    ref = pd.read_csv(help_dir / "reference" / "deployments.csv", encoding="utf-8-sig")
    ref["event"] = pd.to_datetime(
        ref["EventDate"], format="mixed", dayfirst=True, errors="coerce"
    )
    sites = pd.read_csv(help_dir / "site_matching_provisional.csv").dropna(
        subset=["proposed_new_site"]
    )
    legacy = pd.read_csv(help_dir / "deployment_match.csv")["from_files"].dropna()
    legacy = pd.DataFrame({"DropID": legacy})
    legacy["SiteID"] = legacy["DropID"].str.extract(r"_BUV_(.+)_\d+$")[0]
    legacy["event"] = pd.to_datetime(
        legacy["DropID"].str.extract(r"_(\d{8})_BUV")[0],
        format="%Y%m%d",
        errors="coerce",
    )
    legacy = legacy[~legacy["DropID"].isin(ref["DropID"])]

    out = []
    for k in keys.itertuples():
        names = {k.site_id} | set(
            sites.loc[sites["old_site"] == k.site_id, "proposed_new_site"]
        )
        found = ("", "", None)
        for source, table in (("sharepoint", ref), ("legacy_files", legacy)):
            c = table[table["SiteID"].isin(names)]
            days = (c["event"] - k.event).dt.days
            c = c[days.abs() <= window]
            if len(c) == 1:
                found = (c["DropID"].iloc[0], source, int(days[c.index[0]]))
                break
            if len(c) > 1:
                found = ("", f"ambiguous: {', '.join(c['DropID'])}", None)
                break
        out.append(found)
    keys[["drop_id", "drop_id_source", "days_apart"]] = pd.DataFrame(
        out, index=keys.index
    )
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("drops", nargs="+", help="curated volume drops, e.g. volume_30505")
    ap.add_argument(
        "--window",
        type=int,
        default=90,
        help="days a Zooniverse date may differ from the deployment's",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    subjects = subjects_by_uuid()
    rows = []
    for drop in args.drops:
        dirs = [
            d
            for d in config.deployment_data_dir.glob(f"*/{drop}")
            if (d / "labels").is_dir()
        ]
        if len(dirs) != 1:
            raise SystemExit(f"{drop}: expected one folder with labels/, found {dirs}")
        stems = sorted(p.stem for p in (dirs[0] / "labels").glob("*.txt"))
        hit = subjects.reindex(stems)
        missing = hit["site_id"].isna()
        if missing.any():
            raise SystemExit(
                f"{drop}: {int(missing.sum())} of {len(stems)} frames have no Zooniverse subject, e.g. {list(hit.index[missing][:3])}"
            )
        hit = hit.assign(drop=drop, frame_stem=hit.index)
        hit["deployment"] = hit["site_id"] + "_" + hit["event"].dt.strftime("%Y%m%d")
        rows.append(hit)
    frames = pd.concat(rows, ignore_index=True)

    keys = frames.groupby(
        ["drop", "deployment", "site_id", "event", "reserve"], as_index=False
    ).agg(frames=("frame_stem", "size"), video_filename=("video_filename", "first"))
    keys = drop_matches(keys, args.window)
    frames = frames.merge(
        keys[["drop", "deployment", "drop_id"]], on=["drop", "deployment"]
    )

    out = config.training_extra_frame_deployments_file
    cols = [
        "drop",
        "frame_stem",
        "deployment",
        "drop_id",
        "subject_id",
        "site_id",
        "event_date",
        "video_filename",
        "frame_number",
        "time_seconds",
    ]
    frames[cols].sort_values(["drop", "deployment", "frame_number"]).to_csv(
        out, index=False
    )
    summary = out.with_name(out.stem + "_summary.csv")
    keys.sort_values("frames", ascending=False).to_csv(summary, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)
    print(
        keys.sort_values("frames", ascending=False)[
            [
                "drop",
                "deployment",
                "frames",
                "video_filename",
                "drop_id",
                "drop_id_source",
                "days_apart",
            ]
        ].to_string(index=False)
    )
    print(
        f"\n{len(frames)} frames, {len(keys)} deployments; DropID found for "
        f"{(keys['drop_id'] != '').sum()} ({keys.loc[keys['drop_id'] != '', 'frames'].sum()} frames)\n"
        f"wrote {out}\nwrote {summary}"
    )


if __name__ == "__main__":
    main()
