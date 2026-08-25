import logging
import re
from collections import Counter
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.biigle.biigle_parser import BiigleParser
from spyfish.biigle.biigle_to_yolo import biigle_to_yolo
from spyfish.config.base import ExpertStatus
from spyfish.config.species import species_registry
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import (
    AnnotationDatabaseManager,
    null_deployment_row,
)
from spyfish.database.manager import DatabaseManager
from spyfish.utils import seconds_to_time, time_to_seconds


def _keep_countable_rows(
    df: pd.DataFrame,
    species_label_ids: set,
    workflow_label_ids: set,
    substrate_label_ids: set,
) -> pd.DataFrame:
    """Keep the report rows that MaxN should count, drop the rest.

    An allowlist, not a substrate denylist. Excluding one known-bad tree meant
    every other kind of label was countable by default, so a new tree, or a
    label belonging to none, silently became a species. Here a row has to earn
    its place: it is counted if it comes from the species tree, or is one of
    the workflow labels that marks an animal, or belongs to no tree we know
    (legacy and genus-level names, which `_map_biigle_to_spyfish_schema` then
    judges on shape).

    What this removes is substrate, which is measured as percent-cover by
    `process_substrate` rather than counted, and the workflow tree's progress
    markers. Without it each substrate polygon would inflate a bogus "species".

    Falls through unfiltered when the report has no `label_id` or no tree could
    be fetched: the per-row mapper still applies its own name-based checks, so
    this degrades rather than dropping everything.
    """
    known = species_label_ids | workflow_label_ids | substrate_label_ids
    if "label_id" not in df.columns or not known:
        return df

    ids = pd.to_numeric(df["label_id"], errors="coerce")
    # "In no tree we know" must be tested against EVERY known tree, not just
    # the two whose rows we keep. Leaving substrate out of `known` would let
    # every substrate polygon through this clause as an unrecognised label.
    countable = (
        ids.isin(species_label_ids)
        | ids.isin(set(config.biigle_workflow_tree_keep_labels))
        | ~ids.isin(known)
    )
    return df[countable].copy()


def _extract_timestamp_from_filename(row: pd.Series, fname_col: str) -> Optional[str]:
    """Parse timestamps from Biigle's image or video snippet filenames."""
    if not fname_col or fname_col not in row:
        return None

    fname = str(row[fname_col])
    try:
        secs = None
        if "__frame_" in fname:
            match = re.search(r"__frame_([\d\.]+)s\.jpg", fname)
            if match:
                secs = float(match.group(1))
        elif "_clip_" in fname:
            match = re.search(r"_clip_([\d\.]+)s\.", fname)
            if match:
                secs = float(match.group(1))

            if "frames" in row:
                frame_str = str(row["frames"]).strip("[]")
                if frame_str and frame_str != "nan":
                    secs += float(frame_str)  # type: ignore

        if secs is not None:
            return seconds_to_time(secs)
    except (ValueError, IndexError) as e:
        logging.warning(f"Could not parse timestamp from filename '{fname}': {e}")

    return None


def _map_biigle_to_spyfish_schema(
    row: pd.Series,
    label_col: str,
    drop_id: str,
    timestamp: Optional[str],
    frame_key: str,
    workflow_label_ids: Optional[set] = None,
    species_label_ids: Optional[set] = None,
) -> Optional[Tuple[Tuple[str, str], Dict[str, Any]]]:
    """Maps a Biigle annotation row to the Spyfish schema. Returns (aggregation_key, mapped_dict).

    Which LABEL TREE a label belongs to decides what it means, so that is
    checked first and the text second. A tree is assigned by the system; the
    text is typed by a person, so the tree is the stronger signal.

    * **Species tree** — always kept. An expert picked it from the species
      list, which settles the question; if the registry has not heard of it
      that is the registry's gap, so it is stored and reported rather than
      dropped.
    * **Workflow tree** — annotator progress markers ("Done Volume", "In
      progress", "Nothing here"), dropped. Except the entries in
      `biigle_workflow_tree_keep_labels`, which mark a real animal and become
      the class named there: an unidentified fish, or the bait.
    * **No known tree** — a legacy or genus-level name ("Conger sp"). Kept if
      it has a taxonomic shape, reported either way.

    Names the registry knows are resolved to their canonical scientific name
    rather than kept as written: it treats "Fish: final" as an alias of `fish`,
    and storing the alias would put a second spelling of one class in the
    database for every reader to know about.

    `frame_key` is used as the per-frame identifier when `timestamp` is None,
    e.g. for image volumes (like UUID-named uploads) where no clip/frame-seconds
    pattern is in the filename. Preserves per-frame uniqueness for MaxN aggregation.
    """
    label = str(row.get(label_col, "unknown_species")).strip()
    label_id = pd.to_numeric(row.get("label_id"), errors="coerce")
    label_id = None if pd.isna(label_id) else int(label_id)

    # ── Workflow tree: progress markers, not sightings ───────────────────────
    if workflow_label_ids and label_id in workflow_label_ids:
        keep = config.biigle_workflow_tree_keep_labels
        if label_id in keep:
            species = keep[label_id]
        else:
            logging.debug(
                f"{drop_id}: skipping workflow-tree label {label!r} (id {label_id})."
            )
            return None
    else:
        in_species_tree = bool(species_label_ids) and label_id in species_label_ids
        registry = species_registry()
        known = registry.get(label) or registry.get(label.lower())
        if known is not None:
            species = known.scientific_name
        elif in_species_tree or " - " in label:
            # From the species tree, so it is a species by construction. The
            # tree's convention is "Common - Scientific", so the half after the
            # separator is the scientific name; a name without the separator is
            # taken whole. This is how a species new to the tree arrives before
            # anyone adds it to the registry.
            species = label.split(" - ", 1)[1] if " - " in label else label
            logging.warning(
                f"{drop_id}: BIIGLE label {label!r} (id {label_id}) is in the "
                f"species tree but not in the species registry; storing "
                f"{species!r}. Add it to the registry."
            )
        elif re.match(config.genus_level_label_pattern, label):
            # A genus-level identification ("Conger sp"). An expert chose it
            # deliberately, so dropping it would lose a real observation that
            # no re-sync can recover. Matched on shape rather than accepted
            # blanket, so "Interesting Sighting" still cannot get in. The
            # trailing dot goes, so "Arripis sp." and "Arripis sp" land as one
            # name in the database.
            species = label.rstrip(".")
            logging.warning(
                f"{drop_id}: BIIGLE label {label!r} (id {label_id}) is a "
                "genus-level ID unknown to the species registry; storing it "
                "as-is. Add it to the registry."
            )
        else:
            logging.warning(
                f"{drop_id}: skipping BIIGLE annotation with non-species label "
                f"{label!r} (id {label_id}): in no known label tree, unknown to "
                "the species registry, and neither 'Common - Scientific' nor "
                "genus-level in form."
            )
            return None

    sortable_time = timestamp or frame_key
    key = (sortable_time, species)

    time_str = timestamp or frame_key
    try:
        toms = time_to_seconds(time_str) if time_str else None
    except Exception:
        toms = None
    mapped_item = {
        "drop_id": drop_id,
        "scientific_name": species,
        "time_of_max": time_str,
        "time_of_max_seconds": toms,
        "max_interval": 0,
        "annotated_by": "expert",
        "interval_annotation": "",
        "confidence_agreement": 1.0,
        "external_id": str(row.get("annotation_id", row.get("id", ""))),
    }
    return key, mapped_item


def aggregate_raw_to_maxn_rows(
    fish_annotations_df: pd.DataFrame,
    drop_id: str,
    workflow_label_ids: Optional[set] = None,
    species_label_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Aggregate raw Biigle annotation rows into MaxN rows (one per frame × species).

    Public entry point reused by the Biigle sync pipeline AND by the
    download-volume bundle synthesizer. Returns a list of dicts using the
    Spyfish lowercase schema (drop_id, scientific_name, time_of_max, ...);
    use `maxn_rows_to_df` to rename for CSV output.

    The two label-id sets are the trees, fetched once per run by the caller.
    Both optional so the bundle synthesizer, which has no API session, still
    works; without them labels fall back to being judged by name.
    """
    label_col = "label_name"
    fname_col = "filename"

    aggregated_annotations = {}
    for _, row in fish_annotations_df.iterrows():
        fname = str(row.get(fname_col, ""))
        timestamp = _extract_timestamp_from_filename(row, fname_col)
        mapped = _map_biigle_to_spyfish_schema(
            row,
            label_col,
            drop_id,
            timestamp,
            fname,
            workflow_label_ids,
            species_label_ids,
        )
        if mapped is None:
            continue
        key, mapped_item = mapped

        if key not in aggregated_annotations:
            aggregated_annotations[key] = mapped_item

        aggregated_annotations[key]["max_interval"] += 1

    annotations_to_add = list(aggregated_annotations.values())
    annotations_to_add.sort(key=lambda x: (x["drop_id"], x["time_of_max"] or ""))
    return annotations_to_add


def maxn_rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a MaxN CSV-ready DataFrame from `aggregate_raw_to_maxn_rows` output.

    Renames lowercase schema keys to the canonical CSV column names used
    downstream by `prepare_from_annotations` (config-driven).
    """
    return pd.DataFrame(rows).rename(
        columns={
            "drop_id": config.drop_id_column,
            "scientific_name": config.csv_scientific_name_column,
            "time_of_max": config.csv_maxn_time_column,
            "max_interval": config.csv_max_interval_column,
            "annotated_by": config.csv_annotated_by_column,
            "interval_annotation": config.csv_interval_annotation_column,
            "confidence_agreement": config.csv_confidence_agreement_column,
        }
    )


def _fetch_tree_label_ids(handler, tree_id: int, name: str) -> set:
    """Label ids belonging to one BIIGLE label tree, or an empty set.

    An empty set means "this tree's rule cannot be applied this run", which
    every caller degrades to name-based handling. That is weaker than tree
    membership but it keeps a sync going through an API hiccup, and the warning
    says which rule was lost.
    """
    try:
        ids = {int(lbl["id"]) for lbl in handler.get_label_tree_labels(tree_id)}
        logging.info(f"{name.capitalize()} tree {tree_id} has {len(ids)} label(s)")
        return ids
    except Exception as e:
        logging.warning(
            f"Could not fetch {name} label tree {tree_id}: {e}. "
            f"{name.capitalize()} labels will be handled by name only this run."
        )
        return set()


class LabelTrees(NamedTuple):
    """The three label-id sets, passed around together.

    They are always fetched together and always used together, so one value
    beats three parallel arguments that a caller can pass in the wrong order.
    """

    species: set
    substrate: set
    workflow: set


def _fetch_label_trees(handler) -> LabelTrees:
    """Every tree's label ids, fetched once per run.

    Which tree a label belongs to is what decides how its annotation is
    handled: the tree is assigned by BIIGLE, while the label text is typed by a
    person, so the tree is the stronger signal.
    """
    return LabelTrees(
        species=_fetch_tree_label_ids(handler, config.default_label_tree_id, "species"),
        substrate=_fetch_tree_label_ids(
            handler, config.biigle_substrate_label_tree_id, "substrate"
        ),
        workflow=_fetch_tree_label_ids(
            handler, config.biigle_workflow_label_tree_id, "workflow"
        ),
    )


def _ready_media_type(handler, drop_id, volume_id, done_volumes) -> Optional[str]:
    """The volume's media type if it is ready to sync, else None.

    Two gates. Project membership is the real one: only volumes in `done` are
    finished. The `Done Volume` whole-file label is the legacy gate, kept
    behind `require_done_label` as belt-and-braces during the transition.

    Returns None rather than raising, because "not ready yet" is the normal
    case on most runs, not a failure.
    """
    done_project_id = config.biigle_done_project_id
    if volume_id not in done_volumes:
        logging.debug(
            f"  Volume {volume_id} for {drop_id} not in project "
            f"{done_project_id} (done) yet. Skipping."
        )
        return None

    if config.biigle_require_done_label:
        is_done, media_type = handler.volume_is_done(volume_id)
        if not is_done:
            logging.debug(
                f"  Volume {volume_id} for {drop_id} in done project but "
                "Done-label gate enabled and label missing. Skipping."
            )
            return None
        logging.info(
            f"  Volume {volume_id} for {drop_id} is DONE ({media_type}). "
            "Downloading report..."
        )
        return media_type

    # Reuse cached metadata, saves a get_volume_info() round-trip per volume
    media_type = done_volumes[volume_id].get("media_type", "image")
    logging.info(
        f"  Volume {volume_id} for {drop_id} in done project "
        f"({media_type}). Downloading report..."
    )
    return media_type


def _record_null_deployment(db, ann_db, drop_id: str, volume_id: int, why: str) -> None:
    """Record that this expert review found nothing, and complete the drop.

    Reached two ways — an empty report, and a report whose labels were all
    non-fish — and they mean the same thing, so they write the same row. The
    null-deployment row keeps "reviewed, saw nothing" distinguishable from
    "never reviewed", and lets the data-presence rule in sync_annotation_counts
    advance the status like any other completion.

    Volume-scoped external_id so a re-sync replaces it: clear_synced_annotations
    only removes rows with a non-null external_id, which is what protects
    hand-entered expert rows.
    """
    logging.info(f"  {why} for {drop_id} — recording a null-deployment row.")
    ann_db.clear_synced_annotations(drop_id, "expert")
    ann_db.add_annotations(
        [
            null_deployment_row(
                drop_id, "expert", external_id=f"biigle_volume_{volume_id}"
            )
        ]
    )
    db.advance_status(drop_id, ExpertStatus.COLUMN, ExpertStatus.COMPLETE)


def _write_substrate_csv(parser, report: pd.DataFrame, drop_id: str, trees) -> None:
    """Export substrate percent-cover for this drop, if it has any.

    Substrate is an area-cover statistic, not a count, so it is computed and
    exported separately and never reaches the MaxN aggregation.
    """
    substrate_df = parser.process_substrate(report, trees.substrate)
    if substrate_df.empty:
        return
    substrate_path = config.get_biigle_expert_substrate_csv_path(drop_id)
    substrate_df.to_csv(substrate_path, index=False)
    logging.info(f"  Expert substrate ({len(substrate_df)} rows) → {substrate_path}")


def _ingest_species(db, ann_db, drop_id: str, annotations_to_add: list) -> None:
    """Replace this drop's synced expert rows, export its MaxN CSV, complete it.

    Only Biigle-sourced rows are cleared (external_id IS NOT NULL), so
    manually-entered expert annotations survive a re-sync.
    """
    ann_db.clear_synced_annotations(drop_id, "expert")
    ann_db.add_annotations(annotations_to_add)

    maxn_path = config.get_biigle_expert_maxn_csv_path(drop_id)
    maxn_rows_to_df(annotations_to_add).to_csv(maxn_path, index=False)
    logging.info(f"  Expert MaxN → {maxn_path}")
    logging.info(f"  Ingested {len(annotations_to_add)} annotations for {drop_id}")

    db.advance_status(drop_id, ExpertStatus.COLUMN, ExpertStatus.COMPLETE)


def _drop_boundary_pattern(drop_id: str) -> re.Pattern:
    """Match `drop_id` at a token boundary inside a filename.

    A bare substring test false-matches across replicate numbers,
    "..._085_01" is a substring of "..._085_010", the same pitfall
    `db_refresh._volume_name_matches` guards against.
    """
    return re.compile(re.escape(drop_id) + r"(?![0-9A-Za-z])")


def _rows_for_drop(report: pd.DataFrame, drop_id: str) -> pd.DataFrame:
    """Rows of a SHARED (survey-pooled) volume report belonging to one drop.

    Pooled volume filenames carry the ``{drop}/frames/`` segment and their
    basenames start with the drop id, so a boundary match on the filename
    column routes each row. A report with no filename-ish column cannot be
    split; it is returned whole with a warning rather than silently losing
    every row.
    """
    fname_cols = [c for c in report.columns if "filename" in str(c).lower()]
    if not fname_cols:
        logging.warning(
            f"  {drop_id}: shared volume report has no filename column; "
            "cannot split rows per deployment, using the full report."
        )
        return report
    pattern = _drop_boundary_pattern(drop_id)
    mask = report[fname_cols[0]].astype(str).apply(lambda f: bool(pattern.search(f)))
    return report[mask]


def _volume_has_drop_files(handler, volume_id: int, drop_id: str, cache: dict) -> bool:
    """Whether the shared volume holds any file belonging to this drop.

    Distinguishes "reviewed, nothing seen" (files present, no annotation
    rows) from "never uploaded into the pooled volume" (no files), which
    must NOT be recorded as an absence.
    """
    if volume_id not in cache:
        cache[volume_id] = [
            str(img.get("filename", "")) for img in handler.get_volume_images(volume_id)
        ]
    pattern = _drop_boundary_pattern(drop_id)
    return any(pattern.search(name) for name in cache[volume_id])


def _write_universe_csv(handler, volume_id: int, drop_id: str, cache: dict) -> int:
    """Record every frame of `drop_id` present in `volume_id`, annotated or not.

    A Biigle annotation report contains only frames that HAVE annotations, so on
    its own it cannot tell "the expert looked and saw nothing" from "never
    uploaded". Both look like a missing row, and the second must never become a
    training negative. Writing the volume's file list at sync time - when the
    volume is DONE, so every frame in it has been reviewed - is what lets
    `biigle_to_yolo` emit an empty .txt (a YOLO background image) for the
    reviewed-but-empty frames.

    Before this, the corpus held 1 background frame in 5054 while the frame
    selector was deliberately choosing "Blind (False Negative Check)" frames for
    review: the experts were producing negatives and the parser discarded every
    one (found 2026-08-24). Returns the number of frames recorded.
    """
    if volume_id not in cache:
        cache[volume_id] = [
            str(img.get("filename", "")) for img in handler.get_volume_images(volume_id)
        ]
    pattern = _drop_boundary_pattern(drop_id)
    names = sorted(n for n in cache[volume_id] if n and pattern.search(n))
    if not names:
        return 0
    path = config.get_biigle_expert_universe_csv_path(drop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filename": names}).to_csv(path, index=False)
    logging.info(f"  Reviewed-frame universe ({len(names)}) → {path}")
    return len(names)


def sync_biigle_annotations():
    """
    Sync annotations from Biigle volumes that the annotator has marked as done.

    For each deployment with expert_status=uploaded and a biigle_volume_id:
    - Confirm the volume lives in the `done` project (3711). In-progress
      volumes (4942) are skipped so the annotator's WIP isn't pulled prematurely.
    - When `biigle.require_done_label` is true, ALSO require the legacy
      `Done Volume` whole-file label gate (belt-and-braces during transition).
    - Download the annotation report, save raw CSV, aggregate to MaxN, ingest
      into the annotations DB, export MaxN CSV per drop, rebuild YOLO labels.
    """
    logging.info("Starting Biigle annotation sync...")

    db = DatabaseManager()
    ann_db = AnnotationDatabaseManager()
    handler = BiigleHandler()

    deployments = db.get_biigle_volumes_awaiting_sync(ExpertStatus.UPLOADED)

    if not deployments:
        logging.info("No active deployments with Biigle volumes found to check.")
        return

    # Project-membership gate: only volumes currently in `done` (3711) are ready
    # to sync. One API call instead of per-volume get_volume_info.
    done_project_id = config.biigle_done_project_id
    # Cache full volume dicts (not just IDs) so we can reuse media_type below
    # without a per-volume get_volume_info() round-trip.
    done_volumes = {v["id"]: v for v in handler.get_volumes(done_project_id)}
    logging.info(f"Project {done_project_id} (done) has {len(done_volumes)} volume(s)")

    # Each tree falls back to an empty set on API failure, which degrades that
    # one routing rule to name-based handling rather than aborting the sync.
    trees = _fetch_label_trees(handler)

    # Survey-pooled volumes (--survey-volume uploads): several deployments
    # share one volume id. Each volume's report is downloaded once per run,
    # and rows are routed to their deployment by filename.
    volume_share_count = Counter(
        int(d["biigle_volume_id"]) for d in deployments if d.get("biigle_volume_id")
    )
    report_cache: dict = {}
    volume_files_cache: dict = {}

    processed_drops = []
    for dep in deployments:
        drop_id = dep["drop_id"]
        volume_id = int(dep["biigle_volume_id"])

        logging.debug(f"Checking Biigle volume {volume_id} for {drop_id}")

        try:
            media_type = _ready_media_type(handler, drop_id, volume_id, done_volumes)
            if media_type is None:
                continue

            parser = BiigleParser()
            if volume_id in report_cache:
                report = report_cache[volume_id]
            else:
                report = parser.download_volume_annotations(
                    volume_id=volume_id,
                    type_id=(
                        config.annotation_report_type_video
                        if media_type == "video"
                        else config.annotation_report_type_images
                    ),
                )
                report_cache[volume_id] = report

            if volume_share_count[volume_id] > 1:
                report = _rows_for_drop(report, drop_id)
                if report.empty and not _volume_has_drop_files(
                    handler, volume_id, drop_id, volume_files_cache
                ):
                    # No rows AND no files: this drop was never uploaded into
                    # the shared volume. Recording an absence here would fake
                    # a "reviewed, nothing seen"; leave the drop as-is.
                    logging.warning(
                        f"  {drop_id}: shared volume {volume_id} holds no "
                        "files for this drop; skipping (not reviewed)."
                    )
                    continue

            if report.empty:
                _record_null_deployment(
                    db, ann_db, drop_id, volume_id, "No annotations found"
                )
                processed_drops.append(drop_id)
                continue

            # Save raw Biigle report (used by YOLO label generation)
            config.get_drop_annotations_dir(drop_id).mkdir(parents=True, exist_ok=True)
            raw_path = config.get_biigle_expert_raw_csv_path(drop_id)
            report.to_csv(raw_path, index=False)
            logging.info(f"  Raw expert annotations → {raw_path}")

            # Negatives: the frames in this volume that the report does NOT
            # mention are frames the expert reviewed and left empty.
            _write_universe_csv(handler, volume_id, drop_id, volume_files_cache)

            _write_substrate_csv(parser, report, drop_id, trees)

            countable = _keep_countable_rows(
                report, trees.species, trees.workflow, trees.substrate
            )
            annotations_to_add = aggregate_raw_to_maxn_rows(
                countable, drop_id, trees.workflow, trees.species
            )

            if not annotations_to_add:
                _record_null_deployment(
                    db,
                    ann_db,
                    drop_id,
                    volume_id,
                    "No fish annotations after aggregation (only non-fish labels)",
                )
                processed_drops.append(drop_id)
                continue

            _ingest_species(db, ann_db, drop_id, annotations_to_add)
            processed_drops.append(drop_id)

        except Exception as e:
            logging.error(
                f"  Failed to sync volume {volume_id} for {drop_id}: {e}",
                exc_info=True,
            )

    if processed_drops:
        db.sync_annotation_counts(processed_drops)

        # Rebuild YOLO labels from all expert CSVs (class map needs all drops)
        class_map_path = config.local_training_dir / "class_map.json"
        biigle_to_yolo(
            deployment_data_dir=config.deployment_data_dir,
            class_map_path=class_map_path,
        )

        logging.info(f"Biigle sync complete. Processed {len(processed_drops)} drops.")
    else:
        logging.info("No new 'Done' volumes found to process.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync_biigle_annotations()
