"""Data layer for the DOC reporting page.

One place where the reporting tabs get their frames, so every tab counts the
same deployment the same way. Tabs receive a context dict and read nothing from
module scope, see ``build_context``.

Everything here is derived from the two pipeline databases via
``app/ecology_data.py``; nothing reads CSVs or S3 directly.
"""

import pandas as pd
import streamlit as st
from ecology_data import (
    EXPERT_SOURCES,
    NO_SHAREPOINT_RECORD,
    add_display_names,
    add_drop_id_columns,
    join_site_metadata,
    load_common_names,
    load_sites,
    source_bucket,
    source_rank,
)
from utils import CACHE_TTL_SECONDS

from spyfish.config.base import (
    NULL_DEPLOYMENT,
    CitSciStatus,
    ExpertStatus,
    IngestStatus,
    MlStatus,
    ReportingStatus,
    VideoPresence,
)
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import newest_model_only
from spyfish.database.connection import connect_readonly


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_deployments() -> pd.DataFrame:
    """Every deployment row, with drop_id parsed and site metadata joined.

    Coordinates are deliberately NOT requested here. A tab that needs a map asks
    for them separately behind ``check_password()``, see the note in
    ``ecology_data.load_sites``.
    """
    with connect_readonly(config.db_path) as conn:
        base_cols = (
            "drop_id, video_presence, ingest_status, ml_status, "
            "citsci_status, expert_status, reporting_status, is_bad_deployment, "
            "ml_annotations, citsci_annotations, expert_annotations, depth"
        )
        # Columns added by later migrations. The migration runs in the
        # pipeline and this reader is read-only, so an older database may lack
        # them: ask which exist rather than guessing with a failed query.
        optional = [
            "nzmhcs_abiotic",
            "nzmhcs_biotic",
            "zoo_frames_status",
            "zoo_crops_status",
        ]
        present = {row[1] for row in conn.execute("PRAGMA table_info(deployments)")}
        extra = [c for c in optional if c in present]
        df = pd.read_sql(
            f"SELECT {', '.join([base_cols, *extra])} FROM deployments", conn
        )
        # The Habitat panel reads missing habitat columns as empty and says so;
        # the status table skips a missing section column.
        for col in ("nzmhcs_abiotic", "nzmhcs_biotic"):
            if col not in df.columns:
                df[col] = pd.NA
    # Deployments hold no species, so only the two identity steps apply, no
    # invented `scientific_name` column needed.
    return join_site_metadata(add_drop_id_columns(df), load_sites())


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_annotations() -> pd.DataFrame:
    """One row per species observation, enriched the same way as deployments."""
    with connect_readonly(config.annotations_db_path) as conn:
        # `time_of_max` (text) and `external_id` ride along for the discovery
        # curve: legacy expert reviews carry their interval times only in the
        # text column (`time_of_max_seconds` was never backfilled for them),
        # and `external_id` is what identifies a row as a legacy full-video
        # review rather than a BIIGLE frame pick.
        df = pd.read_sql(
            "SELECT drop_id, scientific_name, max_interval, annotated_by, "
            "time_of_max, time_of_max_seconds, confidence_agreement, "
            "external_id FROM annotations",
            conn,
        )
    # One model's account per drop: where several models ran, the newest wins
    # (`newest_model_only`), so no chart mixes counts from two models.
    df = newest_model_only(df)
    # NULL_DEPLOYMENT rows record "reviewed, nothing seen". As NaN they keep
    # working with every notna/groupby path; isna() finds exactly these rows,
    # since a source that never reviewed a drop has no rows at all.
    df.loc[df["scientific_name"] == NULL_DEPLOYMENT, "scientific_name"] = pd.NA
    return add_display_names(
        join_site_metadata(add_drop_id_columns(df), load_sites()),
        load_common_names(),
    )


BEST_AVAILABLE = "Best available"
# "All" keeps every source's rows side by side, which is what a search wants:
# the point is to see what each source recorded for the same deployment. It is
# NOT interchangeable with "Best available", a deployment reviewed by both an
# expert and volunteers contributes two rows, so anything that sums or counts
# rows will double-count under it. Best available stays the default for that
# reason.
ALL_SOURCES = "All"
SOURCE_CHOICES = [BEST_AVAILABLE, ALL_SOURCES, "Expert", "CitSci", "ML"]


def _apply_source(ann: pd.DataFrame, source: str) -> pd.DataFrame:
    """Keep one annotation source, or collapse to the best available.

    `source_bucket` is the one mapping from `annotated_by` to a source. ML
    rows hold only the newest model's rows per drop by this point
    (`load_annotations`); the model name is in `external_id`.

    Best available follows the pipeline's "expert wins, no merge" doctrine
    **per deployment**: the best-ranked source that covered a drop supplies ALL
    of that drop's rows, and the other sources' rows are dropped entirely. An
    expert review is one coherent account of the deployment, mixing in ML rows
    for species the expert did not record would blend two accounts and let ML
    false positives ride along under the expert's name.

    Two things this deliberately does NOT do:

    * It does not collapse to one row per (drop, species). The frame keeps
      every interval row of the winning source, so downstream consumers take
      the peak themselves (`_deployment_maxn` etc.). Collapsing here to a
      single arbitrary interval row would silently understate every MaxN.
    * It does not fall back per species. If the expert covered the drop, a
      citsci-only species on that drop is treated as the expert not confirming
      it, not as missing data.
    """
    if ann.empty or source == ALL_SOURCES:
        return ann
    if source == BEST_AVAILABLE:
        # `source_rank` reads `external_id` too: a legacy expert review ranks
        # below a BIIGLE review of the same drop, so the two expert accounts
        # are never shown together as one.
        ranked = ann.assign(_rank=source_rank(ann))
        best = ranked.groupby("drop_id")["_rank"].transform("min")
        return ranked[ranked["_rank"] == best].drop(columns="_rank")
    if source == "Expert":
        return ann[ann["annotated_by"].isin(EXPERT_SOURCES)]
    if source == "CitSci":
        return ann[ann["annotated_by"] == "citsci"]
    return ann[~ann["annotated_by"].isin(EXPERT_SOURCES + ("citsci",))]


def source_coverage(annotations: pd.DataFrame) -> dict:
    """Deployments covered by each source, for labelling the source picker.

    Shown in the label because an empty chart is otherwise unexplained: the
    usual cause is that the chosen source has barely annotated anything, not
    that the filters are wrong.
    """
    if annotations.empty:
        return {}
    return (
        annotations.assign(_source=source_bucket(annotations["annotated_by"]))
        .groupby("_source")["drop_id"]
        .nunique()
        .to_dict()
    )


# ── Three "done" questions ───────────────────────────────────────────────────
#
# The report asks three different things of a deployment, and they are three
# functions on purpose, each next to the metric it drives. They deliberately
# disagree on some deployments (tests/unit/test_completion.py pins where):
#
# * `stage_flags` (here): has each SOURCE produced annotations? Read from the
#   annotation COUNTS, not the status columns. Drives the pipeline funnel and
#   the survey progress bands, where "ML done" means ML rows exist whatever the
#   status column says.
# * `is_complete` (here): is there nothing left to do? Read from the STATUS
#   columns: expert or reporting complete, or all three sections explicitly
#   skipped. Drives the "needs action" lists. Expert annotations without the
#   expert status are not complete; ML complete alone is not complete.
# * `ecology_data.analysed_by_status`: was it looked at? Ingest ok and any
#   section finished (expert uploaded counts). The denominator of every
#   abundance mean, so an ML-only drop counts as effort there while still
#   needing action here.


def stage_flags(dep: pd.DataFrame) -> pd.DataFrame:
    """Per-deployment booleans for each stage, measured by data presence.

    Drives the pipeline funnel and the survey progress bands. See the
    "Three done questions" note above for how this differs from
    `is_complete` and `analysed_by_status`.

    Presence of annotations rather than the status columns. A drop carrying ML
    annotations must have got through ML whatever its status column says, and
    section statuses drift from the annotation data. Ingest problems found after
    the fact do not unwind progress already made.
    """
    out = dep.copy()
    out["ok"] = out[IngestStatus.COLUMN] == IngestStatus.OK
    out["has_video"] = out["video_presence"] == VideoPresence.PRESENT
    # Present OR archived: archived footage exists and its annotations are
    # real work. Tapuae made the difference concrete — every one of its 103
    # expert-annotated deployments is archived, so a funnel keyed on
    # "present" showed zero expert work for the whole reserve.
    out["footage_exists"] = out["video_presence"].isin(
        [VideoPresence.PRESENT, VideoPresence.ARCHIVED]
    )
    for source in ("ml", "citsci", "expert"):
        # to_numeric first: a count column read from SQL can arrive as object
        # dtype, and fillna on object downcasts with a FutureWarning.
        counts = pd.to_numeric(out[f"{source}_annotations"], errors="coerce")
        out[f"{source}_done"] = counts.fillna(0) > 0
    return out


_PRESENCE_DISPLAY = {
    VideoPresence.PRESENT: "Present",
    VideoPresence.ARCHIVED: "Archived",
    VideoPresence.ABSENT: "Absent",
    VideoPresence.NO_VIDEO_BAD_DEP: "No video (bad dep.)",
}


def is_complete(dep: pd.DataFrame) -> pd.Series:
    """Expert or reporting complete, or every section explicitly skipped.

    Reads the **status columns**, unlike `stage_flags`, because it asks a
    different question: whether any work is left, which is what the status
    columns decide. See the "Three done questions" note above.
    """
    return (
        (dep[ExpertStatus.COLUMN] == ExpertStatus.COMPLETE)
        | (dep[ReportingStatus.COLUMN] == ReportingStatus.COMPLETE)
        | (
            (dep[MlStatus.COLUMN] == MlStatus.SKIPPED)
            & (dep[CitSciStatus.COLUMN] == CitSciStatus.SKIPPED)
            & (dep[ExpertStatus.COLUMN] == ExpertStatus.SKIPPED)
        )
    )


def add_completion_flags(dep: pd.DataFrame) -> pd.DataFrame:
    """`complete`, `total_annotations`, `video_status` and `needs_action`.

    "Complete" is `is_complete` above, which reads the **status columns**,
    unlike the rest of the report, which counts annotations, see the open
    question on the reporting-numbers panel.
    """
    out = dep.copy()
    out["complete"] = is_complete(out)
    counts = out[["expert_annotations", "ml_annotations", "citsci_annotations"]]
    out["total_annotations"] = counts.fillna(0).astype(int).sum(axis=1)
    out["video_status"] = out["video_presence"].map(_PRESENCE_DISPLAY).fillna("Unknown")
    out["needs_action"] = ~(
        out["complete"] | out["is_bad_deployment"].fillna(0).astype(bool)
    )
    return out


# `max_interval` is the count in ONE time interval, and the annotations table
# holds one row per interval. It is NOT MaxN.
#
# MaxN is the PEAK across those intervals. Summing instead of peaking counts the
# same fish once per interval it appears in: on one deployment here that is 698
# instead of 24. The two functions below are the only places the peak is taken,
# so no view can get this wrong on its own.
INTERVAL_COUNT = "max_interval"


def species_maxn(ann: pd.DataFrame, extra_keys: tuple = ()) -> pd.DataFrame:
    """One row per (deployment, *extra_keys, species): the peak across intervals.

    `extra_keys` splits the peak further, e.g. `("Source",)` to keep each
    annotation source's own view of the same deployment separate.

    Absence records are kept: a "reviewed, nothing seen" deployment comes
    through as one row with `scientific_name` NaN and `maxn` 0, so it stays in
    every per-deployment mean downstream. `groupby` defaults to dropping NaN
    keys, which silently removed exactly those rows, and every "mean abundance"
    or "mean richness" built on this frame was really a mean over deployments
    where something was seen. Consumers that count species are unaffected:
    `nunique` and species-keyed groupbys ignore NaN, so an absence row counts
    a deployment without inventing a species.

    `display_name` rides along when the input has it. Grouping returns only the
    keys and the aggregated column, so without this each chart has to map
    scientific names back to display names itself, against its own copy of the
    lookup, and two charts can then label one species differently.

    Reattached rather than grouped on: it is derived from `scientific_name`,
    not part of what makes a row unique. `add_display_names` falls back to the
    scientific name, so it is never null where the species is not, and the two
    give the same result on real data.
    """
    keys = ["drop_id", *extra_keys, "scientific_name"]
    out = ann.groupby(keys, dropna=False)[INTERVAL_COUNT].max().reset_index(name="maxn")
    if "display_name" in ann.columns:
        names = ann[["scientific_name", "display_name"]].drop_duplicates(
            "scientific_name"
        )
        out = out.merge(names, on="scientific_name", how="left")
    return out


def deployment_maxn(ann: pd.DataFrame, extra_keys: tuple = ()) -> pd.Series:
    """Total MaxN per (deployment, *extra_keys): peak per species, then summed.

    Two steps, in this order. The peak comes first because MaxN is the most
    individuals visible at once, so a species' value for a deployment is the
    maximum over its intervals, never the sum. Species are then added, because
    the deployment total is across species.
    """
    per_species = species_maxn(ann, extra_keys)
    return per_species.groupby(["drop_id", *extra_keys])["maxn"].sum()


def richness_by(
    sightings: pd.DataFrame, keys, extra_aggs: dict | None = None
) -> pd.DataFrame:
    """Distinct real species per group, pooled and summed per deployment.

    The ONE place richness is counted. Two columns, for the two things
    "richness" means in this report:

    * `species`: distinct real species recorded anywhere in the group, the
      pooled richness a site map or a per-regime trend shows;
    * `richness_sum`: distinct real species per deployment, summed over the
      group's deployments. Divided by the group's effort it is species per
      deployment (`richness_rollup`), the "what does a typical drop here
      hold" number.

    `real_species` is applied here, so the generic (`fish`) and unidentified
    buckets never count as a species; it is idempotent, so a pre-filtered
    frame is harmless. Absence records (NaN species) count nothing: a drop
    holding only an absence reads 0 species, and a group holding only
    absences keeps its row at 0. `keys` is a column name or list of names;
    `extra_aggs` are further named aggregations in `.agg()` form, grouped on
    the same keys.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    real = real_species(sightings)
    aggs = {"species": ("scientific_name", "nunique"), **(extra_aggs or {})}
    pooled = real.groupby(keys, dropna=False).agg(**aggs).reset_index()
    summed = (
        real.groupby([*keys, "drop_id"], dropna=False)["scientific_name"]
        .nunique()
        .groupby(keys, dropna=False)
        .sum()
        .reset_index(name="richness_sum")
    )
    return pooled.merge(summed, on=keys, how="left")


def richness_rollup(
    sightings: pd.DataFrame,
    effort: pd.DataFrame,
    by,
    attrs: tuple = (),
    extra_aggs: dict | None = None,
) -> pd.DataFrame:
    """Species, analysed deployments and species-per-deployment, per group.

    ONE definition of the per-group richness rollup, because there were two:
    the Sites view's per-site table and the MPA view's per-MPA table had
    near-identical blocks that agreed by luck rather than by construction.
    They differed in two ways that mattered, and this settles both:

    * **Empty groups are kept.** A site or MPA that was analysed and
      recorded nothing has real effort and zero richness; dropping it (what
      the per-site version did, via a left merge) hid 25 surveyed sites and
      quietly flattered the averages. Absences are data everywhere else in
      this report, so they are here too.
    * **`real_species` is applied** (in `richness_by`), not by each caller.
      Richness counts SPECIES, and the generic (`fish`) and unidentified
      buckets are N unknown species wearing one label; a caller that forgot
      the call got a richness one too high.

    `species_per_dep` is richness summed PER DEPLOYMENT and then divided by
    the analysed count, not distinct species over the group divided by
    deployments: the first answers "what does a typical drop here hold",
    which is what the column claims. The counting is `richness_by`; this
    adds the effort denominator. The MPA view's reserve trends go through
    here too, so "species per deployment" is one number everywhere.

    Args:
        sightings: one row per (deployment, species) with `drop_id` and
            `scientific_name`, already filtered to the current selection.
        effort: the deployments analysed in the same selection, the
            denominator (see `effort_per`).
        by: the grouping column, e.g. "site_id" or "reserve", or a list of
            columns for a multi-key cell such as (reserve, year, side).
        attrs: per-group attribute columns carried through unchanged
            (e.g. region, protection status). They must be functionally
            dependent on `by`, since they are grouped on.
        extra_aggs: further named aggregations, in `.agg()` form, e.g.
            `{"sites": ("site_id", "nunique")}`.
    """
    by = [by] if isinstance(by, str) else list(by)
    counted = richness_by(sightings, [*by, *attrs], extra_aggs)
    # Outer, so a group with effort but no sightings keeps its row.
    out = counted.merge(effort_per(effort, by), on=by, how="outer")
    filled = {"species": 0, "richness_sum": 0, "analysed": 0}
    filled.update({name: 0 for name in (extra_aggs or {})})
    out = out.fillna(filled)
    out["species_per_dep"] = out["richness_sum"] / out["analysed"].clip(lower=1)
    return out.drop(columns="richness_sum").sort_values(
        "species_per_dep", ascending=False
    )


def arrival_and_peak(ann: pd.DataFrame) -> pd.DataFrame:
    """Per (deployment, species): when it first appeared, and when it peaked.

    Two different measurements that were previously conflated under one
    timestamp:

    * **Arrival** is the first interval the species was detected at all, and it
      comes from ML only. The model scores every 10-second interval of the
      video, so it is the only source that can say when something first came
      into frame. An expert reviewing ten selected frames cannot, and would
      report whichever of those frames happened to be earliest.
    * **MaxN time** is the interval carrying the species' highest count, from
      whichever source is best for that deployment. Expert wins where it
      exists, so this is the reportable number.

    A species can arrive at 2 minutes and peak at 20, which is exactly the
    behaviour worth seeing, and exactly what a single timestamp hides.

    Takes the UNFILTERED annotations: arrival needs the ML rows even when the
    reader has filtered to expert.
    """
    timed = ann[ann["time_of_max_seconds"].notna() & ann["scientific_name"].notna()]
    if timed.empty:
        return pd.DataFrame(
            columns=["drop_id", "scientific_name", "arrival_s", "peak_s", "peak_source"]
        )

    is_ml = ~timed["annotated_by"].isin(EXPERT_SOURCES + ("citsci",))
    detected = timed[timed[INTERVAL_COUNT] > 0]

    arrival = (
        detected[is_ml.reindex(detected.index, fill_value=False)]
        .groupby(["drop_id", "scientific_name"])["time_of_max_seconds"]
        .min()
        .rename("arrival_s")
    )

    # The peak's timestamp, not the earliest: sort so the highest count per
    # (drop, species) is last, then keep that row.
    best = _apply_source(detected, BEST_AVAILABLE)
    peak = (
        best.sort_values(INTERVAL_COUNT)
        .groupby(["drop_id", "scientific_name"])
        .agg(
            peak_s=("time_of_max_seconds", "last"), peak_source=("annotated_by", "last")
        )
    )
    times = arrival.to_frame().join(peak, how="outer").reset_index()
    # Display names come along so the chart never has to reach back for them.
    names = ann[["scientific_name", "display_name"]].drop_duplicates("scientific_name")
    return times.merge(names, on="scientific_name", how="left")


# The two count paths a discovery curve can be built from. Module constants so
# the chart's radio and the data function cannot drift apart on the labels.
CURVE_ML = "ML detections"
CURVE_EXPERT = "Expert (legacy full-video reviews)"


def _legacy_seconds(rows: pd.DataFrame) -> pd.Series:
    """Seconds-into-deployment for legacy expert rows, from `time_of_max`.

    Thin wrapper over the pipeline's `legacy_time_of_max_seconds`, which owns
    the Excel-coercion repair (minutes-first rule, Kalindi 2026-10-02) and is
    the same function the ingest and the DB backfill use — so a legacy row
    cannot parse to two different moments in the pipeline and the report.

    Only a fallback: rows ingested or backfilled since the repair carry
    `time_of_max_seconds` already, and `discovery_curve` prefers that. This
    covers databases whose legacy rows predate the backfill.

    Local import: the orchestrator module pulls in the S3 stack, which the
    report needs nowhere else.
    """
    from spyfish.orchestrator.legacy_extract import legacy_time_of_max_seconds

    return legacy_time_of_max_seconds(rows["drop_id"], rows["time_of_max"])


def discovery_curve(ann: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    """Per (deployment, species): share of the final MaxN reached by each minute.

    The deployment-length validation from the DOC BUV studies (Brough et al.
    2018, DOC R&D Series 354, call it a discovery curve): plot the best count
    seen so far against time into the deployment. If the mean curve gets close
    to its final value well before the cutoff, the deployments are long enough
    for MaxN to be a stable estimate; a curve still climbing at the end means
    longer footage would have reported more fish.

    Two count paths qualify, returned side by side in the `Source` column:

    * `CURVE_ML`: the model scores every 10-second interval, so its rows trace
      the count's whole path through the video.
    * `CURVE_EXPERT`: legacy full-video reviews (`external_id` = "legacy")
      recorded a count per interval watched, the method of the DOC BUV
      toolbox. Their times live in the text column, see `_legacy_seconds`.

    Everything else is excluded: BIIGLE frame reviews and citsci clips
    describe a handful of selected moments, and a curve built from them would
    mostly reflect which moments were picked.

    Shares, not raw counts, so a deployment with MaxN 2 and one with MaxN 20
    weigh the same: the question is *when* the peak is reached, not how big it
    is. Every curve ends at 1.0 by construction; the finding is how early it
    gets close. Pairs whose counts never exceed zero have no peak to reach and
    are dropped.

    Returns one row per (Source, drop_id, scientific_name, minute) with
    `share` in 0..1, plus `display_name` for labelling.
    """
    empty = pd.DataFrame(
        columns=[
            "Source",
            "drop_id",
            "scientific_name",
            "display_name",
            "minute",
            "share",
        ]
    )
    counted = ann[ann["scientific_name"].notna() & (ann[INTERVAL_COUNT] > 0)]

    ml = counted[
        ~counted["annotated_by"].isin(EXPERT_SOURCES + ("citsci",))
        & counted["time_of_max_seconds"].notna()
    ]
    pieces = [ml.assign(Source=CURVE_ML, _seconds=ml["time_of_max_seconds"])]

    if "external_id" in counted.columns and "time_of_max" in counted.columns:
        legacy = counted[
            (counted["external_id"] == "legacy") & counted["time_of_max"].notna()
        ]
        if not legacy.empty:
            # Stored seconds first (the ingest and backfill write the repaired
            # value), parsing the text only for rows that predate them.
            pieces.append(
                legacy.assign(
                    Source=CURVE_EXPERT,
                    _seconds=pd.to_numeric(
                        legacy["time_of_max_seconds"], errors="coerce"
                    ).fillna(_legacy_seconds(legacy)),
                )
            )

    # Empty pieces are dropped before the concat: pandas warns that it will
    # stop inferring dtypes from all-NA frames, and a source with no rows has
    # nothing to contribute to the dtypes anyway.
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        return empty
    timed = pd.concat(pieces)
    timed = timed[timed["_seconds"].notna()]
    if timed.empty:
        return empty

    # Interval timestamps land in minute buckets 1..`minutes`; a few run past
    # the nominal cutoff (encoder drift, slightly long videos) and clip into
    # the last bucket rather than inventing a 31st minute.
    minute = (timed["_seconds"] // 60 + 1).clip(upper=minutes).astype(int)
    per_minute = (
        timed.assign(minute=minute)
        .groupby(["Source", "drop_id", "scientific_name", "minute"])[INTERVAL_COUNT]
        .max()
        .unstack("minute")
        .reindex(columns=range(1, minutes + 1))
    )
    # Best-so-far: a minute with no detection keeps the previous peak, which is
    # what "the MaxN you would report if the video ended here" means.
    cumulative = per_minute.fillna(0).cummax(axis=1)
    final = cumulative.iloc[:, -1]
    curve = cumulative[final > 0].div(final[final > 0], axis=0)
    if curve.empty:
        return empty

    long = curve.stack().rename("share").reset_index()
    names = ann[["scientific_name", "display_name"]].drop_duplicates("scientific_name")
    return long.merge(names, on="scientific_name", how="left")


def bootstrap_diff_ci(
    a, b, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Difference in means (a minus b) with a 95% bootstrap interval.

    Answers the one question a pair of means cannot: had the cameras landed
    on a different sample of these same areas, how far could this gap have
    moved? Each group's per-deployment values are resampled with replacement
    `n_boot` times, the difference in means recomputed each time, and the
    interval is the middle 95% of those differences. An interval containing
    zero means the data cannot yet tell the gap from sampling luck.

    Bootstrap rather than a t-interval because MaxN-style values are heavily
    skewed with many zeros, and the resampling makes no shape assumptions.
    One assumption remains and callers must say it out loud: every deployment
    is treated as independent, so replicate drops at one site make the
    interval look tighter than it honestly is.

    Fixed seed so the page shows the same interval on every rerun; with 2,000
    resamples the seed-to-seed wobble is in the second decimal.

    Returns (difference, low, high), or NaNs when either side has fewer than
    two values.
    """
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    diffs = rng.choice(a, size=(n_boot, len(a))).mean(axis=1) - rng.choice(
        b, size=(n_boot, len(b))
    ).mean(axis=1)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return (float(a.mean() - b.mean()), float(low), float(high))


# One resample count for every panel that bootstraps, because two panels on
# one page disagreed about the same species: snapper's interval sits on zero,
# so at 1,000 resamples the lower bound came out -0.01 and at 2,000 it came
# out +0.00, and the two charts drew opposite verdicts from the same data.
# Callers do not get to tune this; a comparison's answer must not depend on
# which chart asked.
N_BOOT = 2000


def stratified_diff_ci(
    frame: pd.DataFrame,
    value_col: str,
    group_col: str,
    stratum_col: str,
    inside: str,
    outside: str,
    n_boot: int = N_BOOT,
    seed: int = 0,
    min_per_side: int = 2,
) -> dict:
    """Inside-minus-outside measured WITHIN each stratum, then pooled.

    The pooled version of this comparison (`bootstrap_diff_ci` on two flat
    lists) silently compares areas as well as protection regimes: this
    programme has 73 protected against 244 unprotected deployments at one
    marine reserve and 150 against 110 at another, so a pooled mean weights
    those places differently on the two sides, and a difference between the
    PLACES arrives looking like a difference from protection. Brough et al.
    (2018) refused to pool for exactly this reason, keeping their five survey
    areas as separate levels.

    So the difference is computed inside each stratum, where both sides are
    the same place (and, if the caller strata by year as well, the same
    survey season), and only then combined. Each stratum needs
    `min_per_side` deployments on both sides to contribute; the rest are
    reported as skipped rather than quietly folded in.

    Weighting is by precision: the variance of a difference of means goes as
    1/n_inside + 1/n_outside, so the weight is its reciprocal,
    n_in*n_out/(n_in+n_out). A stratum resting on 3 against 4 deployments
    moves the pooled number far less than one resting on 150 against 110,
    without being thrown away.

    The bootstrap resamples within each (stratum, side) cell and recomputes
    the same weighted combination, so the interval inherits the
    stratification. Weights stay fixed across resamples, which is standard:
    they describe the design, not the data.

    Returns a dict with the pooled `diff`, `lo`, `hi`, the two sides'
    `mean_inside` / `mean_outside`, `share_other_way`, the `strata` frame (one
    row per contributing stratum, for showing the spread behind the pooled
    number), `n_strata`, `n_skipped`, and the deployment counts. The pooled
    values are NaN when no stratum qualifies.

    `share_other_way` is the fraction of resamples whose difference landed on
    the opposite side of zero from the estimate. It is the continuous version
    of "does the interval clear zero", and it is reported because the binary
    is a cliff: snapper on this data sits at a lower bound of 0.00, so the
    tick-box answer flips on resampling noise while `share_other_way` says
    plainly that the direction is held by a hair (about 2.5%) rather than
    established (0.1%).

    The two means carry the SAME weights as the difference, so
    `mean_inside - mean_outside == diff` exactly. Weighting them per side
    instead (each side's own deployment counts) lets the pair contradict the
    difference it sits beside — on this data blue cod read "2.06 inside
    against 2.97 outside, difference -0.02", which is the stratification
    working correctly and looks like a bug.
    """
    import numpy as np

    blank = {
        "diff": float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
        "mean_inside": float("nan"),
        "mean_outside": float("nan"),
        "share_other_way": float("nan"),
        "strata": pd.DataFrame(
            columns=[
                stratum_col,
                "n_inside",
                "n_outside",
                "mean_inside",
                "mean_outside",
                "diff",
            ]
        ),
        "n_strata": 0,
        "n_skipped": 0,
        "n_inside": 0,
        "n_outside": 0,
    }
    needed = {value_col, group_col, stratum_col}
    if frame.empty or not needed <= set(frame.columns):
        return blank

    rows, cells, skipped = [], [], 0
    for stratum, sub in frame.groupby(stratum_col, dropna=True):
        a = sub.loc[sub[group_col] == inside, value_col].to_numpy(dtype=float)
        b = sub.loc[sub[group_col] == outside, value_col].to_numpy(dtype=float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < min_per_side or len(b) < min_per_side:
            skipped += 1
            continue
        weight = len(a) * len(b) / (len(a) + len(b))
        cells.append((a, b, weight))
        rows.append(
            {
                stratum_col: stratum,
                "n_inside": len(a),
                "n_outside": len(b),
                "mean_inside": a.mean(),
                "mean_outside": b.mean(),
                "diff": a.mean() - b.mean(),
            }
        )

    if not cells:
        return {**blank, "n_skipped": skipped}

    total_w = sum(w for _, _, w in cells)
    point = sum(w * (a.mean() - b.mean()) for a, b, w in cells) / total_w
    mean_in = sum(w * a.mean() for a, _, w in cells) / total_w
    mean_out = sum(w * b.mean() for _, b, w in cells) / total_w

    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot)
    for a, b, weight in cells:
        resampled = rng.choice(a, size=(n_boot, len(a))).mean(axis=1) - rng.choice(
            b, size=(n_boot, len(b))
        ).mean(axis=1)
        boot += weight * resampled
    boot /= total_w
    low, high = np.percentile(boot, [2.5, 97.5])
    # How much of the resampled distribution sits on the far side of zero
    # from the estimate: the continuous reading of the same question the
    # interval answers yes/no.
    other_way = (boot < 0).mean() if point >= 0 else (boot > 0).mean()

    strata = pd.DataFrame(rows).sort_values("diff", ascending=False)
    return {
        "diff": float(point),
        "lo": float(low),
        "hi": float(high),
        "mean_inside": float(mean_in),
        "mean_outside": float(mean_out),
        "share_other_way": float(other_way),
        "strata": strata,
        "n_strata": len(rows),
        "n_skipped": skipped,
        "n_inside": int(strata["n_inside"].sum()),
        "n_outside": int(strata["n_outside"].sum()),
    }


def species_value_grid(
    deployments: pd.DataFrame, per_species: pd.DataFrame, species: list
) -> pd.DataFrame:
    """Per-deployment MaxN for each named species, zero-filled.

    `species_maxn` only has a row where a species was recorded, so a mean
    taken straight off it is a mean over the deployments that already had the
    species — which cannot fall and always flatters the population. The
    answer is a full grid: every deployment in `deployments` carries a value
    for every named species, and a deployment that was reviewed without
    recording it carries a real zero.

    `deployments` is the deployment-level frame (one row per `drop_id`, with
    whatever grouping columns the caller needs); its columns ride along so
    the result is ready to group by protection and stratum.
    """
    if deployments.empty or not species:
        return pd.DataFrame(columns=[*deployments.columns, "scientific_name", "maxn"])
    frames = []
    for name in species:
        values = (
            per_species.loc[per_species["scientific_name"] == name]
            .set_index("drop_id")["maxn"]
            .groupby(level=0)
            .max()
        )
        frames.append(
            deployments.assign(
                scientific_name=name,
                maxn=deployments["drop_id"].map(values).fillna(0.0),
            )
        )
    return pd.concat(frames, ignore_index=True)


def unify_unidentified(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge every not-a-species class into one labelled bucket.

    `fish`, `Fish: review required`, `unknown` and the rest all mean the same
    thing: something was seen and not named. Left as they are they appear as
    several separate "species", each with its own bar, colour and row in a
    co-occurrence matrix, and together they are the single most numerous label
    in the database.

    Merged rather than dropped, because "seen but not identified" is a real
    observation and a large one. Charts that count species exclude the bucket
    afterwards with `real_species`; charts that count animals keep it, and show
    honestly how much of the catch has no name yet.

    The list is `reporting.non_species_classes` in config.yaml, because the
    class map changes without a code release.
    """
    if frame.empty or "scientific_name" not in frame.columns:
        return frame
    is_other = frame["scientific_name"].isin(config.non_species_classes)
    if not is_other.any():
        return frame
    out = frame.copy()
    label = config.unidentified_label
    out.loc[is_other, "scientific_name"] = label
    if "display_name" in out.columns:
        out.loc[is_other, "display_name"] = label
    return out


def effort_per(frame: pd.DataFrame, keys, name: str = "analysed") -> pd.DataFrame:
    """Distinct deployments per key, the denominator every rate divides by.

    Eight charts used to derive this individually, with three different
    zero-handling policies between them. One definition, so "per deployment"
    cannot mean different things on neighbouring charts. `keys` is a column
    name or list of names; the count comes back as column `name`.

    Division stays with the caller (`.clip(lower=1)` on the denominator is the
    house rule), because what is being divided differs per chart.
    """
    return frame.groupby(keys)["drop_id"].nunique().reset_index(name=name)


def real_species(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the unidentified bucket, for anything counting distinct species.

    Richness, accumulation and co-occurrence all treat each name as one
    species. The bucket is N unknown species wearing one label, so counting it
    as one both understates richness and puts a meaningless row in every
    matrix.
    """
    if frame.empty or "scientific_name" not in frame.columns:
        return frame
    names = set(config.non_species_classes) | {config.unidentified_label}
    return frame[~frame["scientific_name"].isin(names)]


def experiments_frame(ann: pd.DataFrame) -> pd.DataFrame:
    """One row per (deployment, source, species), with `maxn`.

    The shape the Experiments page works in. Its charts were written against
    `ecology_data.load_maxn`, which has already taken the peak, while the
    report loads the raw interval rows so it can filter by source first. Rather
    than rewrite a dozen working charts as they are copied across, this gives
    them the frame they expect, from the report's already-filtered annotations.

    Selecting the peak ROW, rather than aggregating the peak value and joining
    metadata back on, keeps the columns that describe *that* interval,
    `time_of_max_seconds` above all, which is what makes a MaxN a "time of max"
    and is the only thing the bait-arrival curve reads. Aggregating first threw
    those columns away, and the chart raised `KeyError` on the one it needed.

    NaN species are kept: a row with no species is an absence record,
    "reviewed, nothing seen", and those deployments are the denominator of
    every detection rate. `drop_duplicates` treats NaN as equal, so they
    collapse to one row per deployment rather than vanishing.

    `na_position="first"` puts unmeasured rows at the bottom of the sort, so
    `keep="last"` cannot pick a NaN count as a deployment's peak.
    """
    keys = ["drop_id", "annotated_by", "scientific_name"]
    peak_rows = ann.sort_values(INTERVAL_COUNT, na_position="first").drop_duplicates(
        subset=keys, keep="last"
    )
    # Merged here, so every chart reading this frame sees one bucket rather than
    # six labels for the same thing.
    return unify_unidentified(peak_rows.rename(columns={INTERVAL_COUNT: "maxn"}))


def split_reserves(series: pd.Series) -> set:
    """Distinct MPA names from the comma-joined link column.

    A site between two MPAs carries both, and the same pair appears in both
    orders, so the raw column holds more values than there are MPAs.

    Here rather than in each view: four copies of this loop existed, and a view
    that split on a different character would report a different number of MPAs
    from the one beside it.
    """
    names = set()
    for value in series.dropna():
        names.update(part.strip() for part in str(value).split(",") if part.strip())
    return names


def matches_reserves(series: pd.Series, reserves: list) -> pd.Series:
    """Boolean mask for rows belonging to any of `reserves`.

    `link_to_marine_reserve` holds comma-joined names for sites that sit between
    two reserves, so membership is a split-and-match rather than an `isin`.
    Reserve *names* rather than the DropID reserve code, because that is what
    the Sites view has always filtered on and the two must agree.
    """
    wanted = set(reserves)
    return series.fillna("").apply(
        lambda v: bool({p.strip() for p in str(v).split(",") if p.strip()} & wanted)
    )


def reserve_code_to_names(*frames: pd.DataFrame) -> dict:
    """Map each DropID reserve code to the MPA name(s) it resolves to.

    Built from the rows that carry BOTH a `reserve_code` and a
    `link_to_marine_reserve`, which is the only place the two are tied together.
    A code can map to several names (a code whose sites straddle two MPAs), so
    the value is a set. Used to recover the MPA of a legacy/no-SharePoint drop,
    whose site is not in the sites list and so has no `link` of its own.
    """
    mapping: dict = {}
    for frame in frames:
        if frame.empty or "reserve_code" not in frame.columns:
            continue
        have = frame[frame["link_to_marine_reserve"].notna()]
        for code, link in zip(have["reserve_code"], have["link_to_marine_reserve"]):
            names = {p.strip() for p in str(link).split(",") if p.strip()}
            mapping.setdefault(code, set()).update(names)
    return mapping


def matches_reserves_with_code(
    frame: pd.DataFrame, reserves: list, code_names: dict
) -> pd.Series:
    """Reserve mask that falls back to the DropID code when `link` is missing.

    A row matches if its `link_to_marine_reserve` names any wanted MPA, OR, for
    a row with no link, the legacy/no-SharePoint case, if its `reserve_code`
    resolves (via `code_names`) to a wanted MPA. Without the fallback, picking
    an MPA in the sidebar dropped every legacy drop at that MPA, because its
    link is blank, and the pre-pipeline years vanished from the view.
    """
    by_link = matches_reserves(frame["link_to_marine_reserve"], reserves)
    wanted = set(reserves)
    missing = frame["link_to_marine_reserve"].isna()
    by_code = missing & frame["reserve_code"].map(
        lambda c: bool(code_names.get(c, set()) & wanted)
    )
    return by_link | by_code


def build_context(
    deployments: pd.DataFrame,
    annotations: pd.DataFrame,
    years: tuple,
    reserves: list,
    ingested_only: bool,
    source: str = BEST_AVAILABLE,
    species: list | None = None,
    regions: list | None = None,
    protections: list | None = None,
    include_historical: bool = True,
    exclude_bad_deployments: bool = False,
) -> dict:
    """Apply the page-level filters once and hand every view the same frames.

    Filtering here rather than inside each view keeps the counts consistent.
    Two views filtering separately would disagree about how many deployments
    exist, with both totals on screen at once.

    `ingested_only` drops deployments that were excluded at ingest or hold
    validation errors. They never reach a processing stage, so including them
    makes coverage look worse than it is for the deployments that can actually
    be processed. The reporting shell leaves it False on purpose and explains
    the gap on the front page instead, so no screenshot is ambiguous about which
    way it was set.
    """
    dep = deployments
    ann = annotations

    # Bad deployments (camera over, bait gone, visibility hopeless) leave the
    # ANNOTATION frames: their sightings are as suspect as their blanks, so
    # they belong in neither the numerators nor the deployment denominators
    # of a species statistic. They stay in the deployments frame, where the
    # coverage and pipeline counts deliberately show them.
    #
    # Default is off, and it is a sidebar toggle rather than a hard rule,
    # because the flag is NOT reliably "unusable": 41 flagged drops carried
    # ~3,000 annotation rows when this was written, 2,154 of them full
    # legacy expert reviews. Somebody watched that footage and got data out
    # of it. See `claude_docs/todo.md` ("IsBadDeployment audit").
    if exclude_bad_deployments:
        bad_drops = set(
            deployments.loc[
                deployments["is_bad_deployment"].fillna(False).astype(bool), "drop_id"
            ]
        )
        if bad_drops:
            ann = ann[~ann["drop_id"].isin(bad_drops)]

    if not include_historical:
        # Legacy imports can reference drops with no deployment/survey record
        # on SharePoint. Without the sidebar tick they stay out so every
        # denominator rests on registered deployments.
        ann = ann[ann["drop_id"].isin(dep["drop_id"])]
    else:
        # Ticked: those drops must also COUNT as deployments (surveys per
        # year, activity charts), so synthesize one deployment row per
        # annotation-only drop. The annotations frame already carries the
        # parsed DropID and site-metadata columns, so reuse them; DB-only
        # columns (statuses, counts, depth) stay NaN via reindex. A distinct
        # ingest label keeps status splits honest instead of blending in.
        extra_mask = ~ann["drop_id"].isin(dep["drop_id"])
        if extra_mask.any():
            shared = [c for c in dep.columns if c in ann.columns]
            extra = (
                ann.loc[extra_mask, shared]
                .drop_duplicates("drop_id")
                .reindex(columns=dep.columns)
            )
            extra["ingest_status"] = NO_SHAREPOINT_RECORD
            extra["is_bad_deployment"] = False
            dep = pd.concat([dep, extra], ignore_index=True)

    if years:
        lo, hi = years

        # Undated rows are kept. A DropID whose date cannot be read still
        # describes a real deployment, and dropping it here would quietly change
        # every total on the page relative to `all_deployments`, which is what
        # the Metadata error review page counts them against. The Sites filters
        # have always kept them for the same reason, so this is also what makes
        # the two agree.
        def in_years(frame):
            year = frame["survey_year"]
            return frame[year.between(lo, hi) | year.isna()]

        dep = in_years(dep)
        ann = in_years(ann)
    if reserves:
        # Code→name map from the full frames (before this filter narrows them),
        # so a legacy drop with no `link` is still matched to its MPA by the
        # reserve code in its DropID.
        code_names = reserve_code_to_names(deployments, annotations)
        dep = dep[matches_reserves_with_code(dep, reserves, code_names)]
        ann = ann[matches_reserves_with_code(ann, reserves, code_names)]
    if regions:
        dep = dep[dep["region"].isin(regions)]
        ann = ann[ann["region"].isin(regions)]
    if protections:
        dep = dep[dep["protection_status"].isin(protections)]
        ann = ann[ann["protection_status"].isin(protections)]
    if ingested_only:
        dep = dep[dep[IngestStatus.COLUMN] == IngestStatus.OK]
        ann = ann[ann["drop_id"].isin(dep["drop_id"])]

    # Year/reserve-filtered but source-UNfiltered. The source-comparison panels
    # need this frame: under any source filter, one side of an ML-vs-expert
    # comparison has already been thrown away before the panel runs.
    ann_all_sources = ann
    ann = _apply_source(ann, source)

    # Species narrows the ANNOTATION rows only, and last: deployments are not
    # "about" one species, and absence records (null species) or other species'
    # rows leaving the deployment frame would silently change every effort
    # denominator on the page. The pre-species frame stays available for
    # exactly those denominators.
    ann_all_species = ann
    if species:
        ann = ann[ann["display_name"].isin(species)]
        ann_all_sources = ann_all_sources[ann_all_sources["display_name"].isin(species)]

    return {
        "deployments": dep,
        "annotations": ann,
        "annotations_all_sources": ann_all_sources,
        # Source-filtered but species-UNfiltered: the denominator frame for
        # detection rates and "reviewed, nothing seen" counts.
        "annotations_all_species": ann_all_species,
        "source": source,
        # Kept unfiltered so a tab can show "x of y" without re-loading.
        "all_deployments": deployments,
        "all_annotations": annotations,
        "years": years,
        "reserves": reserves,
        "species": species or [],
        "regions": regions or [],
        "protections": protections or [],
        "ingested_only": ingested_only,
        "exclude_bad_deployments": exclude_bad_deployments,
    }
