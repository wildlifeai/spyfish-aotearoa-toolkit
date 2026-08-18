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
    SOURCE_PRIORITY,
    add_display_names,
    add_drop_id_columns,
    join_site_metadata,
    load_common_names,
    load_sites,
    source_bucket,
)
from utils import CACHE_TTL_SECONDS

from spyfish.config.wrapper import config
from spyfish.database.connection import connect_readonly


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_deployments() -> pd.DataFrame:
    """Every deployment row, with drop_id parsed and site metadata joined.

    Coordinates are deliberately NOT requested here. A tab that needs a map asks
    for them separately behind ``check_password()``, see the note in
    ``ecology_data.load_sites``.
    """
    with connect_readonly(config.db_path) as conn:
        df = pd.read_sql(
            "SELECT drop_id, video_presence, ingest_status, ml_status, "
            "citsci_status, expert_status, reporting_status, is_bad_deployment, "
            "ml_annotations, citsci_annotations, expert_annotations, depth "
            "FROM deployments",
            conn,
        )
    # Deployments hold no species, so only the two identity steps apply, no
    # invented `scientific_name` column needed.
    return join_site_metadata(add_drop_id_columns(df), load_sites())


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_annotations() -> pd.DataFrame:
    """One row per species observation, enriched the same way as deployments."""
    with connect_readonly(config.annotations_db_path) as conn:
        df = pd.read_sql(
            "SELECT drop_id, scientific_name, max_interval, annotated_by, "
            "time_of_max_seconds, confidence_agreement FROM annotations",
            conn,
        )
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

    `annotated_by` holds a model name for ML rows, so ML is "not expert and not
    citsci" rather than a literal value (`source_bucket` is the one mapping).

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
        ranked = ann.assign(
            _rank=ann["annotated_by"].map(SOURCE_PRIORITY).fillna(SOURCE_PRIORITY["ml"])
        )
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
    out = ann.groupby(keys)[INTERVAL_COUNT].max().reset_index(name="maxn")
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


def protection_rank(status: str) -> int:
    """Coarse protection ranking: 0 fully protected, 3 unknown.

    The Experiments page's classifier, moved here whole when its charts were
    copied into the report. Deliberately not merged with
    `ecology_data.protection_group`: that one is config-driven and answers
    protected/unprotected/unknown, while this ranks four levels by substring
    and is what the ported charts sort and split on. Merging them would change
    what those charts show, which is a decision to take on purpose rather than
    as a side effect of moving files.
    """
    status = (status or "").lower()
    if any(k in status for k in ("reserve", "inside")):
        return 0
    if any(k in status for k in ("partial", "buffer")):
        return 1
    if any(k in status for k in ("fished", "outside", "unprotected")):
        return 2
    return 3


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
    metadata back on, keeps the columns that describe *that* interval —
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


def build_context(
    deployments: pd.DataFrame,
    annotations: pd.DataFrame,
    years: tuple,
    reserves: list,
    ingested_only: bool,
    source: str = BEST_AVAILABLE,
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
        dep = dep[matches_reserves(dep["link_to_marine_reserve"], reserves)]
        ann = ann[matches_reserves(ann["link_to_marine_reserve"], reserves)]
    if ingested_only:
        dep = dep[dep["ingest_status"] == "ok"]
        ann = ann[ann["drop_id"].isin(dep["drop_id"])]

    # Year/reserve-filtered but source-UNfiltered. The source-comparison panels
    # need this frame: under any source filter, one side of an ML-vs-expert
    # comparison has already been thrown away before the panel runs.
    ann_all_sources = ann
    ann = _apply_source(ann, source)

    return {
        "deployments": dep,
        "annotations": ann,
        "annotations_all_sources": ann_all_sources,
        "source": source,
        # Kept unfiltered so a tab can show "x of y" without re-loading.
        "all_deployments": deployments,
        "all_annotations": annotations,
        "years": years,
        "reserves": reserves,
        "ingested_only": ingested_only,
    }
