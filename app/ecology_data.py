"""Shared ecology data layer for the Streamlit app.

Single source of truth for loading + enriching annotation data. Used by the
Experiments page (``pages/🧪_Experiments.py``) and the DOC report's data layer
(``doc_report/data.py`` and ``doc_report/species_search.py``). Page-specific viz helpers
stay in their own pages, only the page-agnostic loaders and the drop_id/site
enrichment live here.
"""

import math

import pandas as pd
import streamlit as st
from utils import CACHE_TTL_SECONDS

from spyfish.config.base import (
    NULL_DEPLOYMENT,
    CitSciStatus,
    ExpertStatus,
    IngestStatus,
    MlStatus,
)
from spyfish.config.species import species_registry
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.connection import connect_readonly

# Plausibility floor for survey dates. Looser than the first real Spyfish survey
# (2011) so legacy and partner data with earlier dates is not discarded.
EARLIEST_SURVEY_YEAR = 2000


# Source precedence used wherever we collapse to a single "best" annotation per
# (drop, species): expert beats expert_training beats citsci beats ml.
#
# None of these review a whole deployment, and only ML ever does: it runs over
# every frame, and everything downstream adjudicates the moments ML surfaced.
# Volunteers see ~100 clips chosen from the ML MaxN CSV, not the full 30 minutes
# (`sample_all_clips` would change that and is off). So the ordering is not about
# coverage, all four look at the same candidate set, it is about whose
# judgement to trust on it, which puts both expert sources above citsci.
#
# The corollary is that every source inherits ML's recall ceiling: a fish the
# model never detected reaches no volunteer and no expert, because it was never
# put in front of them.
#
# `expert_training` sits below `expert` because its frames were picked by ML
# alone, without the volunteer signal folded in. (Frames from a pre-`--use-ml`
# run were sampled blind, with no selection pass behind them at all, those
# deserve less trust than this ranking gives them.)
#
# Anything unranked falls to 99 in `best_per_drop_species`, so a new source must
# be added here or it silently sorts below ml.
SOURCE_PRIORITY = {"expert": 0, "expert_training": 1, "citsci": 2, "ml": 3}

# Both are expert judgement; `expert_training` only differs in how its frames
# were picked (see SOURCE_PRIORITY above). Everywhere a view shows or filters
# "Expert", both belong, without this tuple, expert_training rows fall through
# every `== "expert"` check and get displayed and filtered as ML.
EXPERT_SOURCES = ("expert", "expert_training")


def source_bucket(annotated_by: pd.Series) -> pd.Series:
    """Collapse `annotated_by` to the three user-facing buckets.

    ML rows carry the model name, so ML is "neither expert nor citsci" rather
    than a literal value. Every view that labels, filters or counts by source
    must go through this one function, a view that hand-rolls the mapping will
    drift (the old species-search tiles looked up the literal key "ml" and
    showed 0 forever).
    """
    return annotated_by.map(
        lambda s: (
            "Expert" if s in EXPERT_SOURCES else ("CitSci" if s == "citsci" else "ML")
        )
    )


# ── Protection grouping ──────────────────────────────────────────────────────
#
# One definition of "protected", from config.protected_statuses. There used to
# be two: a keyword heuristic in the Species view that bucketed mātaitai /
# taiāpure as Protected, while the MPA view's caption said the opposite. Same
# deployment, opposite sides of the headline comparison, on the same page.

PROTECTED = "Protected"
UNPROTECTED = "Unprotected"
# Neither: a partial or unclear regime. Its own group rather than a side,
# because folding it into either one answers the reserve question with
# deployments that cannot answer it.
OTHER_PROTECTION = "Other"


def protection_group(status: pd.Series) -> pd.Series:
    """Protected / Unprotected / Other / None from `protection_status`.

    Three groups, not two. `config.protected_statuses` and
    `config.unprotected_statuses` each name exactly the statuses that mean what
    they say; everything else (High Protection Area, Type II MPA, taiapure,
    mataitai, Fisheries Act closures, seafloor protection, "Other") is a
    partial or unclear regime and becomes **Other**.

    That third group used to be split across the two sides: the config counted
    High Protection Area and Type II MPA as protected while the MPA view's own
    substring classifier read them as outside, so one deployment sat on
    opposite sides of the comparison depending on which chart was open. Naming
    both ends and leaving the remainder as Other means neither side can absorb
    a deployment nobody has decided about.

    Missing or "unknown" metadata maps to None, still separate from Other: one
    is "we know it is partial", the other is "we do not know".
    """
    protected = set(config.protected_statuses)
    unprotected = set(config.unprotected_statuses)

    def bucket(value):
        if pd.isna(value) or str(value).strip().lower() in ("", "unknown"):
            return None
        if value in protected:
            return PROTECTED
        if value in unprotected:
            return UNPROTECTED
        return OTHER_PROTECTION

    return status.map(bucket)


# ── Diversity indices ────────────────────────────────────────────────────────
#
# Pure functions over a vector of per-species counts (MaxN). They live here
# rather than in a page because two pages needed them and each had grown its own
# copy, the Experiments page called its Pielou's evenness "Evenness", the
# Mussel Insights page called the same formula something else, and neither could
# be changed without silently diverging from the other.
#
# All three take counts in any order; zeros and NaNs are dropped. None of them
# know where the counts came from, so they work equally on a deployment, a site
# or a whole reserve, the caller decides what to aggregate first.
#
# What each one is for:
#   shannon , rewards having many species AND having them in balance. Its
#              ceiling is ln(S), so it is NOT comparable between datasets with
#              different species pools.
#   simpson , probability two individuals drawn at random are different
#              species. Less sensitive to rare species than Shannon.
#   pielou  . Shannon divided by its own maximum, so bounded 0-1 and therefore
#              comparable across datasets. 1 = perfectly even, near 0 = one
#              species dominates.


def _clean_counts(counts) -> list[float]:
    """Positive counts only, as plain floats."""
    return [float(c) for c in counts if pd.notna(c) and float(c) > 0]


def shannon(counts) -> float:
    """Shannon diversity H' = -sum(p * ln p)."""
    vals = _clean_counts(counts)
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log(c / total) for c in vals))


def simpson(counts) -> float:
    """Simpson's index in its 1-D form: higher = more diverse."""
    vals = _clean_counts(counts)
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float(1 - sum((c / total) ** 2 for c in vals))


def pielou(counts) -> float:
    """Pielou's evenness J' = H' / ln(S). Bounded 0-1.

    Zero when only one species is present: with nothing to be uneven against,
    evenness is undefined, and 0 is the conventional reported value.
    """
    vals = _clean_counts(counts)
    if len(vals) < 2:
        return 0.0
    return float(shannon(vals) / math.log(len(vals)))


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_maxn() -> pd.DataFrame:
    df = AnnotationDatabaseManager().get_maxn_summary()
    # The DB speaks NULL_DEPLOYMENT ("reviewed, nothing seen"); app frames
    # speak NaN, which every notna/groupby path here already treats as
    # "no species" while keeping the row. NaN is unambiguous in a frame: a
    # never-reviewed source has no rows at all, so isna() means exactly the
    # reviewed-empty deployments.
    df.loc[df["scientific_name"] == NULL_DEPLOYMENT, "scientific_name"] = pd.NA
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def search_species_annotations(scientific_name: str) -> pd.DataFrame:
    """Every raw annotation row for one species, cached per species.

    Unlike `get_maxn_summary()` (peak per drop only), this returns every
    time-window observation so the species-search view can list every
    timestamp the species was seen. Cached by argument so each species
    selection is fetched once per session.
    """
    with connect_readonly(config.annotations_db_path) as conn:
        return pd.read_sql(
            "SELECT drop_id, scientific_name, time_of_max, time_of_max_seconds, "
            "max_interval, annotated_by, confidence_agreement, external_id "
            "FROM annotations WHERE scientific_name = ? "
            "ORDER BY drop_id, time_of_max_seconds",
            conn,
            params=(scientific_name,),
        )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sites(include_coords: bool = False) -> pd.DataFrame:
    """Site metadata. Coordinates are opt-in and must stay behind check_password().

    `region` is the only geographic grouping the pipeline has, it cannot be
    derived from a DropID or a reserve name, so it comes straight from
    `BUV Survey Sites.csv`.
    """
    columns = "site_id, site_name, protection_status, region, link_to_marine_reserve"
    if include_coords:
        columns += ", latitude, longitude"
    with connect_readonly(config.db_path) as conn:
        return pd.read_sql(f"SELECT {columns} FROM sites", conn)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_effort() -> pd.DataFrame:
    """One row per ANALYSED deployment, the denominator for any mean abundance.

    An abundance index must divide by the deployments someone actually looked at.
    Two wrong denominators to avoid:

    * Dividing by deployments where the species was *seen* discards every zero and
      overstates abundance, worst for rare species, exactly where it matters.
    * Dividing by every ingested deployment treats "nobody has analysed this yet"
      as "no fish here". On this database that is the difference between 114 and
      2370, a twentyfold error, all of it deflating the mean.

    "Analysed" is the pipeline's own record that results exist for a drop, which
    also captures the honest zero: a deployment that was reviewed and had nothing
    in it has no annotation rows but should still count as effort.
    """
    with connect_readonly(config.db_path) as conn:
        by_status = pd.read_sql(
            """
            SELECT drop_id FROM deployments
            WHERE ingest_status = ?
              AND (
                    ml_status = ?
                 OR citsci_status = ?
                 OR expert_status IN (?, ?)
              )
            """,
            conn,
            params=(
                IngestStatus.OK,
                MlStatus.COMPLETE,
                CitSciStatus.COMPLETE,
                ExpertStatus.UPLOADED,
                ExpertStatus.COMPLETE,
            ),
        )
    # Union with drops that actually hold annotations. The two disagree slightly
    # (114 vs 116 here) because section statuses drift from the annotation data.
    # Status alone would put a drop's sightings in the numerator while leaving it
    # out of the denominator, which surfaces as a frequency above 100%.
    with connect_readonly(config.annotations_db_path) as conn:
        annotated = pd.read_sql("SELECT DISTINCT drop_id FROM annotations", conn)
    return pd.concat([by_status, annotated]).drop_duplicates("drop_id")


@st.cache_data(ttl=3600)
def load_common_names() -> dict:
    """scientific_name → 'Common name (Scientific name)' via the species registry.

    Real species only, bucket classes (fish/bait) are excluded, and the dict
    is empty when class_map.json is missing.
    """
    return species_registry().common_names()


# ── Frame enrichment ─────────────────────────────────────────────────────────
#
# Three functions rather than one, so a caller asks for exactly the columns it
# needs: a deployments frame carries no species, and should not have to invent
# a null `scientific_name` just to get its drop_id parsed.
#
# Use these rather than parsing drop_ids in a page. Two pages parsing their own
# will drift, and the same number then disagrees between views.


def _this_year() -> int:
    """Upper bound for a plausible survey year. A survey cannot be in the future."""
    from datetime import date

    return date.today().year


def unparseable_dates(df: pd.DataFrame) -> pd.DataFrame:
    """DropIDs whose date could not be read, after `add_drop_id_columns`.

     Either the field is not a date at all, or it is outside the plausible range
    , most often a date typed DDMMYYYY instead of YYYYMMDD.
    """
    return df[df["survey_year"].isna()]


def add_drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse DropID into reserve_code, survey_id, survey_date/year and site_id.

    DropID is `{Reserve}_{YYYYMMDD}_BUV_{SiteCode}_{Replicate}`, so every one of
    these is derivable without touching another table.
    """
    parts = df["drop_id"].str.split("_", expand=True)
    df = df.copy()
    df["reserve_code"] = parts.get(0, pd.Series("", index=df.index)).fillna("")
    df["survey_id"] = df["drop_id"].str.slice(0, 16)
    df["survey_date"] = pd.to_datetime(
        parts.get(1, pd.Series("", index=df.index)),
        format="%Y%m%d",
        errors="coerce",
    )
    df["survey_year"] = df["survey_date"].dt.year
    # A DropID whose date was typed DDMMYYYY still parses, `23042026` reads as
    # the year 2304, and one such deployment stretches every year axis in the
    # app by three centuries. Anything outside a plausible range is treated as
    # having no date rather than being clamped into a year it might not belong
    # to. `unparseable_dates()` reports them so they can be fixed upstream,
    # which is the only real fix: these deployments pass ingest today.
    plausible = df["survey_year"].between(EARLIEST_SURVEY_YEAR, _this_year())
    implausible = ~plausible.fillna(False)
    # NaT belongs only in the datetime column, assigning it into the float
    # survey_year column coerces the whole column to object dtype.
    df.loc[implausible, "survey_date"] = pd.NaT
    df.loc[implausible, "survey_year"] = float("nan")
    p3 = parts.get(3, pd.Series("", index=df.index)).fillna("")
    p4 = parts.get(4, pd.Series("", index=df.index)).fillna("")
    df["site_id"] = p3 + "_" + p4
    df["site_id"] = df["site_id"].replace("_", pd.NA)
    return df


def join_site_metadata(df: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Join the sites table on site_id and tidy the labels it brings.

    Every column the `sites` frame carries is joined through, so passing
    `load_sites(include_coords=True)` brings coordinates along. Callers are
    responsible for gating those.

    Requires `site_id`, so run `add_drop_id_columns` first.
    """
    df = df.merge(sites, on="site_id", how="left")
    # site_name is ~38% empty upstream and many non-empty values are junk integers
    # ("1"), so the site_id is the more informative label in both cases.
    numeric_name = df["site_name"].astype(str).str.fullmatch(r"\s*\d+\s*")
    df["site_name"] = df["site_name"].mask(numeric_name).fillna(df["site_id"])
    df["protection_status"] = (
        df["protection_status"].replace("", pd.NA).fillna("unknown")
    )
    df["region"] = df["region"].replace("", pd.NA).fillna("unknown")
    return df


def add_display_names(df: pd.DataFrame, common_names: dict) -> pd.DataFrame:
    """Attach `display_name`, 'Common name (Scientific name)' where known.

    Species-only: frames without a `scientific_name` column are returned
    untouched rather than given an empty one, so a deployments frame does not
    have to pretend to hold species.
    """
    if "scientific_name" not in df.columns:
        return df
    df = df.copy()
    df["display_name"] = df["scientific_name"].map(
        lambda s: common_names.get(s, s) if pd.notna(s) else s
    )
    return df
