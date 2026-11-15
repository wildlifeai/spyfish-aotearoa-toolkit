"""Species-presence prior per marine reserve, from expert annotations.

two_stage_classifier_plan.md §6. The classifier's species probabilities for a
box get multiplied by this prior for the reserve the drop was made in, so a
snapper at Akaroa needs more evidence than a snapper at Leigh. Three facts shape
the design:

* **Experts only.** ``annotated_by = 'expert'`` rows (legacy workbook and
  BIIGLE). Citsci picks from a short list and ML is what we are trying to
  correct, so neither can define what lives where.
* **Presence, not counts.** A species' rate at a reserve is the share of
  expert-reviewed drops it was recorded in. A school of 40 jack mackerel in one
  drop is one record. An expert-reviewed drop that saw nothing (the
  ``NULL_DEPLOYMENT`` sentinel) still counts as reviewed.
* **Small reserves borrow.** Te Angiangi has 17 reviewed drops and Okaihae has
  one; the rarefaction output shows a sample that size finds a fraction of a
  reserve's species. So every reserve's rate is shrunk toward a distance-weighted
  mean of its neighbours' rates: ``(n * own + k * neighbours) / (n + k)`` with
  ``k = site_prior.pseudo_drops``. Distances are between reserve centroids (the
  median deployment position); the kernel is ``exp(-d / length_scale_km)``.
  Reserves below ``donor_min_drops`` borrow but never lend.

Every rate is then floored (``site_prior.floor``) so no species is impossible
anywhere: an unexpected species should be flagged for review, not suppressed.

The outputs name species per reserve, which CLAUDE.md lists as sensitive. They
go under ``paths.site_prior`` (gitignored) and carry no coordinates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import (
    BIIGLE_EXTERNAL_ID_PREFIX,
    LEGACY_EXTERNAL_ID,
)
from spyfish.database.connection import connect_readonly

RESERVE = "reserve_code"
SPECIES = "scientific_name"
EARTH_RADIUS_KM = 6371.0


def reserve_code(drop_id: str) -> str:
    """The marine reserve code a DropID was made under: its first segment.

    Legacy workbook drops carry site names instead of site codes
    (``CRP_20030423_BUV_Liitle-waterfall_01``) and fail ``validate_drop_id``.
    Nothing here builds a path from the id, and the reserve segment is still
    well-formed, so the prefix alone is required. Anything without one raises.
    """
    code = config.get_reserve_code_from_drop(drop_id)
    if code is None:
        raise ValueError(f"DropID has no reserve prefix: {drop_id!r}")
    return code


def count_nonstandard_drop_ids(drop_ids: Iterable[str]) -> list[str]:
    """DropIDs that fail the configured pattern, for a one-line warning."""
    bad = []
    for d in set(drop_ids):
        try:
            config.validate_drop_id(d)
        except ValueError:
            bad.append(d)
    return sorted(bad)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ── loading ──────────────────────────────────────────────────────────────────


def load_expert_rows(annotations_db_path) -> pd.DataFrame:
    """Expert rows as ``drop_id, scientific_name, source`` with canonical names.

    ``source`` is ``legacy`` or ``biigle``. Names go through the species
    registry the same way ``add_annotations`` applies it on insert, so legacy
    rows written before canonicalisation (2026-11-03) line up with newer ones.
    Unknown names are kept as written and logged once with their row count.
    """
    from spyfish.config.species import species_registry

    with connect_readonly(annotations_db_path) as conn:
        df = pd.read_sql_query(
            "SELECT drop_id, scientific_name, external_id FROM annotations "
            "WHERE annotated_by = 'expert'",
            conn,
        )
    df["source"] = np.where(
        df["external_id"].fillna("").str.startswith(BIIGLE_EXTERNAL_ID_PREFIX),
        "biigle",
        np.where(df["external_id"] == LEGACY_EXTERNAL_ID, "legacy", "other"),
    )
    registry = species_registry()
    canonical: dict[str, str] = {}
    unknown: dict[str, int] = {}
    counts = df[SPECIES].value_counts(dropna=True)
    for name, n in counts.items():
        resolved, problem = registry.resolve_name(name)
        canonical[name] = resolved
        if problem:
            unknown[name] = int(n)
    if unknown:
        listed = ", ".join(f"{k} ({v})" for k, v in sorted(unknown.items()))
        logging.warning(
            f"Site prior: {len(unknown)} expert names not in the species registry, "
            f"kept as written: {listed}"
        )
    df[SPECIES] = df[SPECIES].map(lambda n: canonical.get(n, n))
    return df.drop(columns=["external_id"])


def load_reserve_centroids(pipeline_db_path) -> pd.DataFrame:
    """Median position per reserve code: ``reserve_code, lat, lon``.

    Deployments first (where drops actually landed); reserves with none in the
    pipeline DB (legacy-only reserves such as PKI) fall back to the planned
    positions of their sites, whose ids share the reserve prefix (``PKI_001``).
    The median, not the mean: a handful of deployments carry typo coordinates
    (AHE averages to the Tasman Sea) and the median ignores them.
    """
    with connect_readonly(pipeline_db_path) as conn:
        drops = pd.read_sql_query(
            "SELECT drop_id, latitude, longitude FROM deployments "
            "WHERE latitude IS NOT NULL AND longitude IS NOT NULL",
            conn,
        )
        sites = pd.read_sql_query(
            "SELECT site_id, latitude, longitude FROM sites "
            "WHERE latitude IS NOT NULL AND longitude IS NOT NULL",
            conn,
        )
    drops[RESERVE] = drops["drop_id"].map(reserve_code)
    sites[RESERVE] = sites["site_id"].map(reserve_code)
    from_drops = drops.groupby(RESERVE)[["latitude", "longitude"]].median()
    from_sites = sites.groupby(RESERVE)[["latitude", "longitude"]].median()
    fallback = from_sites[~from_sites.index.isin(from_drops.index)]
    if len(fallback):
        logging.info(
            f"Site prior: centroids from site positions for reserves without "
            f"deployments: {', '.join(sorted(fallback.index))}"
        )
    return (
        pd.concat([from_drops, fallback])
        .rename(columns={"latitude": "lat", "longitude": "lon"})
        .reset_index()
    )


# ── presence ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Presence:
    """Expert presence per reserve.

    ``table``: one row per (reserve, species) with ``drops_present`` and
    ``sources``. ``n_drops``: expert-reviewed drops per reserve, indexed by
    reserve code, including drops that recorded nothing.
    """

    table: pd.DataFrame
    n_drops: pd.Series


def presence_from_rows(rows: pd.DataFrame, bucket_names: Iterable[str]) -> Presence:
    """Collapse expert rows to drop-level presence per reserve.

    A drop counts as reviewed if it has any expert row, including the absence
    sentinel. The sentinel, nulls and bucket names (``fish``) are not species
    and never appear in ``table``.
    """
    rows = rows.copy()
    nonstandard = count_nonstandard_drop_ids(rows["drop_id"])
    if nonstandard:
        logging.warning(
            f"Site prior: {len(nonstandard)} expert drop ids do not match the DropID "
            f"pattern (legacy site names); kept by reserve prefix. "
            f"e.g. {', '.join(nonstandard[:3])}"
        )
    rows[RESERVE] = rows["drop_id"].map(reserve_code)
    n_drops = rows.groupby(RESERVE)["drop_id"].nunique().rename("n_drops")

    excluded = set(bucket_names) | {NULL_DEPLOYMENT}
    species_rows = rows[rows[SPECIES].notna() & ~rows[SPECIES].isin(excluded)]
    table = (
        species_rows.groupby([RESERVE, SPECIES])
        .agg(
            drops_present=("drop_id", "nunique"),
            sources=("source", lambda s: "+".join(sorted(set(s)))),
        )
        .reset_index()
    )
    return Presence(table=table, n_drops=n_drops)


def rate_matrix(presence: Presence) -> pd.DataFrame:
    """Reserve × species matrix of raw presence rates (share of reviewed drops)."""
    counts = presence.table.pivot(
        index=RESERVE, columns=SPECIES, values="drops_present"
    )
    counts = counts.reindex(index=presence.n_drops.index).fillna(0.0)
    # A reserve with no reviewed drops has no rate; 0 here, the prior is all borrowed.
    return counts.div(presence.n_drops.replace(0, np.nan), axis=0).fillna(0.0)


# ── neighbours ───────────────────────────────────────────────────────────────


def neighbour_weights(
    reserves: Iterable[str],
    centroids: pd.DataFrame,
    n_drops: pd.Series,
    length_scale_km: float,
    donor_min_drops: int,
) -> pd.DataFrame:
    """Row-normalised donor weights: ``weights.loc[r, j]`` is how much reserve
    ``r`` listens to reserve ``j``.

    Donors are reserves with at least ``donor_min_drops`` reviewed drops, never
    the reserve itself. Weight is ``exp(-d / length_scale_km)`` on the centroid
    distance. A reserve with no centroid (no deployment coordinates in the
    pipeline DB) gets equal weight on every donor and a warning: it borrows the
    national average rather than nothing.
    """
    reserves = list(reserves)
    donors = [j for j in reserves if n_drops.get(j, 0) >= donor_min_drops]
    if not donors:
        raise ValueError(
            f"No reserve has {donor_min_drops} expert-reviewed drops; nothing can lend."
        )
    pos = centroids.set_index(RESERVE)[["lat", "lon"]]
    weights = pd.DataFrame(0.0, index=reserves, columns=donors)
    for r in reserves:
        others = [j for j in donors if j != r]
        if not others:
            continue
        if r not in pos.index:
            logging.warning(
                f"Site prior: reserve {r} has no deployment coordinates; "
                f"borrowing equally from all {len(others)} donors."
            )
            weights.loc[r, others] = 1.0 / len(others)
            continue
        lat, lon = pos.loc[r, "lat"], pos.loc[r, "lon"]
        kernel = {}
        for j in others:
            if j not in pos.index:
                continue  # a donor without coordinates cannot be placed
            d = haversine_km(lat, lon, pos.loc[j, "lat"], pos.loc[j, "lon"])
            kernel[j] = math.exp(-d / length_scale_km)
        total = sum(kernel.values())
        if total <= 0:
            continue
        for j, w in kernel.items():
            weights.loc[r, j] = w / total
    return weights


def neighbour_distances(
    reserves: Iterable[str], centroids: pd.DataFrame
) -> pd.DataFrame:
    """Pairwise centroid distance in km (NaN where a side has no centroid)."""
    reserves = list(reserves)
    pos = centroids.set_index(RESERVE)[["lat", "lon"]]
    out = pd.DataFrame(np.nan, index=reserves, columns=reserves)
    for r in reserves:
        if r not in pos.index:
            continue
        for j in reserves:
            if j not in pos.index:
                continue
            out.loc[r, j] = haversine_km(
                pos.loc[r, "lat"],
                pos.loc[r, "lon"],
                pos.loc[j, "lat"],
                pos.loc[j, "lon"],
            )
    return out


# ── the prior ────────────────────────────────────────────────────────────────


def shrink(
    rates: pd.DataFrame,
    n_drops: pd.Series,
    weights: pd.DataFrame,
    pseudo_drops: float,
    floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shrink each reserve's rates toward its neighbours'.

    Returns ``(neighbour_rates, prior)``, both reserve × species. ``prior`` is
    ``(n * own + k * neighbours) / (n + k)`` floored at ``floor``. A species
    seen nowhere in the neighbourhood sits at the floor.
    """
    rates = rates.reindex(index=weights.index).fillna(0.0)
    donor_rates = rates.reindex(index=weights.columns).fillna(0.0)
    # Broadcast-and-sum rather than ``@``: numpy's Accelerate matmul on macOS
    # emits spurious divide-by-zero warnings on these tiny matrices.
    product = (weights.values[:, :, None] * donor_rates.values[None, :, :]).sum(axis=1)
    neighbour = pd.DataFrame(product, index=weights.index, columns=rates.columns)
    n = n_drops.reindex(weights.index).fillna(0.0).astype(float)
    own_share = (n / (n + pseudo_drops)).values[:, None]
    prior = own_share * rates.values + (1 - own_share) * neighbour.values
    prior = pd.DataFrame(prior, index=weights.index, columns=rates.columns).clip(
        lower=floor
    )
    return neighbour, prior


def rarefaction(presence: Presence, sample_sizes: Iterable[int]) -> pd.DataFrame:
    """Expected number of species found in ``n`` of a reserve's reviewed drops.

    Hurlbert's rarefaction: ``E[S_n] = sum_s 1 - C(N - m_s, n) / C(N, n)`` where
    ``N`` is the reserve's reviewed drops and ``m_s`` the drops species ``s``
    appears in. Exact, no resampling. Rows only where ``n <= N``.
    ``share_of_observed`` is ``E[S_n]`` over the species count at ``N``, the
    answer to "how much of a big reserve's list would a small reserve's effort
    have found?".
    """
    rows = []
    for r, group in presence.table.groupby(RESERVE):
        N = int(presence.n_drops[r])
        m = group["drops_present"].astype(int).values
        total = len(m)
        denom_cache = {}
        for n in sample_sizes:
            if n > N or total == 0:
                continue
            denom = denom_cache.setdefault(n, math.comb(N, n))
            expected = sum(1 - math.comb(N - ms, n) / denom for ms in m)
            rows.append(
                {
                    RESERVE: r,
                    "n_drops_total": N,
                    "n_drops_sample": n,
                    "species_observed": total,
                    "expected_species": round(expected, 2),
                    "share_of_observed": round(expected / total, 3),
                }
            )
    return pd.DataFrame(rows)


def neighbour_skill(
    rates: pd.DataFrame, neighbour: pd.DataFrame, n_drops: pd.Series, min_drops: int
) -> pd.Series:
    """Spearman correlation between a reserve's own rates and its neighbours'
    prediction, over all species, for reserves with at least ``min_drops``.

    A leave-self-out check that geography carries information: the neighbour
    rate never uses the reserve's own data, so a high value means the
    neighbourhood would have ranked this reserve's species well before anyone
    reviewed a drop there.
    """
    out = {}
    for r in rates.index:
        if n_drops.get(r, 0) < min_drops:
            continue
        own, nb = rates.loc[r], neighbour.loc[r]
        if own.nunique() < 2 or nb.nunique() < 2:
            continue
        out[r] = round(float(own.corr(nb, method="spearman")), 3)
    return pd.Series(out, name="neighbour_spearman")


# ── assembly ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SitePrior:
    long: pd.DataFrame  # one row per (reserve, species)
    summary: pd.DataFrame  # one row per reserve
    rarefaction: pd.DataFrame
    weights: pd.DataFrame  # long: reserve, donor, weight, distance_km


def build_site_prior(
    rows: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    length_scale_km: Optional[float] = None,
    pseudo_drops: Optional[float] = None,
    donor_min_drops: Optional[int] = None,
    floor: Optional[float] = None,
    rarefaction_sizes: Optional[Iterable[int]] = None,
    bucket_names: Optional[Iterable[str]] = None,
    common_names: Optional[dict[str, str]] = None,
) -> SitePrior:
    """Everything from expert rows and centroids to the tables written to disk.

    Keyword arguments default to ``config.site_prior_*``; tests pass them.
    """
    length_scale_km = (
        config.site_prior_length_scale_km
        if length_scale_km is None
        else length_scale_km
    )
    pseudo_drops = (
        config.site_prior_pseudo_drops if pseudo_drops is None else pseudo_drops
    )
    donor_min_drops = (
        config.site_prior_donor_min_drops
        if donor_min_drops is None
        else donor_min_drops
    )
    floor = config.site_prior_floor if floor is None else floor
    rarefaction_sizes = list(
        config.site_prior_rarefaction_sample_sizes
        if rarefaction_sizes is None
        else rarefaction_sizes
    )
    bucket_names = list(
        config.site_prior_bucket_names if bucket_names is None else bucket_names
    )
    common_names = common_names or {}

    presence = presence_from_rows(rows, bucket_names)
    # A reserve with a position but no expert-reviewed drop (ORA, NOI) still
    # gets a prior: n = 0, so it is entirely its neighbours'. Without this the
    # reserves that most need a prior would have none.
    unreviewed = sorted(set(centroids[RESERVE]) - set(presence.n_drops.index))
    if unreviewed:
        logging.info(
            f"Site prior: {len(unreviewed)} reserves have no expert-reviewed drops and "
            f"borrow everything: {', '.join(unreviewed)}"
        )
        n_drops = (
            presence.n_drops.reindex(list(presence.n_drops.index) + unreviewed)
            .fillna(0)
            .astype(int)
        )
        presence = Presence(table=presence.table, n_drops=n_drops)
    rates = rate_matrix(presence)
    reserves = list(presence.n_drops.index)
    weights = neighbour_weights(
        reserves, centroids, presence.n_drops, length_scale_km, donor_min_drops
    )
    neighbour, prior = shrink(rates, presence.n_drops, weights, pseudo_drops, floor)
    prior_weight = prior.div(prior.sum(axis=1), axis=0)

    long = (
        pd.concat(
            {
                "raw_rate": rates.stack(),
                "neighbour_rate": neighbour.stack(),
                "presence_prob": prior.stack(),
                "prior_weight": prior_weight.stack(),
            },
            axis=1,
        )
        .rename_axis([RESERVE, SPECIES])
        .reset_index()
    )
    long = long.merge(presence.table, on=[RESERVE, SPECIES], how="left")
    long["drops_present"] = long["drops_present"].fillna(0).astype(int)
    long["sources"] = long["sources"].fillna("")
    long["n_drops"] = long[RESERVE].map(presence.n_drops).astype(int)
    long["borrowed_share"] = (pseudo_drops / (long["n_drops"] + pseudo_drops)).round(3)
    long["common_name"] = long[SPECIES].map(lambda s: common_names.get(s, ""))
    for col in ("raw_rate", "neighbour_rate", "presence_prob"):
        long[col] = long[col].round(4)
    long["prior_weight"] = long["prior_weight"].round(5)
    long = long[
        [
            RESERVE,
            SPECIES,
            "common_name",
            "n_drops",
            "drops_present",
            "sources",
            "raw_rate",
            "neighbour_rate",
            "presence_prob",
            "prior_weight",
            "borrowed_share",
        ]
    ].sort_values([RESERVE, "presence_prob", SPECIES], ascending=[True, False, True])

    distances = neighbour_distances(reserves, centroids)
    weights_long = (
        weights.stack().rename("weight").rename_axis([RESERVE, "donor"]).reset_index()
    )
    weights_long = weights_long[weights_long["weight"] > 0]
    weights_long["distance_km"] = [
        None if pd.isna(distances.loc[r, j]) else round(float(distances.loc[r, j]), 1)
        for r, j in zip(weights_long[RESERVE], weights_long["donor"])
    ]
    weights_long["weight"] = weights_long["weight"].round(4)
    weights_long = weights_long.sort_values(
        [RESERVE, "weight"], ascending=[True, False]
    )

    skill = neighbour_skill(rates, neighbour, presence.n_drops, donor_min_drops)
    above_floor = (prior > floor).sum(axis=1)
    top_donors = pd.Series(
        {
            r: ", ".join(
                f"{d}:{w:.2f}" for d, w in zip(g["donor"][:3], g["weight"][:3])
            )
            for r, g in weights_long.groupby(RESERVE)
        },
        name="top_donors",
    )
    summary = pd.DataFrame(
        {
            "n_drops": presence.n_drops,
            "species_observed": (rates > 0).sum(axis=1),
            "species_above_floor": above_floor,
            "borrowed_share": (pseudo_drops / (presence.n_drops + pseudo_drops)).round(
                3
            ),
            "is_donor": presence.n_drops >= donor_min_drops,
            "has_centroid": presence.n_drops.index.isin(centroids[RESERVE]),
        }
    )
    summary = summary.join(skill).join(top_donors).rename_axis(RESERVE).reset_index()
    summary = summary.sort_values("n_drops", ascending=False)

    return SitePrior(
        long=long,
        summary=summary,
        rarefaction=rarefaction(presence, rarefaction_sizes),
        weights=weights_long,
    )


def write_site_prior(result: SitePrior, out_dir) -> dict[str, Path]:
    """Write the four tables under ``out_dir`` and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "species_presence_by_reserve": out_dir / "species_presence_by_reserve.csv",
        "reserve_summary": out_dir / "reserve_summary.csv",
        "rarefaction": out_dir / "rarefaction.csv",
        "neighbour_weights": out_dir / "neighbour_weights.csv",
    }
    result.long.to_csv(paths["species_presence_by_reserve"], index=False)
    result.summary.to_csv(paths["reserve_summary"], index=False)
    result.rarefaction.to_csv(paths["rarefaction"], index=False)
    result.weights.to_csv(paths["neighbour_weights"], index=False)
    for name, p in paths.items():
        logging.info(f"Site prior: wrote {name} -> {p}")
    return paths
