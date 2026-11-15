"""Site prior (`spyfish/ml/classifier/site_prior.py`): the shrinkage arithmetic
and the drop-level presence rules, on hand-built rows.

Pinned here because a wrong prior is invisible downstream: it just makes some
species a little more or less likely, and nothing raises.
"""

import math

import pandas as pd
import pytest

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.ml.classifier import site_prior as sp

SNAPPER = "Chrysophrys auratus"
BLUE_COD = "Parapercis colias"
LOBSTER = "Jasus edwardsii"

# Three reserves: a big northern one, a big southern one, and a tiny one next
# to the southern one. Realistic DropIDs, as CLAUDE.md asks.
BIG_N = "CRP"
BIG_S = "BNP"
TINY = "OKA"


def _drop(code, i):
    return f"{code}_20240124_BUV_{code}_{i:03d}_01"


def _rows():
    rows = []
    # CRP: 10 drops, snapper in all, blue cod in 2
    for i in range(1, 11):
        rows.append((_drop(BIG_N, i), SNAPPER, "legacy"))
        if i <= 2:
            rows.append((_drop(BIG_N, i), BLUE_COD, "legacy"))
    # BNP: 10 drops, blue cod in 8, lobster in 5, one drop that saw nothing,
    # one drop with a duplicate species row and a 'fish' bucket row
    for i in range(1, 9):
        rows.append((_drop(BIG_S, i), BLUE_COD, "legacy"))
    for i in range(1, 6):
        rows.append((_drop(BIG_S, i), LOBSTER, "biigle"))
    rows.append((_drop(BIG_S, 9), NULL_DEPLOYMENT, "legacy"))
    rows.append((_drop(BIG_S, 10), BLUE_COD, "legacy"))
    rows.append((_drop(BIG_S, 10), BLUE_COD, "biigle"))
    rows.append((_drop(BIG_S, 10), "fish", "biigle"))
    # OKA: 1 drop, blue cod only
    rows.append((_drop(TINY, 1), BLUE_COD, "biigle"))
    return pd.DataFrame(rows, columns=["drop_id", "scientific_name", "source"])


def _centroids():
    return pd.DataFrame(
        {
            "reserve_code": [BIG_N, BIG_S, TINY],
            "lat": [-36.3, -43.9, -45.9],
            "lon": [174.8, 173.0, 170.4],
        }
    )


def test_presence_counts_drops_not_rows():
    presence = sp.presence_from_rows(_rows(), bucket_names=["fish"])
    n = presence.n_drops
    assert n[BIG_N] == 10 and n[BIG_S] == 10 and n[TINY] == 1  # absence drop counts
    t = presence.table.set_index(["reserve_code", "scientific_name"])
    assert t.loc[(BIG_S, BLUE_COD), "drops_present"] == 9  # dup rows = one drop
    assert t.loc[(BIG_S, BLUE_COD), "sources"] == "biigle+legacy"
    assert "fish" not in presence.table["scientific_name"].values
    assert NULL_DEPLOYMENT not in presence.table["scientific_name"].values


def test_rates_are_share_of_reviewed_drops():
    presence = sp.presence_from_rows(_rows(), bucket_names=["fish"])
    rates = sp.rate_matrix(presence)
    assert rates.loc[BIG_N, SNAPPER] == 1.0
    assert rates.loc[BIG_S, BLUE_COD] == pytest.approx(0.9)
    assert rates.loc[BIG_S, SNAPPER] == 0.0


def test_tiny_reserve_borrows_mostly_from_its_neighbour():
    presence = sp.presence_from_rows(_rows(), bucket_names=["fish"])
    w = sp.neighbour_weights(
        presence.n_drops.index,
        _centroids(),
        presence.n_drops,
        length_scale_km=250,
        donor_min_drops=10,
    )
    assert TINY not in w.columns  # 1 drop: borrows, never lends
    assert w.loc[TINY].sum() == pytest.approx(1.0)
    assert w.loc[TINY, BIG_S] > 0.9  # BNP ~270 km away, CRP ~1,100 km
    assert w.loc[BIG_S, BIG_S] == 0.0  # never listens to itself


def test_shrinkage_arithmetic_and_floor():
    presence = sp.presence_from_rows(_rows(), bucket_names=["fish"])
    rates = sp.rate_matrix(presence)
    w = sp.neighbour_weights(
        presence.n_drops.index,
        _centroids(),
        presence.n_drops,
        length_scale_km=250,
        donor_min_drops=10,
    )
    neighbour, prior = sp.shrink(rates, presence.n_drops, w, pseudo_drops=4, floor=0.01)
    # OKA: n=1, k=4 -> 1/5 own + 4/5 neighbours
    expected = 0.2 * 1.0 + 0.8 * neighbour.loc[TINY, BLUE_COD]
    assert prior.loc[TINY, BLUE_COD] == pytest.approx(expected)
    # snapper never seen in the south; BNP's neighbour estimate comes from CRP
    # but at 1,100 km with L=250 that weight is tiny, so the prior sits near the floor
    assert prior.loc[BIG_S, SNAPPER] >= 0.01
    assert prior.loc[TINY, SNAPPER] >= 0.01
    # a big reserve keeps most of its own signal
    assert prior.loc[BIG_N, SNAPPER] == pytest.approx(
        10 / 14 * 1.0 + 4 / 14 * neighbour.loc[BIG_N, SNAPPER]
    )


def test_rarefaction_is_exact_hurlbert():
    presence = sp.presence_from_rows(_rows(), bucket_names=["fish"])
    r = sp.rarefaction(presence, [1, 10, 50])
    bnp = r[r["reserve_code"] == BIG_S].set_index("n_drops_sample")
    assert 50 not in bnp.index  # cannot sample more drops than reviewed
    assert bnp.loc[10, "expected_species"] == pytest.approx(2.0)  # all of them
    # one drop: P(blue cod) + P(lobster) = 9/10 + 5/10
    assert bnp.loc[1, "expected_species"] == pytest.approx(1.4)


def test_haversine_sanity():
    assert sp.haversine_km(0, 0, 0, 0) == 0
    # one degree of latitude is ~111 km
    assert math.isclose(sp.haversine_km(-40, 174, -41, 174), 111.2, rel_tol=0.01)


def test_build_site_prior_end_to_end_tables():
    result = sp.build_site_prior(
        _rows(),
        _centroids(),
        length_scale_km=250,
        pseudo_drops=4,
        donor_min_drops=10,
        floor=0.01,
        rarefaction_sizes=[1, 5],
        bucket_names=["fish"],
        common_names={SNAPPER: "Snapper"},
    )
    long = result.long
    assert set(long["reserve_code"]) == {BIG_N, BIG_S, TINY}
    per_reserve = long.groupby("reserve_code")["prior_weight"].sum()
    assert all(abs(v - 1.0) < 1e-4 for v in per_reserve)  # rounded to 5 dp
    assert (
        long.loc[long["scientific_name"] == SNAPPER, "common_name"].eq("Snapper").all()
    )
    summary = result.summary.set_index("reserve_code")
    assert summary.loc[TINY, "borrowed_share"] == pytest.approx(0.8)
    assert not summary.loc[TINY, "is_donor"] and summary.loc[BIG_S, "is_donor"]
    assert summary.loc[TINY, "top_donors"].startswith(BIG_S)
    assert "distance_km" in result.weights.columns
    assert "lat" not in result.long.columns and "lat" not in result.summary.columns


def test_reserve_with_position_but_no_expert_drops_borrows_everything():
    centroids = pd.concat(
        [
            _centroids(),
            pd.DataFrame({"reserve_code": ["ORA"], "lat": [-45.9], "lon": [170.6]}),
        ]
    )
    result = sp.build_site_prior(
        _rows(),
        centroids,
        length_scale_km=250,
        pseudo_drops=4,
        donor_min_drops=10,
        floor=0.01,
        rarefaction_sizes=[1],
        bucket_names=["fish"],
    )
    ora = result.long[result.long["reserve_code"] == "ORA"].set_index("scientific_name")
    assert (ora["n_drops"] == 0).all() and (ora["raw_rate"] == 0).all()
    assert (ora["borrowed_share"] == 1.0).all()
    assert ora.loc[BLUE_COD, "presence_prob"] == pytest.approx(
        ora.loc[BLUE_COD, "neighbour_rate"]
    )
    summary = result.summary.set_index("reserve_code")
    assert summary.loc["ORA", "n_drops"] == 0 and not summary.loc["ORA", "is_donor"]
