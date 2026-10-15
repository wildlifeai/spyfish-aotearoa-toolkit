"""Propose a train/val/test split of BIIGLE training drops, rarest class first.

Input:  process_files/training_lists/biigle_label_by_drop.csv (scripts/biigle_label_inventory.py)
Frozen: process_files/training_lists/test_drops.txt (training.test_drops_file).
        Drops already listed there are locked to test; --freeze appends this
        run's new test drops to it and never removes one, so the test set only
        grows and every model is scored on the same drops.
Output: process_files/training_lists/proposed_split.csv           one row per drop
        process_files/training_lists/proposed_split_coverage.csv  one row per class
"""

import argparse
import math
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.training.prepare_training_data import canonicalize_species  # noqa: E402

SKIP_LABELS = {
    "Bait",
    "Leather jack - Oligoplites saurus",
}  # bait isn't a species; Oligoplites is a mis-click
NOT_STRATIFIED = {"fish", "bait"}
TEST_SHARE, VAL_SHARE = 0.10, 0.08
EMPTY_DROPS = {"test": 3, "val": 2}
MIN_TRAIN_SHARE = 0.5  # never leave a class with fewer than half its drops in train
MAX_TEST_SHARE = (
    0.25  # the test frame floor may take at most this share of a class's drops
)
MIN_TEST_FRAMES = 10  # per model class; 20 put 9 of 33 rock lobster drops in test

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument(
    "--freeze",
    action="store_true",
    help="append this run's new test drops to training.test_drops_file",
)
args = ap.parse_args()

rng = random.Random(config.training_split_seed)
frozen = config.training_test_drops
out = config.training_excluded_drops_file.parent
reg = species_registry()

# ── Load and canonicalise to training classes ──────────────────────────────
L = pd.read_csv(out / "biigle_label_by_drop.csv")
L = L[~L.label_name.isin(SKIP_LABELS) & (L.n_frames > 0)].copy()
L["cls"] = L.label_name.map(
    lambda n: "(empty)" if n == "(empty frame)" else canonicalize_species(n)
)
L = L[~L.cls.isin(config.training_excluded_species)]
L = L[~L.drop_id.isin(config.training_excluded_drops)]

frames = L[L.cls != "(empty)"].groupby(["drop_id", "cls"]).n_frames.sum()
by_drop = {d: g.droplevel(0).to_dict() for d, g in frames.groupby(level=0)}
empty_n = L[L.cls == "(empty)"].groupby("drop_id").n_frames.sum().to_dict()

# Fully empty drops: a drop whose frames were all reviewed and none annotated
# has a header-only raw CSV, so it never reaches the inventory. The BIIGLE file
# list of the survey training volumes (fetched separately) names every frame
# the expert was shown; a drop listed there with no inventory rows is empty.
survey_files = out / "biigle_survey_volume_files.csv"
assumed_empty = set()
if survey_files.exists():
    sf = pd.read_csv(survey_files)
    for d, n in sf.groupby("drop_id").filename.nunique().items():
        if d not in set(L.drop_id):
            empty_n[d] = n
            assumed_empty.add(d)

drops = sorted(set(L.drop_id) | assumed_empty)
for d in drops:
    by_drop.setdefault(d, {})


def common(c):
    sp = reg.get(c)
    return sp.common_name if sp else c


# ── Drop metadata ──────────────────────────────────────────────────────────
con = sqlite3.connect(config.db_path)
meta = pd.read_sql(
    "select drop_id, depth, nzmhcs_abiotic, nzmhcs_biotic from deployments", con
).set_index("drop_id")
meta = meta.reindex(drops)


def depth_band(x):
    if pd.isna(x):
        return "unknown"
    return "<10m" if x < 10 else "10-20m" if x < 20 else "20-30m" if x < 30 else "30m+"


has_location = {d: not d.startswith("volume_") for d in drops}
reserve = {d: d.split("_")[0] if has_location[d] else "(no location)" for d in drops}
band = {d: depth_band(meta.depth.get(d)) for d in drops}

# ── Tiers ──────────────────────────────────────────────────────────────────
n_drops = Counter(c for d in drops for c in by_drop[d])
total_frames = Counter()
for d in drops:
    total_frames.update(by_drop[d])


def quota(n):
    if n >= 15:
        return (
            max(2, round(TEST_SHARE * n)),
            max(1, round(VAL_SHARE * n)),
            "common (>=15)",
        )
    if n >= 10:
        return 2, 1, "rare (10-14)"
    if n >= 6:
        return 1, 1, "rare (6-9)"
    if n >= 3:
        return 1, 0, "scarce (3-5)"
    return 0, 0, "too few (<=2)"


Q = {c: quota(n) for c, n in n_drops.items()}
split = {d: None for d in drops}
reason = {d: [] for d in drops}
placed = {"test": Counter(), "val": Counter()}

# Every class keeps its biggest drop in train: test is counted per drop, training needs the frames.
locked = set()
for c in n_drops:
    if c in NOT_STRATIFIED:
        continue
    pool = [d for d in drops if c in by_drop[d] and d not in frozen]
    if not pool:
        continue
    biggest = max(pool, key=lambda d: by_drop[d][c])
    locked.add(biggest)
    reason[biggest].append(f"biggest {common(c)} drop ({by_drop[biggest][c]} frames)")
for d in drops:
    if not has_location[d]:
        locked.add(d)
        reason[d].append("no DropID, so no location")


def feasible(d):
    for k in by_drop[d]:
        held_out = placed["test"][k] + placed["val"][k] + 1
        if n_drops[k] - held_out < max(1, math.ceil(MIN_TRAIN_SHARE * n_drops[k])):
            return False
    return True


def overshoot(d, s, skip_cls=None):
    """Cost of rare classes this drop would push past their quota in split s."""
    cost = 0.0
    for k in by_drop[d]:
        if k == skip_cls or k in NOT_STRATIFIED:
            continue
        q = Q[k][0] if s == "test" else Q[k][1]
        if placed[s][k] >= q:
            cost += 3 if n_drops[k] < 15 else 0.2
    return cost


def assign(d, s, why):
    split[d] = s
    reason[d].append(why)
    for k in by_drop[d]:
        placed[s][k] += 1


def candidates(pool, s):
    return [d for d in pool if split[d] is None and d not in locked and feasible(d)]


def needs_more_test_frames(c, s):
    """A model class with too few test frames can't be measured; top it up."""
    if s != "test" or c not in config.training_class_order or n_drops[c] < 6:
        return False
    test_frames = sum(
        by_drop[d][c] for d in drops if split[d] == "test" and c in by_drop[d]
    )
    return test_frames < MIN_TEST_FRAMES and placed["test"][c] < math.ceil(
        MAX_TEST_SHARE * n_drops[c]
    )


# ── Phase 0: the frozen test list ───────────────────────────────────────────
for d in sorted(frozen & set(drops)):
    assign(d, "test", "test: frozen (test_drops.txt)")
missing_frozen = sorted(frozen - set(drops))
if missing_frozen:
    print(
        f"WARNING: {len(missing_frozen)} frozen test drop(s) not in the inventory: {missing_frozen[:10]}"
    )

# ── Phase 1: rarest class first ────────────────────────────────────────────
for c in sorted(
    (c for c in n_drops if c not in NOT_STRATIFIED), key=lambda c: (n_drops[c], c)
):
    for s, q in (("test", Q[c][0]), ("val", Q[c][1])):
        while placed[s][c] < q or needs_more_test_frames(c, s):
            pool = candidates([d for d in drops if c in by_drop[d]], s)
            if not pool:
                break
            reserves_used = {
                reserve[d]
                for d in drops
                if split[d] in ("test", "val") and c in by_drop[d]
            }

            # Rare classes: prefer small drops, training needs their frames.
            # Common classes: prefer typical drops, or test gets 1-frame drops
            # and can't measure the class (rock lobster got 4 test frames).
            sizes = sorted(by_drop[x][c] for x in drops if c in by_drop[x])
            median = sizes[len(sizes) // 2]

            def size_cost(d):
                if n_drops[c] < 15:
                    return by_drop[d][c] / total_frames[c]
                return abs(by_drop[d][c] - median) / max(median, 1)

            def score(d):
                return (
                    overshoot(d, s, skip_cls=c)
                    + (1.0 if reserve[d] in reserves_used else 0.0)
                    + size_cost(d)
                    + (0.3 if band[d] == "unknown" else 0.0)
                    + rng.random() * 0.01
                )

            d = min(pool, key=score)
            assign(
                d,
                s,
                f"{s}: {common(c)} ("
                + (f"{placed[s][c] + 1} of {q}" if placed[s][c] < q else "frame floor")
                + f", class has {n_drops[c]} drops)",
            )

# ── Phase 2: fully empty drops (false-positive check) ──────────────────────
empties = [d for d in drops if not by_drop[d] and has_location[d]]
rng.shuffle(empties)
for s, k in EMPTY_DROPS.items():
    # Count empties already there (frozen ones), or every rerun adds k more.
    k -= sum(1 for d in empties if split[d] == s)
    for d in sorted(
        (d for d in empties if split[d] is None),
        key=lambda d: reserve[d] in {reserve[x] for x in drops if split[x] == s},
    )[: max(k, 0)]:
        how = (
            "header-only CSV on NeSI, from BIIGLE file list"
            if d in assumed_empty
            else "empty labels"
        )
        assign(d, s, f"{s}: fully empty drop ({empty_n.get(d, 0)} frames, {how})")

# ── Phase 3: fill to target share, spread over reserve x depth band ────────
eligible = [d for d in drops if has_location[d]]
strata = Counter((reserve[d], band[d]) for d in eligible)
for s, share in (("test", TEST_SHARE), ("val", VAL_SHARE)):
    target = round(share * len(eligible))
    while sum(split[d] == s for d in drops) < target:
        have = Counter((reserve[d], band[d]) for d in drops if split[d] == s)
        pool = [
            d for d in candidates(eligible, s) if by_drop[d]
        ]  # empties only via phase 2
        if not pool:
            break
        # most under-filled stratum relative to its share of all drops
        d = min(
            pool,
            key=lambda d: (
                have[(reserve[d], band[d])] / strata[(reserve[d], band[d])]
                + overshoot(d, s)
                + rng.random() * 0.01
            ),
        )
        assign(d, s, f"{s}: fill, {reserve[d]} {band[d]}")

for d in drops:
    if split[d] is None:
        split[d] = "train"


# ── Write ──────────────────────────────────────────────────────────────────
def species_str(d):
    items = sorted(by_drop[d].items(), key=lambda kv: (n_drops[kv[0]], -kv[1]))
    return "; ".join(f"{common(c)} {n}" for c, n in items)


def rare_str(d):
    return "; ".join(
        common(c)
        for c in sorted(by_drop[d], key=lambda c: n_drops[c])
        if c not in NOT_STRATIFIED and n_drops[c] < 15
    )


order = {"test": 0, "val": 1, "train": 2}
rows = pd.DataFrame(
    {
        "drop_id": drops,
        "proposed_split": [split[d] for d in drops],
        "reason": ["; ".join(reason[d]) for d in drops],
        "reserve": [reserve[d] for d in drops],
        "survey": [
            "_".join(d.split("_")[:3]) if has_location[d] else "" for d in drops
        ],
        "depth_m": [meta.depth.get(d) for d in drops],
        "depth_band": [band[d] for d in drops],
        "nzmhcs_abiotic": [meta.nzmhcs_abiotic.get(d) for d in drops],
        "nzmhcs_biotic": [meta.nzmhcs_biotic.get(d) for d in drops],
        "habitat_tag": "",  # fill by hand
        "n_classes": [len(by_drop[d]) for d in drops],
        "empty_frames": [empty_n.get(d, 0) for d in drops],
        "rare_classes": [rare_str(d) for d in drops],
        "classes_frames": [species_str(d) for d in drops],
    }
)
rows = rows.sort_values(
    ["proposed_split", "reserve", "depth_m"],
    key=lambda s: s.map(order) if s.name == "proposed_split" else s,
)
rows.to_csv(out / "proposed_split.csv", index=False)

cov = []
for c in sorted(n_drops, key=lambda c: n_drops[c]):
    per = {
        s: [d for d in drops if split[d] == s and c in by_drop[d]]
        for s in ("train", "val", "test")
    }
    cov.append(
        {
            "class": common(c),
            "scientific": c,
            "tier": Q[c][2],
            "n_drops": n_drops[c],
            "in_model": c in config.training_class_order,
            **{f"{s}_drops": len(per[s]) for s in per},
            **{f"{s}_frames": sum(by_drop[d][c] for d in per[s]) for s in per},
            "test_reserves": ",".join(sorted({reserve[d] for d in per["test"]})),
        }
    )
pd.DataFrame(cov).to_csv(out / "proposed_split_coverage.csv", index=False)

if args.freeze:
    new_test = sorted(set(rows.drop_id[rows.proposed_split == "test"]) - frozen)
    path = config.training_test_drops_file
    lines = (
        []
        if path.exists()
        else [
            "# FROZEN test set (training.test_drops_file): these drops are always test,",
            "# never train or val, in every retrain. Append only: removing a drop makes",
            "# test scores incomparable with earlier models. Built by",
            "# scripts/propose_test_split.py --freeze; edit by hand freely before the",
            "# first model is scored on it.",
            "",
        ]
    )
    why = dict(zip(rows.drop_id, rows.reason))
    for d in new_test:
        picked = [
            p.split(": ", 1)[1] for p in why[d].split("; ") if p.startswith("test:")
        ]
        lines.append(f"{d}   # {'; '.join(picked)}")
    with path.open("a") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(
        f"froze {len(new_test)} new test drop(s) into {path} ({len(frozen)} already there)"
    )

print(rows.proposed_split.value_counts().to_string())
print(f"wrote {out}/proposed_split.csv and proposed_split_coverage.csv")
