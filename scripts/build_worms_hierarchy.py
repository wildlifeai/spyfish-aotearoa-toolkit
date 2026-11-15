"""Build hierarchy.json (child -> parent) for the training class rollup from the
species registry's AphiaIDs via the WoRMS REST API.

Every class-map taxon gets a chain leaf -> genus -> family -> "fish" (fish
lineages) or -> "invertebrate" (everything else), with "fish", "invertebrate"
and "bait" parented to "object". Family is the coarsest rank that can become
an automatic group class: an order such as Centrarchiformes (trumpeters,
sweeps and morwongs together) is not a thing a detector class should mean, so
orders appear only as explicit pool nodes. The pool nodes in
`training.rollup_pool_nodes` are kept wherever they occur in a lineage, at
whatever rank WoRMS gives them (Anguilliformes is an order, Elasmobranchii a
class, Batoidea a superorder), so the rollup can stop at them. A taxon labelled
above species level ("Conger sp.", a genus; Triglidae, a family) chains from
the rank below its own.

Nothing is skipped on account of training.species_canonicalization: the rollup
(`spyfish/ml/training/rollup.py`) applies that map before it reads the
hierarchy, so a merged-away name is simply never looked up.

WoRMS responses are cached in worms_cache.json next to the output so re-runs
are offline; only AphiaIDs new to the cache are fetched (one request each, the
API has no bulk classification endpoint).

Usage:
    python scripts/build_worms_hierarchy.py [--out process_files/training/hierarchy.json]

Run it after every class-map reseed (new BIIGLE labels), then commit the file:
the rollup reads it on NeSI.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spyfish.config.wrapper import config  # noqa: E402

WORMS_URL = "https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/{}"

# WoRMS lineage names that mark a fish lineage (paraphyletic, so a name set).
FISH_MARKERS = {
    "Actinopteri",
    "Actinopterygii",
    "Teleostei",
    "Elasmobranchii",
    "Chondrichthyes",
    "Holocephali",
    "Myxini",
    "Petromyzonti",
    "Pisces",
}
KEPT_RANKS = ("Family", "Genus")
ROOTS = {"fish": "object", "invertebrate": "object", "bait": "object"}


def lineage(aphia_id: int, cache: dict) -> list:
    """[(rank, name), ...] from kingdom down to the queried taxon."""
    key = str(aphia_id)
    if key not in cache:
        resp = requests.get(WORMS_URL.format(aphia_id), timeout=30)
        resp.raise_for_status()
        cache[key] = resp.json()
        time.sleep(0.15)  # politeness to the free API
    node, chain = cache[key], []
    while isinstance(node, dict) and node.get("scientificname"):
        chain.append((node.get("rank", ""), node["scientificname"]))
        node = node.get("child")
    return chain


def chain_for(name: str, lineage_nodes: list, pool_nodes: set) -> list:
    """Ancestors of ``name`` bottom-up: kept ranks and pool nodes, then the root."""
    names = {n for _, n in lineage_nodes}
    root = "fish" if names & FISH_MARKERS else "invertebrate"
    kept = [
        n
        for rank, n in lineage_nodes
        if (rank in KEPT_RANKS or n in pool_nodes)
        and n != name
        and "incertae sedis" not in n  # WoRMS placeholder, not a taxon
    ]
    return list(reversed(kept)) + [root]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=config.training_hierarchy_file)
    args = ap.parse_args()

    registry = json.loads(config.class_map_path.read_text())
    pool_nodes = set(config.training_rollup_pool_nodes)
    cache_path = args.out.parent / "worms_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    before = json.loads(args.out.read_text()) if args.out.exists() else {}

    edges: dict = dict(ROOTS)
    failed = []
    entries = [v for v in registry.values() if isinstance(v, dict)]
    n_fetched = 0
    for entry in entries:
        name = entry.get("scientific_name")
        aphia = entry.get("aphia_id")
        if not name or name in ROOTS:
            continue
        if not aphia:
            failed.append((name, "no aphia_id"))
            continue
        cached = str(aphia) in cache
        try:
            nodes = lineage(int(aphia), cache)
        except Exception as e:  # noqa: BLE001 report per taxon, keep going
            failed.append((name, str(e)))
            continue
        n_fetched += 0 if cached else 1
        child = name
        for parent in chain_for(name, nodes, pool_nodes):
            edges.setdefault(child, parent)
            child = parent

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    args.out.write_text(json.dumps(edges, indent=2, sort_keys=True) + "\n")

    added = sorted(set(edges) - set(before))
    removed = sorted(set(before) - set(edges))
    changed = sorted(k for k in set(edges) & set(before) if edges[k] != before[k])
    missing_pools = sorted(pool_nodes - set(edges))
    print(
        f"Wrote {args.out}: {len(edges)} edges, {n_fetched} WoRMS lookups, "
        f"{len(failed)} failed; vs previous file: +{len(added)} -{len(removed)} "
        f"~{len(changed)} reparented."
    )
    for k in added:
        print(f"  + {k} -> {edges[k]}")
    for k in removed:
        print(f"  - {k} (was -> {before[k]})")
    for k in changed:
        print(f"  ~ {k}: {before[k]} -> {edges[k]}")
    for name, why in failed:
        print(f"  FAILED {name}: {why}")
    if missing_pools:
        print(f"  WARNING pool node(s) absent from every lineage: {missing_pools}")


if __name__ == "__main__":
    main()
