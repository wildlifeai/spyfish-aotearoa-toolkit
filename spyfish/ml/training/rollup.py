"""Taxonomy-driven class rollup: which training class each labelled taxon joins.

One rule, computed once per data prep from what the model will actually train
on, written to ``rollup_map.json`` beside ``data.yaml`` and read back by every
consumer (snapshot, test records, planning scripts). It replaces the hand-kept
list of merges that used to live in ``training.species_canonicalization``
(2026-11-10): that list named every eel, shark and ray species individually,
nine of them "unseen in data yet", and the floor that merged thin species into
``fish`` counted frames differently in the val balancer and after assembly.

The rule, applied to post-cap TRAIN frame counts per taxon (the same count the
old post-assembly floor used):

1. **Exempt** names are never moved: the catch-all ``fish`` and ``bait``.
2. **Pools** (``training.rollup_pool_nodes``, e.g. ``Anguilliformes``): every
   taxon under a pool node trains as the pool, however many frames it has,
   unless it is listed in ``training.rollup_keep_separate``. A pool with fewer
   than ``class_floor_min_images`` frames in total is not a class; its members
   fall through to step 4.
3. **Standalone**: any remaining taxon with at least the floor stands as its
   own class.
4. **Residual walk** for what is left (below the floor): walk the taxon's
   ancestors nearest first. An ancestor that is already a class takes it. An
   ancestor that is not yet a class becomes one only if the taxa it would
   collect, which have NOT already earned a class of their own, reach the floor
   together. Counting only that residual is what stops a rare species borrowing
   its common sibling's frames to earn a class it cannot fill. Nothing reached:
   the taxon trains as ``fish``.

Two taxa that experts label at different levels but that denote one animal in
this data (red gurnard / Triglidae) are still merged by hand in
``training.species_canonicalization`` BEFORE this rule sees them: a rule over
counts cannot know that two labels mean the same fish.

The hierarchy is ``hierarchy.json`` (child -> parent over WoRMS genus and
family plus the pool nodes at their own rank), built by
``scripts/build_worms_hierarchy.py``. Family is therefore the coarsest class the
walk can create on its own; an order-level class only exists as a pool node.
A labelled name the hierarchy does not know is logged and treated as a direct
child of ``fish``: it can stand alone or floor, never join a group.
"""

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from spyfish.ml.training.yolo_labels import (  # noqa: F401  (SPLITS re-exported)
    SPLITS,
    iter_class_ids,
    read_class_names,
    remap_label_file,
    write_class_names,
)

CATCH_ALL = "fish"
EXEMPT: Tuple[str, ...] = ("fish", "bait")
ROLLUP_MAP_FILENAME = "rollup_map.json"


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def load_hierarchy(path: Path) -> Dict[str, str]:
    """``{child: parent}`` as written by ``scripts/build_worms_hierarchy.py``."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(f"{path} must be a flat {{child: parent}} JSON object")
    return data


def hierarchy_hash(hierarchy: Dict[str, str]) -> str:
    return _hash(hierarchy)


def _terminal_nodes(hierarchy: Dict[str, str], catch_all: str) -> Set[str]:
    """Nodes the ancestor walk never passes and never targets: the catch-all,
    the roots (``object``), and the roots' direct children (``fish``,
    ``invertebrate``, ``bait``). An invertebrate below the floor therefore
    rolls to ``fish`` like everything else: there is one catch-all."""
    roots = set(hierarchy.values()) - set(hierarchy)
    return {catch_all} | roots | {n for n, p in hierarchy.items() if p in roots}


def ancestors(
    name: str, hierarchy: Dict[str, str], terminal: Optional[Set[str]] = None
) -> List[str]:
    """Ancestors of ``name``, nearest first, stopping before any terminal node.
    A name the hierarchy does not know has none."""
    if terminal is None:
        terminal = _terminal_nodes(hierarchy, CATCH_ALL)
    out: List[str] = []
    seen = {name}
    cur = hierarchy.get(name)
    while cur is not None and cur not in terminal and cur not in seen:
        out.append(cur)
        seen.add(cur)
        cur = hierarchy.get(cur)
    return out


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@dataclass
class Rollup:
    """Result of ``compute_rollup``: every input taxon maps to its class."""

    mapping: Dict[str, str]
    reasons: Dict[str, str]
    frames: Dict[str, int]
    floor: int
    pool_nodes: List[str]
    keep_separate: List[str]
    catch_all: str = CATCH_ALL
    unknown: List[str] = field(default_factory=list)

    def classes(self) -> List[str]:
        return sorted(set(self.mapping.values()))

    def class_frames(self) -> Dict[str, int]:
        """Frames per class as the sum of its taxa's frames (an upper bound
        when one frame holds two taxa of the same class)."""
        out: Counter = Counter()
        for taxon, cls in self.mapping.items():
            out[cls] += self.frames.get(taxon, 0)
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def merged(self) -> Dict[str, str]:
        """Only the taxa that do not train under their own name."""
        return {t: c for t, c in self.mapping.items() if t != c}

    def map_hash(self) -> str:
        return _hash(self.mapping)


def compute_rollup(
    frames: Dict[str, int],
    hierarchy: Dict[str, str],
    floor: int,
    pool_nodes: Sequence[str],
    keep_separate: Sequence[str] = (),
    catch_all: str = CATCH_ALL,
    exempt: Sequence[str] = EXEMPT,
) -> Rollup:
    """Map every taxon in ``frames`` to the class it trains as. See the module
    docstring for the rule; ``frames`` is ``{taxon: train frames containing it}``
    and should be post-cap (what the model sees), the same count as the floor."""
    clash = set(pool_nodes) & set(keep_separate)
    if clash:
        raise ValueError(
            f"rollup: {sorted(clash)} listed as both pool node and keep-separate"
        )
    terminal = _terminal_nodes(hierarchy, catch_all)
    names = list(frames)
    anc = {n: ancestors(n, hierarchy, terminal) for n in names}
    exempt_set = set(exempt) | {catch_all}
    unknown = sorted(n for n in names if n not in hierarchy and n not in exempt_set)
    if unknown:
        logging.warning(
            f"rollup: {len(unknown)} labelled name(s) missing from the hierarchy, "
            f"treated as direct children of {catch_all!r} (rebuild hierarchy.json "
            f"or add a synonym): {unknown}"
        )

    mapping: Dict[str, str] = {}
    reasons: Dict[str, str] = {}

    def assign(taxon: str, cls: str, why: str) -> None:
        mapping[taxon] = cls
        reasons[taxon] = why

    # 1. exempt
    for n in names:
        if n in exempt_set:
            assign(n, n, "exempt: never rolled up")

    # 2. pools: nearest pool ancestor wins; a pool node labelled directly is its
    #    own member; keep-separate taxa skip this step (they may still join a
    #    pool in step 4 if they cannot stand alone).
    members: Dict[str, List[str]] = {p: [] for p in pool_nodes}
    for n in names:
        if n in mapping or n in keep_separate:
            continue
        if n in members:
            members[n].append(n)
            continue
        pool = next((a for a in anc[n] if a in members), None)
        if pool is not None:
            members[pool].append(n)
    for pool, taxa in members.items():
        total = sum(frames[t] for t in taxa)
        if not taxa:
            continue
        if total >= floor:
            for t in taxa:
                assign(
                    t,
                    pool,
                    (
                        f"pool node: {total} frames in pool"
                        if t == pool
                        else f"pooled under {pool} ({total} frames in pool)"
                    ),
                )
        else:
            logging.info(
                f"rollup: pool {pool!r} has {total} frames in total, under the "
                f"floor of {floor}; its {len(taxa)} member(s) fall through"
            )

    # 3. standalone
    for n in names:
        if n not in mapping and frames[n] >= floor:
            assign(n, n, f"standalone: {frames[n]} frames >= {floor}")

    # 4. residual walk, nearest ancestors first (deepest candidates first)
    pending = [n for n in names if n not in mapping]
    classes_so_far = set(mapping.values())
    candidates = {a for n in pending for a in anc[n]}

    def depth(node: str) -> int:
        return len(ancestors(node, hierarchy, terminal))

    for node in sorted(candidates, key=lambda a: (-depth(a), a)):
        group = [n for n in pending if n not in mapping and node in anc[n]]
        if not group:
            continue
        if node in classes_so_far:
            for n in group:
                assign(n, node, f"joined existing class {node} ({frames[n]} < {floor})")
            continue
        if node in frames and node not in mapping:
            group.append(node)  # the group label's own frames count too
        residual = sum(frames[n] for n in group)
        if residual < floor:
            continue
        classes_so_far.add(node)
        for n in group:
            assign(
                n,
                node,
                (
                    f"group class: residual {residual} frames >= {floor}"
                    if n == node
                    else f"rolled up to {node} (residual {residual} >= {floor})"
                ),
            )
    for n in pending:
        if n not in mapping:
            assign(
                n,
                catch_all,
                f"below floor ({frames[n]} < {floor}), no group reaches the floor",
            )

    return Rollup(
        mapping=dict(sorted(mapping.items())),
        reasons=dict(sorted(reasons.items())),
        frames=dict(sorted(frames.items())),
        floor=floor,
        pool_nodes=list(pool_nodes),
        keep_separate=list(keep_separate),
        catch_all=catch_all,
        unknown=unknown,
    )


# ---------------------------------------------------------------------------
# Counting frames
# ---------------------------------------------------------------------------


def frame_counts(label_dir: Path, class_names: Sequence[str]) -> Dict[str, int]:
    """``{name: distinct frames containing it}`` over one directory of YOLO
    ``.txt`` files whose ids index ``class_names``. Every name is present, with
    0 when it has no frame."""
    counts = Counter({n: 0 for n in class_names})
    if label_dir.is_dir():
        for txt in label_dir.glob("*.txt"):
            for cid in set(iter_class_ids(txt)):
                if 0 <= cid < len(class_names):
                    counts[class_names[cid]] += 1
    return dict(counts)


def capped_frame_counts(
    labels_staged_dir: Path,
    class_names: Sequence[str],
    drops: Iterable[str],
    cap: int,
) -> Dict[str, int]:
    """Pre-split estimate of post-cap train frames: per drop, a taxon can keep
    at most ``cap`` frames (``cap_frames_per_drop``), so its contribution is
    ``min(frames in drop, cap)``. An upper bound (val drops will leave train,
    and the cap is per frame not per taxon), used only to steer the val
    balancer; the binding rollup is computed after assembly."""
    totals = Counter({n: 0 for n in class_names})
    for drop in drops:
        d = labels_staged_dir / drop
        if not d.is_dir():
            continue
        per_drop = Counter()
        for txt in d.glob("*.txt"):
            for cid in set(iter_class_ids(txt)):
                if 0 <= cid < len(class_names):
                    per_drop[class_names[cid]] += 1
        for name, n in per_drop.items():
            totals[name] += min(n, cap)
    return dict(totals)


# ---------------------------------------------------------------------------
# Applying it to an assembled dataset
# ---------------------------------------------------------------------------


def apply_rollup(
    species_dir: Path,
    hierarchy: Dict[str, str],
    floor: int,
    pool_nodes: Sequence[str],
    keep_separate: Sequence[str],
    same_animal: Optional[Dict[str, str]] = None,
    hierarchy_path: Optional[Path] = None,
) -> Tuple[Optional[Rollup], List[str]]:
    """Compute the rollup from ``labels/train`` and rewrite every split's labels
    so each taxon's boxes carry its class id. Writes ``rollup_map.json`` beside
    ``data.yaml``.

    Class ids are left alone: a taxon rolled into a group keeps its name and id
    in ``data.yaml`` and simply ends up with no instances (Ultralytics leaves
    an empty class out of mAP). Removing names renumbered every class after
    them and made checkpoints incomparable, so this never does. A group class
    that has no id yet (a family the hierarchy created) is APPENDED with a
    warning asking for it to be pinned in ``training.class_order``.

    Returns ``(rollup, class_names)``; ``(None, [])`` when there is no dataset.
    """
    data_yaml_path = species_dir / "data.yaml"
    if not data_yaml_path.exists():
        return None, []
    names: List[str] = read_class_names(data_yaml_path)

    train = frame_counts(species_dir / "labels" / "train", names)
    present: Set[str] = set()
    for split in SPLITS:
        counts = (
            train
            if split == "train"
            else frame_counts(species_dir / "labels" / split, names)
        )
        present |= {n for n, c in counts.items() if c > 0}
    present |= set(pool_nodes) & set(names)
    frames = {n: train[n] for n in names if n in present}

    rollup = compute_rollup(frames, hierarchy, floor, pool_nodes, keep_separate)

    new_names = list(names)
    appended = [c for c in rollup.classes() if c not in new_names]
    for cls in appended:
        new_names.append(cls)
    if appended:
        logging.warning(
            f"rollup: {len(appended)} group class(es) created by the hierarchy have "
            f"no frozen id and were appended: {appended}. Pin them in "
            "training.class_order so their ids stop moving."
        )
    unpinned_pools = [p for p in pool_nodes if p not in names]
    if unpinned_pools:
        logging.warning(
            f"rollup: pool node(s) {unpinned_pools} are not in training.class_order"
        )

    id_map = {
        names.index(taxon): new_names.index(cls)
        for taxon, cls in rollup.mapping.items()
    }
    changed = {o: n for o, n in id_map.items() if o != n}
    if changed:
        for split in SPLITS:
            split_dir = species_dir / "labels" / split
            if not split_dir.is_dir():
                continue
            for txt in split_dir.glob("*.txt"):
                # `changed` lists only the ids that move: the rest keep theirs.
                remap_label_file(txt, changed, missing="keep")

    if new_names != names:
        write_class_names(data_yaml_path, new_names)

    write_rollup_map(
        species_dir / ROLLUP_MAP_FILENAME,
        rollup,
        hierarchy=hierarchy,
        hierarchy_path=hierarchy_path,
        same_animal=same_animal or {},
    )
    _log_summary(rollup)
    return rollup, new_names


def _log_summary(rollup: Rollup) -> None:
    merged = rollup.merged()
    by_class: Dict[str, List[str]] = {}
    for taxon, cls in merged.items():
        by_class.setdefault(cls, []).append(taxon)
    live = {c: n for c, n in rollup.class_frames().items() if n > 0}
    logging.info(
        f"rollup: {len(live)} class(es) with train frames at floor {rollup.floor}: "
        + ", ".join(f"{c} {n}" for c, n in live.items())
    )
    for cls, taxa in sorted(by_class.items()):
        logging.info(
            f"rollup: -> {cls}: "
            + ", ".join(f"{t} ({rollup.frames[t]})" for t in sorted(taxa))
        )
    if not merged:
        logging.info("rollup: every labelled taxon trains under its own name")


# ---------------------------------------------------------------------------
# The artefact
# ---------------------------------------------------------------------------


def write_rollup_map(
    path: Path,
    rollup: Rollup,
    hierarchy: Optional[Dict[str, str]] = None,
    hierarchy_path: Optional[Path] = None,
    same_animal: Optional[Dict[str, str]] = None,
) -> Path:
    """The per-data-prep decision log every consumer reads. ``map`` is the
    taxon -> class dict; ``map_hash`` identifies it (two runs with the same
    hash trained the same class vocabulary)."""
    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "map_hash": rollup.map_hash(),
        "floor": rollup.floor,
        "catch_all": rollup.catch_all,
        "pool_nodes": rollup.pool_nodes,
        "keep_separate": rollup.keep_separate,
        "same_animal": dict(same_animal or {}),
        "hierarchy_file": str(hierarchy_path) if hierarchy_path else None,
        "hierarchy_hash": hierarchy_hash(hierarchy) if hierarchy else None,
        "unknown_to_hierarchy": rollup.unknown,
        "classes": rollup.class_frames(),
        "train_frames": rollup.frames,
        "map": rollup.mapping,
        "reasons": rollup.reasons,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    logging.info(f"Wrote rollup map ({doc['map_hash']}) -> {path}")
    return path


def read_rollup_map(path: Path) -> Optional[dict]:
    """The artefact as a dict, or None when no data prep has written one."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def rollup_map_hash(path: Path) -> Optional[str]:
    doc = read_rollup_map(path)
    return doc["map_hash"] if doc else None


def _hash(obj) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


__all__ = [
    "CATCH_ALL",
    "EXEMPT",
    "ROLLUP_MAP_FILENAME",
    "Rollup",
    "ancestors",
    "apply_rollup",
    "capped_frame_counts",
    "compute_rollup",
    "frame_counts",
    "hierarchy_hash",
    "load_hierarchy",
    "read_rollup_map",
    "rollup_map_hash",
    "write_rollup_map",
]
