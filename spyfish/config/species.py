"""
Single owner of species identity.

It owns the reconciliation of five sources that describe the same species and
do not agree:

  * ``class_map.json``     , class_id, aphia_id, scientific, common, aliases
  * ``species_labels.csv`` , the BIIGLE label tree ("Common - Scientific" → id)
  * ``config.yaml`` ``label_mapping``      , per-species BIIGLE label overrides
  * ``config.yaml`` ``non_species_classes``, which names are not real species
  * ``BUV Species.csv``    , declared in config, never read for name lookup

…with five loaders on top (``load_class_map``, ``load_class_map_by_id``,
``load_species_labels``, the app's ``load_common_names``, plus ad-hoc dict
comprehensions). They had already drifted: ``non_species_classes`` lists
``unknown``, which exists as a class nowhere.

This module is the only place that should read ``class_map.json`` or
``species_labels.csv``.

Identity rules
--------------
* **scientific_name is the join key.** AphiaID would be better but the bucket
  classes have none.
* **Bucket classes** (``fish``, ``bait``) have ``common_name ==
  scientific_name`` and ``is_bucket=True``. They are legitimate model outputs,
  ``fish`` means "an animal, species unidentified", not errors.
* **A miss never raises.** Lookups return ``None`` or echo the input back. An
  unmapped species must still yield a usable label rather than killing a render
  or an upload.

The optional path arguments
---------------------------
``species_registry()`` normally reads the configured class map. Exactly one
caller passes a different one: ``prepare_training_data`` decodes a drop's YOLO
labels against a per-drop **sidecar** ``class_map.json`` when one exists beside
them.

That matters because YOLO label files store bare integers (``0 0.51 0.33 0.12
0.08``) which are meaningless without the map that produced them, and the IDs
are not stable. ``class_map._build_registry_from_labels`` assigns them by
sorting species on AphiaID and enumerating, so adding one species to the BIIGLE
label tree shifts every subsequent ID. Decoding an older label file with the
current map yields systematically wrong species, silently.

Every other caller passes the configured path (sometimes rebuilt by hand, e.g.
``retrain_runner``), and the sidecars written by ``biigle_to_yolo`` and
``train`` are provenance, "the map used at download time", rather than live
decode inputs.

It lives under ``config/`` because it loads reference data whose paths come
from ``config``. Note it *consumes* the config singleton rather than being a
mixin composed into it, so the import is one-way and deferred to call time.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd


def normalise_zoo_choice(choice: str) -> str:
    """Normalise a Zooniverse ALL_CAPS choice key for lookup.

    Lowercase, strip whitespace, hyphens and underscores. Must stay aligned
    with how ``by_zoo_choice`` keys are built below or lookups silently miss.
    """
    return re.sub(r"[\s\-_]", "", (choice or "").lower())


@dataclass(frozen=True)
class Species:
    """One species (or bucket class) as known to every part of the pipeline."""

    scientific_name: str
    common_name: str
    class_id: Optional[int] = None  # YOLO class index
    aphia_id: Optional[int] = None  # WoRMS AphiaID, the only global identifier
    biigle_label_id: Optional[int] = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    is_bucket: bool = False

    @property
    def biigle_label_name(self) -> str:
        """The ``"Common - Scientific"`` form BIIGLE label trees use."""
        return f"{self.common_name} - {self.scientific_name}"

    @property
    def display_name(self) -> str:
        """``"Common (Scientific)"`` for dashboards.

        Bucket classes return the bare name, "fish (fish)" helps nobody.
        """
        if self.is_bucket or self.common_name == self.scientific_name:
            return self.scientific_name
        return f"{self.common_name} ({self.scientific_name})"

    @property
    def label(self) -> str:
        """The name to show a volunteer.

        For a bucket class this is "fish", which is correct: the question is
        "Is this a fish?" and that is precisely what the model claimed.
        """
        return self.common_name or self.scientific_name


@dataclass(frozen=True)
class SpeciesRegistry:
    """Every species lookup, built once from every source that describes one."""

    by_scientific: dict[str, Species]
    by_class_id: dict[int, Species]
    by_zoo_choice: dict[str, Species]
    by_any_name: dict[str, Species]  # scientific, "Common - Scientific", aliases

    def get(self, name: str) -> Optional[Species]:
        """Look up by any known name form. ``None`` when unknown."""
        if not name:
            return None
        return self.by_any_name.get(str(name).strip())

    def from_class_id(self, class_id: int) -> Optional[Species]:
        try:
            return self.by_class_id.get(int(class_id))
        except (TypeError, ValueError):
            return None

    def from_zoo_choice(self, choice: str) -> Optional[Species]:
        return self.by_zoo_choice.get(normalise_zoo_choice(choice))

    def label_for(self, class_name: str) -> str:
        """Volunteer-facing label for an ML class name; echoes back on a miss."""
        sp = self.get(class_name)
        return sp.label if sp else class_name

    def display_for(self, scientific_name: str) -> str:
        """``"Common (Scientific)"`` for a scientific name; echoes back on a miss."""
        sp = self.by_scientific.get(scientific_name)
        return sp.display_name if sp else scientific_name

    def scientific_for_class_id(self, class_id: int) -> Optional[str]:
        sp = self.from_class_id(class_id)
        return sp.scientific_name if sp else None

    def class_id_for(self, name: str) -> Optional[int]:
        sp = self.get(name)
        return sp.class_id if sp else None

    def biigle_label_id_for(self, name: str) -> Optional[int]:
        sp = self.get(name)
        return sp.biigle_label_id if sp else None

    def is_species(self, name: str) -> bool:
        """True only for real species. False for buckets and unknown names."""
        sp = self.get(name)
        return bool(sp and not sp.is_bucket)

    # ── plain-dict views ──────────────────────────────────────────────────
    # For call sites that genuinely want a dict, they pass it around, mutate a
    # copy, or use it in a comprehension, rather than querying the registry.

    def name_to_class_id(self) -> dict[str, int]:
        return {
            name: sp.class_id
            for name, sp in self.by_any_name.items()
            if sp.class_id is not None
        }

    def class_id_to_scientific(self) -> dict[int, str]:
        return {cid: sp.scientific_name for cid, sp in self.by_class_id.items()}

    def name_to_biigle_label_id(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for sp in self.by_scientific.values():
            if sp.biigle_label_id is not None:
                out[sp.scientific_name] = sp.biigle_label_id
                out[sp.biigle_label_name] = sp.biigle_label_id
        return out

    def zoo_choice_to_scientific(self) -> dict[str, str]:
        return {k: sp.scientific_name for k, sp in self.by_zoo_choice.items()}

    def common_names(self) -> dict[str, str]:
        """scientific → ``"Common (Scientific)"``, real species only.

        Bucket classes are excluded: the dashboard uses this for display names
        and "fish (fish)" is noise.
        """
        return {
            sp.scientific_name: sp.display_name
            for sp in self.by_scientific.values()
            if not sp.is_bucket
        }


def _load_class_map_entries(path: Optional[Path]) -> list[dict]:
    if not path or not Path(path).exists():
        logging.warning(
            f"class_map.json not found at {path}, species lookups will fall "
            "back to raw class names."
        )
        return []
    with open(path) as f:
        return list(json.load(f).values())


def _load_biigle_label_ids(path: Optional[Path]) -> dict[str, int]:
    """scientific name → BIIGLE label id, from the label-tree export CSV."""
    if not path or not Path(path).exists():
        logging.warning(
            f"species_labels.csv not found at {path}. BIIGLE annotations will "
            "fall back to default_fish_label_id and Zooniverse choice keys will "
            "not be normalised to scientific names."
        )
        return {}
    df = pd.read_csv(path)
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        full = str(row.get("name", "")).strip()
        if " - " not in full:
            continue
        _, scientific = full.split(" - ", 1)
        try:
            out[scientific.strip()] = int(row["id"])
        except (ValueError, TypeError, KeyError):
            continue
    return out


@lru_cache(maxsize=8)
def _build_registry(
    class_map_path: Optional[Path], labels_path: Optional[Path]
) -> SpeciesRegistry:
    """Cached per path pair, a training run may hold a sidecar open too."""
    entries = _load_class_map_entries(class_map_path)
    label_ids = _load_biigle_label_ids(labels_path)

    by_scientific: dict[str, Species] = {}
    by_class_id: dict[int, Species] = {}
    by_zoo_choice: dict[str, Species] = {}
    by_any_name: dict[str, Species] = {}

    for entry in entries:
        scientific = (entry.get("scientific_name") or "").strip()
        if not scientific:
            continue
        common = (entry.get("common_name") or "").strip() or scientific
        aliases = tuple(entry.get("aliases", []) or ())
        class_id = entry.get("class_id")
        sp = Species(
            scientific_name=scientific,
            common_name=common,
            class_id=int(class_id) if class_id is not None else None,
            aphia_id=int(entry["aphia_id"]) if entry.get("aphia_id") else None,
            biigle_label_id=label_ids.get(scientific),
            aliases=aliases,
            is_bucket=(common == scientific),
        )

        by_scientific[scientific] = sp
        if sp.class_id is not None:
            by_class_id[sp.class_id] = sp
        by_zoo_choice[normalise_zoo_choice(common)] = sp

        by_any_name[scientific] = sp
        by_any_name[sp.biigle_label_name] = sp
        for alias in aliases:
            by_any_name[alias] = sp

    if entries:
        logging.info(
            f"Species registry ({Path(class_map_path).name}): "
            f"{len(by_scientific)} classes "
            f"({sum(1 for s in by_scientific.values() if s.is_bucket)} buckets), "
            f"{sum(1 for s in by_scientific.values() if s.biigle_label_id)} with "
            "BIIGLE label ids."
        )
    return SpeciesRegistry(by_scientific, by_class_id, by_zoo_choice, by_any_name)


def species_registry(
    class_map_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
) -> SpeciesRegistry:
    """The species registry, cached.

    Both paths default to the configured ones. Pass ``class_map_path`` only to
    decode against a per-drop sidecar, see the module docstring.
    """
    from spyfish.config.wrapper import config

    if class_map_path is None:
        class_map_path = config.class_map_path
    if labels_path is None:
        labels_path = config.species_labels_csv_path
    return _build_registry(Path(class_map_path), Path(labels_path))
