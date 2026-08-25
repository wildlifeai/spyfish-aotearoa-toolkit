"""Class ids must mean the same species in every run.

Before `training.class_order` existed, ids came from `sorted(all_species)`, so
they were alphabetical over whatever species happened to have data. Between the
2026-08-21 and 2026-08-22 runs three species entered and every id moved: snapper
10→12, blue cod 11→13, rock lobster 4→7. Two checkpoints a day apart could not
be compared, and `compare_with_production` was evaluating a 17-output model
against a 20-class data.yaml where class 0 had changed meaning.

These tests pin the two places that used to renumber.
"""

import yaml

from spyfish.config.species import species_registry
from spyfish.config.wrapper import config
from spyfish.ml.training.prepare_training_data import (
    apply_post_assembly_floor,
    canonicalize_species,
)

ROSTER = ("fish", "Pagrus auratus", "Parapercis colias", "Jasus edwardsii", "bait")


def _registry(roster=ROSTER):
    return species_registry(class_order=roster)


# ── the roster itself ────────────────────────────────────────────────────


def test_configured_roster_pins_the_indicator_species_low():
    """The ordering is a human-legibility contract, so assert it, not just its shape."""
    order = config.training_class_order
    assert order[0] == "fish"
    assert order[1] == "Pagrus auratus"  # snapper
    assert order[2] == "Parapercis colias"  # blue cod
    assert order[3] == "Jasus edwardsii"  # rock lobster
    assert order[-1] == "bait"  # not a species, last on purpose


def test_every_indicator_species_is_on_the_roster():
    """An indicator species with an unpinned id defeats the point of the roster."""
    order = set(config.training_class_order)
    missing = [s for s in config.indicator_species if s not in order]
    assert not missing, f"indicator species missing from class_order: {missing}"


# ── ordering a run's species ─────────────────────────────────────────────


def test_order_is_the_roster_not_alphabetical():
    ordered = _registry().order_training_classes(
        {"Jasus edwardsii", "Pagrus auratus", "fish"}
    )
    assert ordered == list(ROSTER)
    # Alphabetical would have put "Jasus edwardsii" first and "fish" last.
    assert ordered != sorted(ordered)


def test_species_with_no_data_this_run_keeps_its_slot():
    """The stability guarantee: a class vanishing must not renumber the rest."""
    full = _registry().order_training_classes(set(ROSTER))
    thin = _registry().order_training_classes({"fish", "bait"})
    assert thin == full
    assert thin.index("bait") == full.index("bait")


def test_unknown_species_are_appended_not_inserted(caplog):
    ordered = _registry().order_training_classes({"fish", "Aaa bbb", "Zzz yyy"})
    assert ordered[: len(ROSTER)] == list(ROSTER)
    assert ordered[len(ROSTER) :] == ["Aaa bbb", "Zzz yyy"]
    assert "Aaa bbb" in caplog.text


def test_adding_a_species_to_the_end_does_not_move_anyone():
    before = _registry().order_training_classes({"fish"})
    after = _registry(ROSTER + ("Squalus acanthias",)).order_training_classes({"fish"})
    assert after[: len(before)] == before


def test_training_class_id_is_the_roster_index():
    reg = _registry()
    assert reg.training_class_id_for("Pagrus auratus") == 1
    assert reg.training_class_id_for("Jasus edwardsii") == 3


def test_training_class_id_is_not_the_class_map_id():
    """Two different id spaces for the same animal; conflating them mislabels data."""
    sp = _registry().by_scientific.get("Pagrus auratus")
    assert sp is not None
    assert sp.training_class_id == 1
    # class_map.json sorts on AphiaID, so snapper is nowhere near 1 there.
    # Assert it is a real, different id rather than a vacuously unequal None.
    assert isinstance(sp.class_id, int)
    assert sp.class_id != sp.training_class_id


def test_roster_entry_absent_from_class_map_still_resolves():
    """Batoidea and Elasmobranchii were in the BIIGLE tree but not the stale class
    map, so the registry had no record and their detections fell through to
    default_fish_label_id at frame upload."""
    reg = species_registry(class_order=("fish", "Batoidea"))
    sp = reg.by_scientific.get("Batoidea")
    assert sp is not None
    assert sp.training_class_id == 1
    assert reg.name_to_biigle_label_id().get("Batoidea") is not None


# ── name resolution feeding the class list ───────────────────────────────


def test_biigle_workflow_labels_are_fish_not_species():
    """'Fish: final' means an animal identified as far as it goes, i.e. `fish`.

    These reach the expert MaxN CSVs as literal ScientificName strings. Before
    canonicalize_species consulted the registry they became phantom classes in
    data.yaml, since nothing else on the MaxN path resolves aliases.
    """
    for label in ("Fish: final", "Fish - Final", "Fish: review required", "To review"):
        assert canonicalize_species(label) == "fish", label


def test_deleted_workflow_label_still_resolves():
    """'Interesting Sighting' was removed from BIIGLE tree 3511, so it has no
    class_map alias to resolve through and is named in config instead."""
    assert canonicalize_species("Interesting Sighting") == "fish"


def test_canonicalize_resolves_name_form_before_merging_taxa():
    """The registry step must run first so the synonym map only needs to list
    canonical scientific names, not every "Common - Scientific" spelling."""
    assert canonicalize_species("Red gurnard - Chelidonichthys kumu") == "Triglidae"
    assert canonicalize_species("Chelidonichthys kumu") == "Triglidae"


def test_canonicalize_passes_unknown_names_through():
    """Unknown names must survive to the unpinned tail, not be silently dropped."""
    assert canonicalize_species("Nonexistent species") == "Nonexistent species"


# ── the post-assembly floor ──────────────────────────────────────────────


def _dataset(tmp_path, class_names, train_labels):
    """Minimal assembled dataset: data.yaml plus train/val label dirs."""
    species_dir = tmp_path / "species"
    for split in ("train", "val"):
        (species_dir / "labels" / split).mkdir(parents=True)
    with open(species_dir / "data.yaml", "w") as f:
        yaml.dump({"nc": len(class_names), "names": list(class_names)}, f)
    for i, ids in enumerate(train_labels):
        rows = "\n".join(f"{cid} 0.5 0.5 0.1 0.1" for cid in ids)
        (species_dir / "labels" / "train" / f"f{i}.txt").write_text(rows)
    return species_dir


def test_floor_merges_weak_classes_into_fish():
    """Sanity: the merge still happens, it is only the renumbering that stopped."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        # class 1 (snapper) appears in 3 frames, class 2 (blue cod) in 1.
        species_dir = _dataset(
            Path(td),
            ["fish", "Pagrus auratus", "Parapercis colias", "bait"],
            [[1], [1], [1], [2]],
        )
        merged, names = apply_post_assembly_floor(species_dir, min_images=3)

        assert merged == {"Parapercis colias"}
        weak_frame = (species_dir / "labels" / "train" / "f3.txt").read_text()
        assert weak_frame.startswith("0 "), "weak class should now be fish (id 0)"


def test_floor_leaves_class_ids_and_names_untouched():
    """The regression this whole file exists for."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        class_names = ["fish", "Pagrus auratus", "Parapercis colias", "bait"]
        species_dir = _dataset(Path(td), class_names, [[1], [1], [1], [2]])
        merged, names = apply_post_assembly_floor(species_dir, min_images=3)

        assert merged  # something was merged, so the old code would have renumbered
        assert names == class_names, "the floor must not drop or reorder names"

        on_disk = yaml.safe_load((species_dir / "data.yaml").read_text())
        assert on_disk["names"] == class_names
        assert on_disk["nc"] == len(class_names)

        # Snapper is still 1 in the rewritten labels, not shifted down to 1-of-3.
        surviving = (species_dir / "labels" / "train" / "f0.txt").read_text()
        assert surviving.startswith("1 ")


def test_floor_never_merges_fish_or_bait():
    """bait is in every frame; folding it into fish inflates every drop's MaxN."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        species_dir = _dataset(
            Path(td), ["fish", "Pagrus auratus", "bait"], [[1], [1], [1], [2]]
        )
        merged, _ = apply_post_assembly_floor(species_dir, min_images=3)
        assert "bait" not in merged
        assert "fish" not in merged
