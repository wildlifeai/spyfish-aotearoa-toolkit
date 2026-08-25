"""Tests for prune_unpinned_empty_classes (the 50 -> 22 class-list fix)."""

import yaml

from spyfish.ml.training.prepare_training_data import prune_unpinned_empty_classes

ROSTER = ["fish", "Pagrus auratus", "Parapercis colias", "bait"]


def _dataset(tmp_path, names, labels_by_split):
    species_dir = tmp_path / "species"
    (species_dir).mkdir()
    (species_dir / "data.yaml").write_text(
        yaml.safe_dump({"nc": len(names), "names": names}, sort_keys=False)
    )
    for split, files in labels_by_split.items():
        d = species_dir / "labels" / split
        d.mkdir(parents=True)
        for i, lines in enumerate(files):
            (d / f"KSF_20240124_BUV_KSF_085_01__frame_{i}.txt").write_text(
                "\n".join(lines)
            )
    return species_dir


def test_prunes_only_empty_extras(tmp_path):
    """Empty extras go; empty ROSTER entries keep their frozen seats."""
    names = ROSTER + ["Zeus faber", "Thyrsites atun"]
    # id 1 (roster) and both extras have no boxes; only the extras may be cut.
    species_dir = _dataset(
        tmp_path,
        names,
        {"train": [["0 .5 .5 .1 .1", "2 .4 .4 .1 .1"], ["3 .2 .2 .1 .1"]]},
    )

    pruned, class_names = prune_unpinned_empty_classes(species_dir, len(ROSTER))

    assert sorted(pruned) == ["Thyrsites atun", "Zeus faber"]
    assert class_names == ROSTER
    written = yaml.safe_load((species_dir / "data.yaml").read_text())
    assert written["nc"] == 4
    assert written["names"] == ROSTER
    # Roster ids must be untouched - that is the whole point of the frozen list.
    assert (species_dir / "labels" / "train").glob("*.txt")
    lines = sorted(
        line
        for f in (species_dir / "labels" / "train").glob("*.txt")
        for line in f.read_text().splitlines()
    )
    assert lines == ["0 .5 .5 .1 .1", "2 .4 .4 .1 .1", "3 .2 .2 .1 .1"]


def test_surviving_extra_is_renumbered_and_labels_follow(tmp_path):
    """A populated extra shifts down into the gap, and its labels shift with it."""
    names = ROSTER + ["Zeus faber", "Thyrsites atun"]  # ids 4, 5
    species_dir = _dataset(
        tmp_path,
        names,
        {"train": [["5 .5 .5 .1 .1", "0 .1 .1 .1 .1"]]},  # only id 5 has data
    )

    pruned, class_names = prune_unpinned_empty_classes(species_dir, len(ROSTER))

    assert pruned == ["Zeus faber"]
    assert class_names == ROSTER + ["Thyrsites atun"]
    lines = sorted(
        line
        for f in (species_dir / "labels" / "train").glob("*.txt")
        for line in f.read_text().splitlines()
    )
    assert lines == ["0 .1 .1 .1 .1", "4 .5 .5 .1 .1"]  # 5 -> 4, roster 0 stays


def test_val_only_class_is_kept(tmp_path):
    """A class with boxes in val but not train still owns its id - pruning it
    would leave those labels pointing at another species."""
    names = ROSTER + ["Zeus faber"]
    species_dir = _dataset(
        tmp_path,
        names,
        {"train": [["0 .5 .5 .1 .1"]], "val": [["4 .5 .5 .1 .1"]]},
    )

    pruned, class_names = prune_unpinned_empty_classes(species_dir, len(ROSTER))

    assert pruned == []
    assert class_names == names


def test_roster_only_list_is_a_noop(tmp_path):
    species_dir = _dataset(tmp_path, ROSTER, {"train": [["0 .5 .5 .1 .1"]]})
    pruned, class_names = prune_unpinned_empty_classes(species_dir, len(ROSTER))
    assert pruned == []
    assert class_names == ROSTER
