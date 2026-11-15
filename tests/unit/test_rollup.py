"""The taxonomy rollup: one rule, computed from post-cap train frames, that
decides which class every labelled taxon trains as (built 2026-11-10).

The hand list it replaced named every eel, shark and ray species individually
and could not be told apart from the floor that merged thin species into
'fish'; these tests pin the rule's four steps and the artefact it writes.
"""

import json

import pytest
import yaml

from spyfish.ml.training import rollup as ru

# A small slice of the real tree: child -> parent, roots parented to "object".
H = {
    "fish": "object",
    "invertebrate": "object",
    "bait": "object",
    # eels (pool)
    "Anguilliformes": "fish",
    "Muraenidae": "Anguilliformes",
    "Gymnothorax": "Muraenidae",
    "Gymnothorax prionodon": "Gymnothorax",
    "Congridae": "Anguilliformes",
    "Conger": "Congridae",
    "Conger verreauxi": "Conger",
    "Conger sp.": "Conger",
    # sharks and rays (nested pools)
    "Elasmobranchii": "fish",
    "Carcharhiniformes": "Elasmobranchii",
    "Triakidae": "Carcharhiniformes",
    "Mustelus": "Triakidae",
    "Mustelus lenticulatus": "Mustelus",
    "Scyliorhinidae": "Carcharhiniformes",
    "Cephaloscyllium": "Scyliorhinidae",
    "Cephaloscyllium isabella": "Cephaloscyllium",
    "Batoidea": "Elasmobranchii",
    "Rajiformes": "Batoidea",
    "Rajidae": "Rajiformes",
    "Myliobatiformes": "Batoidea",
    "Dasyatidae": "Myliobatiformes",
    "Bathytoshia": "Dasyatidae",
    "Bathytoshia brevicaudata": "Bathytoshia",
    # wrasses (no pool)
    "Labriformes": "fish",
    "Labridae": "Labriformes",
    "Bodianus": "Labridae",
    "Bodianus unimaculatus": "Bodianus",
    "Notolabrus": "Labridae",
    "Notolabrus celidotus": "Notolabrus",
    "Notolabrus fucicola": "Notolabrus",
    "Pseudolabrus": "Labridae",
    "Pseudolabrus miles": "Pseudolabrus",
    # gurnards
    "Perciformes": "fish",
    "Triglidae": "Perciformes",
    "Chelidonichthys": "Triglidae",
    "Chelidonichthys kumu": "Chelidonichthys",
    # an invertebrate
    "Decapoda": "invertebrate",
    "Palinuridae": "Decapoda",
    "Jasus": "Palinuridae",
    "Jasus edwardsii": "Jasus",
}
POOLS = ("Anguilliformes", "Elasmobranchii", "Batoidea")
KEEP = ("Cephaloscyllium isabella",)


def _roll(frames, floor=50, pools=POOLS, keep=KEEP):
    return ru.compute_rollup(frames, H, floor, pools, keep)


# ── the rule ─────────────────────────────────────────────────────────────


def test_standalone_above_floor_and_fish_below_when_nothing_catches_it():
    r = _roll({"Notolabrus celidotus": 400, "Bodianus unimaculatus": 26, "fish": 9})
    assert r.mapping["Notolabrus celidotus"] == "Notolabrus celidotus"
    assert r.mapping["Bodianus unimaculatus"] == "fish"
    assert r.mapping["fish"] == "fish"
    assert "below floor" in r.reasons["Bodianus unimaculatus"]


def test_residual_counting_does_not_let_a_rare_species_borrow_its_sibling():
    """Notolabrus celidotus (400) stands alone; fucicola (24) must not become a
    'Notolabrus' class on the strength of its sibling's frames."""
    r = _roll({"Notolabrus celidotus": 400, "Notolabrus fucicola": 24})
    assert r.mapping["Notolabrus fucicola"] == "fish"
    assert "Notolabrus" not in r.classes()


def test_rare_relatives_together_make_a_group_class_at_the_nearest_rank():
    """Three thin wrasses of different genera: no genus reaches the floor, the
    family does, so they train as Labridae."""
    r = _roll(
        {
            "Bodianus unimaculatus": 26,
            "Notolabrus fucicola": 20,
            "Pseudolabrus miles": 10,
        }
    )
    assert set(r.mapping.values()) == {"Labridae"}
    assert "group class" in r.reasons["Labridae"] if "Labridae" in r.reasons else True
    assert r.class_frames() == {"Labridae": 56}


def test_group_label_frames_count_toward_their_own_group():
    """Frames labelled at family level ('Labridae', an unidentified wrasse) are
    residual too: 30 family frames + 25 pigfish make a Labridae class."""
    r = _roll({"Labridae": 30, "Bodianus unimaculatus": 25})
    assert r.mapping == {"Bodianus unimaculatus": "Labridae", "Labridae": "Labridae"}


def test_thin_species_joins_an_ancestor_that_is_already_a_class():
    """Triglidae stands alone (171). A gurnard species with 5 frames joins it:
    no residual floor applies when the class already exists."""
    r = _roll({"Triglidae": 171, "Chelidonichthys kumu": 5})
    assert r.mapping["Chelidonichthys kumu"] == "Triglidae"
    assert "joined existing class" in r.reasons["Chelidonichthys kumu"]


def test_pool_takes_every_member_however_many_frames():
    """Morays at 195 would stand alone; the eel pool is unconditional."""
    r = _roll(
        {
            "Muraenidae": 195,
            "Conger verreauxi": 64,
            "Gymnothorax prionodon": 5,
            "Conger sp.": 4,
        }
    )
    assert set(r.mapping.values()) == {"Anguilliformes"}
    assert r.class_frames() == {"Anguilliformes": 268}


def test_keep_separate_stands_alone_and_nested_pool_takes_the_rays():
    r = _roll(
        {
            "Cephaloscyllium isabella": 247,
            "Mustelus lenticulatus": 55,
            "Rajidae": 6,
            "Bathytoshia brevicaudata": 44,
        }
    )
    assert r.mapping["Cephaloscyllium isabella"] == "Cephaloscyllium isabella"
    assert (
        r.mapping["Mustelus lenticulatus"] == "Elasmobranchii"
    )  # 55 >= floor, pooled anyway
    assert r.mapping["Rajidae"] == "Batoidea"  # nearest pool wins over Elasmobranchii
    assert r.mapping["Bathytoshia brevicaudata"] == "Batoidea"


def test_keep_separate_species_that_cannot_stand_alone_rejoins_its_pool():
    r = _roll({"Cephaloscyllium isabella": 12, "Mustelus lenticulatus": 60})
    assert r.mapping["Cephaloscyllium isabella"] == "Elasmobranchii"
    assert "joined existing class" in r.reasons["Cephaloscyllium isabella"]


def test_pool_under_the_floor_is_not_a_class_and_members_fall_through():
    """30 ray frames: Batoidea is too thin to be a class, so the rays walk on
    up and land in the shark pool, which is live."""
    r = _roll({"Bathytoshia brevicaudata": 30, "Mustelus lenticulatus": 60})
    assert r.mapping["Bathytoshia brevicaudata"] == "Elasmobranchii"
    # and with no live pool above them either, they are fish
    r2 = _roll({"Bathytoshia brevicaudata": 30})
    assert r2.mapping["Bathytoshia brevicaudata"] == "fish"


def test_invertebrates_floor_into_fish_the_single_catch_all():
    r = _roll({"Jasus edwardsii": 10})
    assert r.mapping["Jasus edwardsii"] == "fish"
    assert "invertebrate" not in r.classes()


def test_exempt_names_never_move_and_unknown_names_are_reported(caplog):
    r = _roll({"bait": 3, "fish": 2, "Octopus tetricus": 4})
    assert r.mapping["bait"] == "bait" and r.mapping["fish"] == "fish"
    assert r.mapping["Octopus tetricus"] == "fish"
    assert r.unknown == ["Octopus tetricus"]
    assert "missing from the hierarchy" in caplog.text


def test_pool_and_keep_separate_may_not_overlap():
    with pytest.raises(ValueError):
        _roll({"Muraenidae": 5}, pools=("Anguilliformes",), keep=("Anguilliformes",))


def test_map_hash_depends_only_on_the_mapping():
    a = _roll({"Muraenidae": 195, "Conger verreauxi": 64})
    b = _roll({"Conger verreauxi": 70, "Muraenidae": 180})
    assert a.map_hash() == b.map_hash()
    assert a.map_hash() != _roll({"Muraenidae": 195}).map_hash()


# ── counting ─────────────────────────────────────────────────────────────


def test_capped_frame_counts_cap_each_drop(tmp_path):
    names = ["fish", "Parapercis colias", "Jasus edwardsii"]
    for drop, n in (
        ("KSF_20240124_BUV_KSF_085_01", 80),
        ("KSF_20240124_BUV_KSF_086_01", 10),
    ):
        d = tmp_path / drop
        d.mkdir()
        for i in range(n):
            (d / f"{drop}__frame_{i}.txt").write_text("1 .5 .5 .1 .1\n2 .5 .5 .1 .1\n")
    counts = ru.capped_frame_counts(
        tmp_path, names, [d.name for d in tmp_path.iterdir()], cap=50
    )
    assert counts == {"fish": 0, "Parapercis colias": 60, "Jasus edwardsii": 60}


# ── applying it to an assembled dataset ──────────────────────────────────


def _dataset(tmp_path, names, labels):
    """labels: {split: {stem: [class ids]}}"""
    sd = tmp_path / "species"
    for split, files in labels.items():
        (sd / "labels" / split).mkdir(parents=True)
        for stem, ids in files.items():
            (sd / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(f"{c} .5 .5 .1 .1" for c in ids)
            )
    (sd / "data.yaml").write_text(yaml.safe_dump({"nc": len(names), "names": names}))
    return sd


def test_apply_rewrites_every_split_keeps_ids_and_writes_the_map(tmp_path):
    names = [
        "fish",
        "Muraenidae",
        "Conger verreauxi",
        "Anguilliformes",
        "Notolabrus celidotus",
    ]
    train = {f"m{i}": [1] for i in range(60)} | {f"c{i}": [2] for i in range(10)}
    train |= {f"w{i}": [4] for i in range(60)}
    sd = _dataset(
        tmp_path, names, {"train": train, "val": {"v0": [1, 2, 4]}, "test": {"t0": [2]}}
    )

    rollup, out_names = ru.apply_rollup(sd, H, 50, POOLS, KEEP, same_animal={"x": "y"})

    assert out_names == names  # pooled names keep their seats
    assert rollup.mapping["Muraenidae"] == "Anguilliformes"
    assert (sd / "labels" / "val" / "v0.txt").read_text().split("\n") == [
        "3 .5 .5 .1 .1",
        "3 .5 .5 .1 .1",
        "4 .5 .5 .1 .1",
    ]
    assert (sd / "labels" / "test" / "t0.txt").read_text().startswith("3 ")
    doc = json.loads((sd / ru.ROLLUP_MAP_FILENAME).read_text())
    assert doc["map"]["Conger verreauxi"] == "Anguilliformes"
    assert doc["same_animal"] == {"x": "y"}
    assert doc["train_frames"]["Conger verreauxi"] == 10
    assert (
        doc["map_hash"]
        == rollup.map_hash()
        == ru.rollup_map_hash(sd / ru.ROLLUP_MAP_FILENAME)
    )
    assert "hierarchy_hash" in doc and doc["floor"] == 50


def test_apply_appends_a_group_class_that_has_no_seat(tmp_path, caplog):
    names = ["fish", "Bodianus unimaculatus", "Notolabrus fucicola"]
    train = {f"b{i}": [1] for i in range(30)} | {f"n{i}": [2] for i in range(30)}
    sd = _dataset(tmp_path, names, {"train": train})

    rollup, out_names = ru.apply_rollup(sd, H, 50, (), ())

    assert out_names == names + ["Labridae"]
    assert yaml.safe_load((sd / "data.yaml").read_text())["names"] == out_names
    assert (sd / "labels" / "train" / "b0.txt").read_text().startswith("3 ")
    assert "Pin them in training.class_order" in caplog.text


def test_val_only_taxon_is_rolled_up_too(tmp_path):
    """A taxon with boxes in val but none in train has 0 train frames: its val
    boxes must be relabelled as whatever it rolls to, not left under an id the
    model never trains."""
    names = ["fish", "Notolabrus fucicola"]
    sd = _dataset(tmp_path, names, {"train": {"a": [0]}, "val": {"v": [1]}})
    rollup, _ = ru.apply_rollup(sd, H, 50, (), ())
    assert rollup.mapping["Notolabrus fucicola"] == "fish"
    assert (sd / "labels" / "val" / "v.txt").read_text().startswith("0 ")


def test_apply_without_a_dataset_is_a_noop(tmp_path):
    assert ru.apply_rollup(tmp_path, H, 50, (), ()) == (None, [])
