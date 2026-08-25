"""Tests for the per-species val floor in balance_val_drops.

val_pct alone is scale-invariant, so it is right for common species and useless
for rare ones: 8% of Batoidea's 71 boxes is 6, and 5 val boxes is what the
2026-08-23 run actually produced for it.
"""

from spyfish.ml.training.split_data import balance_val_drops

SPECIES = ["fish", "Pagrus auratus", "Batoidea"]


def _corpus(tmp_path, layout):
    """layout: {drop_id: {class_id: n_boxes}} -> one .txt per box, 1 box each."""
    root = tmp_path / "labels_staged"
    for drop, counts in layout.items():
        d = root / drop
        d.mkdir(parents=True)
        i = 0
        for cid, n in counts.items():
            for _ in range(n):
                (d / f"{drop}__frame_{i}.txt").write_text(f"{cid} .5 .5 .1 .1\n")
                i += 1
    return root


def _rare_corpus(tmp_path):
    # Batoidea (id 2): 10 boxes spread over 10 drops, one each - plenty of
    # drops, tiny box count. Pagrus (id 1) is common.
    layout = {}
    for i in range(10):
        layout[f"KSF_20240124_BUV_KSF_0{80+i}_01"] = {1: 40, 2: 1}
    return _corpus(tmp_path, layout), sorted(layout)


def test_percentage_alone_starves_a_rare_species(tmp_path):
    """Baseline: without a floor, 8% of 10 boxes rounds to 1."""
    root, drops = _rare_corpus(tmp_path)
    train, val, _ = balance_val_drops(
        root, SPECIES, drops, val_pct=0.08, tolerance=0.2, max_share=0.4
    )
    assert len(val) <= 2


def test_floor_lifts_the_rare_species_target(tmp_path):
    """With a floor, the greedy keeps picking drops until the rare class is covered."""
    root, drops = _rare_corpus(tmp_path)
    train, val, _ = balance_val_drops(
        root, SPECIES, drops, val_pct=0.08, tolerance=0.2, max_share=0.4, min_boxes=4
    )
    # 4 Batoidea boxes need 4 drops (one box each), vs 1 without the floor.
    assert len(val) >= 4
    assert train, "train must not be emptied"


def test_floor_is_clamped_by_max_share(tmp_path):
    """An unreachable floor must not drag every drop into val.

    Batoidea has 10 boxes; max_share=0.4 permits at most 4 in val. A floor of
    50 is impossible, and without the clamp the greedy would consume the whole
    candidate pool chasing it.
    """
    root, drops = _rare_corpus(tmp_path)
    train, val, _ = balance_val_drops(
        root, SPECIES, drops, val_pct=0.08, tolerance=0.2, max_share=0.4, min_boxes=50
    )
    assert len(val) <= 4, f"val bloated to {len(val)} drops chasing an impossible floor"
    assert len(train) >= 6


def test_floor_does_not_disturb_a_common_species(tmp_path):
    """A species already above the floor keeps its percentage target."""
    root, drops = _rare_corpus(tmp_path)
    _, val_without, _ = balance_val_drops(
        root,
        ["fish", "Pagrus auratus"],
        drops,
        val_pct=0.08,
        tolerance=0.2,
        max_share=0.4,
    )
    _, val_with, _ = balance_val_drops(
        root,
        ["fish", "Pagrus auratus"],
        drops,
        val_pct=0.08,
        tolerance=0.2,
        max_share=0.4,
        min_boxes=4,
    )
    # Pagrus has 400 boxes; 8% = 32, already far above a floor of 4.
    assert len(val_with) == len(val_without)


def test_floor_skips_species_the_class_floor_will_merge(tmp_path):
    """Lifting the val target for a species that gets merged into 'fish' after
    assembly just pulls drops out of train for a class that ceases to exist.

    Batoidea (71 frames) survives class_floor_min_images=50; Notolabrus fucicola
    (24 frames) does not, and must keep its plain percentage target.
    """
    species = ["fish", "Batoidea", "Notolabrus fucicola"]
    layout = {}
    for i in range(10):
        # id 1 survives the floor (10 drops x 8 frames = 80), id 2 does not
        # (10 drops x 2 frames = 20).
        layout[f"KSF_20240124_BUV_KSF_0{80+i}_01"] = {0: 30, 1: 8, 2: 2}
    root = _corpus(tmp_path, layout)
    drops = sorted(layout)

    _, val_lifted, _ = balance_val_drops(
        root,
        species,
        drops,
        val_pct=0.08,
        tolerance=0.2,
        max_share=0.4,
        min_boxes=20,
        floor_min_frames=50,
    )
    _, val_all, _ = balance_val_drops(
        root,
        species,
        drops,
        val_pct=0.08,
        tolerance=0.2,
        max_share=0.4,
        min_boxes=20,
        floor_min_frames=0,  # no guard: lift everything
    )
    # Without the guard the doomed species drags in extra drops.
    assert len(val_lifted) < len(val_all)
