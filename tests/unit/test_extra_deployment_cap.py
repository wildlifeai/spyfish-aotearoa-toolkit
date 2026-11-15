"""Curated volumes are capped per SOURCE deployment when their frames are mapped.

Volume 30505 holds 1,887 Zooniverse frames from 29 deployments, up to 328 from
one video, against ~10 frames for a pipeline drop. Treated as one uncapped
"drop" it outweighs ~70 drops while adding the variety of 29 (2026-11-06).
"""

import random

from spyfish.ml.training.prepare_training_data import select_frames_per_drop

VOLUME = "volume_30505"
COD, SNAPPER, LOBSTER = 1, 2, 3


def _volume(tmp_path, per_dep):
    """per_dep: {deployment: [(class_id, n_frames), ...]} -> stems by deployment."""
    d = tmp_path / VOLUME
    d.mkdir(parents=True)
    mapping, i = {}, 0
    for dep, spec in per_dep.items():
        for cid, n in spec:
            for _ in range(n):
                stem = f"f{i:04d}"
                (d / f"{stem}.txt").write_text(f"{cid} .5 .5 .1 .1\n")
                mapping[stem] = dep
                i += 1
    return mapping


def _run(root, mapping, cap=10, rare=frozenset()):
    return select_frames_per_drop(
        drops={VOLUME},
        species_labels_dir=root,
        extras_set={VOLUME},
        dominant_class_ids={COD},
        rare_class_ids=set(rare),
        cap=cap,
        rng=random.Random(0),
        extra_deployments={VOLUME: mapping} if mapping is not None else None,
    )


def test_unmapped_volume_stays_uncapped(tmp_path):
    _volume(tmp_path, {"A": [(SNAPPER, 40)], "B": [(SNAPPER, 40)]})
    ff, _, stats = _run(tmp_path, None)
    assert len(ff[VOLUME]) == 80
    assert stats["extras_uncapped"] == 1


def test_each_source_deployment_gets_the_normal_cap(tmp_path):
    mapping = _volume(
        tmp_path, {"A": [(SNAPPER, 40)], "B": [(SNAPPER, 40)], "C": [(SNAPPER, 4)]}
    )
    ff, _, stats = _run(tmp_path, mapping)
    kept = ff[VOLUME]
    by_dep = {dep: sum(mapping[s] == dep for s in kept) for dep in "ABC"}
    assert by_dep == {"A": 10, "B": 10, "C": 4}
    assert stats["extra_deployments_capped"] == 3


def test_blue_cod_only_frames_go_first_within_a_deployment(tmp_path):
    mapping = _volume(tmp_path, {"A": [(SNAPPER, 6), (COD, 30)]})
    ff, _, _ = _run(tmp_path, mapping)
    kept = ff[VOLUME]
    snapper = [
        s
        for s in kept
        if (tmp_path / VOLUME / f"{s}.txt").read_text().startswith(f"{SNAPPER} ")
    ]
    assert len(kept) == 10 and len(snapper) == 6


def test_rare_frames_survive_and_unmapped_frames_are_kept(tmp_path):
    mapping = _volume(tmp_path, {"A": [(SNAPPER, 40), (LOBSTER, 5)]})
    lost = next(iter(mapping))
    del mapping[lost]
    ff, _, stats = _run(tmp_path, mapping, rare={LOBSTER})
    kept = ff[VOLUME]
    lobster = [
        s
        for s in kept
        if (tmp_path / VOLUME / f"{s}.txt").read_text().startswith(f"{LOBSTER} ")
    ]
    assert len(lobster) == 5, "rare frames are exempt from the cap"
    assert lost in kept and stats["extra_frames_unmapped"] == 1
