"""Frames holding a scarce species must survive the per-drop cap.

`dominant_species` already drops blue-cod-only frames first, but once a drop has
more than `cap` frames containing SOME non-dominant species, the sampler thins
them at random — so a deployment's only lobster frames compete on equal terms
with its snapper frames and can be discarded (2026-08-25). Keeping real, varied
frames of the scarce animal beats duplicating one frame, which only adds
memorisation pressure.
"""

import random

from spyfish.ml.training.prepare_training_data import select_frames_per_drop

DROP = "KSF_20240124_BUV_KSF_085_01"
DOMINANT, SNAPPER, LOBSTER = {1}, 2, 3


def _drop(tmp_path, snapper=40, lobster=5, dominant=40, background=0):
    d = tmp_path / DROP
    d.mkdir(parents=True)
    i = 0
    for cid, n in ((SNAPPER, snapper), (LOBSTER, lobster), (1, dominant)):
        for _ in range(n):
            (d / f"s{cid}_{i}.txt").write_text(f"{cid} .5 .5 .1 .1\n")
            i += 1
    for _ in range(background):
        (d / f"bg_{i}.txt").write_text("")
        i += 1
    return tmp_path


def _run(root, rare_ids, cap=10):
    return select_frames_per_drop(
        drops={DROP},
        species_labels_dir=root,
        extras_set=set(),
        dominant_class_ids=DOMINANT,
        rare_class_ids=rare_ids,
        cap=cap,
        rng=random.Random(42),
    )


def _kept_of(root, kept, cid):
    return sum(
        1
        for stem in kept
        if (root / DROP / f"{stem}.txt").read_text().startswith(f"{cid} ")
    )


def test_without_the_exemption_lobster_frames_are_thinned(tmp_path):
    """Baseline: 5 lobster frames compete with 40 snapper frames for 10 slots."""
    root = _drop(tmp_path)
    ff, _, stats = _run(root, rare_ids=set())
    kept = ff[DROP]
    assert len(kept) == 10, "cap applies"
    assert _kept_of(root, kept, LOBSTER) < 5, "some lobster frames were dropped"
    assert stats["rare_exempt"] == 0


def test_exempt_lobster_frames_all_survive_and_do_not_consume_the_cap(tmp_path):
    root = _drop(tmp_path)
    ff, _, stats = _run(root, rare_ids={LOBSTER})
    kept = ff[DROP]
    assert _kept_of(root, kept, LOBSTER) == 5, "every lobster frame kept"
    # 5 exempt + a full cap of 10 from the rest, so the cap is not eaten by them
    assert len(kept) == 15
    assert stats["rare_exempt"] == 5


def test_dominant_only_frames_still_go_first(tmp_path):
    """The exemption must not disturb the existing blue-cod deprioritisation."""
    root = _drop(tmp_path, snapper=12, lobster=3, dominant=40)
    ff, _, _ = _run(root, rare_ids={LOBSTER})
    kept = ff[DROP]
    assert _kept_of(root, kept, LOBSTER) == 3
    assert _kept_of(root, kept, 1) == 0, "blue-cod-only frames trimmed first"


def test_drop_under_budget_is_untouched(tmp_path):
    root = _drop(tmp_path, snapper=3, lobster=2, dominant=2)
    ff, _, stats = _run(root, rare_ids={LOBSTER})
    assert len(ff[DROP]) == 7 and stats["under_budget"] == 1


def test_background_frames_are_reported_separately(tmp_path):
    """Empty .txt files are negatives, decided globally by background_ratio,
    and must never count toward the cap."""
    root = _drop(tmp_path, snapper=12, lobster=2, dominant=0, background=6)
    ff, bg, _ = _run(root, rare_ids={LOBSTER})
    assert len(bg[DROP]) == 6
    assert not (set(bg[DROP]) & ff[DROP]), "backgrounds are not in the frame filter"


def test_extras_bypass_the_cap_entirely(tmp_path):
    root = _drop(tmp_path)
    ff, _, stats = select_frames_per_drop(
        drops={DROP},
        species_labels_dir=root,
        extras_set={DROP},
        dominant_class_ids=DOMINANT,
        rare_class_ids={LOBSTER},
        cap=10,
        rng=random.Random(42),
    )
    assert len(ff[DROP]) == 85 and stats["extras_uncapped"] == 1


def test_bucket_classes_are_never_rare(tmp_path):
    """'fish' and 'bait' must not be cap-exempt however few frames they have.

    'fish' is the catch-all that absorbs every floored species and 'bait' is in
    essentially every frame, so treating them as rare exempts almost every frame
    and the cap stops applying. Measured 2026-08-25: drops sampled down by the
    cap fell 20 -> 6 and 2541 extra frames came in.
    """
    from spyfish.config.species import species_registry

    reg = species_registry()
    for bucket in ("fish", "bait"):
        sp = reg.get(bucket)
        assert sp is not None and sp.is_bucket, f"{bucket} must be a bucket class"
    # a real species must NOT be flagged as a bucket, or the guard would
    # silently exclude everything
    sp = reg.get("Jasus edwardsii")
    assert sp is not None and not sp.is_bucket
