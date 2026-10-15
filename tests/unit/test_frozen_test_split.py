"""Frozen test drops are test in every retrain, whatever the splitter picks.

`balance_val_drops` always returned an empty test split and nothing read a test
list, so a hand-checked test set would have been trained on at the next
retrain (found 2026-10-15). Two models are only comparable on test if the
same drops are test both times.
"""

from spyfish.ml.training.split_data import apply_frozen_test

A, B, C, D = (
    "SLI_20260114_BUV_SLI_001_01",
    "SLI_20260114_BUV_SLI_003_01",
    "TUK_20250124_BUV_TUK_002_01",
    "BNP_20250118_BUV_BNP_121_01",
)


def test_frozen_drops_leave_train_and_val():
    train, val, test = apply_frozen_test([A, B], [C], [], {A, C})
    assert (train, val, test) == ([B], [], [A, C])


def test_no_frozen_list_changes_nothing():
    assert apply_frozen_test([A, B], [C], [D], set()) == ([A, B], [C], [D])


def test_listed_drop_without_training_data_is_skipped_and_logged(caplog):
    with caplog.at_level("WARNING"):
        train, val, test = apply_frozen_test([A], [B], [], {C})
    assert test == [] and train == [A] and val == [B]
    assert any(C in r.message for r in caplog.records)


def test_drop_already_in_test_is_not_duplicated():
    _, _, test = apply_frozen_test([A], [], [B, B], {B})
    assert test == [B]
