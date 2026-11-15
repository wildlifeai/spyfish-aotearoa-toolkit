"""The frozen-test promotion gate in evaluate.py must tell "no production
model" (first model: promote) from "production model could not be scored"
(keep existing). Reading the second as the first promoted whatever had just
been trained because scoring the incumbent hiccupped."""

from pathlib import Path
from unittest.mock import patch

import pytest

from spyfish.ml.training import evaluate as ev

NEW = {"all": {"mAP50": 0.9}, "test_set": {"version": "v1"}}


def test_unscorable_production_is_a_refusal_not_a_first_model():
    ok, why, record = ev.frozen_test_decision(NEW, ev.PRODUCTION_UNSCORABLE)
    assert ok is False
    assert "could not be scored" in why
    assert record is None  # nothing comparable to tabulate


def test_no_production_still_promotes_first_model():
    ok, why, record = ev.frozen_test_decision(NEW, None)
    assert ok is True and record is None


def test_record_on_frozen_test_marks_production_unscorable(tmp_path):
    prod = tmp_path / "prod.pt"
    prod.write_bytes(b"")

    def score(path, dataset_dir):
        if path == str(prod):
            raise RuntimeError("CUDA out of memory")
        return NEW

    with patch.object(ev, "evaluate_on_test", side_effect=score):
        new, prod_record = ev._record_on_frozen_test("new.pt", str(prod), Path("."))
    assert new == NEW
    assert prod_record == ev.PRODUCTION_UNSCORABLE


def test_record_on_frozen_test_new_model_failure_propagates(tmp_path):
    with patch.object(ev, "evaluate_on_test", side_effect=RuntimeError("corrupt")):
        with pytest.raises(RuntimeError, match="corrupt"):
            ev._record_on_frozen_test("new.pt", "", Path("."))
