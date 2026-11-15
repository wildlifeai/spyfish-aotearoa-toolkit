"""
Tests for spyfish.zooniverse.panoptes_fetch.

Panoptes is never called: pages come from a fake ``get`` that records the
params it was asked for, and subjects from a dict. What is checked is the
cursor (what the second pull asks for, what lands on disk), the join (only
the set's subjects, once each, in the parser's dict shape) and the refusal
to join a set with no subjects.
"""

import json

import pandas as pd
import pytest

from spyfish.config.wrapper import config
from spyfish.zooniverse import panoptes_fetch as pf

PROJECT = 14054
WORKFLOW = "32835"


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_project_root", tmp_path)
    return tmp_path


def _raw(cid, subject_id, workflow=WORKFLOW, user="2817676", session="s1"):
    return {
        "id": str(cid),
        "created_at": "2026-10-15T19:38:06.935Z",
        "annotations": [{"task": "T0", "value": 1, "taskType": "single"}],
        "metadata": {"workflow_version": "18.58", "session": session},
        "links": {
            "project": str(PROJECT),
            "user": user,
            "workflow": workflow,
            "subjects": [str(subject_id)],
        },
    }


class FakeApi:
    """Serves ``rows`` NEWEST FIRST (as Panoptes does), ``page_size`` at a time,
    only rows with id greater than ``last_id``, with ``next_page`` in meta."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def __call__(self, path, params):
        self.calls.append(dict(params))
        after = int(params.get("last_id", 0))
        matching = sorted(
            (r for r in self.rows if int(r["id"]) > after),
            key=lambda r: int(r["id"]),
            reverse=True,
        )
        size, page = params["page_size"], params.get("page", 1)
        chunk = matching[(page - 1) * size : page * size]
        more = page * size < len(matching)
        return {
            "classifications": chunk,
            "meta": {"classifications": {"next_page": page + 1 if more else None}},
        }


def test_pull_walks_every_page_before_moving_the_cursor(local_root, monkeypatch):
    monkeypatch.setattr(pf, "PAGE_SIZE", 2)
    api = FakeApi([_raw(10, 1), _raw(11, 1), _raw(12, 2)])

    assert pf.pull_workflow(PROJECT, WORKFLOW, get=api) == 3
    rows = [
        json.loads(line) for line in pf._rows_path(WORKFLOW).read_text().splitlines()
    ]
    # Newest first from the API, and the older row on page two is NOT skipped.
    assert sorted(r["id"] for r in rows) == ["10", "11", "12"]
    assert pf._cursor_path(WORKFLOW).read_text() == "12"
    # No cursor on a first pull; the second page keeps the same (absent) cursor.
    assert [c.get("last_id") for c in api.calls] == [None, None]
    assert [c["page"] for c in api.calls] == [1, 2]
    assert api.calls[0]["project_id"] == str(PROJECT)
    assert api.calls[0]["workflow_id"] == WORKFLOW

    # A later pull sends the stored cursor and appends only what is new.
    api.rows.append(_raw(13, 3))
    api.calls.clear()
    assert pf.pull_workflow(PROJECT, WORKFLOW, get=api) == 1
    assert api.calls[0]["last_id"] == "12"
    assert len(pf._rows_path(WORKFLOW).read_text().splitlines()) == 4
    assert pf._cursor_path(WORKFLOW).read_text() == "13"


def test_pull_with_nothing_new_touches_nothing(local_root):
    pf._rows_path(WORKFLOW).parent.mkdir(parents=True)
    pf._cursor_path(WORKFLOW).write_text("99")
    api = FakeApi([_raw(50, 1)])  # older than the cursor
    assert pf.pull_workflow(PROJECT, WORKFLOW, get=api) == 0
    assert not pf._rows_path(WORKFLOW).exists()
    assert pf._cursor_path(WORKFLOW).read_text() == "99"


def test_classifications_for_set_joins_only_the_sets_subjects_once(local_root):
    other_workflow = "32407"
    api = FakeApi(
        [
            _raw(1, 101, user=None, session="anon-a"),
            _raw(2, 102),
            _raw(3, 999),  # a subject from another set on the same workflow
        ]
    )
    pf.pull_workflow(PROJECT, WORKFLOW, get=api)
    pf.pull_workflow(
        PROJECT, other_workflow, get=FakeApi([_raw(4, 101, other_workflow)])
    )
    # The same row cached under both workflows must come out once.
    with pf._rows_path(other_workflow).open("a") as fh:
        fh.write(json.dumps(_raw(1, 101, user=None, session="anon-a")) + "\n")

    subjects = {
        "101": {"metadata": {"#FrameIndex": 37377}, "locations": [{"image/jpeg": "u"}]},
        "102": {"metadata": {"#FrameIndex": 40000}, "locations": []},
    }
    records = pf.classifications_for_set("139847", [WORKFLOW, other_workflow], subjects)

    assert sorted(r["classification_id"] for r in records) == ["1", "2", "4"]
    first = next(r for r in records if r["classification_id"] == "1")
    assert first["subject_id"] == "101"
    assert first["subject_set_id"] == "139847"
    assert first["workflow_id"] == WORKFLOW
    assert first["subject_metadata"] == {"#FrameIndex": 37377}
    assert first["subject_locations"] == [{"image/jpeg": "u"}]
    assert first["classification_metadata"]["workflow_version"] == "18.58"
    assert first["user_id"] is None and first["session"] == "anon-a"
    assert first["annotations"][0]["taskType"] == "single"


def test_set_without_subjects_is_refused(local_root):
    with pytest.raises(ValueError, match="no subjects"):
        pf.classifications_for_set("139847", [WORKFLOW], subjects={})


def test_subjects_in_set_reads_id_metadata_locations():
    class S:
        def __init__(self, sid, meta, locs):
            self.id = sid
            self.metadata = meta
            self.raw = {"locations": locs}

    found = pf.subjects_in_set(
        "139847",
        where=lambda sid: [S(126452337, {"DropID": "x"}, [{"image/jpeg": "u"}])],
    )
    assert found == {
        "126452337": {"metadata": {"DropID": "x"}, "locations": [{"image/jpeg": "u"}]}
    }


def test_workflows_of_unions_the_sets_workflows():
    completion = pd.DataFrame(
        [
            {"project_id": 14054, "subject_set_id": 1, "workflow_ids": ["a", "b"]},
            {"project_id": 14054, "subject_set_id": 2, "workflow_ids": ["b"]},
            {"project_id": 30327, "subject_set_id": 3, "workflow_ids": ["c"]},
            {"project_id": 30327, "subject_set_id": 4, "workflow_ids": []},
        ]
    )
    assert pf.workflows_of(completion, ["1", 2, 4]) == {(14054, "a"), (14054, "b")}
    assert pf.workflows_of(completion, []) == set()


def test_pull_workflows_isolates_a_failing_workflow(local_root):
    """One workflow's 5xx must not stop the others; the failed pair is returned
    so the caller can skip the drops that needed it (their cache may be a
    partial prefix) while cached drops proceed."""
    calls = []

    def flaky(path, params):
        calls.append(params["workflow_id"])
        if params["workflow_id"] == "2":
            raise RuntimeError("502 Bad Gateway")
        return {"classifications": [], "meta": {"classifications": {}}}

    failed = pf.pull_workflows({(PROJECT, "1"), (PROJECT, "2"), (PROJECT, "3")}, flaky)
    assert failed == {(PROJECT, "2")}
    assert calls == ["1", "2", "3"]
