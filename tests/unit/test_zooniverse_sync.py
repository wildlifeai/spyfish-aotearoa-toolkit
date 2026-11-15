"""
Tests for the clips sync (spyfish.zooniverse.sync): what one drop does with
its completion row. Panoptes, the parser and the annotations DB are patched;
the point is the gating around the fetch, above all that a retired set with
no classifications raises rather than becoming an absence record.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spyfish.config.wrapper import config
from spyfish.zooniverse import sync

DROP = "HOR_20251204_BUV_HOR_024_01"


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_project_root", tmp_path)
    return tmp_path


def _completion(pct, set_type="clips"):
    return pd.DataFrame(
        [
            {
                "project_id": 14054,
                "subject_set_id": "147931",
                "subject_set_type": set_type,
                "drop_id": DROP,
                "total": 36,
                "retired": int(36 * pct / 100),
                "pct_retired": pct,
                "workflow_ids": ["23924"],
                "fully_complete": pct >= 100,
            }
        ]
    )


def test_not_retired_or_missing_set_does_nothing(local_root):
    with patch.object(sync, "classifications_for_set") as fetch:
        assert not sync._sync_one_drop(DROP, _completion(60.0), False)
        assert not sync._sync_one_drop(DROP, _completion(100.0, "vframes"), False)
        fetch.assert_not_called()


def test_retired_set_without_classifications_raises(local_root):
    with patch.object(sync, "classifications_for_set", return_value=[]) as fetch:
        with patch.object(sync, "write_empty_zooniverse_maxn_csv") as empty:
            with pytest.raises(RuntimeError, match="Refusing to record absence"):
                sync._sync_one_drop(DROP, _completion(100.0), False)
    fetch.assert_called_once_with("147931", ["23924"])
    empty.assert_not_called()


def test_retired_set_is_fetched_parsed_and_advanced(local_root):
    raw = [{"classification_id": "1"}]
    parsed = pd.DataFrame([{"classification_id": "1", "drop_id": DROP}])
    aggregated = pd.DataFrame([{"drop_id": DROP, "species": "fish"}])
    with (
        patch.object(sync, "classifications_for_set", return_value=raw),
        patch.object(sync, "parse_classifications", return_value=parsed),
        patch.object(sync, "aggregate_by_subject_species", return_value=aggregated),
        patch.object(sync, "write_zooniverse_maxn_csv") as write,
        patch.object(sync, "ingest_zooniverse_annotations") as ingest,
    ):
        assert sync._sync_one_drop(DROP, _completion(100.0), False)
    write.assert_called_once()
    ingest.assert_called_once_with(DROP)
    assert config.get_zooniverse_raw_csv_path(DROP).exists()

    # Second run: the raw CSV is the cache, nothing is fetched.
    with (
        patch.object(sync, "classifications_for_set") as fetch,
        patch.object(sync, "aggregate_by_subject_species", return_value=aggregated),
        patch.object(sync, "write_zooniverse_maxn_csv"),
        patch.object(sync, "ingest_zooniverse_annotations"),
    ):
        assert sync._sync_one_drop(DROP, _completion(100.0), False)
        fetch.assert_not_called()


def test_run_pulls_only_workflows_of_retired_uncached_sets(local_root):
    db = MagicMock()
    db.get_deployments_eligible.return_value = [{"drop_id": DROP}]
    with (
        patch.object(sync, "DatabaseManager", return_value=db),
        patch.object(sync, "connect_to_zooniverse"),
        patch.object(
            sync, "subject_completion_from_api", return_value=_completion(100.0)
        ),
        patch.object(sync, "pull_workflows", return_value=set()) as pull,
        patch.object(sync, "_sync_one_drop", return_value=True),
    ):
        sync.sync_zooniverse_drops()
    pull.assert_called_once_with({(14054, "23924")})


class _FakeSet:
    def __init__(self, sid, name, count, completeness, workflows):
        self.id = sid
        self.display_name = name
        self.raw = {
            "set_member_subjects_count": count,
            "completeness": completeness,
            "links": {"workflows": workflows},
        }


def test_completion_ignores_completeness_of_unlinked_workflows():
    """Panoptes keeps a stale 1.0 for a workflow a set was unlinked from."""
    from spyfish.zooniverse import parse_classifications as pc

    sets = [
        _FakeSet("1", f"clips_{DROP}", 91, {"23924": 1.0}, []),  # unlinked, stale
        _FakeSet(
            "2", "clips_HOR_20251204_BUV_HOR_012_01", 30, {"23924": 1.0}, ["23924"]
        ),
        _FakeSet(
            "3",
            "vframes_HOR_20251204_BUV_HOR_003_01",
            37,
            {"1": 1.0, "32835": 0.5},
            ["32835"],
        ),
        _FakeSet("4", "clips_HOR_20251204_BUV_HOR_036_01", 0, {}, ["23924"]),  # empty
    ]
    with (
        patch("panoptes_client.SubjectSet") as SubjectSet,
        patch.object(
            type(config),
            "zooniverse_source_project_ids",
            property(lambda self: [14054]),
        ),
    ):
        SubjectSet.where.return_value = sets
        df = pc.subject_completion_from_api().set_index("subject_set_id")
    assert df.loc["1", "pct_retired"] == 0.0 and not df.loc["1", "fully_complete"]
    assert df.loc["1", "workflow_ids"] == []
    assert df.loc["2", "fully_complete"] and df.loc["2", "workflow_ids"] == ["23924"]
    assert df.loc["3", "pct_retired"] == 50.0 and not df.loc["3", "fully_complete"]
    assert df.loc["3", "subject_set_type"] == "vframes"
    assert not df.loc["4", "fully_complete"]


def test_drop_whose_workflow_pull_failed_is_skipped_not_synced(local_root):
    """A failed pull may leave a partial cache; aggregating it would write a
    MaxN from half the votes. The drop stays at clips_uploaded for next run."""
    db = MagicMock()
    db.get_deployments_eligible.return_value = [{"drop_id": DROP}]
    with (
        patch.object(sync, "DatabaseManager", return_value=db),
        patch.object(sync, "connect_to_zooniverse"),
        patch.object(
            sync, "subject_completion_from_api", return_value=_completion(100.0)
        ),
        patch.object(sync, "pull_workflows", return_value={(14054, "23924")}),
        patch.object(sync, "_sync_one_drop") as one,
    ):
        sync.sync_zooniverse_drops()
    one.assert_not_called()
