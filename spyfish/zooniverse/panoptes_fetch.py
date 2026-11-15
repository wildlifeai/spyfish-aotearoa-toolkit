"""
Incremental Panoptes classification fetch: one cursor per workflow.

What the classifications endpoint does and does not offer (checked live,
2026-11-03, project 14054):

* ``GET /classifications/project`` IGNORES ``subject_set_id``. A filter on an
  empty set still streamed the whole project, which is why the old per-set
  fetch took half an hour per drop. It then read retirement from a
  ``subject_data`` field that only the CSV export has, found nothing, and
  returned an empty list.
* It HONOURS ``workflow_id``, and ``last_id`` when ``project_id`` is also
  given: only rows with a greater id. That is a cursor. Rows come back in no
  id order (``sort`` is ignored), so a pull walks every page under one fixed
  ``last_id`` and moves the cursor only at the end.
* A classification carries no subject metadata, only ``links.subjects``. The
  subjects endpoint does filter by set: ``Subject.where(subject_set_id=...)``
  returns every subject's id, metadata and locations in one paged call.

So classifications are pulled per WORKFLOW, new rows only, and appended to one
JSON-lines file per workflow with the last id seen beside it. A set's
classifications are the cached rows whose subject is in that set, joined in
memory to the subject's metadata. The first pull of a busy legacy workflow is
the one large transfer; after that a run moves only what volunteers did since
the previous one. The CSV exports (``scripts/fetch_zooniverse_exports.py``)
remain the bulk backfill and recovery path.

Every function here raises on an API failure. A workflow file missing rows is
wrong data, not partial data.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import pandas as pd

from spyfish.config.wrapper import config

PAGE_SIZE = 100


# ── storage ──────────────────────────────────────────────────────────────────


def _rows_path(workflow_id: str) -> Path:
    return config.zooniverse_classifications_dir / f"workflow_{workflow_id}.jsonl"


def _cursor_path(workflow_id: str) -> Path:
    return config.zooniverse_classifications_dir / f"workflow_{workflow_id}.last_id"


# subject_id → byte offsets of that subject's rows, per workflow file. Keyed by
# (path, mtime_ns, size) so an appending pull invalidates it. One full parse of
# a workflow's history per run instead of one per drop: workflow 14054 is years
# of classifications, and a sync of thirty retired drops was thirty full scans.
_SUBJECT_OFFSETS: dict = {}


def _subject_offsets(workflow_id: str) -> dict:
    path = _rows_path(workflow_id)
    if not path.exists():
        return {}
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in _SUBJECT_OFFSETS:
        return _SUBJECT_OFFSETS[key]
    index: dict = {}
    with path.open("rb") as fh:
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            if line.strip():
                subject_ids = (json.loads(line).get("links") or {}).get(
                    "subjects"
                ) or []
                if subject_ids:
                    index.setdefault(str(subject_ids[0]), []).append(offset)
    # Drop stale entries for this file so the cache holds one index per workflow.
    for stale in [k for k in _SUBJECT_OFFSETS if k[0] == str(path)]:
        del _SUBJECT_OFFSETS[stale]
    _SUBJECT_OFFSETS[key] = index
    logging.info(
        f"  workflow {workflow_id}: indexed {sum(map(len, index.values()))} cached "
        f"classification(s) on {len(index)} subject(s)."
    )
    return index


def _rows_at(workflow_id: str, offsets: Iterable[int]) -> Iterator[dict]:
    offsets = list(offsets)
    if not offsets:
        return
    with _rows_path(workflow_id).open("rb") as fh:
        for offset in sorted(offsets):
            fh.seek(offset)
            yield json.loads(fh.readline())


# ── pulling ──────────────────────────────────────────────────────────────────


def _api_get(path: str, params: dict) -> dict:
    """One authenticated GET against Panoptes; needs ``connect_to_zooniverse``."""
    from panoptes_client.panoptes import Panoptes

    return Panoptes.client().get(path, params=params)[0]


def pull_workflow(
    project_id: int, workflow_id: str, get: Callable[[str, dict], dict] = _api_get
) -> int:
    """Append every classification on ``workflow_id`` newer than the cursor.

    ``last_id`` stays fixed for the whole pull while every page after it is
    walked: Panoptes does not return rows in id order, so advancing the cursor
    page by page would skip the older rows of later pages (it did, on the
    first live run). The cursor moves to the largest id seen only once the
    last page is in. Rows are appended as they arrive, so an interrupted pull
    leaves a complete prefix that the next pull simply re-fetches; the reader
    de-duplicates by id. Returns the number of rows appended.
    """
    rows_path, cursor_path = _rows_path(workflow_id), _cursor_path(workflow_id)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    last_id = cursor_path.read_text().strip() if cursor_path.exists() else None
    n_new, newest, page = 0, int(last_id or 0), 1
    while True:
        params = {
            "project_id": str(project_id),
            "workflow_id": str(workflow_id),
            "page_size": PAGE_SIZE,
            "page": page,
        }
        if last_id:
            params["last_id"] = last_id
        response = get("/classifications/project", params)
        rows = response.get("classifications") or []
        if not rows:
            break
        with rows_path.open("a") as fh:
            for raw in rows:
                fh.write(json.dumps(raw) + "\n")
        n_new += len(rows)
        newest = max(newest, *(int(raw["id"]) for raw in rows))
        meta = (response.get("meta") or {}).get("classifications") or {}
        if not meta.get("next_page"):
            break
        page += 1
    if newest:
        cursor_path.write_text(str(newest))
    logging.info(
        f"  workflow {workflow_id}: {n_new} new classification(s), "
        f"cursor at {newest or 'none'} → {rows_path.name}"
    )
    return n_new


def workflows_of(completion: pd.DataFrame, subject_set_ids: Iterable) -> set:
    """``(project_id, workflow_id)`` pairs the given subject sets are attached to.

    ``completion`` is ``subject_completion_from_api()``'s frame, whose
    ``workflow_ids`` column is the set's ``links.workflows``.
    """
    wanted = {str(s) for s in subject_set_ids}
    rows = completion[completion["subject_set_id"].astype(str).isin(wanted)]
    return {
        (int(row["project_id"]), str(workflow_id))
        for _, row in rows.iterrows()
        for workflow_id in (row["workflow_ids"] or [])
    }


def pull_workflows(
    pairs: Iterable[tuple], get: Callable[[str, dict], dict] = _api_get
) -> set:
    """Pull each ``(project_id, workflow_id)`` once, in a stable order.

    One workflow's failure (Panoptes 5xx under load) must not stop the pull
    of the others or the sync of drops whose sets are already cached, so each
    pull is isolated and the failed pairs are returned. A caller must not
    sync a drop whose workflow failed to pull: its cache may be a partial
    prefix, and aggregating that would write a MaxN from half the votes.
    """
    failed: set = set()
    for project_id, workflow_id in sorted(set(pairs)):
        try:
            pull_workflow(project_id, workflow_id, get)
        except Exception as e:  # noqa: BLE001, isolate per workflow
            logging.error(
                f"  workflow {workflow_id}: pull failed ({type(e).__name__}: {e}); "
                "drops on this workflow are skipped this run."
            )
            failed.add((project_id, workflow_id))
    return failed


# ── joining a set's classifications to its subjects ──────────────────────────


def subjects_in_set(
    subject_set_id: str, where: Optional[Callable[[str], Iterable]] = None
) -> dict:
    """``{subject_id: {"metadata": ..., "locations": [...]}}`` for one set.

    One paged bulk call (``Subject.where(subject_set_id=...)``), the only
    set-filtered endpoint Panoptes offers for this join.
    """
    if where is None:
        from panoptes_client import Subject

        def where(sid: str):
            return Subject.where(subject_set_id=sid)

    subjects = {}
    for subject in where(str(subject_set_id)):
        raw = getattr(subject, "raw", None) or {}
        subjects[str(subject.id)] = {
            "metadata": dict(getattr(subject, "metadata", None) or {}),
            "locations": list(raw.get("locations") or []),
        }
    return subjects


def _to_record(raw: dict, subject_set_id: str, subject_id: str, subject: dict) -> dict:
    """The dict shape ``parse_classifications`` and the validation sync read."""
    links = raw.get("links") or {}
    meta = raw.get("metadata") or {}
    return {
        "classification_id": raw.get("id"),
        "created_at": raw.get("created_at"),
        # The API has neither user_name nor the hashed user_ip the CSV export
        # carries. Anonymous volunteers are told apart by the browser session
        # token instead (the volunteer key in parse_classifications falls
        # through to it).
        "user_name": None,
        "user_id": links.get("user"),
        "user_ip": None,
        "session": meta.get("session"),
        "annotations": raw.get("annotations") or [],
        "subject_id": subject_id,
        "subject_set_id": str(subject_set_id),
        "workflow_id": str(links.get("workflow")),
        "subject_metadata": subject["metadata"],
        "subject_locations": subject["locations"],
        "classification_metadata": meta,
    }


def classifications_for_set(
    subject_set_id: str,
    workflow_ids: Iterable[str],
    subjects: Optional[dict] = None,
) -> list:
    """Every cached classification on a subject of ``subject_set_id``.

    Call ``pull_workflows`` for the set's workflows first; this reads only the
    on-disk cache. ``subjects`` (as ``subjects_in_set`` returns it) is fetched
    when not given. Raises if the set has no subjects: a retired set with
    nothing in it cannot be joined, and treating that as "no answers" is how
    an absence record gets written by mistake.
    """
    if subjects is None:
        subjects = subjects_in_set(subject_set_id)
    if not subjects:
        raise ValueError(f"subject set {subject_set_id} has no subjects on Zooniverse")

    seen: set = set()
    records = []
    for workflow_id in workflow_ids:
        index = _subject_offsets(str(workflow_id))
        offsets = [o for sid in subjects if sid in index for o in index[sid]]
        for raw in _rows_at(str(workflow_id), offsets):
            subject_id = str(((raw.get("links") or {}).get("subjects") or [None])[0])
            if raw.get("id") in seen:
                continue
            seen.add(raw.get("id"))
            records.append(
                _to_record(raw, subject_set_id, subject_id, subjects[subject_id])
            )
    logging.info(
        f"  subject set {subject_set_id}: {len(records)} classification(s) on "
        f"{len(subjects)} subject(s) from workflow(s) {list(workflow_ids)}."
    )
    return records
