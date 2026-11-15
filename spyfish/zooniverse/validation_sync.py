"""
Zooniverse validation sync-back: crop verdicts (precision), missed fish (recall).

For each drop at ``zoo_crops_uploaded`` / ``zoo_frames_uploaded``, check whether
its ``crops_{drop}`` / ``vframes_{drop}`` subject set is fully retired. If yes,
fetch the set's classifications, read every answer through the workflow
VERSION it was given under, join it back to the ML box or frame the subject
was rendered from (the manifest ``--zooniverse-crops`` / ``--zooniverse-frames``
wrote), write per-drop verdict CSVs, and advance the section to ``*_complete``.

Outputs, all in the drop's annotations dir:

  crops   ``{drop}_zooniverse_crops_verdicts.csv``   one row per ML box shown:
          votes, yes/no counts, ``verdict`` (correct / wrong / split), and the
          box's species, label, confidence and pixels from the manifest.
  frames  ``{drop}_zooniverse_frames_verdicts.csv``  one row per frame shown:
          were the boxes right, was anything missed, how many ML boxes, how
          many consensus missed fish.
          ``{drop}_zooniverse_missed_boxes.csv``     one row per cluster of
          volunteer-drawn boxes that overlap no ML box, in VIDEO pixels, with
          how many volunteers drew it and whether that is a majority.

Why answers go through the workflow version
-------------------------------------------
Panoptes stores a single-answer task as an index. The missed-fish workflow's
first question flipped polarity on 2026-09-20 ("Are ALL fish in a box?" No →
draw, became "Any fish NOT in a box?" Yes → draw), so the same index means the
opposite thing under two versions. Every classification carries the version it
was given under; the version holds the question text, the answer labels and
the routing. This module reads labels and routing from that version, never a
bare index, and finds the tasks by their shape (the drawing task, the question
that routes to it, the other question) rather than by hard-coded ids.

Entry point is ``sync_validation_drops``, wired into ``run_pipeline.py`` behind
``--zooniverse-validation-sync``. Classifications come from the per-workflow
cache ``panoptes_fetch`` keeps current (new rows only); the set's subjects are
one bulk call. Idempotent: the joined classifications and the workflow
versions are cached on disk per drop; ``force=True`` rebuilds them.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from spyfish.config.base import ZooCropsStatus, ZooFramesStatus
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.render_annotations import iou
from spyfish.zooniverse.panoptes_fetch import (
    classifications_for_set,
    pull_workflows,
    workflows_of,
)
from spyfish.zooniverse.parse_classifications import (
    connect_to_zooniverse,
    subject_completion_from_api,
)
from spyfish.zooniverse.subject_keys import CROP_PATH_COLUMN, FRAME_PATH_COLUMN

# Subject metadata the upload wrote (upload._crop_meta / _validation_frame_meta).
DETECTION_KEY = "#DetectionKey"
FRAME_INDEX_KEY = "#FrameIndex"

VERDICT_CORRECT = "correct"
VERDICT_WRONG = "wrong"
VERDICT_SPLIT = "split"
VERDICT_NO_VOTES = "no_votes"

# The two polarities every single-choice answer is reduced to, as written in
# the vote CSVs. Version labels map onto them via zooniverse.validation.sync.
YES = "Yes"
NO = "No"

# ── workflow versions ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkflowVersion:
    """One version of a workflow: its tasks (with routing) and display strings."""

    tasks: dict
    strings: dict
    first_task: str

    @classmethod
    def from_raw(cls, raw: dict) -> "WorkflowVersion":
        return cls(
            tasks=raw.get("tasks") or {},
            strings=raw.get("strings") or {},
            first_task=raw.get("first_task") or "",
        )

    def question(self, task: str) -> str:
        return self.strings.get(f"{task}.question", "").strip()

    def answer_label(self, task: str, index: int) -> str:
        return self.strings.get(f"{task}.answers.{index}.label", str(index)).strip()

    def single_tasks(self) -> list[str]:
        return [t for t, d in self.tasks.items() if d.get("type") == "single"]

    def drawing_task(self) -> Optional[str]:
        return next(
            (t for t, d in self.tasks.items() if d.get("type") == "drawing"), None
        )

    def gate_task(self) -> Optional[str]:
        """The single-answer task with an answer that routes to the drawing task."""
        drawing = self.drawing_task()
        if drawing is None:
            return None
        for task in self.single_tasks():
            answers = self.tasks[task].get("answers") or []
            if any(a.get("next") == drawing for a in answers):
                return task
        return None

    def boxes_task(self) -> Optional[str]:
        """The single-answer task that is NOT the gate: 'are the boxes all right?'."""
        gate = self.gate_task()
        others = [t for t in self.single_tasks() if t != gate]
        return others[0] if others else None


def version_key(raw: dict) -> str:
    return f"{raw['major_number']}.{raw['minor_number']}"


class WorkflowVersions:
    """Workflow versions by id, cached as one JSON per workflow.

    A version missing from the cache triggers one re-fetch of that workflow's
    full version list (new versions are appended on Zooniverse as the team
    edits the workflow); a version still missing afterwards is an error.
    """

    def __init__(self, cache_dir: Path, fetch: Optional[Callable] = None):
        self.cache_dir = cache_dir
        self._fetch = fetch or fetch_workflow_versions
        self._cache: dict[str, dict[str, WorkflowVersion]] = {}

    def _path(self, workflow_id: str) -> Path:
        return self.cache_dir / f"workflow_{workflow_id}_versions.json"

    def _load(self, workflow_id: str, refresh: bool) -> dict[str, WorkflowVersion]:
        path = self._path(workflow_id)
        if refresh or not path.exists():
            raw_versions = self._fetch(workflow_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw_versions))
        else:
            raw_versions = json.loads(path.read_text())
        self._cache[workflow_id] = {
            version_key(v): WorkflowVersion.from_raw(v) for v in raw_versions
        }
        return self._cache[workflow_id]

    def get(self, workflow_id: str, version: str) -> WorkflowVersion:
        versions = self._cache.get(workflow_id) or self._load(workflow_id, False)
        if version not in versions:
            versions = self._load(workflow_id, True)
        if version not in versions:
            raise KeyError(
                f"workflow {workflow_id} has no version {version!r} on Zooniverse"
            )
        return versions[version]


def fetch_workflow_versions(workflow_id: str) -> list[dict]:
    """Every version of a workflow from Panoptes, oldest first. Needs a connection."""
    from panoptes_client.panoptes import Panoptes

    versions: list[dict] = []
    page = 1
    while True:
        resp = Panoptes.client().get(
            "/workflow_versions",
            params={"workflow_id": workflow_id, "page": page, "page_size": 100},
        )[0]
        versions += resp["workflow_versions"]
        if not resp["meta"]["workflow_versions"].get("next_page"):
            break
        page += 1
    return versions


# ── reading one classification ───────────────────────────────────────────────


def _single_answer(annotations: list, task: str) -> Optional[int]:
    for a in annotations:
        if a.get("task") == task and a.get("taskType") == "single":
            value = a.get("value")
            return int(value) if value is not None else None
    return None


def answer_label(
    annotations: list, task: str, version: WorkflowVersion
) -> Optional[str]:
    """The answer's label ("Yes"/"No"/...) for ``task`` under ``version``."""
    index = _single_answer(annotations, task)
    return None if index is None else version.answer_label(task, index)


def answer_polarity(label: Optional[str]) -> Optional[str]:
    """``YES`` / ``NO`` for a resolved answer label, ``None`` for no answer.

    The label text comes from the workflow version the answer was given
    under, so it can be reworded between versions. The accepted spellings
    live in ``zooniverse.validation.sync.yes_labels`` / ``no_labels``; a label
    in neither raises, because treating it as "no vote" would let a reworded
    answer empty every verdict without a trace.
    """
    if label is None:
        return None
    if label in config.zooniverse_validation_yes_labels:
        return YES
    if label in config.zooniverse_validation_no_labels:
        return NO
    raise ValueError(
        f"answer label {label!r} is neither a yes nor a no label; add it to "
        "zooniverse.validation.sync.yes_labels or no_labels"
    )


def routes_to_drawing(annotations: list, version: WorkflowVersion) -> Optional[bool]:
    """True if the gate answer sent the volunteer to draw: "some fish are missed".

    Routing, not label: the gate question's polarity flipped between versions,
    so "Yes" meant "none missed" under one and "some missed" under the next.
    """
    gate = version.gate_task()
    if gate is None:
        return None
    index = _single_answer(annotations, gate)
    if index is None:
        return None
    answers = version.tasks[gate].get("answers") or []
    return index < len(answers) and answers[index].get("next") == version.drawing_task()


def drawn_rectangles(annotations: list) -> list[tuple[float, float, float, float]]:
    """Volunteer rectangles as ``(x1, y1, x2, y2)`` in the subject image's pixels."""
    boxes = []
    for a in annotations:
        if a.get("taskType") != "drawing":
            continue
        for b in a.get("value") or []:
            if b.get("toolType") != "rectangle":
                continue
            hw, hh = b["width"] / 2, b["height"] / 2
            boxes.append(
                (
                    b["x_center"] - hw,
                    b["y_center"] - hh,
                    b["x_center"] + hw,
                    b["y_center"] + hh,
                )
            )
    return boxes


def _seconds(meta: dict) -> Optional[float]:
    try:
        start = datetime.fromisoformat(meta["started_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(meta["finished_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError):
        return None
    return round((end - start).total_seconds(), 1)


def _user(c: dict) -> str:
    """Who voted: user id, else the export's hashed ip, else the API's session
    token, else the classification itself (never collapse anonymous votes)."""
    return str(
        c.get("user_id")
        or c.get("user_ip")
        or c.get("session")
        or f"anon_{c.get('classification_id')}"
    )


def _version_for(c: dict, versions: WorkflowVersions) -> WorkflowVersion:
    meta = c.get("classification_metadata") or {}
    return versions.get(str(c["workflow_id"]), str(meta.get("workflow_version")))


# ── crops → precision ────────────────────────────────────────────────────────


def crop_votes(classifications: list[dict], versions: WorkflowVersions) -> pd.DataFrame:
    """One row per answer on a crop subject: who, which box, and the answer label.

    The crop workflow has one question; it is read from the version's
    ``first_task`` so a renamed task id does not break the sync. The question
    text is kept per row: the pilot's first wording asked about ALL fish in
    the crop rather than the boxed one, and anyone re-reading the votes can
    filter on it.
    """
    rows = []
    for c in classifications:
        meta = c.get("subject_metadata") or {}
        key = meta.get(DETECTION_KEY)
        if not key:
            continue
        version = _version_for(c, versions)
        task = version.first_task or next(iter(version.single_tasks()), None)
        if task is None:
            continue
        rows.append(
            {
                "classification_id": c.get("classification_id"),
                "user": _user(c),
                "DetectionKey": key,
                "workflow_version": (c.get("classification_metadata") or {}).get(
                    "workflow_version"
                ),
                "question": version.question(task),
                "answer": answer_polarity(
                    answer_label(c.get("annotations") or [], task, version)
                ),
                "seconds": _seconds(c.get("classification_metadata") or {}),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "classification_id",
            "user",
            "DetectionKey",
            "workflow_version",
            "question",
            "answer",
            "seconds",
        ],
    )


def _majority(n_yes: int, n_no: int) -> str:
    if n_yes + n_no == 0:
        return VERDICT_NO_VOTES
    if n_yes > n_no:
        return VERDICT_CORRECT
    if n_no > n_yes:
        return VERDICT_WRONG
    return VERDICT_SPLIT


def crop_verdicts(votes: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per crop that was shown: vote counts, verdict, and the box it judged.

    Manifest rows without a crop image (boxes never uploaded) are left out;
    a shown crop with no Yes/No vote gets ``no_votes``.
    """
    shown = manifest[manifest[CROP_PATH_COLUMN].notna()].copy()
    decided = votes[votes["answer"].isin([YES, NO])]
    counts = (
        decided.assign(yes=(decided["answer"] == YES).astype(int))
        .groupby("DetectionKey")
        .agg(n_votes=("yes", "size"), n_yes=("yes", "sum"))
        .reset_index()
    )
    out = shown.merge(counts, on="DetectionKey", how="left")
    out["n_votes"] = out["n_votes"].fillna(0).astype(int)
    out["n_yes"] = out["n_yes"].fillna(0).astype(int)
    out["n_no"] = out["n_votes"] - out["n_yes"]
    out["verdict"] = [_majority(y, n) for y, n in zip(out["n_yes"], out["n_no"])]
    keep = [
        config.drop_id_column,
        "DetectionKey",
        "FrameIndex",
        config.csv_clip_max_time_column,
        config.csv_scientific_name_column,
        "LabelShown",
        config.csv_confidence_agreement_column,
        "BBox",
        "ModelName",
        CROP_PATH_COLUMN,
        "n_votes",
        "n_yes",
        "n_no",
        "verdict",
    ]
    return out[[k for k in keep if k in out.columns]].reset_index(drop=True)


def crop_precision_by_label(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Per shown label: crops, correct, wrong, split, precision over the decided ones."""
    if verdicts.empty:
        return pd.DataFrame()
    by = verdicts.groupby("LabelShown")["verdict"]
    table = pd.DataFrame(
        {
            "crops": by.size(),
            "correct": by.apply(lambda v: int((v == VERDICT_CORRECT).sum())),
            "wrong": by.apply(lambda v: int((v == VERDICT_WRONG).sum())),
            "split": by.apply(lambda v: int((v == VERDICT_SPLIT).sum())),
        }
    )
    decided = table["correct"] + table["wrong"]
    table["precision"] = (
        (table["correct"] / decided.replace(0, pd.NA)).astype(float).round(2)
    )
    return table.sort_values("crops", ascending=False)


# ── frames → recall ──────────────────────────────────────────────────────────


def _manifest_boxes(row) -> list[tuple[float, float, float, float]]:
    """ML corner boxes of a manifest frame row, in video pixels."""
    return [tuple(b["bbox"]) for b in json.loads(row["Boxes"] or "[]")]


def frame_votes(
    classifications: list[dict],
    versions: WorkflowVersions,
    manifest: pd.DataFrame,
    miss_iou: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-answer rows and per-drawn-box rows for the validation frames.

    Drawn boxes come back in the subject image's pixels; ``RenderScale`` from
    the manifest maps them to video pixels, the space the ML boxes are in. A
    drawn box overlapping an ML box at IoU >= ``miss_iou`` is the volunteer
    redrawing that box, not a new fish; anything else is a miss.
    """
    by_frame = {int(r["FrameIndex"]): r for _, r in manifest.iterrows()}
    answers, drawn = [], []
    for c in classifications:
        meta = c.get("subject_metadata") or {}
        if meta.get(FRAME_INDEX_KEY) in (None, ""):
            continue
        frame = int(float(meta[FRAME_INDEX_KEY]))
        row = by_frame.get(frame)
        if row is None:
            logging.warning(f"frame {frame} is on Zooniverse but not in the manifest")
            continue
        version = _version_for(c, versions)
        annotations = c.get("annotations") or []
        scale = float(row["RenderScale"])
        ml_boxes = _manifest_boxes(row)
        n_drawn = n_miss = 0
        for sx1, sy1, sx2, sy2 in drawn_rectangles(annotations):
            box = (sx1 / scale, sy1 / scale, sx2 / scale, sy2 / scale)
            best = max((iou(box, m) for m in ml_boxes), default=0.0)
            is_miss = best < miss_iou
            n_drawn += 1
            n_miss += int(is_miss)
            drawn.append(
                {
                    "classification_id": c.get("classification_id"),
                    "user": _user(c),
                    "FrameIndex": frame,
                    "x1": round(box[0], 1),
                    "y1": round(box[1], 1),
                    "x2": round(box[2], 1),
                    "y2": round(box[3], 1),
                    "best_iou_with_ml": round(best, 3),
                    "is_miss": is_miss,
                }
            )
        boxes_task = version.boxes_task()
        missed = routes_to_drawing(annotations, version)
        answers.append(
            {
                "classification_id": c.get("classification_id"),
                "user": _user(c),
                "FrameIndex": frame,
                "workflow_version": (c.get("classification_metadata") or {}).get(
                    "workflow_version"
                ),
                "seconds": _seconds(c.get("classification_metadata") or {}),
                "n_ml_boxes": len(ml_boxes),
                "boxes_all_correct": (
                    answer_polarity(answer_label(annotations, boxes_task, version))
                    if boxes_task
                    else None
                ),
                "fish_missed": None if missed is None else (YES if missed else NO),
                "n_drawn": n_drawn,
                "n_new_fish": n_miss,
            }
        )
    answer_cols = [
        "classification_id",
        "user",
        "FrameIndex",
        "workflow_version",
        "seconds",
        "n_ml_boxes",
        "boxes_all_correct",
        "fish_missed",
        "n_drawn",
        "n_new_fish",
    ]
    drawn_cols = [
        "classification_id",
        "user",
        "FrameIndex",
        "x1",
        "y1",
        "x2",
        "y2",
        "best_iou_with_ml",
        "is_miss",
    ]
    return (
        pd.DataFrame(answers, columns=answer_cols),
        pd.DataFrame(drawn, columns=drawn_cols),
    )


def _mean_box(boxes: list[tuple]) -> tuple:
    n = len(boxes)
    return tuple(sum(b[i] for b in boxes) / n for i in range(4))


def cluster_missed_boxes(
    drawn: pd.DataFrame, votes_per_frame: dict[int, int], miss_iou: float
) -> pd.DataFrame:
    """Group volunteer miss boxes on a frame into one row per fish.

    Greedy: a box joins the first cluster whose mean box it overlaps at IoU >=
    ``miss_iou``, otherwise starts a new one. Supporters are distinct
    classifications (one volunteer drawing two boxes on one fish counts once).
    ``consensus`` is a strict majority of that frame's answers.
    """
    cols = [
        "FrameIndex",
        "x1",
        "y1",
        "x2",
        "y2",
        "n_supporters",
        "n_votes",
        "consensus",
    ]
    rows = []
    misses = drawn[drawn["is_miss"]] if len(drawn) else drawn
    for frame, group in misses.groupby("FrameIndex"):
        clusters: list[dict] = []
        for _, b in group.iterrows():
            box = (b["x1"], b["y1"], b["x2"], b["y2"])
            for cl in clusters:
                if iou(box, _mean_box(cl["boxes"])) >= miss_iou:
                    cl["boxes"].append(box)
                    cl["supporters"].add(b["classification_id"])
                    break
            else:
                clusters.append(
                    {"boxes": [box], "supporters": {b["classification_id"]}}
                )
        n_votes = votes_per_frame.get(int(frame), 0)
        for cl in clusters:
            x1, y1, x2, y2 = _mean_box(cl["boxes"])
            n_sup = len(cl["supporters"])
            rows.append(
                {
                    "FrameIndex": int(frame),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                    "n_supporters": n_sup,
                    "n_votes": n_votes,
                    "consensus": n_sup * 2 > n_votes,
                }
            )
    return pd.DataFrame(rows, columns=cols)


def frame_verdicts(
    answers: pd.DataFrame, missed: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    """One row per frame shown: boxes verdict, missed-fish verdict, consensus misses."""
    shown = manifest[manifest[FRAME_PATH_COLUMN].notna()].copy()
    shown["FrameIndex"] = shown["FrameIndex"].astype(int)
    if len(answers):
        agg = (
            answers.assign(
                ok=(answers["boxes_all_correct"] == YES).astype(int),
                not_ok=(answers["boxes_all_correct"] == NO).astype(int),
                miss=(answers["fish_missed"] == YES).astype(int),
                no_miss=(answers["fish_missed"] == NO).astype(int),
            )
            .groupby("FrameIndex")
            .agg(
                n_votes=("classification_id", "size"),
                n_boxes_ok=("ok", "sum"),
                n_boxes_not_ok=("not_ok", "sum"),
                n_say_missed=("miss", "sum"),
                n_say_none_missed=("no_miss", "sum"),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame(
            columns=[
                "FrameIndex",
                "n_votes",
                "n_boxes_ok",
                "n_boxes_not_ok",
                "n_say_missed",
                "n_say_none_missed",
            ]
        )
    out = shown.merge(agg, on="FrameIndex", how="left")
    for col in (
        "n_votes",
        "n_boxes_ok",
        "n_boxes_not_ok",
        "n_say_missed",
        "n_say_none_missed",
    ):
        out[col] = out[col].fillna(0).astype(int)
    out["n_ml_boxes"] = [len(_manifest_boxes(r)) for _, r in out.iterrows()]
    out["boxes_verdict"] = [
        _majority(y, n) for y, n in zip(out["n_boxes_ok"], out["n_boxes_not_ok"])
    ]
    # "correct" here means the majority said something WAS missed.
    missed_majority = [
        _majority(y, n) for y, n in zip(out["n_say_missed"], out["n_say_none_missed"])
    ]
    out["missed_verdict"] = [
        {VERDICT_CORRECT: "missed", VERDICT_WRONG: "none_missed"}.get(v, v)
        for v in missed_majority
    ]
    consensus = (
        missed[missed["consensus"]].groupby("FrameIndex").size()
        if len(missed)
        else pd.Series(dtype=int)
    )
    out["n_consensus_missed"] = out["FrameIndex"].map(consensus).fillna(0).astype(int)
    keep = [
        config.drop_id_column,
        "FrameIndex",
        config.csv_clip_max_time_column,
        config.selection_reason_column,
        "ModelName",
        FRAME_PATH_COLUMN,
        "n_ml_boxes",
        "n_votes",
        "n_boxes_ok",
        "n_boxes_not_ok",
        "boxes_verdict",
        "n_say_missed",
        "n_say_none_missed",
        "missed_verdict",
        "n_consensus_missed",
    ]
    return out[[k for k in keep if k in out.columns]].reset_index(drop=True)


# ── per-drop processing ──────────────────────────────────────────────────────


def _load_or_fetch(
    drop_id: str, kind: str, subject_set_id: str, workflow_ids: list, force: bool
) -> list:
    """The set's classifications, from the per-drop cache unless ``force``."""
    cache = config.get_validation_classifications_path(drop_id, kind)
    if cache.exists() and not force:
        logging.info(f"  {drop_id}: {kind} classifications from {cache.name}")
        return json.loads(cache.read_text())
    raw = classifications_for_set(subject_set_id, workflow_ids)
    if raw:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw))
    return raw


def process_crops(
    drop_id: str, classifications: list, versions: WorkflowVersions, miss_iou: float
) -> pd.DataFrame:
    manifest = pd.read_csv(config.get_validation_manifest_path(drop_id, "crops"))
    verdicts = crop_verdicts(crop_votes(classifications, versions), manifest)
    path = config.get_validation_verdicts_path(drop_id, "crops")
    path.parent.mkdir(parents=True, exist_ok=True)
    verdicts.to_csv(path, index=False)
    counts = verdicts["verdict"].value_counts().to_dict()
    logging.info(f"  {drop_id}: {len(verdicts)} crop verdict(s) {counts} → {path.name}")
    table = crop_precision_by_label(verdicts)
    if len(table):
        logging.info(f"  {drop_id}: precision by label\n{table.to_string()}")
    return verdicts


def process_frames(
    drop_id: str, classifications: list, versions: WorkflowVersions, miss_iou: float
) -> pd.DataFrame:
    manifest = pd.read_csv(config.get_validation_manifest_path(drop_id, "frames"))
    answers, drawn = frame_votes(classifications, versions, manifest, miss_iou)
    votes_per_frame = (
        answers.groupby("FrameIndex").size().to_dict() if len(answers) else {}
    )
    missed = cluster_missed_boxes(drawn, votes_per_frame, miss_iou)
    missed.insert(0, config.drop_id_column, drop_id)
    verdicts = frame_verdicts(answers, missed, manifest)

    missed_path = config.get_validation_missed_boxes_path(drop_id)
    missed_path.parent.mkdir(parents=True, exist_ok=True)
    missed.to_csv(missed_path, index=False)
    path = config.get_validation_verdicts_path(drop_id, "frames")
    verdicts.to_csv(path, index=False)

    n_consensus = int(missed["consensus"].sum()) if len(missed) else 0
    n_ml = int(verdicts["n_ml_boxes"].sum()) if len(verdicts) else 0
    logging.info(
        f"  {drop_id}: {len(verdicts)} frame verdict(s), {len(answers)} answer(s), "
        f"{len(missed)} drawn-box cluster(s), {n_consensus} by consensus "
        f"(recall upper bound ≈ {n_ml}/({n_ml}+{n_consensus})) → {path.name}, "
        f"{missed_path.name}"
    )
    return verdicts


@dataclass(frozen=True)
class _Kind:
    kind: str  # manifest / path kind: "crops" or "frames"
    set_type: str  # subject set display_name prefix without the underscore
    status: type  # ZooCropsStatus / ZooFramesStatus
    process: Callable


KINDS = (
    _Kind("crops", "crops", ZooCropsStatus, process_crops),
    _Kind("frames", "vframes", ZooFramesStatus, process_frames),
)


def _sync_one(
    k: _Kind,
    drop_id: str,
    completion: pd.DataFrame,
    versions: WorkflowVersions,
    db: DatabaseManager,
    force: bool,
) -> bool:
    """One drop, one kind. Returns True if the section advanced to COMPLETE."""
    rows = completion[
        (completion["drop_id"] == drop_id)
        & (completion["subject_set_type"] == k.set_type)
    ]
    if rows.empty:
        logging.info(f"  {drop_id}: no {k.set_type}_ set on Zooniverse yet, skipping.")
        return False
    row = rows.iloc[0]
    if not row["fully_complete"]:
        logging.info(
            f"  {drop_id}: {k.set_type} {int(row['retired'])}/{int(row['total'])} "
            f"retired ({row['pct_retired']:.0f}%), not ready yet."
        )
        return False
    manifest = config.get_validation_manifest_path(drop_id, k.kind)
    if not manifest.exists():
        logging.error(
            f"  {drop_id}: {k.kind} set is retired but {manifest.name} is not on "
            "disk; answers cannot be joined to boxes. Sync the annotations dir."
        )
        return False
    classifications = _load_or_fetch(
        drop_id, k.kind, row["subject_set_id"], row["workflow_ids"], force
    )
    if not classifications:
        raise RuntimeError(
            f"{drop_id}: {k.set_type} set {row['subject_set_id']} is fully retired "
            f"({int(row['total'])} subjects) but has no classifications on "
            f"workflow(s) {row['workflow_ids']}; nothing to join."
        )
    k.process(drop_id, classifications, versions, config.zooniverse_validation_miss_iou)
    db.advance_status(drop_id, k.status.COLUMN, k.status.COMPLETE)
    logging.info(f"  {drop_id}: → {k.status.COMPLETE}")
    return True


def sync_validation_drops(force: bool = False) -> None:
    """Run the validation sync-back for every drop at ``*_uploaded``.

    One completion query up front (every subject set in the source projects),
    one incremental pull per workflow that a retired, uncached set is attached
    to, then per drop and kind: join, write verdicts, advance.
    Per-drop isolation as in the clips sync: a Panoptes 5xx on one drop must
    not stop the others, and a failed drop stays at ``*_uploaded`` for the
    next run.
    """
    db = DatabaseManager()
    todo = {
        k: [
            r["drop_id"]
            for r in db.get_deployments_eligible(k.status.COLUMN, [k.status.UPLOADED])
        ]
        for k in KINDS
    }
    if not any(todo.values()):
        logging.info("No drops eligible for zooniverse-validation-sync.")
        return
    for k, drops in todo.items():
        logging.info(f"{len(drops)} drop(s) at {k.status.UPLOADED}.")

    connect_to_zooniverse()
    completion = subject_completion_from_api()
    if completion is None or completion.empty:
        logging.info("Completion data unavailable from Panoptes, nothing to do.")
        return
    versions = WorkflowVersions(config.zooniverse_workflow_versions_dir)

    to_fetch: dict = {}  # subject_set_id → (kind, drop_id) needing a pull
    for k, drops in todo.items():
        ready = completion[
            completion["drop_id"].isin(drops)
            & (completion["subject_set_type"] == k.set_type)
            & completion["fully_complete"]
        ]
        for _, r in ready.iterrows():
            cache = config.get_validation_classifications_path(r["drop_id"], k.kind)
            if force or not cache.exists():
                to_fetch[r["subject_set_id"]] = (k.kind, r["drop_id"])
    # A failed workflow pull may leave a partial cache; the drops that needed
    # it are skipped this run and stay at *_uploaded. Cached drops proceed.
    failed_pulls = (
        set(pull_workflows(workflows_of(completion, to_fetch))) if to_fetch else set()
    )
    blocked = {
        kind_drop
        for set_id, kind_drop in to_fetch.items()
        if workflows_of(completion, [set_id]) & failed_pulls
    }

    failed: list[str] = [f"{d} ({kind})" for kind, d in sorted(blocked)]
    for k, drops in todo.items():
        for drop_id in drops:
            if (k.kind, drop_id) in blocked:
                logging.error(
                    f"  {drop_id}: {k.kind} workflow pull failed, skipped this run."
                )
                continue
            try:
                _sync_one(k, drop_id, completion, versions, db, force)
            except Exception as e:  # noqa: BLE001, isolate per drop
                logging.error(
                    f"  {drop_id}: {k.kind} validation sync failed ({e}), continuing."
                )
                failed.append(f"{drop_id} ({k.kind})")
    if failed:
        logging.warning(
            f"zooniverse-validation-sync: {len(failed)} failed and remain at "
            f"uploaded: {failed}. Re-run to retry."
        )
