"""Which species the second-stage classifier names, and the record of that choice.

One rule, shared by the probe evaluation, the fine-tune, the hard-negative
review and the demo so they never disagree: a species is a class when it has
at least ``training.classifier.min_drops`` drops and ``min_crops`` crops;
everything else trains and scores as ``other_class``. Drops matter because the
fine-tune is held out by drop. The UoA source fixes the paper's four species
instead, whatever their counts.

``write_class_table`` puts the decision beside the checkpoint as
``classes.json``, the way the YOLO data prep writes ``rollup_map.json``: which
species became classes, which fell to ``other``, with their counts and the
thresholds that decided it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from spyfish.config.wrapper import config
from spyfish.ml.classifier import artifacts
from spyfish.ml.classifier.crops import SOURCE, UOA_CLASSES, UOA_OTHER, UOA_STILLS

# The catch-all class. UoA uses the paper's name for it, `fish`, so the
# classifier's output can be written straight into a raw CSV `class` column
# and compared with the 5-class YOLO's own `fish` class.
OTHER = UOA_OTHER if SOURCE == "uoa" else config.classifier_training_other_class
N_FOLDS = config.classifier_training_folds
CLASS_TABLE_FILENAME = artifacts.CLASS_TABLE
# Groups that train in every fold and are never scored: the UoA snapper stills
# are other people's footage, so a score on them says nothing about UoA video.
ALWAYS_TRAIN_GROUPS = (UOA_STILLS,)


def fit_probe(X: np.ndarray, y: np.ndarray):
    """The linear probe on frozen embeddings, fitted: standardise, then a
    class-balanced logistic regression (``training.classifier.probe``).

    One recipe for every script that compares something with "the probe".
    Returns the fitted pipeline: ``.predict(X)``, ``.predict_proba(X)`` and
    ``.classes_`` work on raw embeddings.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    recipe = config.classifier_probe
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=recipe.max_iter, C=recipe.c, class_weight="balanced"
        ),
    ).fit(X, y)


def probe_probabilities(probe, X: np.ndarray, classes: list) -> np.ndarray:
    """``probe.predict_proba(X)`` laid out in ``classes`` order, with a zero
    column for any class the probe never saw in training (a fold can miss one)."""
    p = probe.predict_proba(X)
    full = np.zeros((len(X), len(classes)))
    for ci, c in enumerate(probe.classes_):
        full[:, classes.index(c)] = p[:, ci]
    return full


def grouped_folds(
    groups: np.ndarray, n_splits: int = N_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold (train, test) index pairs over ``groups``.

    Rows of ``ALWAYS_TRAIN_GROUPS`` join every train set and no test set; the
    folds rotate over the other groups only. Without such rows this is exactly
    ``GroupKFold(min(n_splits, n_groups)).split``. Callers score only rows that
    were in a test set (``scored_rows``).
    """
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    fixed = np.isin(groups, ALWAYS_TRAIN_GROUPS)
    idx, extra = np.flatnonzero(~fixed), np.flatnonzero(fixed)
    n = min(n_splits, len(np.unique(groups[idx])))
    return [
        (np.sort(np.concatenate([idx[tr], extra])), idx[te])
        for tr, te in GroupKFold(n_splits=n).split(idx, groups=groups[idx])
    ]


def scored_rows(groups: np.ndarray) -> np.ndarray:
    """Boolean mask of the rows ``grouped_folds`` puts in a test set."""
    return ~np.isin(np.asarray(groups), ALWAYS_TRAIN_GROUPS)


def classes_for(
    meta: pd.DataFrame,
    source: str = SOURCE,
    min_drops: Optional[int] = None,
    min_crops: Optional[int] = None,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """``(target, classes, stats)``: which species get their own class.

    ``meta`` is ``crops_meta.csv`` (one row per crop with ``species`` and
    ``drop``). ``target`` are the species with a class of their own,
    ``classes`` is ``target + [OTHER]`` in training order, ``stats`` has crops,
    drops and the tier each species landed in, busiest first.
    """
    min_drops = config.classifier_training_min_drops if min_drops is None else min_drops
    min_crops = config.classifier_training_min_crops if min_crops is None else min_crops
    stats = meta.groupby("species").agg(
        crops=("crop_id", "size"), drops=("drop", "nunique")
    )
    other = UOA_OTHER if source == "uoa" else OTHER
    if source == "uoa":
        target = [c for c in UOA_CLASSES if c in stats.index]
        missing = [c for c in UOA_CLASSES if c not in stats.index]
        if missing:
            logging.warning(f"UoA classes with no crops on disk: {missing}")
        stats["tier"] = np.where(stats.index.isin(target), "class (fixed)", other)
    else:
        stats["tier"] = np.where(
            (stats["drops"] >= min_drops) & (stats["crops"] >= min_crops),
            "class",
            other,
        )
        target = [s for s in stats.index if stats.loc[s, "tier"] == "class"]
    stats = stats.sort_values("crops", ascending=False)
    return target, [*target, other], stats


def write_class_table(
    path: Path,
    target: list[str],
    classes: list[str],
    stats: pd.DataFrame,
    source: str = SOURCE,
    extra: Optional[dict] = None,
) -> Path:
    """The decision log beside a checkpoint: classes in training order, the
    rule that made them, and every species' counts and tier."""
    common = _registry().common_name_map()
    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "min_drops": config.classifier_training_min_drops,
        "min_crops": config.classifier_training_min_crops,
        "other_class": classes[-1] if classes else OTHER,
        "classes": list(classes),
        "target": list(target),
        "species": [
            {
                "species": s,
                "common_name": common.get(s, ""),
                "crops": int(r["crops"]),
                "drops": int(r["drops"]),
                "tier": str(r["tier"]),
            }
            for s, r in stats.iterrows()
        ],
        **(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    logging.info(
        f"Wrote classifier class table: {len(target)} classes + {doc['other_class']!r} -> {path}"
    )
    return path


def read_class_table(path: Path) -> Optional[dict]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _registry():
    from spyfish.config.species import species_registry

    return species_registry()
