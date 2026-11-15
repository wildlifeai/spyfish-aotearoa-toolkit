"""Read an ML raw CSV the way every downstream stage should see it.

The raw CSV is the record of what the model saw, so it keeps every box,
including the ones the second-stage classifier named
``ml_inference.classifier.reject_label`` (``not a fish``: the class trained on
boxes BIIGLE experts deleted). Those are not detections. Every reader that
turns the raw CSV into something people see or count (MaxN, BIIGLE and QA
frames, Zooniverse validation frames and crops) reads through here so they
drop out in one place. Analysis of the classifier itself
(``scripts/compare_two_stage_names.py``) reads the file directly, on purpose.
"""

import logging
from pathlib import Path

import pandas as pd

from spyfish.config.wrapper import config


def drop_rejected(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """``df`` without the rows the classifier named the reject label."""
    if df.empty or "class" not in df.columns:
        return df
    rejected = df["class"] == config.classifier_reject_label
    if rejected.any():
        logging.info(
            f"{source or 'raw CSV'}: {int(rejected.sum())} of {len(df)} boxes are "
            f"'{config.classifier_reject_label}', left out."
        )
        return df[~rejected]
    return df


def read_raw_detections(path) -> pd.DataFrame:
    """The raw CSV at ``path`` minus rejected boxes."""
    return drop_rejected(pd.read_csv(path), Path(path).name)
