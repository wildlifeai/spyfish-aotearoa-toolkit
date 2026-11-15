"""`site_data.best_per_drop_species` must stay one row per (drop, species).

A drop reviewed in its own BIIGLE volume and again in a survey volume keeps a
peak per volume, both ranked "expert". Both rows used to survive, so the drop
counted twice in every site total. MaxN over both sets of frames is the larger
peak, so that is the row kept.
"""

import sys
from pathlib import Path

import pandas as pd

APP = Path(__file__).resolve().parents[2] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from doc_report.site_data import best_per_drop_species  # noqa: E402

DROP = "KSF_20240124_BUV_KSF_085_01"
COD = "Parapercis colias"


def _row(source, ext, maxn, species=COD):
    return {
        "drop_id": DROP,
        "scientific_name": species,
        "annotated_by": source,
        "external_id": ext,
        "maxn": maxn,
    }


def test_two_biigle_volumes_of_one_drop_count_once_at_the_larger_peak():
    df = pd.DataFrame(
        [
            _row("expert", "biigle_volume_1", 3),
            _row("expert", "biigle_volume_2", 5),
            _row("ml", "species_20260601", 9),
        ]
    )
    out = best_per_drop_species(df)
    assert len(out) == 1
    assert out.iloc[0]["maxn"] == 5
    assert out.iloc[0]["annotated_by"] == "expert"


def test_reviewed_empty_rows_from_two_volumes_collapse_too():
    df = pd.DataFrame(
        [
            _row("expert", "biigle_volume_1", 0, species=pd.NA),
            _row("expert", "biigle_volume_2", 0, species=pd.NA),
        ]
    )
    assert len(best_per_drop_species(df)) == 1
