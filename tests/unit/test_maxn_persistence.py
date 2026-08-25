"""
Persistence filter in process_maxn — the rolling-min / gap-fill semantics.

Measured basis (19 drops, 2026-08-21, see claude_docs/todo.md): 40% of
detection runs last exactly one sampled frame (detector misfires), while 49%
of gaps BETWEEN runs are one frame (threshold flicker on a present animal).
The filter must therefore kill isolated blips, forgive single-frame dropouts,
and — critically — leave everything else byte-identical, which is what the
persistence_seconds=0 defaults guarantee (verified against hand-computed
legacy output here and in tests/integration/test_process_maxn.py).

Fixtures simulate a 30 fps video sampled every 10th frame (ml_fps 3), so one
grid slot is 1/3 s: persistence_seconds=1.0 → window of 3 sampled frames,
gap_fill_seconds=0.4 → gaps of 1 frame close.
"""

import pandas as pd

from spyfish.config.base import NULL_DEPLOYMENT
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.ml.process_ml_annotations import _ingest_ml_annotations, process_maxn

DROP = "KSF_20240124_BUV_KSF_085_01"
MODEL = "species_test"
FPS = 30.0
STRIDE = 10  # video frames between sampled frames


def _det(slot, cls="Pseudolabrus miles", conf=0.8, n=1):
    """`n` detection rows of `cls` at sampled-grid slot `slot`."""
    return [
        {
            "frame": slot * STRIDE,
            "time_seconds": slot * STRIDE / FPS,
            "class": cls,
            "confidence": conf,
            "x": 100.0,
            "y": 100.0,
            "w": 40.0,
            "h": 30.0,
        }
        for _ in range(n)
    ]


def _slot_time(slot):
    return slot * STRIDE / FPS


def _run(tmp_path, rows, **kwargs):
    return process_maxn(
        raw_df=pd.DataFrame(rows),
        output_csv_path=str(tmp_path / f"{DROP}_{MODEL}_maxn.csv"),
        drop_id=DROP,
        interval_seconds=10,
        confidence_threshold=0.4,
        model_name=MODEL,
        **kwargs,
    )


def _row_for(result, species):
    matches = result[result[config.csv_scientific_name_column] == species]
    assert len(matches) == 1, f"expected exactly one {species} row, got {len(matches)}"
    return matches.iloc[0]


def test_blip_suppressed_sustained_kept(tmp_path):
    """A single-frame detection amid a sustained visit of another class is
    zeroed but keeps its row with spike provenance; the sustained class is
    untouched."""
    rows = []
    for s in range(9):
        rows += _det(s, "Pseudolabrus miles", conf=0.8)
    rows += _det(4, "Jasus edwardsii", conf=0.9)

    result = _run(tmp_path, rows, persistence_seconds=1.0, gap_fill_seconds=0.4)

    sustained = _row_for(result, "Pseudolabrus miles")
    assert sustained[config.csv_max_interval_column] == 1
    assert sustained[config.csv_raw_max_interval_column] == 1
    assert not sustained[config.csv_spike_flag_column]

    blip = _row_for(result, "Jasus edwardsii")
    assert blip[config.csv_max_interval_column] == 0
    assert blip[config.csv_raw_max_interval_column] == 1
    assert blip[config.csv_spike_flag_column]
    assert blip[config.csv_spike_time_seconds_column] == _slot_time(4)


def test_gap_fill_forgives_single_dropout(tmp_path):
    """Detection pattern 1,1,0,1,1: threshold flicker on a present animal.
    With gap fill the zero closes and the visit sustains; without it, two
    2-frame runs both fail a 3-frame window and the interval zeroes."""
    rows = []
    for s in (0, 1, 3, 4):
        rows += _det(s)

    filled = _run(tmp_path, rows, persistence_seconds=1.0, gap_fill_seconds=0.4)
    row = _row_for(filled, "Pseudolabrus miles")
    assert row[config.csv_max_interval_column] == 1
    assert not row[config.csv_spike_flag_column]

    unfilled = _run(tmp_path, rows, persistence_seconds=1.0, gap_fill_seconds=0.0)
    row = _row_for(unfilled, "Pseudolabrus miles")
    assert row[config.csv_max_interval_column] == 0
    assert row[config.csv_spike_flag_column]


def test_gap_fill_never_invents_an_isolated_blip(tmp_path):
    """Gap fill closes gaps BETWEEN detections; an isolated blip has zeros on
    both sides and must stay exposed to the window."""
    rows = []
    for s in range(9):
        rows += _det(s, "Pseudolabrus miles")  # establishes the grid
    rows += _det(4, "Jasus edwardsii")

    result = _run(tmp_path, rows, persistence_seconds=1.0, gap_fill_seconds=0.4)
    assert _row_for(result, "Jasus edwardsii")[config.csv_max_interval_column] == 0


def test_count_spike_reduced_to_sustained_level(tmp_path):
    """Counts 1,1,1,3,1,1,1: the 3 lasts one frame, so MaxN reports the
    sustained 1 while RawMaxInterval keeps the 3 for review selection."""
    rows = []
    for s in range(9):
        rows += _det(s, n=3 if s == 4 else 1)

    result = _run(tmp_path, rows, persistence_seconds=1.0, gap_fill_seconds=0.4)
    row = _row_for(result, "Pseudolabrus miles")
    assert row[config.csv_max_interval_column] == 1
    assert row[config.csv_raw_max_interval_column] == 3
    assert row[config.csv_spike_flag_column]
    assert row[config.csv_spike_time_seconds_column] == _slot_time(4)


def test_persistence_computed_across_interval_boundary(tmp_path):
    """A 3-frame visit straddling the 10s boundary (9.67s, 10.0s, 10.33s)
    sustains in the interval holding the window's middle frame. Per-interval
    computation would see two fragments and count zero everywhere."""
    rows = []
    for s in (29, 30, 31):
        rows += _det(s)

    result = _run(tmp_path, rows, persistence_seconds=1.0, gap_fill_seconds=0.4)
    by_interval = {
        (r[config.csv_maxn_time_seconds_column] // 10) * 10: r
        for _, r in result.iterrows()
    }
    assert by_interval[10.0][config.csv_max_interval_column] == 1
    # The boundary interval that only saw the visit's first frame is reduced,
    # not lost: its row survives with spike provenance.
    assert by_interval[0.0][config.csv_max_interval_column] == 0
    assert by_interval[0.0][config.csv_spike_flag_column]


def test_persistence_zero_reproduces_legacy_output(tmp_path):
    """persistence 0 / gap fill 0 must reproduce the pre-filter behaviour
    exactly: single-frame max per interval, ties broken by mean confidence."""
    rows = []
    rows += _det(3, conf=0.85, n=2)  # interval 0: MaxN 2, mean conf 0.85
    rows += _det(31, conf=0.72)  # interval 10: tie on count...
    rows += _det(35, conf=0.95)  # ...frame with higher mean conf wins

    result = _run(tmp_path, rows)  # defaults: persistence 0, no fill

    first = _row_for(
        result[result[config.csv_maxn_time_seconds_column] < 10], "Pseudolabrus miles"
    )
    assert first[config.csv_max_interval_column] == 2
    assert first[config.csv_confidence_agreement_column] == 0.85
    assert first[config.csv_maxn_time_seconds_column] == _slot_time(3)

    second = _row_for(
        result[result[config.csv_maxn_time_seconds_column] >= 10], "Pseudolabrus miles"
    )
    assert second[config.csv_max_interval_column] == 1
    assert second[config.csv_confidence_agreement_column] == 0.95
    assert second[config.csv_maxn_time_seconds_column] == _slot_time(35)
    # No spikes at window=1: raw and persistent are the same number.
    assert not result[config.csv_spike_flag_column].any()
    assert (
        result[config.csv_raw_max_interval_column]
        == result[config.csv_max_interval_column]
    ).all()


def test_excluded_class_gets_no_row(tmp_path):
    """Bait is a real fish strapped to the canister: it must not appear as its
    own MaxN row. Other classes are unaffected."""
    rows = []
    for s in range(9):
        rows += _det(s, "bait", conf=0.9)
        rows += _det(s, "Pagrus auratus", conf=0.8)

    result = _run(
        tmp_path,
        rows,
        persistence_seconds=1.0,
        gap_fill_seconds=0.4,
        exclude_classes=("bait",),
    )
    names = set(result[config.csv_scientific_name_column])
    assert "bait" not in names
    assert _row_for(result, "Pagrus auratus")[config.csv_max_interval_column] == 1


def test_every_class_keeps_its_own_row(tmp_path):
    """One row per class per interval, nothing derived: the 'fish' catch-all
    means "unidentified fish only", and a total-fish figure is a downstream
    sum, not a pipeline row (decided 2026-08-21, union row deleted)."""
    rows = []
    for s in range(9):
        rows += _det(s, "Pagrus auratus")
        rows += _det(s, "Parapercis colias")
        rows += _det(s, "fish")

    result = _run(tmp_path, rows, persistence_seconds=1.0, gap_fill_seconds=0.4)
    per_interval = result[result[config.csv_maxn_time_seconds_column] < 10]
    names = set(per_interval[config.csv_scientific_name_column])
    assert names == {"Pagrus auratus", "Parapercis colias", "fish"}
    for name in names:
        assert _row_for(per_interval, name)[config.csv_max_interval_column] == 1


def test_spike_only_deployment_ingests_null_row(tmp_path):
    """A deployment whose every detection is a suppressed spike is a real
    zero: the CSV keeps the spike rows, the annotations DB gets the
    NULL_DEPLOYMENT sentinel, and no species row with max_interval 0 leaks in."""
    rows = _det(4, "Jasus edwardsii", conf=0.9)
    result = _run(tmp_path, rows, persistence_seconds=1.0, gap_fill_seconds=0.4)

    assert len(result) == 1
    assert result.iloc[0][config.csv_max_interval_column] == 0
    assert result.iloc[0][config.csv_spike_flag_column]

    ann_db = AnnotationDatabaseManager(db_path=str(tmp_path / "annotations.db"))
    _ingest_ml_annotations(ann_db, DROP, result, MODEL)
    db_rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert len(db_rows) == 1
    assert db_rows[0]["scientific_name"] == NULL_DEPLOYMENT
    assert db_rows[0]["max_interval"] == 0


def test_mixed_real_and_spike_ingest_skips_only_spikes(tmp_path):
    """Suppressed rows stay CSV-only; sustained rows ingest normally."""
    rows = []
    for s in range(9):
        rows += _det(s, "Pseudolabrus miles")
    rows += _det(4, "Jasus edwardsii", conf=0.9)

    result = _run(tmp_path, rows, persistence_seconds=1.0, gap_fill_seconds=0.4)
    ann_db = AnnotationDatabaseManager(db_path=str(tmp_path / "annotations.db"))
    _ingest_ml_annotations(ann_db, DROP, result, MODEL)

    db_rows = ann_db.get_annotations_for_drop(DROP, "ml")
    assert {r["scientific_name"] for r in db_rows} == {"Pseudolabrus miles"}
