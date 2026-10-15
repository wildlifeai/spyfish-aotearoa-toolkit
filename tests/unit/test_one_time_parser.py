"""Guards that legacy TimeOfMax text has exactly ONE reader.

The bug this prevents already happened in slow motion. Legacy interval times
are stored as text, the derived seconds column was NULL for every legacy row,
and three places wanted the number: the ingest, a backfill script and the
report dashboard. Each would have parsed the text itself, each copy written
from the previous one's docstring rather than its code, and the next edge case
(the Excel MM:SS coercion) fixed in only one of them. A deployment would then
sit at minute 1 on one page and minute 17 on another, with no error anywhere.

Two guards, because they fail on different mistakes:

* `test_every_path_agrees` catches a FORK: a second implementation that
  drifts. It feeds the nastiest fixture through every public path and demands
  identical numbers.
* `test_no_second_parser` catches an ADDITION: new time-parsing code landing
  outside the canonical module. It greps the tree, so it fires on code no
  test calls yet.

If a guard fails, the fix is almost never to extend the allowlist. It is to
call `legacy_time_of_max_seconds` (or `legacy_time_reading`, when the caller
also needs to report what was repaired) from
`spyfish/orchestrator/legacy_extract.py`.
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from doc_report import data as report_data  # noqa: E402

from spyfish.orchestrator.legacy_extract import (  # noqa: E402
    legacy_time_of_max_seconds,
    legacy_time_reading,
    parse_legacy_rows,
)

# Every shape the real import holds, including the ones that disagree between
# a naive read and the repair: an Excel-coerced MM:SS, an innocent-looking
# sibling in the same deployment, a nonzero-seconds row that must stay as
# written, a clean deployment, and unparseable text.
COERCED_DROP = "BNP_20170223_BUV_BNP_155_02"
CLEAN_DROP = "TON_20211026_BUV_TON_001_01"
FIXTURE = pd.DataFrame(
    {
        "drop_id": [
            COERCED_DROP,
            COERCED_DROP,
            COERCED_DROP,
            CLEAN_DROP,
            CLEAN_DROP,
        ],
        "time_of_max": [
            "23:45:00",  # coerced: 23 min 45 s
            "00:17:00",  # same sheet, so 17 s
            "00:25:30",  # nonzero seconds: genuine 25.5 min
            "00:10:00",  # clean sheet: 10 min
            "NULL DEPLOYMENT",  # unparseable
        ],
    }
)
EXPECTED = [23 * 60 + 45, 17.0, 25 * 60 + 30, 600.0, None]


def test_the_canonical_reader_is_right_about_the_fixture():
    """Pin the expected answers themselves, so "all paths agree" means agree
    on the CORRECT value rather than agreeing on a shared mistake."""
    seconds = legacy_time_of_max_seconds(
        FIXTURE["drop_id"], FIXTURE["time_of_max"]
    ).tolist()
    for got, want in zip(seconds, EXPECTED):
        if want is None:
            assert pd.isna(got)
        else:
            assert got == want


def test_every_path_agrees():
    """The ingest, the report and the workings-exposed variant are one reader.

    Any fork shows up here as a mismatch rather than as two different charts
    in production.
    """
    canonical = legacy_time_of_max_seconds(FIXTURE["drop_id"], FIXTURE["time_of_max"])

    # The variant that also reports what it repaired must not re-derive.
    reading = legacy_time_reading(FIXTURE["drop_id"], FIXTURE["time_of_max"])
    assert reading["seconds"].equals(canonical)
    # The repair fired on the two coerced rows and nothing else.
    assert reading["repaired"].tolist() == [True, True, False, False, False]

    # The report's accessor, which takes a frame rather than two series.
    via_report = report_data._legacy_seconds(FIXTURE)
    assert via_report.equals(canonical)

    # The ingest, which writes the value into every annotation record.
    csv = pd.DataFrame(
        {
            "DropID": FIXTURE["drop_id"],
            "ScientificName": "Pagrus auratus",
            "TimeOfMax": FIXTURE["time_of_max"],
            "MaxInterval": 1,
            "AnnotatedBy": "expert",
            "IntervalAnnotation": 30,
            "ConfidenceAgreement": "NA",
        }
    )
    annotations, _ = parse_legacy_rows(csv)
    via_ingest = [a["time_of_max_seconds"] for a in annotations]
    for got, want in zip(via_ingest, canonical.tolist()):
        assert (got is None and pd.isna(want)) or got == want


# Patterns that mean "someone is reading a time string". Deliberately broad:
# a guard that only catches the exact spelling already in the tree catches
# nothing new.
PARSER_PATTERNS = (
    re.compile(r"to_timedelta"),
    re.compile(r"strptime"),
    re.compile(r"\.str\.extract\(.*\\d\+\)?:"),
    re.compile(r"split\(['\"]:['\"]\)"),
)

# The only files allowed to parse time text, each for a stated reason. Three
# readers exist on purpose, at three different stages, and the reasons are the
# point: a new entry here needs one as good.
ALLOWED = {
    # The canonical reader for legacy hand-entered text, Excel repair included.
    "spyfish/orchestrator/legacy_extract.py",
    # `time_to_seconds`: the generic HH:MM:SS converter for timestamps the
    # PIPELINE wrote (BIIGLE frame filenames, Zooniverse clip offsets) and the
    # round-trip partner of `seconds_to_time`. It must NOT gain the legacy
    # repair: those timestamps are machine-generated and already correct, and
    # it returns 0.0 for missing input, which in the legacy path would turn
    # "time unknown" into "seen in the first minute".
    "spyfish/utils.py",
    # A DIFFERENT stage, and legitimately separate: `fix_time_of_max`
    # normalises the spelling of one cell while extracting from the source
    # workbooks (';' for ':', the '@' shift-typo) and runs per value, so it
    # cannot see the deployment's other rows — which is exactly what the
    # Excel-coercion rule needs. It produces the text the canonical reader
    # later reads. The script is one-off tooling that lives in the gitignored
    # `scripts/wip/` folder (not in the repo); the entry is kept so the guard
    # stays correct on a machine where that folder exists, and a checkout
    # without it simply has nothing there to scan.
    "scripts/wip/legacy_extract/legacy_annotations_extract.py",
    # This guard itself.
    "tests/unit/test_one_time_parser.py",
}

SEARCH_DIRS = ("spyfish", "app", "scripts", "in_progress")


def test_no_second_parser():
    """Fail when time-parsing code appears outside the canonical module.

    Catches the next copy at the moment it is written, which no behavioural
    test can do while nothing calls it yet.
    """
    offenders = []
    for folder in SEARCH_DIRS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "__pycache__" in path.parts or "_archive" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in ALLOWED:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in PARSER_PATTERNS:
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if pattern.search(line):
                        offenders.append(f"{relative}:{line_no}: {line.strip()}")

    assert not offenders, (
        "Time text is being parsed outside "
        "spyfish/orchestrator/legacy_extract.py. Call "
        "legacy_time_of_max_seconds (or legacy_time_reading) instead of "
        "parsing it again:\n  " + "\n  ".join(offenders)
    )
