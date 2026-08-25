"""Collapse a frame's `SelectionReason` onto the BIIGLE label that explains it.

`SelectionReason` is written per frame by the selection strategy in
`spyfish/extraction/select_frames.py`, carried through the selections CSV, and
reaches Biigle upload on the frames DataFrame. The strings are *parameterised*
("MaxN (Parapercis colias)", "Fish size band 2", "ML peak (conf>=0.25,
count=4)"), because they were written to be read by a human debugging a
selection, not consumed by an API.

Biigle attaches label IDs from a label tree, never free text, so that open
string space has to collapse onto a fixed vocabulary before it can be uploaded.
That is all this module does: reason string → canonical key → configured
Biigle label ID.

The canonical keys are internal and stable; the human-readable label names live
only in the Biigle label tree, so renaming a label there needs no code change.
Matching is on the leading token rather than the whole string precisely because
the tail carries the parameters, and a new species or a retuned threshold must
not silently stop matching.
"""

import logging
import re
from typing import Optional

from spyfish.config.wrapper import config

# Canonical key → pattern matched against the start of the reason string.
#
# Ordered, first match wins. Anchored at the start so a species name that
# happens to contain another bucket's word cannot cross-match: the parameters
# always live in the tail, never the head.
#
# Every producer in select_frames.py is covered:
#   _run_selection        → MaxN / Uncertain / Fish size band / Video Start
#   _blind_selections     → Blind (False Negative Check)
#   _ml_peak_selections   → ML peak (conf>=..., count=...)
#
# The `Confusing` / `Absolute MaxN` / `Empty` rules are LEGACY SPELLINGS, not
# dead code. `upsert_selections` merges each prior pass's selections CSV into
# the fresh one, so a drop first selected under older code keeps its original
# reason strings on disk forever, and a second pass re-uploads them. There are
# rows reading "Confusing (fish)" in the repo today; the bucket was renamed to
# "Uncertain" without rewriting history. `select_clips.py` still uses the older
# vocabulary for the citsci clip path, which is the other way these reach a
# frames DataFrame.
_REASON_RULES: tuple[tuple[str, str], ...] = (
    ("maxn_peak", r"(Absolute )?MaxN\b"),
    ("uncertain_id", r"(Uncertain|Confusing)\b"),
    ("ml_peak", r"ML peak\b"),
    ("fish_variety", r"Fish size band\b"),
    ("spot_check", r"(Blind|(Global )?Empty)\b"),
    ("video_start", r"(Global )?Video Start\b"),
)

# Reasons seen once and then not again, so the log stays one line per unknown
# shape per process rather than one per frame.
_warned_unknown: set = set()


def canonical_reason(reason: Optional[str]) -> Optional[str]:
    """Canonical key for a raw `SelectionReason`, or None if unrecognised.

    None (rather than a fallback key) is deliberate: an unmatched reason means
    the selection strategy grew a bucket this module has not been taught, and
    labelling those frames with the wrong reason is worse than labelling them
    with nothing. The caller logs and skips.
    """
    if reason is None:
        return None
    text = str(reason).strip()
    if not text:
        return None

    for key, pattern in _REASON_RULES:
        if re.match(pattern, text, flags=re.IGNORECASE):
            return key

    if text not in _warned_unknown:
        _warned_unknown.add(text)
        logging.warning(
            f"SelectionReason {text!r} matches no known bucket, no Biigle "
            "'Pick' label will be attached for these frames. Add a rule to "
            "spyfish/biigle/selection_reason.py if this is a new bucket."
        )
    return None


# Buckets whose parenthetical names the species the frame is EVIDENCE FOR.
# The others parameterise something else entirely ("ML peak (conf>=0.25,
# count=4)") or nothing, so parsing them for species would produce garbage.
_SPECIES_BEARING_BUCKETS = frozenset({"maxn_peak", "uncertain_id"})

# "MaxN (Parapercis colias, Pagrus auratus)" → the inner list. Non-greedy up to
# the FIRST closing paren so a species name containing one cannot run away.
_PARENTHETICAL = re.compile(r"\(([^)]*)\)")


def species_in_reason(reason: Optional[str]) -> list:
    """Species named by a reason string, in the order written.

    Only the MaxN and Uncertain buckets carry species; everything else returns
    empty. `collapse_peaks` merges peaks that land on one frame, so a single
    MaxN reason can name several species and each is returned separately.

    Note this is the species the frame is *evidence for*, which is not the same
    as everything visible in it: a blue cod peak frame may also show a spotty
    that the boxes label but this list does not.
    """
    if canonical_reason(reason) not in _SPECIES_BEARING_BUCKETS:
        return []

    match = _PARENTHETICAL.search(str(reason))
    if not match:
        return []
    return [name.strip() for name in match.group(1).split(",") if name.strip()]


def reason_label_id(reason: Optional[str]) -> Optional[int]:
    """Biigle label ID for a raw `SelectionReason`, or None to skip.

    None covers both "reason not recognised" and "recognised but no label has
    been created in Biigle yet", which are the same thing to the caller: do not
    attach anything. Leaving a key unset in config is the supported way to roll
    the feature out one bucket at a time.
    """
    key = canonical_reason(reason)
    if key is None:
        return None
    return config.selection_reason_label_ids.get(key)
