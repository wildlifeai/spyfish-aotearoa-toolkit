"""
Strategy-based frame selection for BIIGLE expert review and training.

Groups raw detections by frame to compute per-frame count and mean confidence,
then applies the per-species bucket strategy (`extraction.frame_strategy`).

Two entry points:
- `select_frames`, picks straight from the ML raw CSV. This is the usual
  path: the training-frames flow always uses it, and the biigle-direct
  pipeline path uses it whenever a drop skipped citsci.
- `select_frames_from_zooniverse`, starts from volunteer clip consensus
  (the Zooniverse MaxN CSV) and folds in ML peaks the volunteers never saw.
  Used on the full-pipeline path, where `citsci_complete` is the
  prerequisite for the BIIGLE upload stage.
"""

import bisect
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.select_clips import _sample


def spread_timestamps(
    start: float, end: float, n: int, power: float = 2.0
) -> list[float]:
    """Generate N timestamps across `[start, end]`, `power` controlling the skew.

    t_i = start + (end - start) × (1 − ((N − i) / N)^power)

    `power=1.0` is even spacing, an unbiased sample of the window, which is what
    a false-negative probe wants. `power=2.0` back-loads toward `end`, putting
    roughly half the points in the final third, matching BUV bait-attraction
    dynamics, right when the goal is to FIND fish rather than sample fairly.
    Use `power` between 1 and 2 for milder back-loading, above 2 for heavier.

    The last timestamp is exactly `end`; all timestamps are strictly increasing.

    Doubling N produces a superset (every old t_i appears as the new t_{2i}), so
    re-running with a larger N adds new frames without colliding with the
    original ones at the S3/Biigle filename layer. Verified in unit tests.

    Raises:
        ValueError: n < 1, end <= start, or power <= 0.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start})")
    if power <= 0:
        raise ValueError(f"power must be > 0, got {power}")
    span = end - start
    return [start + span * (1.0 - ((n - i) / n) ** power) for i in range(1, n + 1)]


def per_frame_species_counts(raw_csv: str, min_confidence: float) -> pd.DataFrame:
    """Detections per (timestamp, class) from a raw YOLO CSV, above a confidence floor.

    Columns: `time_seconds`, `class`, `count`, `mean_conf`, `diag`, `elong`.

    `diag` is the mean box diagonal, the only orientation-independent size
    signal on a DOWNWARD-facing camera, where in-plane rotation is free.
    `elong` is `max(w,h)/min(w,h)`, symmetric so a tall-narrow box and a
    wide-flat one score alike, rather than one animal scoring 0.25 or 4.0
    depending which way it happened to be swimming. No bucket consumes `elong`
    yet; if one ever does, note the signal is one-sided, an elongated box
    means a genuinely long animal, but a square box proves nothing (a long
    fish at 45° produces one), so extremes may be selected but non-extremes
    must never rule anything out.

    Empty frame when the CSV is missing or holds nothing above the floor.
    """
    cols = ["time_seconds", "class", "count", "mean_conf", "diag", "elong"]
    if not Path(raw_csv).exists():
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(raw_csv)
    if raw.empty:
        return pd.DataFrame(columns=cols)
    raw = raw[raw["confidence"] >= min_confidence]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    long_side = raw[["w", "h"]].max(axis=1)
    short_side = raw[["w", "h"]].min(axis=1).replace(0, pd.NA)
    raw = raw.assign(
        _diag=(raw["w"] ** 2 + raw["h"] ** 2) ** 0.5,
        _elong=long_side / short_side,
    )
    return (
        raw.groupby(["time_seconds", "class"])
        .agg(
            count=("confidence", "size"),
            mean_conf=("confidence", "mean"),
            diag=("_diag", "mean"),
            elong=("_elong", "max"),
        )
        .reset_index()
    )


def collapse_peaks(peaks: list, counts: pd.DataFrame, spacing: float) -> list:
    """Merge peaks that can share one frame *without losing any species' count*.

    Two species peaking near each other can often share a single JPEG, one
    expert review instead of two. But merging blindly reads species B's MaxN off
    a frame up to `spacing` seconds from B's true peak, under-counting B. The
    error grows with `spacing`, so widening that window without this check would
    actively degrade counting.

    A merge is therefore allowed only when the surviving frame holds the merged
    species at its *own* peak count. A second pass then drops any frame whose
    every species is already fully represented elsewhere, the symmetric half,
    which is what lets B's frame absorb A's rather than only ever the reverse.

    Args:
        peaks: `(timestamp, species, count)` tuples.
        counts: output of `per_frame_species_counts`.
        spacing: seconds within which a merge may be attempted.

    Returns:
        `(timestamp, [(species, count), ...])` per surviving frame, chronological.
    """
    if not peaks:
        return []

    at = {
        (float(row["time_seconds"]), str(row["class"])): int(row["count"])
        for _, row in counts.iterrows()
    }

    def holds(frame_t: float, species: str, peak_count: int) -> bool:
        """True when `frame_t` shows `species` at its full peak count."""
        return at.get((frame_t, species), 0) >= peak_count

    frames: dict = {}
    # Strongest peaks first, so the frames that anchor the set carry the most.
    for t, species, count in sorted(peaks, key=lambda p: -p[2]):
        host = next(
            (f for f in frames if abs(f - t) < spacing and holds(f, species, count)),
            None,
        )
        frames.setdefault(t if host is None else host, []).append((species, count))

    for ft in sorted(frames, key=lambda f: len(frames[f])):
        # Only a frame close enough to stand in for `ft` may absorb it. Without
        # the distance check this drops genuinely separate moments, two peaks
        # of one species minutes apart both "hold" it, but they are different
        # observations and the caller wants both.
        others = [o for o in frames if o != ft and abs(o - ft) < spacing]
        if not others:
            continue
        if all(any(holds(o, sp, c) for o in others) for sp, c in frames[ft]):
            for sp, c in frames[ft]:
                target = next(o for o in others if holds(o, sp, c))
                if (sp, c) not in frames[target]:
                    frames[target].append((sp, c))
            del frames[ft]

    return sorted(frames.items())


# NOTE: an earlier generation of standalone bucket helpers
# (`size_band_timestamps`, `outlier_timestamps`) was deleted here, the live
# size-band logic inside `_select_frames_with_strategy` superseded them with
# coverage-aware semantics (a band already shown by a selected peak frame is
# not re-selected), and the elong outlier bucket was never wired in. The
# one-sided elong insight lives in `per_frame_species_counts`'s docstring.


def _select_frames_with_strategy(
    frame_df: pd.DataFrame,
    drop_id: str,
    sampling_start: float,
    strategy_params: dict,
    video_start_threshold: int,
    frame_cap: int,
) -> pd.DataFrame:
    """
    Select individual frames using temporal spacing only, no clip bucketing.

    Each row in frame_df represents one video frame with its total detection
    count and mean confidence. Selections are deduplicated by
    temporal_spacing_seconds, allowing multiple frames within the same 10s
    clip window.

    Parallel implementation: select_clips.select_clips_with_strategy() applies the
    same MaxN/confusing/start strategy to clip buckets rather than individual frames.
    Key intentional divergences:
      - This function deduplicates by float spacing; clips use 10s bucket keys.
      - No "empty" bucket here, raw CSVs only contain detected frames, so empty
        intervals cannot be sampled at this stage.
      - Cap/priority logic is simpler (no full-video health-check weighting).
    If you change the core strategy logic here, check whether select_clips.py needs
    the same update.
    """
    columns = [
        config.drop_id_column,
        config.csv_sampling_start_column,
        config.csv_clip_max_time_column,
        config.csv_scientific_name_column,
        "SelectionReason",
        config.csv_max_interval_column,
        config.csv_confidence_agreement_column,
    ]

    if frame_df.empty:
        return pd.DataFrame(columns=columns)

    time_col = config.csv_time_seconds_column
    frame_df = frame_df.copy()

    # Kept in sorted order so the spacing check is O(log n) via bisect instead
    # of O(n) by scanning every previously-selected time. Frame selection can
    # consider thousands of candidates on dense videos, so the quadratic form
    # was the hot path.
    selected_times_sorted: list[float] = []
    per_class_times: dict = {}
    rows: list[dict] = []

    def _spaced(
        t: float, spacing: float, species: str = None, same_frame: float = 0.0
    ) -> bool:
        """True iff `t` is far enough from what is already selected.

        Spacing applies WITHIN a class, not across. Two frames of the same
        species 10s apart on a fixed camera show the same individual against the
        same background, redundant. Two *different* classes at the same instant
        are one image serving both, so blocking there would starve whichever
        bucket runs last. On real data that was the `fish` catch-all: every one
        of its detections sat within 30s of a named-species pick, so the
        highest-value bucket returned nothing at all.

        Across classes only a same-frame tolerance applies, so genuinely
        duplicate timestamps still collapse. `species=None` keeps the old
        global behaviour for callers that have no class to key on.
        """
        if species is None:
            times = selected_times_sorted
        else:
            times = sorted(per_class_times.get(species, []))
            near = min(
                (abs(t - o) for o in selected_times_sorted), default=float("inf")
            )
            if near < same_frame:
                return False
        idx = bisect.bisect_left(times, t)
        if idx > 0 and t - times[idx - 1] < spacing:
            return False
        if idx < len(times) and times[idx] - t < spacing:
            return False
        return True

    def _add(row, reason: str):
        t = float(row[time_col])
        bisect.insort(selected_times_sorted, t)
        per_class_times.setdefault(
            str(row[config.csv_scientific_name_column]), []
        ).append(t)
        rows.append(
            {
                config.drop_id_column: drop_id,
                config.csv_sampling_start_column: sampling_start,
                config.csv_clip_max_time_column: t,
                config.csv_scientific_name_column: row[
                    config.csv_scientific_name_column
                ],
                "SelectionReason": reason,
                config.csv_max_interval_column: row[config.csv_max_interval_column],
                config.csv_confidence_agreement_column: row[
                    config.csv_confidence_agreement_column
                ],
            }
        )

    # Divides spacing ONLY, quotas are untouched. Tightening the spacing does
    # not ask for more frames; it lets quotas that spacing was blocking actually
    # fill, so the count rises toward what the quotas already requested rather
    # than past it.
    spacing = (
        strategy_params["temporal_spacing_seconds"] / strategy_params["spacing_divisor"]
    )
    effort = strategy_params["effort_multiplier"]
    n_maxn_per_sp = strategy_params["per_species_maxn"] * effort
    n_confusing_per_sp = strategy_params["per_species_confusing"] * effort
    n_start = strategy_params["start_export"]
    fish_class = config.catchall_class
    n_bands = strategy_params["fish_bands"] * effort
    same_frame = strategy_params["same_frame_seconds"]
    merge_window = strategy_params["peak_merge_seconds"]

    # No binary/multiclass fork: a single-class drop is the per-species case with
    # N=1, so the branch bought nothing but a second set of quota names to keep
    # in sync.
    # Peaks are gathered before being added, so `collapse_peaks` can merge the
    # ones that genuinely share a frame. Selecting them one at a time could only
    # ever BLOCK a near-simultaneous peak, never merge it, and blocking loses
    # the species entirely, where merging keeps both on one JPEG.
    peak_rows: list = []

    for species in frame_df[config.csv_scientific_name_column].unique():
        sp_df = frame_df[frame_df[config.csv_scientific_name_column] == species]

        added_maxn = 0
        # `_spaced` only checks against already-ADDED selections, and nothing
        # has been added yet in this first bucket, so spacing between this
        # species' own gathered peaks must be enforced here. Without it the
        # top-N rows are adjacent samples of one event (0.33s apart at 3fps),
        # which `collapse_peaks` then merges into a single frame, silently
        # halving the per-species peak quota.
        gathered: list[float] = []
        for _, row in sp_df.nlargest(
            n_maxn_per_sp * 3, config.csv_max_interval_column
        ).iterrows():
            if added_maxn >= n_maxn_per_sp:
                break
            t = float(row[time_col])
            if any(abs(t - o) < spacing for o in gathered):
                continue
            if _spaced(t, spacing, species, same_frame):
                peak_rows.append((t, species, row))
                gathered.append(t)
                added_maxn += 1

    # Merge peaks that can share one frame without losing any species' count,
    # then add the survivors. `same_frame_seconds` is too strict a test on its
    # own: two species peaking 1.001s apart missed it by a millisecond and each
    # took a frame, when one image showed both.
    if peak_rows:
        counts = frame_df.rename(
            columns={
                time_col: "time_seconds",
                config.csv_scientific_name_column: "class",
                config.csv_max_interval_column: "count",
            }
        )[["time_seconds", "class", "count"]]
        merged = collapse_peaks(
            [(t, sp, int(r[config.csv_max_interval_column])) for t, sp, r in peak_rows],
            counts,
            spacing=merge_window,
        )
        by_time = {t: r for t, _, r in peak_rows}
        for t, labels in merged:
            names = ", ".join(sorted({sp for sp, _ in labels}))
            _add(by_time[t], f"MaxN ({names})")

    # Uncertainty frames come AFTER the peaks are registered, so they can see
    # what is already selected and avoid re-picking the same moments.
    for species in frame_df[config.csv_scientific_name_column].unique():
        sp_df = frame_df[frame_df[config.csv_scientific_name_column] == species]

        # Rank by LOW confidence, count as tie-break. NOT by
        # `count / confidence` (ConfusionScore). That score measured 0.982
        # correlated with raw count and only 0.133 with confidence, so it
        # picked the busiest frames, a second MaxN bucket rather than an
        # uncertainty one. The agreement column carries mean model confidence
        # on the ML path and volunteer agreement on the citsci path; low
        # means uncertain in both.
        ranked_conf = sp_df.sort_values(
            [
                config.csv_confidence_agreement_column,
                config.csv_max_interval_column,
            ],
            ascending=[True, False],
        )

        added_conf = 0
        for _, row in ranked_conf.iterrows():
            if added_conf >= n_confusing_per_sp:
                break
            if _spaced(float(row[time_col]), spacing, species, same_frame):
                _add(row, f"Uncertain ({species})")
                added_conf += 1

    # Geometry buckets need box dimensions, which only the ML raw CSV carries, a
    # citsci MaxN CSV has no `w`/`h`, so these contribute nothing there rather
    # than raising. Guarded on column presence, not on which caller we are.
    if "diag" in frame_df.columns:
        unid = frame_df[
            frame_df[config.csv_scientific_name_column] == fish_class
        ].dropna(subset=["diag"])
        if not unid.empty and n_bands >= 1:
            lo = float(unid["diag"].min())
            hi = float(unid["diag"].max())
            width = (hi - lo) / n_bands if hi > lo else 0.0

            def _band_of(d: float) -> int:
                if not width:
                    return 0
                return min(int((d - lo) / width), n_bands - 1)

            # Bands the mandatory peaks already cover. Peaks run first because a
            # species without one has no MaxN at all, but a peak frame still
            # SHOWS an animal of some size, so it fills its band, and only the
            # uncovered bands need their own frame. Without this the two buckets
            # compete for the same spacing budget and whichever runs second gets
            # nothing, which loses either the peak or the largest unidentified
            # animal in the deployment.
            # A band is covered if ANY already-selected frame shows a fish of
            # that size, not just frames labelled `fish`. At 0.4s apart the
            # animal is still on screen, so a frame picked as another species'
            # peak already answers "what is that big unidentified thing". Only
            # frames within the same-frame tolerance count; at 30s the fish has
            # very likely moved on, which is why the two tolerances differ.
            covered = set()
            for _, cand in unid.iterrows():
                ct = float(cand[time_col])
                if any(abs(ct - o) < same_frame for o in selected_times_sorted):
                    covered.add(_band_of(float(cand["diag"])))

            for i in range(n_bands if width else 1):
                if i in covered:
                    continue
                band = unid if not width else unid[unid["diag"].apply(_band_of) == i]
                for _, row in band.sort_values(
                    config.csv_max_interval_column, ascending=False
                ).iterrows():
                    if _spaced(float(row[time_col]), spacing, fish_class, same_frame):
                        _add(row, f"Fish size band {i + 1}")
                        covered.add(i)
                        break

    added_start = 0
    for _, row in _sample(
        frame_df[frame_df[time_col] < sampling_start + video_start_threshold],
        n_start,
    ).iterrows():
        if added_start >= n_start:
            break
        # Deliberately NOT spacing-checked: this asks "did the rig deploy
        # properly", not "is this moment novel". Letting the diversity buckets
        # crowd it out means a failed deployment goes unnoticed to save a frame
        # that is near-duplicate anyway.
        _add(row, "Video Start")
        added_start += 1

    result = pd.DataFrame(rows, columns=columns)

    if frame_cap and len(result) > frame_cap:
        priority = result[result["SelectionReason"].str.contains("MaxN|Video Start")]
        other = result[~result["SelectionReason"].str.contains("MaxN|Video Start")]
        if len(priority) >= frame_cap:
            result = priority.iloc[:frame_cap]
        else:
            n_needed = min(frame_cap - len(priority), len(other))
            result = pd.concat([priority, other.sample(n_needed, random_state=42)])

    return result.sort_values(config.csv_clip_max_time_column).reset_index(drop=True)


def blind_selections(
    drop_id: str,
    sampling_start: float,
    sampling_end: float,
    taken_times: pd.Series,
    spacing: float,
    n: int,
) -> pd.DataFrame:
    """Evenly-spaced frames chosen without consulting the ML detections.

    Every other selection descends from the raw CSV, so a fish the model never
    detected is not merely missed but unreviewable, nothing puts it in front of
    anyone. These are the only frames that can surface that, and the frame path's
    counterpart to the clip path's ``empty_export`` bucket, which frame selection
    cannot reproduce (raw CSVs hold only frames that HAD detections, so there are
    no empty rows to sample).

    Uses `spread_timestamps` at ``power=1.0``, the same generator the ML plan
    fills with, asked for an even spread rather than a back-loaded one.
    Back-loading chases bait-attraction density, which is right when the goal is
    to FIND fish and wrong when it is to sample the window without bias.

    Spacing is enforced against everything already chosen, so fewer than `n` may
    survive on a dense drop, the right trade, since the model already had
    something to say about those moments.
    """
    columns = [
        config.drop_id_column,
        config.csv_sampling_start_column,
        config.csv_clip_max_time_column,
        config.csv_scientific_name_column,
        "SelectionReason",
        config.csv_max_interval_column,
        config.csv_confidence_agreement_column,
    ]
    if n < 1 or sampling_end <= sampling_start:
        return pd.DataFrame(columns=columns)

    taken = sorted(float(t) for t in taken_times.dropna())
    rows = []
    # Size the spread to the slots. NEVER take a prefix of a longer candidate
    # list. `spread_timestamps` returns points in time order, so keeping the
    # first k of a longer list clusters them at the start of the deployment.
    #
    # A blocked point SHIFTS to the nearest free moment rather than being
    # dropped. Dropping it silently thinned this bucket from 3 frames to 1 as
    # the effort multiplier rose and detection frames filled the timeline,
    # exactly backwards, since a dense drop is where checking what the model
    # missed matters most.
    step = (sampling_end - sampling_start) / (n + 1)
    for t in spread_timestamps(sampling_start, sampling_end, n=n, power=1.0):
        placed = None
        for offset in (
            0.0,
            *(o for k in range(1, 21) for o in (k * spacing, -k * spacing)),
        ):
            cand = t + offset
            if not sampling_start <= cand <= sampling_end:
                continue
            if abs(offset) > step:
                break
            if all(abs(cand - o) >= spacing for o in taken):
                placed = cand
                break
        if placed is None:
            continue
        taken.append(placed)
        rows.append(
            {
                config.drop_id_column: drop_id,
                config.csv_sampling_start_column: sampling_start,
                config.csv_clip_max_time_column: placed,
                # No species and no count: nothing was detected here, and that is
                # the entire point of the bucket.
                config.csv_scientific_name_column: pd.NA,
                "SelectionReason": "Blind (False Negative Check)",
                config.csv_max_interval_column: 0,
                config.csv_confidence_agreement_column: pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _run_selection(
    frame_df: pd.DataFrame,
    drop_id: str,
    output_selections_path: str,
    source_label: str,
    allow_fallback_window: bool = False,
) -> pd.DataFrame:
    """Apply strategy, write selections CSV, return result.

    Shared by select_frames (ML) and select_frames_from_zooniverse (citsci).
    Callers are responsible for building and validating frame_df; this
    function owns the strategy selection, _select_frames_with_strategy call,
    and file write.

    `allow_fallback_window` decides what happens when a deployment has no
    `sampling_end`, which is the case for ~41% of rows. False (default, the
    expert-review path) raises: a review with no blind frames is a silently
    weaker review, and expert time is too scarce to spend on one. True (the
    training path) falls back to the expected deployment duration, since those
    frames become training labels rather than a reported abundance figure.
    """
    deployment = DatabaseManager().get_deployment(drop_id)
    if not deployment or deployment.get("sampling_start") is None:
        raise ValueError(f"Missing sampling_start for {drop_id}, cannot select frames.")
    sampling_start = float(deployment["sampling_start"])

    selections_df = _select_frames_with_strategy(
        frame_df=frame_df,
        drop_id=drop_id,
        sampling_start=sampling_start,
        strategy_params=config.frame_strategy,
        video_start_threshold=config.video_start_threshold,
        frame_cap=config.clip_cap,
    )

    # Top up to the floor with frames the model had no say in. Peaks may already
    # have carried the total past it, in which case nothing is added.
    n_blind = 0
    sampling_end = deployment.get("sampling_end")
    # `blind_export` is taken ALWAYS, then topped up if the floor is still short.
    # Treating blind frames as pure filler meant a drop dense with detections got
    # none, backwards, since that is exactly where checking what the model
    # missed matters most.
    strategy = config.frame_strategy
    shortfall = max(
        strategy["blind_export"] * strategy["effort_multiplier"],
        config.min_frames_per_drop - len(selections_df),
    )
    if sampling_end is None:
        if not allow_fallback_window:
            # Expert review: refuse rather than degrade. Without a window there
            # are no blind frames, and a review built only from ML-nominated
            # moments can confirm precision but can never reveal a fish the
            # model missed, while looking entirely normal from the outside.
            raise ValueError(
                f"Missing sampling_end for {drop_id}, refusing to select frames "
                "for expert review: with no sampling window the review would "
                "contain only ML-nominated moments and could never reveal a "
                "missed fish. Fix the deployment metadata, or run the training "
                "path, which falls back to the expected deployment duration."
            )
        # Training: the frames feed labels, not a reported abundance figure, so
        # an assumed window is better than no blind frames at all.
        sampling_end = float(config.buv_video_duration_seconds)
        logging.warning(
            f"{drop_id}: no sampling_end; assuming the expected deployment "
            f"duration ({sampling_end:.0f}s) so blind frames still get taken. "
            "Fix the deployment metadata to make this exact."
        )

    if shortfall > 0:
        blind_df = blind_selections(
            drop_id=drop_id,
            sampling_start=sampling_start,
            sampling_end=float(sampling_end),
            taken_times=selections_df[config.csv_clip_max_time_column],
            spacing=strategy["temporal_spacing_seconds"] / strategy["spacing_divisor"],
            n=shortfall,
        )
        n_blind = len(blind_df)
        if n_blind:
            selections_df = pd.concat([selections_df, blind_df], ignore_index=True)
            selections_df = selections_df.sort_values(
                config.csv_clip_max_time_column
            ).reset_index(drop=True)

    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    selections_df.to_csv(output_selections_path, index=False)
    logging.info(
        f"{len(selections_df)} frame selections for {drop_id} "
        f"({len(frame_df)} candidates from {source_label}, {n_blind} blind)."
    )
    return selections_df


def _ml_peak_selections(
    raw_csv_path: str,
    drop_id: str,
    sampling_start: float,
) -> pd.DataFrame:
    """Top-K per-species frame-level peaks from a raw ML CSV.

     Citsci-driven frame selection is locked to integer-second precision
     (Zooniverse only captures volunteer clicks at whole-second resolution),
     so it can miss multi-fish moments that exist for less than a second.
     This helper builds extra selection rows directly from the YOLO raw CSV
    , preserving the inference frame's sub-second timestamp.

     Counts detections at conf >= ``config.ml_peak_min_confidence`` (lower
     than the MaxN counting threshold of 0.5 on purpose), groups by frame
     and species, keeps top-K per species by count with mean confidence as
     a tiebreaker. Returns rows in the same schema as ``_run_selection``'s
     output so the caller can concat them with citsci-derived rows.
    """
    columns = [
        config.drop_id_column,
        config.csv_sampling_start_column,
        config.csv_clip_max_time_column,
        config.csv_scientific_name_column,
        "SelectionReason",
        config.csv_max_interval_column,
        config.csv_confidence_agreement_column,
    ]
    if not Path(raw_csv_path).exists():
        return pd.DataFrame(columns=columns)

    raw = pd.read_csv(raw_csv_path)
    min_conf = config.ml_peak_min_confidence
    raw = raw[raw["confidence"] >= min_conf]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    per_frame_species = (
        raw.groupby(["time_seconds", "class"])
        .agg(count=("confidence", "count"), mean_conf=("confidence", "mean"))
        .reset_index()
    )
    top_k = (
        per_frame_species.sort_values(["count", "mean_conf"], ascending=[False, False])
        .groupby("class")
        .head(config.ml_peak_top_k_per_species)
    )

    rows = []
    for _, r in top_k.iterrows():
        rows.append(
            {
                config.drop_id_column: drop_id,
                config.csv_sampling_start_column: sampling_start,
                config.csv_clip_max_time_column: float(r["time_seconds"]),
                config.csv_scientific_name_column: str(r["class"]),
                "SelectionReason": (
                    f"ML peak (conf>={min_conf}, count={int(r['count'])})"
                ),
                config.csv_max_interval_column: int(r["count"]),
                config.csv_confidence_agreement_column: round(float(r["mean_conf"]), 4),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def select_frames_from_zooniverse(
    maxn_csv_path: str,
    output_selections_path: str,
    drop_id: str,
    ml_raw_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Select frames for BIIGLE upload from a Zooniverse volunteer MaxN CSV.

    One row per (clip, species) already, no grouping needed. MaxInterval is
    the volunteer mode count; ConfidenceAgreement is agreement_pct / 100.
    The confusion score (high count / low agreement) surfaces suspicious clips
    without needing the separate suspicious_minority_find flag.

    When ``ml_raw_csv_path`` is provided and points to an existing raw YOLO
    CSV, the selection is augmented with top-K per-species ML peak frames
    (see ``_ml_peak_selections``). Volunteer-clicked timestamps are integer
    seconds, but a transient multi-fish moment can live in a sub-second
    window. ML peaks surface those frames the volunteers couldn't pin.
    ML peaks within ``config.ml_peak_citsci_dedupe_tolerance_seconds`` of
    any citsci-selected frame are dropped to avoid near-duplicate uploads.
    """
    if not Path(maxn_csv_path).exists():
        raise FileNotFoundError(f"Zooniverse MaxN CSV not found: {maxn_csv_path}")

    frame_df = pd.read_csv(maxn_csv_path)
    if frame_df.empty:
        raise ValueError(
            f"Empty Zooniverse MaxN CSV for {drop_id}, no volunteer consensus to select from."
        )

    # _run_selection expects config.csv_time_seconds_column ("TimeAbsoluteSeconds");
    # the MaxN CSV uses config.csv_maxn_time_seconds_column ("TimeOfMaxAbsoluteSeconds").
    frame_df = frame_df.rename(
        columns={config.csv_maxn_time_seconds_column: config.csv_time_seconds_column}
    )

    citsci_selections = _run_selection(
        frame_df, drop_id, output_selections_path, "Zooniverse MaxN CSV"
    )

    if not ml_raw_csv_path or not Path(ml_raw_csv_path).exists():
        return citsci_selections

    sampling_start = float(DatabaseManager().get_deployment(drop_id)["sampling_start"])
    ml_rows = _ml_peak_selections(ml_raw_csv_path, drop_id, sampling_start)
    if ml_rows.empty:
        return citsci_selections

    tolerance = config.ml_peak_citsci_dedupe_tolerance_seconds
    citsci_times = citsci_selections[config.csv_clip_max_time_column].to_numpy()
    if citsci_times.size:
        keep_mask = ml_rows[config.csv_clip_max_time_column].apply(
            lambda t: bool(abs(citsci_times - t).min() > tolerance)
        )
        ml_extra = ml_rows[keep_mask]
    else:
        ml_extra = ml_rows

    if ml_extra.empty:
        logging.info(
            f"{drop_id}: ML peak augmentation found {len(ml_rows)} candidate(s), "
            f"all within {tolerance}s of a citsci selection, no extra frames added."
        )
        return citsci_selections

    combined = pd.concat([citsci_selections, ml_extra], ignore_index=True)
    cap = config.clip_cap
    if cap and len(combined) > cap:
        # Cap by trimming the tail (ML-peak rows are appended last, so citsci
        # selections are preserved first). Keeps volunteer intent the priority.
        combined = combined.iloc[:cap]
    combined = combined.sort_values(config.csv_clip_max_time_column).reset_index(
        drop=True
    )
    Path(output_selections_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_selections_path, index=False)
    logging.info(
        f"{drop_id}: augmented selection, {len(citsci_selections)} citsci "
        f"+ {len(ml_extra)} ML peak frame(s) (capped at {cap}) → "
        f"{len(combined)} total."
    )
    return combined


def select_frames(
    raw_csv_path: str,
    output_selections_path: str,
    drop_id: str,
    allow_fallback_window: bool = False,
) -> pd.DataFrame:
    """
    Select frames from the raw ML CSV using the export strategy.

    Groups raw detections by frame to get per-frame count and mean confidence,
    then runs the strategy to pick MaxN and confusing frames. The empty bucket
    is not used, raw CSVs only contain detected frames.

    Args:
        raw_csv_path: Path to the raw YOLO CSV produced by ML inference.
        output_selections_path: Path to write the selections CSV.
        drop_id: Deployment identifier.
        allow_fallback_window: When the deployment has no `sampling_end`,
            assume the expected deployment duration instead of raising. Set by
            the training-frame extractor only — the expert-review path must
            fail, because a review without blind frames cannot reveal a fish
            the model missed and nothing downstream would show that.

    Returns:
        DataFrame of selected frame moments.
    """
    if not Path(raw_csv_path).exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")

    # One bounding box per raw-CSV row; grouping gives per-frame count
    # (MaxInterval), mean confidence (ConfidenceAgreement), and box geometry
    # for the unidentified-size bucket. Only the raw ML CSV has `w`/`h`,
    # volunteer MaxN CSVs carry counts, not boxes, so the geometry columns
    # are absent on the citsci path and those buckets contribute nothing there.
    counts = per_frame_species_counts(raw_csv_path, config.confidence_threshold)
    if counts.empty:
        if pd.read_csv(raw_csv_path).empty:
            raise ValueError(
                f"Empty raw CSV for {drop_id}, no detections available for "
                "frame selection."
            )
        raise ValueError(f"No detections above confidence threshold for {drop_id}.")

    frame_df = counts.rename(
        columns={
            "time_seconds": config.csv_time_seconds_column,
            "class": config.csv_scientific_name_column,
            "count": config.csv_max_interval_column,
            "mean_conf": config.csv_confidence_agreement_column,
        }
    )
    frame_df[config.csv_confidence_agreement_column] = frame_df[
        config.csv_confidence_agreement_column
    ].round(4)

    return _run_selection(
        frame_df,
        drop_id,
        output_selections_path,
        "ML raw CSV",
        allow_fallback_window=allow_fallback_window,
    )


def write_blind_selections(
    drop_id: str, output_path: Path, n_frames: Optional[int] = None
) -> pd.DataFrame:
    """Selections CSV built without consulting the model, the --test-frames path.

    Produces the same CSV `select_frames` does, so everything downstream is
    shared; only where the timestamps came from differs.
    """
    deployment = DatabaseManager().get_deployment(drop_id)
    if deployment is None:
        raise ValueError(f"{drop_id}: not found in deployments DB")
    start = deployment.get("sampling_start")
    end = deployment.get("sampling_end")
    if start is None or end is None:
        raise ValueError(
            f"{drop_id}: missing sampling window (start={start}, end={end})"
        )

    n = n_frames or config.training_extraction_n_frames
    df = blind_selections(
        drop_id=drop_id,
        sampling_start=float(start),
        sampling_end=float(end),
        taken_times=pd.Series(dtype=float),
        spacing=config.frame_strategy["temporal_spacing_seconds"],
        n=n,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"{drop_id}: {len(df)} blind selection(s) → {output_path.name}")
    return df


def upsert_selections(
    prior: Optional[pd.DataFrame], fresh: pd.DataFrame
) -> pd.DataFrame:
    """Merge `fresh` into the prior pass's selections, keyed on timestamp.

    Biigle volumes APPEND rather than replace, so a second pass over a drop adds
    frames to the volume. A selections CSV that got clobbered would then no
    longer describe what the volume holds. Upserting keeps it a true record of
    every frame ever sent for this drop, with `SelectionReason` distinguishing
    which pass each came from.

    `prior` must be captured BEFORE the selection step runs: both selection
    functions write `fresh` to the selections CSV themselves, so reading the
    file here would always see `fresh` and the merge would keep nothing.
    """
    if prior is None or prior.empty:
        return fresh
    key = config.csv_clip_max_time_column
    kept = prior[~prior[key].round(3).isin(fresh[key].round(3))]
    merged = pd.concat([kept, fresh], ignore_index=True).sort_values(key)
    if len(kept):
        logging.info(
            f"Selections upsert: kept {len(kept)} selection(s) from an earlier "
            f"pass, added {len(fresh)}."
        )
    return merged.reset_index(drop=True)
