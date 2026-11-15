"""
MaxN timeline visualisation: one PNG per drop, saved in the drop folder.

Layout, top to bottom, all rows sharing the time axis (mm:ss from video start,
sampling start marked with a dashed line):

  * a fixed-height context row: every detection at the base threshold, any
    class, on its own scale (background, not the measure);
  * one row per species present at the MaxN threshold, ordered by MaxN, on a
    SHARED vertical scale (one fish is the same height in every row, so rows
    grow with their MaxN). Solid fill is the persistence-filtered count, the
    one MaxN actually uses; the thin line is the per-frame count. One marker
    and one text line per species for the deployment's MaxN. Past
    ``MAX_SPECIES_ROWS`` the tail folds into an "Other" row;
  * review tracks: selected clip windows (bars, merged into runs; the MaxN
    intervals when the clips stage has not run yet), then the frames picked
    for expert review and for Zooniverse validation (ticks), each only when
    that selection file exists.

Counts are computed on the inference frame grid (frames without detections are
zeros, so absence reads as absence) and drawn at one point per second.

Usage (standalone; redraws the PNG from the CSVs on disk, no ML run):
    python spyfish/visualisations/maxn_visualisation.py KSF_20240124_BUV_KSF_085_01
    # a drop processed under an older model: name it, the CSVs carry it
    python spyfish/visualisations/maxn_visualisation.py KSF_20240124_BUV_KSF_085_01 --model cfd_binary_water_20260301

Called programmatically:
    from spyfish.visualisations.maxn_visualisation import plot_maxn_timeline
    plot_maxn_timeline(raw_df, maxn_df, drop_id, output_dir, base_conf, maxn_conf,
                       interval_seconds, sampling_start=..., sampling_end=...)
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from spyfish.config.species import species_registry
from spyfish.config.wrapper import config
from spyfish.ml.model_meta import resolve_maxn_threshold
from spyfish.ml.raw_detections import read_raw_detections

matplotlib.use("Agg")

# ── layout (visual constants, not pipeline config) ───────────────────────────
MAX_SPECIES_ROWS = 8  # past this the tail folds into "Other"; also the palette size
FIG_W_IN = 14
UNIT_IN = 0.26  # inches per fish, identical in every species row
PAD_IN = 0.46  # headroom above the tallest spike for the row label
CONTEXT_IN = 0.9  # context row: fixed height, own scale
TRACK_IN = 0.72  # each review track: marks + its own label headroom
TITLE_IN = 0.95  # title + subtitle above the first row
FOOT_IN = 0.75  # time axis + its label below the last row
MAJOR_TICK_S, MINOR_TICK_S = 300, 60

# fixed categorical order, assigned by rank within the drop, never cycled
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def _mmss(x, _=None) -> str:
    return f"{int(x // 60):02d}:{int(x % 60):02d}"


def _runs(intervals) -> list[list[float]]:
    """Merge touching or overlapping (start, end) pairs into runs."""
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class _Counts:
    """Per-frame and per-second count series on the inference grid."""

    def __init__(self, raw_df: pd.DataFrame, t0: float, t1: float):
        times = np.sort(raw_df["time_seconds"].unique())
        self.step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        self.t0, self.t1 = t0, t1
        self.frame_grid = np.arange(t0, t1 + self.step, self.step)
        self.sec_grid = np.arange(np.floor(t0), np.ceil(t1) + 1)
        self._sec_of_frame = np.clip(
            np.floor(self.frame_grid - self.sec_grid[0]).astype(int),
            0,
            len(self.sec_grid) - 1,
        )
        self.window = max(1, round(config.maxn_persistence_seconds / self.step))
        self.max_gap = round(config.maxn_gap_fill_seconds / self.step)

    def per_frame(self, df: pd.DataFrame) -> np.ndarray:
        c = df.groupby("time_seconds").size()
        idx = np.round((c.index.values - self.t0) / self.step).astype(int)
        # Detections outside [t0, t1] (a raw CSV from a run that seeked to a
        # different sampling_start) are dropped, not folded into the edge
        # bins: clipping drew a false spike exactly on the dashed start line.
        inside = (idx >= 0) & (idx < len(self.frame_grid))
        y = np.zeros(len(self.frame_grid))
        np.add.at(y, idx[inside], c.values[inside])
        return y

    def per_second(self, y_frames: np.ndarray) -> np.ndarray:
        """Max per whole second: what the eye resolves at about a pixel per second."""
        out = np.zeros(len(self.sec_grid))
        np.maximum.at(out, self._sec_of_frame, y_frames)
        return out

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.per_second(self.per_frame(df))

    def persistent(self, df: pd.DataFrame) -> np.ndarray:
        """What MaxN counts: gap-filled, then rolling-min over the persistence window."""
        # imported here: process_ml_annotations imports this module
        from spyfish.ml.process_ml_annotations import _centered_rolling_min, _gap_fill

        y = _centered_rolling_min(
            _gap_fill(self.per_frame(df), self.max_gap), self.window
        )
        return self.per_second(y)


def plot_maxn_timeline(
    raw_df: pd.DataFrame,
    maxn_df: pd.DataFrame,
    drop_id: str,
    output_dir: str | Path,
    base_conf: float,
    maxn_conf: float,
    interval_seconds: int,
    sampling_start: Optional[float] = None,
    sampling_end: Optional[float] = None,
    clips_df: Optional[pd.DataFrame] = None,
    frames_df: Optional[pd.DataFrame] = None,
    zoo_frames_df: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    """Render and save the MaxN timeline PNG; returns its path, or None if nothing to draw.

    Args:
        raw_df: raw YOLO detections (``time_seconds``, ``class``, ``confidence``).
        maxn_df: the drop's ML MaxN CSV (one row per interval per species).
        sampling_start/end: the deployment's sampling window in seconds from video
            start. The axis runs from 0 to ``sampling_end``; missing values fall
            back to the raw CSV's extent.
        clips_df: clip selections (``ClipStartAbsoluteSeconds``/``ClipEndAbsoluteSeconds``)
            after the selection strategy; None draws the MaxN intervals instead.
        frames_df / zoo_frames_df: frame selections for BIIGLE / Zooniverse
            validation (``TimeOfMaxAbsoluteSeconds``, ``SelectionReason``); a
            track is drawn only when given.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tcol = config.csv_maxn_time_seconds_column
    ccol = config.csv_max_interval_column
    scol = config.csv_scientific_name_column
    acol = config.csv_confidence_agreement_column
    reason_col = config.selection_reason_column

    base = raw_df[raw_df["confidence"] >= base_conf]
    if base.empty:
        logging.warning(
            f"No detections above base_conf={base_conf} for {drop_id}. Skipping plot."
        )
        return None

    t0 = (
        float(sampling_start) if sampling_start else float(raw_df["time_seconds"].min())
    )
    t1 = float(sampling_end) if sampling_end else float(raw_df["time_seconds"].max())
    counts = _Counts(raw_df, t0, t1)
    registry = species_registry()

    counted = raw_df[raw_df["confidence"] >= maxn_conf]
    maxn = maxn_df[maxn_df[ccol] > 0] if ccol in maxn_df else maxn_df
    peak = (
        maxn.sort_values([ccol, acol], ascending=False)
        .groupby(scol)
        .head(1)
        .set_index(scol)
        if not maxn.empty
        else pd.DataFrame(columns=[ccol, acol, tcol])
    )
    n_boxes = counted.groupby("class").size()
    order = (
        peak.assign(n=n_boxes.reindex(peak.index).fillna(0))
        .sort_values([ccol, "n"], ascending=False)
        .index.astype(str)
        .tolist()
    )
    rows = order[:MAX_SPECIES_ROWS]
    other = order[MAX_SPECIES_ROWS:]
    color = {sp: SERIES[i] for i, sp in enumerate(rows)}

    base_y = counts.raw(base)
    row_max = {sp: max(1, int(peak.loc[sp, ccol])) for sp in rows}
    other_sub = counted[counted["class"].astype(str).isin(other)] if other else None
    other_y = counts.raw(other_sub) if other else None
    if other:
        row_max["other"] = max(1, int(other_y.max()))

    pad = PAD_IN / UNIT_IN
    count_rows = [row_max[sp] for sp in rows] + ([row_max["other"]] if other else [])
    tracks = [("clips", clips_df)] + [
        (k, df)
        for k, df in (("frames", frames_df), ("zoo", zoo_frames_df))
        if df is not None
    ]
    heights = (
        [CONTEXT_IN / UNIT_IN]
        + [m + pad for m in count_rows]
        + [TRACK_IN / UNIT_IN] * len(tracks)
    )
    n_rows = len(heights)
    fig_h = TITLE_IN + FOOT_IN + UNIT_IN * sum(heights) + 0.12 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(FIG_W_IN, fig_h),
        sharex=True,
        gridspec_kw=dict(height_ratios=heights, hspace=0.0),
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    def style(ax, ymax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(1)
        ax.set_xlim(0, t1)
        ax.set_ylim(0, ymax + pad)
        ax.set_yticks([ymax])
        ax.tick_params(axis="y", length=0, labelsize=9, colors=MUTED)
        ax.xaxis.set_major_locator(MultipleLocator(MAJOR_TICK_S))
        ax.xaxis.set_minor_locator(MultipleLocator(MINOR_TICK_S))
        ax.grid(axis="x", which="major", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9)
        if sampling_start:
            ax.axvline(t0, color=INK2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)

    def labels(ax, left, right, italic=False):
        ax.text(
            0.0,
            0.98,
            left,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color=INK,
            style="italic" if italic else "normal",
        )
        if right:
            ax.text(
                1.0,
                0.98,
                right,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=INK2,
            )

    # context row: own scale
    ax = axes[0]
    ax.fill_between(
        counts.sec_grid, base_y, step="mid", color=MUTED, alpha=0.35, linewidth=0
    )
    cmax = max(1, int(base_y.max()))
    style(ax, cmax)
    ax.set_ylim(0, cmax * (1 + PAD_IN / CONTEXT_IN))
    labels(
        ax,
        "All detections",
        f"conf ≥ {base_conf:g}, any class · {len(base)} boxes · own scale, peak {cmax}",
    )
    if sampling_start:
        ax.text(
            t0,
            0.02 * ax.get_ylim()[1],
            f" sampling start {_mmss(t0)}",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=INK2,
        )

    # species rows: shared scale, solid = persistent, thin = per frame
    for i, sp in enumerate(rows):
        ax = axes[1 + i]
        sub = counted[counted["class"].astype(str) == sp]
        col = color[sp]
        ax.fill_between(
            counts.sec_grid,
            counts.persistent(sub),
            step="mid",
            color=col,
            alpha=0.75,
            linewidth=0,
        )
        ax.step(
            counts.sec_grid,
            counts.raw(sub),
            where="mid",
            color=col,
            linewidth=0.7,
            alpha=0.5,
        )
        style(ax, row_max[sp])
        p = peak.loc[sp]
        ax.scatter(
            [p[tcol]],
            [p[ccol]],
            s=70,
            color=col,
            edgecolors=SURFACE,
            linewidths=2,
            zorder=5,
        )
        labels(
            ax,
            registry.display_for(sp),
            f"MaxN {int(p[ccol])} at {_mmss(p[tcol])}  ·  conf {p[acol]:.2f}",
            italic=True,
        )
    if other:
        ax = axes[1 + len(rows)]
        ax.fill_between(
            counts.sec_grid, other_y, step="mid", color=MUTED, alpha=0.45, linewidth=0
        )
        style(ax, row_max["other"])
        labels(
            ax,
            f"Other ({len(other)} species)",
            ", ".join(registry.label_for(s) for s in other),
        )

    # review tracks
    def track(ax, left, right):
        style(ax, 1)
        ax.set_yticks([])
        ax.set_ylim(-0.6, 1.6)  # marks in the lower half, label in the upper
        labels(ax, left, right)

    first_track = 1 + len(count_rows)
    ax = axes[first_track]
    if clips_df is not None and not clips_df.empty:
        a_col, b_col = (
            config.csv_clip_start_absolute_column,
            config.csv_clip_end_absolute_column,
        )
        src = clips_df.rename(columns={a_col: "a", b_col: "b"})
        title = "Selected clips"
        what = f"{len(clips_df)} clips (Zooniverse, after selection strategy)"
    else:
        src = maxn.assign(a=(maxn[tcol] // interval_seconds) * interval_seconds)
        src = src.assign(b=src["a"] + interval_seconds)
        title = "MaxN intervals"
        what = (
            f"{len(maxn)} × {interval_seconds} s with fish "
            "(clip candidates; no selection file yet)"
        )
    n_runs = 0
    for sp, grp in src.groupby(scol):
        for a, b in _runs(zip(grp["a"], grp["b"])):
            n_runs += 1
            ax.barh(
                0,
                b - a - 0.6,
                left=a + 0.3,
                height=0.9,
                color=color.get(str(sp), MUTED),
                linewidth=0,
            )
    track(ax, title, f"{what} · {n_runs} runs")

    for k, (kind, sel) in enumerate(tracks[1:], start=first_track + 1):
        ax = axes[k]
        is_maxn = sel[reason_col].astype(str).str.startswith("MaxN")
        for _, r in sel.iterrows():
            is_peak = str(r[reason_col]).startswith("MaxN")
            col = color.get(str(r[scol]), MUTED) if is_peak else MUTED
            ax.vlines(r[tcol], -0.45, 0.45, color=col, linewidth=2.2)
        where = "BIIGLE" if kind == "frames" else "Zooniverse"
        title = (
            "Selected frames for expert review"
            if kind == "frames"
            else "Selected frames for validation"
        )
        track(
            ax,
            title,
            f"{len(sel)} frames ({where}) · {int(is_maxn.sum())} MaxN picks, "
            f"{int((~is_maxn).sum())} other reasons",
        )

    # section dividers: context | species rows | review tracks
    for a in (axes[0], axes[first_track - 1]):
        a.spines["bottom"].set_color(INK2)
        a.spines["bottom"].set_linewidth(2.2)

    axes[-1].xaxis.set_major_formatter(FuncFormatter(_mmss))
    axes[-1].set_xlabel("Time from video start (mm:ss)", color=INK2, fontsize=10)
    window = f"sampling window {_mmss(t0)} to {_mmss(t1)}"
    if sampling_start:
        window += " (dashed line)"
    fig.text(
        0.04,
        1 - 0.30 / fig_h,
        f"{drop_id}  ·  MaxN timeline",
        ha="left",
        va="top",
        fontsize=13,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.04,
        1 - 0.58 / fig_h,
        f"Counts per sampled frame at conf ≥ {maxn_conf:g} · {window} · solid: held ≥ "
        f"{config.maxn_persistence_seconds:g} s (counts for MaxN), thin: every frame",
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK2,
    )
    fig.subplots_adjust(
        left=0.04, right=0.99, top=1 - TITLE_IN / fig_h, bottom=FOOT_IN / fig_h
    )

    out_path = output_dir / f"{drop_id}_maxn_timeline.png"
    fig.savefig(str(out_path), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    logging.info(f"Saved MaxN timeline plot to {out_path}")
    return out_path


def load_selection_frames(drop_id: str) -> dict[str, Optional[pd.DataFrame]]:
    """The drop's selection files, as DataFrames, None for each that does not exist."""
    paths = {
        "clips_df": config.get_selections_csv_path(drop_id),
        "frames_df": config.get_biigle_selections_csv_path(drop_id),
        "zoo_frames_df": config.get_validation_selection_path(drop_id),
    }
    return {k: pd.read_csv(p) if p.exists() else None for k, p in paths.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the MaxN timeline for one drop."
    )
    parser.add_argument("drop_id", type=str, help="The Drop ID to process.")
    parser.add_argument(
        "--model",
        default=None,
        help="model name as in the CSV file names; default: the pipeline model",
    )
    args = parser.parse_args()
    from spyfish.database.manager import DatabaseManager

    drop_id = args.drop_id
    # CSV filenames embed the model *name* (stem), not the full .pt path,
    # matches how the live pipeline builds them (Path(model).stem).
    model_name = args.model or config.pipeline_model_name
    raw_df = read_raw_detections(config.get_raw_csv_path(drop_id, model_name))  # type: ignore
    maxn_df = pd.read_csv(config.get_maxn_csv_path(drop_id, model_name))  # type: ignore
    dep = DatabaseManager().get_deployment(drop_id) or {}
    plot_maxn_timeline(
        raw_df,
        maxn_df,
        drop_id,
        config.get_drop_dir(drop_id),
        float(config.confidence_threshold),
        resolve_maxn_threshold(config.model_path_for_name(model_name)),
        config.interval_seconds,
        sampling_start=dep.get("sampling_start"),
        sampling_end=dep.get("sampling_end"),
        **load_selection_frames(drop_id),
    )
