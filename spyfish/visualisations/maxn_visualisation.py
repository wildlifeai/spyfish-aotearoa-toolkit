"""
MaxN timeline visualisation, produces a PNG saved to the data_quality folder.

Usage (standalone):
    python spyfish/visualisations/maxn_visualisation.py

Called programmatically:
    from spyfish.visualisations.maxn_visualisation import plot_maxn_timeline
    plot_maxn_timeline(raw_df, maxn_df, drop_id, output_dir, base_conf, maxn_conf, interval_seconds)
"""

import argparse
import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from spyfish.config.wrapper import config


def plot_maxn_timeline(
    raw_df: pd.DataFrame,
    maxn_df: pd.DataFrame,
    drop_id: str,
    output_dir: str | Path,
    base_conf: float,
    maxn_conf: float,
    interval_seconds: int,
) -> Path:
    """
    Renders and saves a MaxN detection timeline plot.

    Args:
        raw_df: Raw YOLO detections DataFrame (columns: time_seconds, confidence).
        maxn_df: MaxN results DataFrame (columns: TimeOfMaxAbsSeconds, MaxInterval, ConfidenceAgreement).
        drop_id: Deployment identifier used for the title and output filename.
        output_dir: Directory to save the PNG into (created if absent).
        base_conf: Base inference confidence threshold (shown as light grey line).
        maxn_conf: MaxN counting threshold (shown as blue line).
        interval_seconds: Width of each time window in seconds.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Total fish (all classes) at base confidence, kept as a faint context line.
    base_total = (
        raw_df[raw_df["confidence"] >= base_conf]
        .groupby("time_seconds")
        .size()
        .reset_index(name="count")
    )

    if base_total.empty:
        logging.warning(
            f"No detections above base_conf={base_conf} for {drop_id}. Skipping plot."
        )
        return None  # type: ignore

    max_time = base_total["time_seconds"].max()

    # Per-species detections at the MaxN counting threshold. Only species that
    # actually appear in this deployment get a line, so the plot scales with
    # what's present (typically a handful) rather than the full class list.
    counted = raw_df[raw_df["confidence"] >= maxn_conf]
    species_list = sorted(counted["class"].astype(str).unique())
    cmap = plt.get_cmap("tab10" if len(species_list) <= 10 else "tab20")
    species_colors = {sp: cmap(i % cmap.N) for i, sp in enumerate(species_list)}

    fig, ax = plt.subplots(figsize=(16, 6))

    # 1. Alternating interval background bands
    for i in range(int(max_time // interval_seconds) + 1):
        start = i * interval_seconds
        color = "#f5f5f5" if i % 2 == 0 else "#ebebeb"
        ax.axvspan(start, start + interval_seconds, alpha=0.6, color=color, zorder=0)

    # 2. Green highlight: MaxN clip windows
    for _, row in maxn_df.iterrows():
        t = row[config.csv_maxn_time_seconds_column]
        interval_start = (t // interval_seconds) * interval_seconds
        ax.axvspan(
            interval_start,
            interval_start + interval_seconds,
            alpha=0.15,
            color="#2ecc71",
            zorder=1,
        )

    # 3. Faint total-fish line (all species, base conf) for context.
    ax.fill_between(
        base_total["time_seconds"],
        base_total["count"],
        alpha=0.08,
        color="#95a5a6",
        zorder=2,
    )
    ax.plot(
        base_total["time_seconds"],
        base_total["count"],
        color="#bdc3c7",
        linewidth=1,
        alpha=0.5,
        zorder=2,
        label=f"All fish (conf ≥ {base_conf})",
    )

    # 4. One coloured line per species (counts at the MaxN threshold).
    for sp in species_list:
        per_sp = (
            counted[counted["class"].astype(str) == sp]
            .groupby("time_seconds")
            .size()
            .reset_index(name="count")
        )
        if per_sp.empty:
            continue
        ax.plot(
            per_sp["time_seconds"],
            per_sp["count"],
            color=species_colors[sp],
            linewidth=1.8,
            zorder=5,
            label=sp,
        )

    # 5. MaxN peak dots, coloured to match the species' line.
    for _, row in maxn_df.iterrows():
        t = row[config.csv_maxn_time_seconds_column]
        count = row[config.csv_max_interval_column]
        species = str(row[config.csv_scientific_name_column])
        dot_color = species_colors.get(species, "#e74c3c")
        ax.scatter(
            t,
            count,
            color=dot_color,
            s=120,
            zorder=6,
            edgecolors="white",
            linewidth=1.5,
        )
        ax.annotate(
            f"{species}\nMaxN={count} ({row[config.csv_confidence_agreement_column]:.2f})",
            xy=(t, count),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=8,
            color=dot_color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=dot_color,
                alpha=0.85,
            ),
        )

    # Styling
    ax.set_xlabel("Time (seconds from video start)", fontsize=12)
    ax.set_ylabel("Fish Count per Frame", fontsize=12)
    ax.set_title(f"MaxN Timeline, {drop_id}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max_time + 1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Collect the auto-labelled total + per-species lines, then append the
    # fixed marker/patch entries. ncol grows with the species count so a
    # species-rich drop doesn't produce a single tall column.
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        plt.scatter(
            [],
            [],
            color="#333333",
            s=80,
            edgecolors="white",
            label="MaxN peak (species colour)",
        ),
        mpatches.Patch(facecolor="#2ecc71", alpha=0.15, label="Selected clip window"),
        mpatches.Patch(
            facecolor="#ebebeb", alpha=0.6, label=f"{interval_seconds}s intervals"
        ),
    ]
    ncol = 2 if len(species_list) > 6 else 1
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, ncol=ncol)

    plt.tight_layout()
    out_path = output_dir / f"{drop_id}_maxn_timeline.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved MaxN timeline plot to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MaxN visualisations.")
    parser.add_argument("drop_id", type=str, help="The Drop ID to process.")
    args = parser.parse_args()

    drop_id = args.drop_id
    # CSV filenames embed the model *name* (stem), not the full .pt path,
    # matches how the live pipeline builds them (Path(model).stem), e.g.
    # "species_20260603" → SLI_..._ml_species_20260603_maxn.csv.
    model_name = Path(config.pipeline_model_path).stem
    base_conf = float(config.confidence_threshold)
    maxn_conf = float(config.maxn_confidence_threshold)
    interval_seconds = config.interval_seconds

    output_dir = config.get_drop_dir(drop_id)

    raw_df = pd.read_csv(config.get_raw_csv_path(drop_id, model_name))  # type: ignore
    maxn_df = pd.read_csv(config.get_maxn_csv_path(drop_id, model_name))  # type: ignore

    plot_maxn_timeline(
        raw_df, maxn_df, drop_id, output_dir, base_conf, maxn_conf, interval_seconds
    )
