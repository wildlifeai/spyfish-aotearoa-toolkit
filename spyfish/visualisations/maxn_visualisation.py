"""
MaxN timeline visualisation — produces a PNG saved to the data_quality folder.

Usage (standalone):
    python spyfish/visualisations/maxn_visualisation.py

Called programmatically:
    from spyfish.visualisations.maxn_visualisation import plot_maxn_timeline
    plot_maxn_timeline(raw_df, maxn_df, drop_id, output_dir, base_conf, maxn_conf, interval_seconds)
"""
import logging
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


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
        maxn_df: MaxN results DataFrame (columns: time_of_maxn_ms, MaxInterval, ConfidenceAgreement).
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

    # Aggregate fish counts at both thresholds
    base_fish = (
        raw_df[raw_df["confidence"] >= base_conf]
        .groupby("time_seconds").size().reset_index(name="count")
    )
    maxn_fish = (
        raw_df[raw_df["confidence"] >= maxn_conf]
        .groupby("time_seconds").size().reset_index(name="count")
    )

    if base_fish.empty:
        logging.warning(f"No detections above base_conf={base_conf} for {drop_id}. Skipping plot.")
        return None

    max_time = base_fish["time_seconds"].max()

    fig, ax = plt.subplots(figsize=(16, 6))

    # 1. Alternating interval background bands
    for i in range(int(max_time // interval_seconds) + 1):
        start = i * interval_seconds
        color = "#f5f5f5" if i % 2 == 0 else "#ebebeb"
        ax.axvspan(start, start + interval_seconds, alpha=0.6, color=color, zorder=0)

    # 2. Green highlight: MaxN clip windows
    for _, row in maxn_df.iterrows():
        t = row["time_of_maxn_ms"]
        interval_start = (t // interval_seconds) * interval_seconds
        ax.axvspan(interval_start, interval_start + interval_seconds,
                   alpha=0.15, color="#2ecc71", zorder=1)

    # 3. Base confidence line (light grey)
    ax.fill_between(base_fish["time_seconds"], base_fish["count"],
                    alpha=0.15, color="#95a5a6", zorder=2)
    ax.plot(base_fish["time_seconds"], base_fish["count"],
            color="#95a5a6", linewidth=1, alpha=0.6, zorder=3,
            label=f"Fish (conf ≥ {base_conf})")

    # 4. MaxN confidence line (solid blue)
    if not maxn_fish.empty:
        ax.fill_between(maxn_fish["time_seconds"], maxn_fish["count"],
                        alpha=0.25, color="#3498db", zorder=4)
        ax.plot(maxn_fish["time_seconds"], maxn_fish["count"],
                color="#2980b9", linewidth=2, zorder=5,
                label=f"Fish (conf ≥ {maxn_conf}) — used for MaxN")

    # 5. Red dots: MaxN peaks with annotation
    for _, row in maxn_df.iterrows():
        t = row["time_of_maxn_ms"]
        count = row["MaxInterval"]
        ax.scatter(t, count, color="#e74c3c", s=120, zorder=6,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(
            f"MaxN={count}\n({row['ConfidenceAgreement']:.2f})",
            xy=(t, count), xytext=(5, 10), textcoords="offset points",
            fontsize=8, color="#c0392b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#e74c3c", alpha=0.8),
        )

    # Styling
    ax.set_xlabel("Time (seconds from SamplingStart)", fontsize=12)
    ax.set_ylabel("Fish Count per Frame", fontsize=12)
    ax.set_title(f"MaxN Timeline — {drop_id}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max_time + 1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        plt.Line2D([0], [0], color="#95a5a6", linewidth=1,
                   label=f"Fish (conf ≥ {base_conf})"),
        plt.Line2D([0], [0], color="#2980b9", linewidth=2,
                   label=f"Fish (conf ≥ {maxn_conf}) — MaxN"),
        plt.scatter([], [], color="#e74c3c", s=80, edgecolors="white",
                    label="MaxN peak"),
        mpatches.Patch(facecolor="#2ecc71", alpha=0.15,
                       label="Selected clip window"),
        mpatches.Patch(facecolor="#ebebeb", alpha=0.6,
                       label=f"{interval_seconds}s intervals"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path = output_dir / f"{drop_id}_maxn_timeline.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved MaxN timeline plot to {out_path}")
    return out_path


if __name__ == "__main__":
    import logging
    from pathlib import Path
    from spyfish.config import config

    drop_id = config.test_drops[0][0]
    model_name = Path(config.model_path or config.mock_model_path).stem
    base_conf = float(config.confidence_threshold)
    maxn_conf = float(config.maxn_confidence_threshold)
    interval_seconds = config.interval_seconds

    repo_root = Path(__file__).parent.parent.parent
    annotations_dir = repo_root / config.local_manifest_dir_path
    output_dir = repo_root / config.local_data_quality_dir / drop_id

    raw_df = pd.read_csv(annotations_dir / f"{drop_id}_{model_name}_raw.csv")
    maxn_df = pd.read_csv(annotations_dir / f"{drop_id}_{model_name}_maxn.csv")

    plot_maxn_timeline(raw_df, maxn_df, drop_id, output_dir, base_conf, maxn_conf, interval_seconds)
