"""
sweep_report.py — Generate a Markdown report from a sweep directory.

Produces sweep_dir/report.md with:
  - Final metrics table (sorted by mAP@0.5)
  - Grouped bar chart across runs
  - Training-curve overlay (mAP@0.5 + val box loss per epoch, one line per run)
  - Per-run confusion matrix and PR curve
  - Per-run example predictions vs ground truth (YOLO val_batch0 images)

Assumes sweep.py has already run and produced:
  sweep_dir/comparison.csv
  sweep_dir/<run_name>/results.csv
  sweep_dir/<run_name>_eval/{confusion_matrix,PR_curve,val_batch0_labels,val_batch0_pred}.*

Usage:
    python -m spyfish.ml.training.sweep_report --sweep-dir process_files/training/runs/sweep_20260424_120000
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _collect_run_dirs(sweep_dir: Path) -> List[Path]:
    """Return per-run training subdirs (exclude *_eval and report_assets)."""
    runs = []
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir() or p.name.endswith("_eval") or p.name == "report_assets":
            continue
        if (p / "results.csv").exists():
            runs.append(p)
    return runs


def _successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with no 'error' value (treats both NaN and empty string as success)."""
    if "error" not in df.columns:
        return df
    return df[df["error"].fillna("").astype(str).str.strip() == ""].copy()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_metric_bars(comparison_csv: Path, assets_dir: Path) -> Optional[str]:
    """Grouped bar chart of precision/recall/mAP50/mAP50-95. One group per metric."""
    import matplotlib.pyplot as plt

    df = _successful_rows(pd.read_csv(comparison_csv))
    if df.empty:
        return None

    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    runs = df["run"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / len(runs)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, run in enumerate(runs):
        vals = [float(df[df["run"] == run][m].iloc[0]) for m in metrics]
        offset = i * width - 0.4 + width / 2
        bars = ax.bar(x + offset, vals, width, label=run)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Metric comparison across runs")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out = assets_dir / "metric_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out.relative_to(assets_dir.parent).as_posix()


def _plot_training_curves(run_dirs: List[Path], assets_dir: Path) -> Optional[str]:
    """Overlay validation mAP@0.5 and val box loss per epoch for all runs."""
    import matplotlib.pyplot as plt

    if not run_dirs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plotted_map, plotted_loss = 0, 0

    for run_dir in run_dirs:
        try:
            df = pd.read_csv(run_dir / "results.csv")
        except Exception as e:
            logging.warning(f"Could not read {run_dir}/results.csv: {e}")
            continue

        df.columns = [c.strip() for c in df.columns]
        epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

        map_col = next(
            (
                c
                for c in df.columns
                if "mAP50" in c and "95" not in c and "metrics" in c
            ),
            None,
        )
        loss_col = next(
            (c for c in df.columns if "val" in c.lower() and "box_loss" in c.lower()),
            None,
        )

        if map_col:
            axes[0].plot(df[epoch_col], df[map_col], label=run_dir.name)
            plotted_map += 1
        if loss_col:
            axes[1].plot(df[epoch_col], df[loss_col], label=run_dir.name)
            plotted_loss += 1

    axes[0].set(xlabel="Epoch", ylabel="mAP@0.5", title="Validation mAP@0.5")
    axes[1].set(xlabel="Epoch", ylabel="val/box_loss", title="Validation box loss")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if plotted_map == 0 and plotted_loss == 0:
        plt.close(fig)
        return None

    out = assets_dir / "training_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out.relative_to(assets_dir.parent).as_posix()


# ---------------------------------------------------------------------------
# YOLO-generated image collection
# ---------------------------------------------------------------------------


def _collect_eval_images(
    sweep_dir: Path, assets_dir: Path, filename: str
) -> List[Tuple[str, str]]:
    """
    Copy a per-run YOLO eval image (e.g. 'confusion_matrix.png') into assets
    with a run-prefixed name. Returns [(run_name, markdown_relative_path)].
    """
    pairs = []
    for eval_dir in sorted(sweep_dir.glob("*_eval")):
        src = eval_dir / filename
        if not src.exists():
            continue
        run_name = eval_dir.name.replace("_eval", "")
        dst = assets_dir / f"{run_name}__{filename}"
        shutil.copy2(src, dst)
        pairs.append((run_name, dst.relative_to(assets_dir.parent).as_posix()))
    return pairs


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _to_md_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Render a DataFrame as a Markdown table (no tabulate dep)."""
    if df.empty:
        return "_(no rows)_"

    def _fmt(v):
        if isinstance(v, float) and np.isfinite(v):
            return format(v, float_fmt)
        return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)

    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in headers) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(sweep_dir: Path) -> Path:
    comparison_csv = sweep_dir / "comparison.csv"
    if not comparison_csv.exists():
        raise FileNotFoundError(
            f"comparison.csv not found in {sweep_dir} — run sweep.py first."
        )

    assets_dir = sweep_dir / "report_assets"
    assets_dir.mkdir(exist_ok=True)

    df_all = pd.read_csv(comparison_csv)
    df_ok = _successful_rows(df_all)
    df_failed = df_all[~df_all.index.isin(df_ok.index)]

    numeric_cols = ["mAP50", "mAP50_95", "precision", "recall"]
    for c in numeric_cols:
        if c in df_ok.columns:
            df_ok[c] = pd.to_numeric(df_ok[c], errors="coerce")
    df_ok = df_ok.sort_values("mAP50", ascending=False)

    lines = [f"# Sweep report: `{sweep_dir.name}`", ""]

    # Val-fallback callout — surface when the test split was empty and we
    # evaluated on val instead. Val was used for early stopping during training,
    # so these metrics are optimistic, not a held-out evaluation.
    if "eval_split" in df_ok.columns:
        val_runs = df_ok[df_ok["eval_split"].astype(str) == "val"]["run"].tolist()
        if val_runs:
            lines += [
                "> ⚠️ **Metrics evaluated on val split, not test.**",
                "> ",
                f"> The following runs had no test images and fell back to val: "
                f"`{', '.join(val_runs)}`. ",
                "> Val was used for early stopping during training, so these numbers "
                "over-estimate held-out performance. Add more drops or extract more "
                "frames so `assemble_yolo_dataset` can carve out a real test split.",
                "",
            ]

    # Final metrics
    lines += ["## Final metrics (sorted by mAP@0.5)", ""]
    display_cols = [
        c
        for c in [
            "run",
            "mAP50",
            "mAP50_95",
            "precision",
            "recall",
            "imgsz",
            "batch",
            "eval_split",
        ]
        if c in df_ok.columns
    ]
    lines.append(_to_md_table(df_ok[display_cols]))
    lines.append("")

    if not df_failed.empty:
        lines += ["### Failed runs", ""]
        fail_cols = [c for c in ["run", "error"] if c in df_failed.columns]
        lines.append(_to_md_table(df_failed[fail_cols]))
        lines.append("")

    # Metric bar chart
    bar_path = _plot_metric_bars(comparison_csv, assets_dir)
    if bar_path:
        lines += ["## Metric comparison", "", f"![Metric bars]({bar_path})", ""]

    # Training curves
    run_dirs = _collect_run_dirs(sweep_dir)
    curve_path = _plot_training_curves(run_dirs, assets_dir)
    if curve_path:
        lines += [
            "## Training curves",
            "",
            "Validation mAP@0.5 and box loss per epoch. Flat tails → converged; still-climbing → under-trained; diverging loss while mAP flat → overfitting.",
            "",
            f"![Training curves]({curve_path})",
            "",
        ]

    # Confusion matrices
    cm_pairs = _collect_eval_images(sweep_dir, assets_dir, "confusion_matrix.png")
    cm_norm = dict(
        _collect_eval_images(sweep_dir, assets_dir, "confusion_matrix_normalized.png")
    )
    if cm_pairs:
        lines += ["## Confusion matrices", ""]
        for run_name, rel in cm_pairs:
            lines += [f"### {run_name}", "", f"![{run_name} CM]({rel})", ""]
            if run_name in cm_norm:
                lines += [f"![{run_name} CM normalized]({cm_norm[run_name]})", ""]

    # PR curves
    pr_pairs = _collect_eval_images(sweep_dir, assets_dir, "PR_curve.png")
    if pr_pairs:
        lines += ["## Precision–recall curves", ""]
        for run_name, rel in pr_pairs:
            lines += [f"### {run_name}", "", f"![{run_name} PR]({rel})", ""]

    # Example predictions vs ground truth
    pred_pairs = _collect_eval_images(sweep_dir, assets_dir, "val_batch0_pred.jpg")
    label_pairs = dict(
        _collect_eval_images(sweep_dir, assets_dir, "val_batch0_labels.jpg")
    )
    if pred_pairs:
        lines += [
            "## Example predictions vs ground truth",
            "",
            "Each grid is 16 images from batch 0 of the validation set. "
            "`labels` = human annotations, `pred` = model output at the evaluation threshold.",
            "",
        ]
        for run_name, pred_rel in pred_pairs:
            lines += [f"### {run_name}", ""]
            if run_name in label_pairs:
                lines += [
                    "**Ground truth:**",
                    "",
                    f"![{run_name} labels]({label_pairs[run_name]})",
                    "",
                ]
            lines += [
                "**Predictions:**",
                "",
                f"![{run_name} predictions]({pred_rel})",
                "",
            ]

    report_path = sweep_dir / "report.md"
    report_path.write_text("\n".join(lines))
    logging.info(f"Wrote report → {report_path}")
    logging.info(f"Assets → {assets_dir}")
    return report_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report from a sweep directory."
    )
    parser.add_argument(
        "--sweep-dir",
        required=True,
        type=Path,
        help="Path to the sweep output directory "
        "(e.g. process_files/training/runs/sweep_20260424_120000).",
    )
    args = parser.parse_args()
    build_report(args.sweep_dir)


if __name__ == "__main__":
    main()
