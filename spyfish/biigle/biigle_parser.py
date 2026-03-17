"""
Biigle annotation parser for Spyfish Aotearoa.

Downloads annotation reports from Biigle and processes them into:
- MaxN per 30s interval per species
- Overall MaxN per species
- Size measurements (from LineString annotations with a scale bar)

Ported from: Spyfish-Aotearoa-toolkit_old/sftk/biigle_parser.py
Changes: sftk imports replaced by spyfish equivalents; cache directory uses config.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.config.wrapper import config

SCALE_BAR_LENGTH_CM = 10
SCALE_BAR_LABEL_NAME = "Scale bar"


class BiigleParser:
    """Parser for downloading and processing Biigle annotation data."""

    def __init__(
        self,
        email: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        drop_id: Optional[str] = None,
    ):
        self.biigle_handler = BiigleHandler(email=email, token=token)
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        elif drop_id:
            self.cache_dir = config.get_biigle_cache_dir(drop_id)
        else:
            # No drop context — use a shared root-level biigle cache
            self.cache_dir = config.data_quality_dir / config._sub("biigle_cache")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Biigle cache directory: {self.cache_dir}")

    # ── Caching ───────────────────────────────────────────────────────────────

    def _get_cached_zip_path(self, resource: str, resource_id: int) -> Path:
        return self.cache_dir / f"{resource_id}_{resource}_report.zip"

    def _export_report_with_cache(
        self,
        resource: str,
        resource_id: int,
        type_id: int = config.annotation_report_type_video,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Export report from Biigle API with optional local ZIP cache."""
        cache_path = self._get_cached_zip_path(resource, resource_id)

        if use_cache and cache_path.exists():
            logging.info(f"Using cached report: {cache_path}")
            zip_bytes = cache_path.read_bytes()
        else:
            logging.info(
                f"Downloading report from Biigle API ({resource} {resource_id})"
            )
            report_id = self.biigle_handler.create_report(
                resource, resource_id, type_id  # type: ignore
            )
            zip_bytes = self.biigle_handler.download_report_zip_bytes(report_id)
            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(zip_bytes)
                logging.info(f"Cached report: {cache_path}")

        allow_nested = resource == "projects"
        dfs = self.biigle_handler.read_csvs_from_zip_bytes(
            zip_bytes, allow_nested=allow_nested
        )
        if not dfs:
            return pd.DataFrame()

        result_df = pd.concat(dfs.values(), ignore_index=True)
        logging.info(
            f"Loaded {len(dfs)} CSV(s), {len(result_df)} total rows from {resource} {resource_id}"
        )
        return result_df

    # ── Main processing ───────────────────────────────────────────────────────

    def process_video_annotations(
        self,
        volume_id: int,
        resource: str,
        type_id: int = config.annotation_report_type_video,
        local_csv_path: Optional[str] = None,
        use_cache: bool = True,
    ) -> dict:
        """
        Download and process video annotations from Biigle.

        Args:
            volume_id: Volume or project ID.
            resource: "volumes", "projects", or "local" (use local_csv_path).
            type_id: Biigle annotation report type ID.
            local_csv_path: Path to a local CSV (only used when resource="local").
            use_cache: Cache downloaded ZIP files locally to avoid repeated API calls.

        Returns:
            Dict with keys:
                raw_annotations_df, max_n_30s_df, max_n_df, sizes_df, maxn_csv_path
        """
        if resource == "local":
            if not local_csv_path:
                raise ValueError("local_csv_path required when resource='local'")
            p = Path(local_csv_path)
            if not p.exists():
                raise FileNotFoundError(f"Local CSV not found: {p}")
            logging.info(f"Loading annotations from local CSV: {p}")
            annotations_df = pd.read_csv(p)
        else:
            annotations_df = self._export_report_with_cache(
                resource=resource,
                resource_id=volume_id,
                type_id=type_id,
                use_cache=use_cache,
            )

        if annotations_df.empty:
            logging.info(f"No annotations found for {resource} {volume_id}.")
            return {}

        # Derive DropID from video filename
        annotations_df[config.drop_id_column] = annotations_df[
            "video_filename"
        ].str.replace(r"\.mp4.*", "", regex=True)

        raw_annotations_df = annotations_df.copy()

        required_columns = [
            config.drop_id_column,
            "label_name",
            "video_id",
            "video_filename",
            "shape_id",
            "shape_name",
            "points",
            "frames",
        ]
        annotations_df = annotations_df[required_columns]
        annotations_df = self.extract_time_values(annotations_df)
        annotations_df = annotations_df.sort_values(
            ["video_filename", "start_seconds", "frame_seconds"]
        ).reset_index(drop=True)

        max_n_30s_df = self.process_30s_count(annotations_df)
        max_n_df = self.process_max_count(max_n_30s_df)
        sizes_df = self.process_sizes(annotations_df)

        # 1. Save maxn CSV to cache
        maxn_csv_path = self.cache_dir / f"{volume_id}_{resource}_maxn.csv"
        max_n_df.to_csv(maxn_csv_path, index=False)
        logging.info(f"Saved MaxN data → {maxn_csv_path}")

        # 2. Side-car export to drop-specific annotations folder
        # We find uniquely active drops in this report
        unique_drops = annotations_df[config.drop_id_column].unique()
        for d_id in unique_drops:
            drop_ann_dir = config.get_drop_annotations_dir(d_id)
            drop_ann_dir.mkdir(parents=True, exist_ok=True)

            # Filter for this drop
            d_raw = raw_annotations_df[
                raw_annotations_df[config.drop_id_column] == d_id
            ]
            d_maxn = max_n_df[max_n_df[config.drop_id_column] == d_id]

            if not d_raw.empty:
                raw_path = drop_ann_dir / f"{d_id}_biigle_expert_raw.csv"
                d_raw.to_csv(raw_path, index=False)
                logging.info(f"Exported expert raw annotations → {raw_path}")

            if not d_maxn.empty:
                maxn_path = drop_ann_dir / f"{d_id}_biigle_expert_maxn.csv"
                # Standardize to mirror DB export
                d_maxn_formatted = self.format_count_annotations_output(d_maxn)
                d_maxn_formatted.to_csv(maxn_path, index=False)
                logging.info(f"Exported expert MaxN annotations → {maxn_path}")

        return {
            "raw_annotations_df": raw_annotations_df,
            "max_n_30s_df": max_n_30s_df,
            "max_n_df": max_n_df,
            "sizes_df": sizes_df,
            "maxn_csv_path": str(maxn_csv_path),
        }

    # ── Aggregation helpers ───────────────────────────────────────────────────

    def process_30s_count(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """Count species occurrences per 30s interval (Rectangle annotations only)."""
        count_df = annotations_df[annotations_df["shape_name"] == "Rectangle"].copy()
        grouped = (
            count_df.groupby(
                [
                    config.drop_id_column,
                    "video_filename",
                    "label_name",
                    "start_seconds",
                    "frame_seconds",
                    "time_of_max",
                ]
            )
            .size()
            .reset_index(name="max_count")
        )
        return grouped.sort_values(["start_seconds", "frame_seconds"]).reset_index(
            drop=True
        )

    def process_max_count(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the single row with the highest count per (DropID, label_name).
        Ties resolved by earliest frame time.
        """
        ordered = annotations_df.sort_values(
            [
                config.drop_id_column,
                "max_count",
                "start_seconds",
                "frame_seconds",
            ],
            ascending=[True, False, True, True],
        )
        result = ordered.drop_duplicates(
            subset=[config.drop_id_column, "label_name"], keep="first"
        )
        return result.sort_values(["start_seconds", "frame_seconds"]).reset_index(
            drop=True
        )

    def process_sizes(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fish sizes from LineString annotations using a scale bar.
        Returns an empty DataFrame if no scale bar is found.
        """
        sizes_df = annotations_df[annotations_df["shape_name"] == "LineString"].copy()
        if not (sizes_df["label_name"] == SCALE_BAR_LABEL_NAME).any():
            return pd.DataFrame()

        sizes_df["size_px"] = sizes_df["points"].apply(self.get_size)
        scale_size = sizes_df[sizes_df["label_name"] == SCALE_BAR_LABEL_NAME][
            "size_px"
        ].mean()
        sizes_df["size_cm"] = sizes_df["size_px"] * SCALE_BAR_LENGTH_CM / scale_size
        sizes_df = sizes_df[sizes_df["label_name"] != SCALE_BAR_LABEL_NAME]

        return (
            sizes_df[
                [
                    config.drop_id_column,
                    "label_name",
                    "video_filename",
                    "start_seconds",
                    "frame_seconds",
                    "time_of_max",
                    "size_px",
                    "size_cm",
                ]
            ]
            .sort_values(["start_seconds", "frame_seconds"])
            .reset_index(drop=True)
        )

    # ── Geometry helpers ─────────────────────────────────────────────────────

    def get_size(self, coordinates) -> float:
        return self.sum_distances(self.parse_points(coordinates))

    def parse_points(self, points_str: str) -> list:
        nums = [float(x) for x in str(points_str).strip("[]").split(",")]
        return list(zip(nums[0::2], nums[1::2]))

    def sum_distances(self, points: list) -> float:
        return sum(
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(points[:-1], points[1:])
        )

    def extract_time_values(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add start_seconds, frame_seconds, and time_of_max columns.
        start_seconds: extracted from clip filename (_clip_<start>_<dur>).
        frame_seconds: the annotation frame timestamp from Biigle's 'frames' column.
        time_of_max: HH:MM:SS of start + frame.
        """
        annotations_df["start_seconds"] = pd.to_numeric(
            annotations_df["video_filename"].str.extract(r"_clip_(\d+)_", expand=False),
            errors="coerce",
        ).astype("Int64")

        annotations_df["frame_seconds"] = (
            annotations_df["frames"].str.strip("[]").astype(float)
        )

        total_seconds = (
            annotations_df["start_seconds"] + annotations_df["frame_seconds"]
        )
        annotations_df["time_of_max"] = pd.to_datetime(
            total_seconds, unit="s"
        ).dt.strftime("%H:%M:%S")
        return annotations_df

    # ── Output formatting ─────────────────────────────────────────────────────

    def format_count_annotations_output(
        self, annotations_df: pd.DataFrame, interval_annotation_s: int = 30
    ) -> pd.DataFrame:
        """
        Format MaxN results into the standard Spyfish annotation output format,
        matching the column names expected by ingest.py.
        """
        annotations_df = annotations_df.copy()
        annotations_df[config.csv_scientific_name_column] = (
            annotations_df["label_name"]
            .str.split(" - ")
            .str[1]
            .fillna(annotations_df["label_name"])
        )
        return annotations_df.rename(
            columns={
                "max_count": config.csv_max_interval_column,
                "time_of_max": config.csv_maxn_time_column,
            }
        ).assign(
            **{
                config.csv_annotated_by_column: "expert",
                config.csv_interval_annotation_column: interval_annotation_s,
                config.csv_confidence_agreement_column: "NA",
            }
        )[
            [
                config.drop_id_column,
                config.csv_scientific_name_column,
                config.csv_maxn_time_column,
                config.csv_max_interval_column,
                config.csv_annotated_by_column,
                config.csv_interval_annotation_column,
                config.csv_confidence_agreement_column,
            ]
        ]
