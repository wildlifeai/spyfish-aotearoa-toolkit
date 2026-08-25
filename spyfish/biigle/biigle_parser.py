"""
Biigle annotation parser for Spyfish Aotearoa.

Downloads annotation reports from Biigle and processes them into:
- MaxN per 30s interval per species
- Overall MaxN per species
- Size measurements (from LineString annotations with a scale bar)

Ported from: Spyfish-Aotearoa-toolkit_old/sftk/biigle_parser.py
Changes: sftk imports replaced by spyfish equivalents; cache directory uses config.
"""

import io
import json
import logging
import math
import zipfile
from typing import List, Optional, Tuple

import pandas as pd

from spyfish.biigle.biigle_handler import BiigleHandler
from spyfish.config.wrapper import config


class BiigleParser:
    """Parser for downloading and processing Biigle annotation data."""

    def __init__(
        self,
        email: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.biigle_handler = BiigleHandler(email=email, token=token)

    def _export_report(
        self,
        resource: str,
        resource_id: int,
        type_id: int = config.annotation_report_type_video,
    ) -> pd.DataFrame:
        """Download an annotation report from the Biigle API."""
        logging.info(f"Downloading report from Biigle API ({resource} {resource_id})")
        report_id = self.biigle_handler.create_report(
            resource, resource_id, type_id  # type: ignore
        )
        zip_bytes = self.biigle_handler.download_report_zip_bytes(report_id)
        if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
            raise ValueError(
                f"Downloaded report for {resource} {resource_id} is not a valid ZIP "
                f"({len(zip_bytes)} bytes)."
            )

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

    def download_volume_annotations(
        self,
        volume_id: int,
        type_id: int,
    ) -> pd.DataFrame:
        """Download the annotation report for a volume and return it as a DataFrame."""
        return self._export_report(
            resource="volumes",
            resource_id=volume_id,
            type_id=type_id,
        )

    # ── Main processing ───────────────────────────────────────────────────────

    def process_video_annotations(
        self,
        volume_id: int,
        resource: str,
        type_id: int = config.annotation_report_type_video,
    ) -> dict:
        """
        Download and process video annotations from Biigle.

        Args:
            volume_id: Volume or project ID.
            resource: "volumes" or "projects".
            type_id: Biigle annotation report type ID (default: video annotations).

        Returns:
            Dict with keys: raw_annotations_df, max_n_30s_df, max_n_df, sizes_df.
        """
        annotations_df = self._export_report(
            resource=resource,
            resource_id=volume_id,
            type_id=type_id,
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

        # Export per-drop annotation CSVs
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
                raw_path = config.get_biigle_expert_raw_csv_path(d_id)
                d_raw.to_csv(raw_path, index=False)
                logging.info(f"Exported expert raw annotations → {raw_path}")

            if not d_maxn.empty:
                maxn_path = config.get_biigle_expert_maxn_csv_path(d_id)
                # Standardize to mirror DB export
                d_maxn_formatted = self.format_count_annotations_output(d_maxn)
                d_maxn_formatted.to_csv(maxn_path, index=False)
                logging.info(f"Exported expert MaxN annotations → {maxn_path}")

        return {
            "raw_annotations_df": raw_annotations_df,
            "max_n_30s_df": max_n_30s_df,
            "max_n_df": max_n_df,
            "sizes_df": sizes_df,
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
        scale_bar_label = config.scale_bar_label_name
        if not (sizes_df["label_name"] == scale_bar_label).any():
            return pd.DataFrame()

        sizes_df["size_px"] = sizes_df["points"].apply(self.get_size)
        scale_size = sizes_df[sizes_df["label_name"] == scale_bar_label][
            "size_px"
        ].mean()
        sizes_df["size_cm"] = (
            sizes_df["size_px"] * config.scale_bar_length_cm / scale_size
        )
        sizes_df = sizes_df[sizes_df["label_name"] != scale_bar_label]

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

    def process_substrate(
        self, annotations_df: pd.DataFrame, substrate_label_ids: set
    ) -> pd.DataFrame:
        """Compute substrate percent-cover per image from CMECS annotations.

        Substrate is identified by label-tree membership, a row's ``label_id``
        is in ``substrate_label_ids`` (the labels of
        ``config.biigle_substrate_label_tree_id``). NOT by shape. A magic-wand
        ``Polygon`` and a traced ``LineString`` are both valid substrate
        outlines, while a fish-SIZE LineString carries a species / Scale bar
        label (a different tree) and is excluded here (it routes to
        ``process_sizes`` instead). Tree membership is used rather than a
        label-name prefix because CMECS hierarchy paths start at generic roots
        ("Substrate Component", ...) with no Spyfish marker.

        Both shapes are treated as a closed polygon ring and measured with the
        shoelace formula. Two percentages are produced per (image, substrate):

          - ``pct_of_annotated`` (primary): area ÷ total annotated substrate
            area on that image. Categories sum to 100%. CMECS composition.
          - ``pct_of_image``: area ÷ full image area (from Biigle's per-row
            ``attributes`` width×height). Can sum to <100% (unannotated
            seafloor) or >100% (overlapping outlines).

        ``substrate_label_ids`` is supplied by the caller (one API fetch via
        ``BiigleHandler.get_label_tree_labels``) to keep this a pure transform.

        Returns an empty DataFrame if no substrate annotations are present.
        Columns: DropID, image, substrate, pct_of_annotated, pct_of_image,
        area_px, image_area_px.
        """
        if (
            annotations_df.empty
            or "shape_name" not in annotations_df.columns
            or "label_id" not in annotations_df.columns
            or not substrate_label_ids
        ):
            return pd.DataFrame()

        # Image annotation reports use `filename`; video reports `video_filename`.
        image_col = (
            "filename" if "filename" in annotations_df.columns else "video_filename"
        )

        is_substrate = pd.to_numeric(annotations_df["label_id"], errors="coerce").isin(
            substrate_label_ids
        )
        substrate_df = annotations_df[is_substrate].copy()
        if substrate_df.empty:
            logging.info(
                f"No substrate annotations found "
                f"(tree {config.biigle_substrate_label_tree_id})."
            )
            return pd.DataFrame()

        substrate_df["area_px"] = substrate_df["points"].apply(self.get_polygon_area)
        substrate_df["image_area_px"] = substrate_df.get(
            "attributes", pd.Series(index=substrate_df.index, dtype=object)
        ).apply(self._image_area_from_attributes)
        # Leaf label is the substrate category (e.g. "Sand").
        substrate_df["substrate"] = substrate_df["label_name"].astype(str)

        # The raw Biigle report has no DropID column, derive it from the image
        # filename ("{drop_id}__frame_<t>s.jpg" or "{drop_id}_clip_<t>..."), the
        # same convention biigle_to_yolo uses. Prefer an existing DropID column
        # if a caller (e.g. process_video_annotations) already added one. This
        # also lets one volume span multiple drops without mislabelling.
        if config.drop_id_column in substrate_df.columns:
            substrate_df[config.drop_id_column] = substrate_df[
                config.drop_id_column
            ].astype(str)
        else:
            substrate_df[config.drop_id_column] = (
                substrate_df[image_col]
                .astype(str)
                .str.split("__frame_")
                .str[0]
                .str.split("_clip_")
                .str[0]
                .str.replace(r"\.(mp4|jpg|jpeg|png).*$", "", regex=True)
            )

        # Sum the area of all patches of the same substrate on the same image.
        grouped = (
            substrate_df.groupby(
                [config.drop_id_column, image_col, "substrate"], as_index=False
            )
            .agg(area_px=("area_px", "sum"), image_area_px=("image_area_px", "first"))
            .rename(columns={image_col: "image"})
        )

        # Normalised cover (primary): share of the total annotated area per image.
        total_per_image = grouped.groupby([config.drop_id_column, "image"])[
            "area_px"
        ].transform("sum")
        grouped["pct_of_annotated"] = (
            grouped["area_px"] / total_per_image * 100
        ).round(2)
        grouped["pct_of_image"] = (
            grouped["area_px"] / grouped["image_area_px"] * 100
        ).round(2)

        return (
            grouped[
                [
                    config.drop_id_column,
                    "image",
                    "substrate",
                    "pct_of_annotated",
                    "pct_of_image",
                    "area_px",
                    "image_area_px",
                ]
            ]
            .sort_values(
                [config.drop_id_column, "image", "pct_of_annotated"],
                ascending=[True, True, False],
            )
            .reset_index(drop=True)
        )

    # ── Geometry helpers ─────────────────────────────────────────────────────

    def get_size(self, coordinates) -> float:
        return self.sum_distances(self.parse_points(coordinates))

    def get_polygon_area(self, coordinates) -> float:
        """Pixel area enclosed by a Biigle Polygon/LineString, via the shoelace
        formula. The ring is auto-closed (last point → first), so an unclosed
        traced LineString measures the same area as the equivalent Polygon.
        Degenerate shapes (<3 distinct vertices) return 0.0."""
        return self.shoelace_area(self.parse_geometry_points(coordinates))

    def parse_geometry_points(self, points_raw) -> List[Tuple[float, float]]:
        """Parse a Biigle ``points`` field into (x, y) pairs.

        Handles both report shapes: flat ``[x1,y1,x2,y2,...]`` (image
        annotations) and nested ``[[x1,y1,...]]`` (a single video keyframe,
        which is unwrapped). Multi-keyframe video shapes, points that move
        over time, can't be reduced to one ring and return ``[]``.
        """
        if isinstance(points_raw, str):
            try:
                points_raw = json.loads(points_raw)
            except (json.JSONDecodeError, TypeError):
                # Fall back to the bracket-strip parser for bare "[1,2,3,4]".
                return self.parse_points(points_raw)
        if not isinstance(points_raw, (list, tuple)) or not points_raw:
            return []
        # Unwrap a single video keyframe: [[...]] -> [...]; skip multi-keyframe.
        if isinstance(points_raw[0], (list, tuple)):
            if len(points_raw) != 1:
                return []
            points_raw = points_raw[0]
        nums = [float(x) for x in points_raw]
        return list(zip(nums[0::2], nums[1::2]))

    def shoelace_area(self, points: List[Tuple[float, float]]) -> float:
        """Absolute polygon area via the shoelace formula. The ring is closed
        implicitly by iterating modulo len. Returns 0.0 for <3 vertices."""
        n = len(points)
        if n < 3:
            return 0.0
        s = sum(
            points[i][0] * points[(i + 1) % n][1]
            - points[(i + 1) % n][0] * points[i][1]
            for i in range(n)
        )
        return abs(s) / 2.0

    def _image_area_from_attributes(self, attributes_raw) -> Optional[float]:
        """Pixel area (width×height) from Biigle's per-row ``attributes`` JSON,
        or None if the column is absent/unparseable. This is the resolution the
        expert drew on, the authoritative denominator for percent-of-image."""
        if attributes_raw is None or (
            isinstance(attributes_raw, float) and math.isnan(attributes_raw)
        ):
            return None
        try:
            meta = (
                json.loads(attributes_raw)
                if isinstance(attributes_raw, str)
                else attributes_raw
            )
            w, h = meta.get("width"), meta.get("height")
            return float(w) * float(h) if w and h else None
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None

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

        Resolves label_name → scientific_name via class_map. Unknown workflow
        labels (e.g. 'Final', 'Interesting Sighting', 'Done Video') route to the
        'fish' bucket, same fallback discipline as biigle_to_yolo.py and
        discover_extra_drops, so workflow markers can never leak into the
        unified species list.
        """
        from spyfish.config.species import species_registry

        annotations_df = annotations_df.copy()

        registry = species_registry()
        name_to_id = registry.name_to_class_id()
        id_to_name = registry.class_id_to_scientific()

        def _resolve(label_name: str) -> str:
            if " - " in label_name:
                return label_name.split(" - ", 1)[1].strip()
            cid = name_to_id.get(label_name)
            return id_to_name.get(cid, "fish") if cid is not None else "fish"

        annotations_df[config.csv_scientific_name_column] = (
            annotations_df["label_name"].astype(str).apply(_resolve)
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
