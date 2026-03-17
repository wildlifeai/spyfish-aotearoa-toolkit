import os
from pathlib import Path
import re
import logging
from spyfish.config.base import BaseConfig


def _require(d: dict, key: str, section: str = "") -> object:
    """Raise a clear error if a required config key is missing."""
    if key not in d or d[key] is None:
        location = f"config.yaml [{section}.{key}]" if section else f"config.yaml [{key}]"
        raise KeyError(f"Required config key missing: {location}")
    return d[key]


class PathsConfig(BaseConfig):

    # ── Top-level sections ──────────────────────────────────────────────────

    @property
    def csv_mapping(self) -> dict:
        return _require(self._yaml_config, "csv_mapping", "")

    @property
    def paths(self) -> dict:
        return _require(self._yaml_config, "paths", "")

    @property
    def sub_dirs(self) -> dict:
        return _require(self.paths, "sub_dirs", "paths")

    @property
    def metadata(self) -> dict:
        return _require(self.paths, "metadata", "paths")

    @property
    def metadata_files(self) -> dict:
        return _require(self.metadata, "files", "paths.metadata")

    @property
    def orchestrator(self) -> dict:
        return _require(self._yaml_config, "orchestrator", "")

    # ── Paths ───────────────────────────────────────────────────────────────

    @property
    def base_dir(self) -> str:
        return _require(self.paths, "base_dir", "paths")

    @property
    def s3_bucket(self) -> str:
        return _require(self.paths, "bucket_name", "paths")

    @property
    def pipeline_targets_csv(self) -> str | None:
        """Default CSV path for --set-targets. None if not configured."""
        return self.paths.get("pipeline_targets_csv")

    # ── Metadata / S3 keys ─────────────────────────────────────────────────

    @property
    def metadata_root(self) -> str:
        return _require(self.metadata, "root", "paths.metadata")

    @property
    def sharepoint_root(self) -> str:
        return f"{self.metadata_root}/{_require(self.metadata, 'sharepoint_dir', 'paths.metadata')}"

    @property
    def status_root(self) -> str:
        return f"{self.metadata_root}/{_require(self.metadata, 'status_dir', 'paths.metadata')}"

    @property
    def s3_sharepoint_path(self) -> str:
        return self.sharepoint_root

    @property
    def s3_status_path(self) -> str:
        return self.status_root

    @property
    def s3_sharepoint_deployment_csv(self) -> str:
        return f"{self.sharepoint_root}/{_require(self.metadata_files, 'deployment_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_survey_csv(self) -> str:
        return f"{self.sharepoint_root}/{_require(self.metadata_files, 'survey_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_site_csv(self) -> str:
        return f"{self.sharepoint_root}/{_require(self.metadata_files, 'site_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_species_csv(self) -> str:
        return f"{self.sharepoint_root}/{_require(self.metadata_files, 'species_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_reserves_csv(self) -> str:
        return f"{self.sharepoint_root}/{_require(self.metadata_files, 'reserves_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_annotations_legacy_experts_csv(self) -> str:
        return f"{self.sharepoint_root}/{_require(self.metadata_files, 'legacy_experts_csv', 'paths.metadata.files')}"

    @property
    def test_deployment_metadata_csv(self) -> Path:
        return self.project_root / _require(self.metadata_files, "test_deployment_csv", "paths.metadata.files")

    @property
    def s3_missing_files(self) -> str:
        return f"{self.s3_status_path}/{_require(self.paths, 'missing_files_filename', 'paths')}"

    @property
    def s3_extra_files(self) -> str:
        return f"{self.s3_status_path}/{_require(self.paths, 'extra_files_filename', 'paths')}"

    @property
    def s3_deployment_status_csv(self) -> str:
        return f"{self.s3_status_path}/deployments_status.csv"

    @property
    def s3_survey_status_csv(self) -> str:
        return f"{self.s3_status_path}/surveys_status.csv"

    # ── Sub-directories (local + S3) ────────────────────────────────────────

    def _sub(self, key: str) -> str:
        return _require(self.sub_dirs, key, "paths.sub_dirs")

    @property
    def media_dir(self) -> Path:
        return self.project_root / self._sub("media")

    @property
    def data_quality_dir(self) -> Path:
        return self.project_root / self.base_dir / self._sub("data_quality")

    @property
    def logs_dir(self) -> Path:
        return self.project_root / self.base_dir / self._sub("logs")

    @property
    def models_root_dir(self) -> Path:
        return self.project_root / self.base_dir / self._sub("models")

    @property
    def base_model_dir(self) -> Path:
        return self.models_root_dir / self._sub("base_model")

    @property
    def pipeline_model_dir(self) -> Path:
        return self.models_root_dir / self._sub("pipeline_model")

    @property
    def trained_model_dir(self) -> Path:
        return self.models_root_dir / self._sub("trained")

    @property
    def biigle_s3_images_prefix(self) -> str:
        return f"{self.base_dir}/{self._sub('biigle_images')}"

    @property
    def s3_data_quality_dir(self) -> str:
        return f"{self.base_dir}/{self._sub('data_quality')}"

    @property
    def s3_annotations_dir(self) -> str:
        return f"{self.base_dir}/{self._sub('annotations')}"

    @property
    def s3_training_output_prefix(self) -> str:
        return f"{self.base_dir}/{self._sub('training')}/"

    # ── DB paths ────────────────────────────────────────────────────────────

    def db_rel_path(self, filename: str) -> str:
        return f"{self.base_dir}/{self._sub('db')}/{filename}"

    def get_db_path(self, filename: str) -> Path:
        path = self.project_root / self.db_rel_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_s3_db_key(self, filename: str) -> str:
        return self.db_rel_path(filename)

    @property
    def db_path(self) -> Path:
        return self.get_db_path("spyfish_pipeline.db")

    @property
    def s3_db_key(self) -> str:
        return self.get_s3_db_key("spyfish_pipeline.db")

    @property
    def annotations_db_path(self) -> Path:
        return self.get_db_path("spyfish_annotations.db")

    @property
    def s3_annotations_db_key(self) -> str:
        return self.get_s3_db_key("spyfish_annotations.db")

    # ── Model paths ─────────────────────────────────────────────────────────

    @property
    def pipeline_model_path(self) -> Path:
        if self.pipeline_model_dir.exists():
            pt_files = list(self.pipeline_model_dir.glob("*.pt"))
            if pt_files:
                return pt_files[0]
        raise FileNotFoundError(f"No model file found in {self.pipeline_model_dir}")

    @property
    def s3_model_key(self) -> str:
        return f"{self.base_dir}/models/pipeline_model/{self.pipeline_model_path.name}"

    @property
    def s3_training_base_model_key(self) -> str:
        if self.base_model_dir.exists():
            pt_files = list(self.base_model_dir.glob("*.pt"))
            if pt_files:
                return f"{self.base_dir}/models/base_model/{pt_files[0].name}"
        return f"{self.base_dir}/models/base_model/model.pt"

    # ── CSV column names ────────────────────────────────────────────────────

    def _col(self, key: str) -> str:
        return _require(self.csv_mapping, key, "csv_mapping")

    @property
    def drop_id_column(self) -> str:
        return self._col("drop_id_column")

    @property
    def pipeline_status_column(self) -> str:
        return self._col("pipeline_status_column")

    @property
    def survey_id_column(self) -> str:
        return self._col("survey_id_column")

    @property
    def site_id_column(self) -> str:
        return self._col("site_id_column")

    @property
    def replicate_column(self) -> str:
        return self._col("replicate_column")

    @property
    def file_name_column(self) -> str:
        return self._col("file_name_column")

    @property
    def link_to_marine_reserve_column(self) -> str:
        return self._col("link_to_marine_reserve_column")

    @property
    def site_name_column(self) -> str:
        return self._col("site_name_column")

    @property
    def csv_video_file_link_column(self) -> str:
        return self._col("video_file_link_column")

    @property
    def csv_sampling_start_column(self) -> str:
        return self._col("sampling_start_column")

    @property
    def csv_sampling_end_column(self) -> str:
        return self._col("sampling_end_column")

    @property
    def csv_clip_start_column(self) -> str:
        return self._col("clip_start_column")

    @property
    def csv_clip_end_column(self) -> str:
        return self._col("clip_end_column")

    @property
    def csv_clip_max_time_column(self) -> str:
        return self._col("clip_max_time_column")

    @property
    def csv_confidence_agreement_column(self) -> str:
        return self._col("confidence_agreement_column")

    @property
    def csv_confusion_score_column(self) -> str:
        return self._col("confusion_score_column")

    @property
    def csv_scientific_name_column(self) -> str:
        return self._col("scientific_name_column")

    @property
    def csv_maxn_time_column(self) -> str:
        return self._col("maxn_time_column")

    @property
    def csv_maxn_time_ms_column(self) -> str:
        return self._col("maxn_time_ms_column")

    @property
    def csv_max_interval_column(self) -> str:
        return self._col("max_interval_column")

    @property
    def csv_annotated_by_column(self) -> str:
        return self._col("annotated_by_column")

    @property
    def csv_interval_annotation_column(self) -> str:
        return self._col("interval_annotation_column")

    @property
    def csv_time_seconds_column(self) -> str:
        return self._col("time_seconds_column")

    # ── Validation ──────────────────────────────────────────────────────────

    @property
    def validation_patterns(self) -> dict:
        patterns = _require(self._yaml_config, "validation_patterns", "")
        return {
            self.drop_id_column:   _require(patterns, "drop_id",   "validation_patterns"),
            self.survey_id_column: _require(patterns, "survey_id", "validation_patterns"),
            self.site_id_column:   _require(patterns, "site_id",   "validation_patterns"),
        }

    @property
    def validation_rules(self) -> dict:
        return _require(self._yaml_config, "validation_rules", "")

    @property
    def movie_extensions(self) -> list:
        return _require(self.paths, "movie_extensions", "paths")

    @property
    def file_presence_rules(self) -> dict:
        return {
            "file_presence": {
                "bucket": self.s3_bucket,
                "s3_sharepoint_path": self.s3_sharepoint_path,
                "csv_filename": _require(self.metadata_files, "deployment_csv", "paths.metadata.files"),
                "csv_column_to_extract": self.csv_video_file_link_column,
                "column_filter": None,
                "column_value": None,
                "valid_extensions": self.movie_extensions,
                "path_prefix": _require(self.sub_dirs, "media", "paths.sub_dirs"),
            }
        }

    def validate_drop_id(self, drop_id: str) -> str:
        pattern = self.validation_patterns[self.drop_id_column]
        if not re.match(pattern, drop_id):
            raise ValueError(f"Invalid DropID format: '{drop_id}'. Must match {pattern}")
        if ".." in drop_id or "/" in drop_id or "\\" in drop_id:
            raise ValueError(f"Security Alert: Malicious DropID detected (potential path traversal): '{drop_id}'")
        return drop_id

    # ── Orchestrator ────────────────────────────────────────────────────────

    @property
    def log_output(self) -> str:
        return _require(self.orchestrator, "log_output", "orchestrator").lower()

    # ── Per-drop helpers ────────────────────────────────────────────────────

    def get_drop_dir(self, drop_id: str) -> Path:
        return self.data_quality_dir / self.validate_drop_id(drop_id)

    def get_biigle_cache_dir(self, drop_id: str) -> Path:
        return self.get_drop_dir(drop_id) / self._sub("biigle_cache")

    def get_drop_annotations_dir(self, drop_id: str) -> Path:
        return self.data_quality_dir / self.validate_drop_id(drop_id) / "annotations"

    def get_video_path(self, drop_id: str) -> Path:
        return self.media_dir / f"{self.validate_drop_id(drop_id)}.mp4"

    def get_maxn_csv_path(self, drop_id: str, model_name: str) -> Path:
        return self.get_drop_annotations_dir(drop_id) / f"{self.validate_drop_id(drop_id)}_{model_name}_maxn.csv"

    def get_selections_csv_path(self, drop_id: str) -> Path:
        return self.get_drop_annotations_dir(drop_id) / f"{self.validate_drop_id(drop_id)}_frames_selection.csv"

    def get_raw_csv_path(self, drop_id: str, model_name: str) -> Path:
        return self.get_drop_annotations_dir(drop_id) / f"{self.validate_drop_id(drop_id)}_{model_name}_raw.csv"

    def get_clips_dir(self, drop_id: str, target: str = "") -> Path:
        sub_path = f"{target}_clips" if target else "clips"
        return self.data_quality_dir / self.validate_drop_id(drop_id) / sub_path

    def get_frames_dir(self, drop_id: str, target: str = "") -> Path:
        sub_path = f"{target}_frames" if target else "frames"
        return self.data_quality_dir / self.validate_drop_id(drop_id) / sub_path

    # ── Training ────────────────────────────────────────────────────────────

    @property
    def training_config(self) -> dict:
        return _require(self._yaml_config, "training", "")

    # ── FFmpeg ──────────────────────────────────────────────────────────────

    @property
    def ffmpeg_config(self) -> dict:
        return _require(self._yaml_config, "ffmpeg", "")

    @property
    def ffmpeg_crf(self) -> str:
        return str(_require(self.ffmpeg_config, "crf", "ffmpeg"))

    @property
    def ffmpeg_preset(self) -> str:
        return str(_require(self.ffmpeg_config, "preset", "ffmpeg"))

    @property
    def ffmpeg_codec(self) -> str:
        return str(_require(self.ffmpeg_config, "codec", "ffmpeg"))


config = PathsConfig()
