import re
from pathlib import Path
from typing import Optional

from spyfish.config.base import BaseConfig, get_required


class PathsConfig(BaseConfig):

    # ── Top-level sections ──────────────────────────────────────────────────

    @property
    def paths(self) -> dict:
        return get_required(self._yaml_config, "paths", "")

    @property
    def sub_dirs(self) -> dict:
        return get_required(self.paths, "sub_dirs", "paths")

    @property
    def metadata(self) -> dict:
        return get_required(self.paths, "metadata", "paths")

    @property
    def metadata_files(self) -> dict:
        return get_required(self.metadata, "files", "paths.metadata")

    @property
    def orchestrator(self) -> dict:
        return get_required(self._yaml_config, "orchestrator", "")

    # ── Paths ───────────────────────────────────────────────────────────────

    @property
    def base_dir(self) -> str:
        return get_required(self.paths, "base_dir", "paths")

    @property
    def pipeline_targets_csv(self) -> str | None:
        """Default CSV path for --set-targets."""
        return self.paths.get("deployment_targets_csv")

    # ── Legacy directories (at project root) ────────────────────────────────

    @property
    def legacy_paths(self) -> dict:
        return get_required(self.paths, "legacy", "paths")

    @property
    def legacy_zooniverse_dir(self) -> Path:
        return self.project_root / get_required(
            self.legacy_paths, "zooniverse", "paths.legacy"
        )

    @property
    def legacy_experts_dir(self) -> Path:
        return self.project_root / get_required(
            self.legacy_paths, "experts", "paths.legacy"
        )

    # ── Metadata / S3 keys ─────────────────────────────────────────────────

    @property
    def metadata_root(self) -> str:
        return get_required(self.metadata, "root", "paths.metadata")

    @property
    def sharepoint_root(self) -> str:
        return f"{self.metadata_root}/{get_required(self.metadata, 'sharepoint_dir', 'paths.metadata')}"

    @property
    def status_root(self) -> str:
        return f"{self.metadata_root}/{get_required(self.metadata, 'status_dir', 'paths.metadata')}"

    @property
    def s3_sharepoint_deployment_csv(self) -> str:
        return f"{self.sharepoint_root}/{get_required(self.metadata_files, 'deployment_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_survey_csv(self) -> str:
        return f"{self.sharepoint_root}/{get_required(self.metadata_files, 'survey_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_site_csv(self) -> str:
        return f"{self.sharepoint_root}/{get_required(self.metadata_files, 'site_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_species_csv(self) -> str:
        return f"{self.sharepoint_root}/{get_required(self.metadata_files, 'species_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_reserves_csv(self) -> str:
        return f"{self.sharepoint_root}/{get_required(self.metadata_files, 'reserves_csv', 'paths.metadata.files')}"

    @property
    def s3_sharepoint_annotations_legacy_experts_csv(self) -> str:
        return f"{self.sharepoint_root}/{get_required(self.metadata_files, 'legacy_experts_csv', 'paths.metadata.files')}"

    @property
    def test_deployment_metadata_csv(self) -> Path:
        return self.project_root / get_required(
            self.paths, "test_deployment_csv", "paths"
        )

    @property
    def s3_missing_files(self) -> str:
        return f"{self.status_root}/{get_required(self.paths, 'missing_files_filename', 'paths')}"

    @property
    def s3_extra_files(self) -> str:
        return f"{self.status_root}/{get_required(self.paths, 'extra_files_filename', 'paths')}"

    # ── Sub-directories (local + S3) ────────────────────────────────────────

    def _sub(self, key: str) -> str:
        return get_required(self.sub_dirs, key, "paths.sub_dirs")

    @property
    def media_dir(self) -> Path:
        media_base = self.paths.get("media_base_dir")
        if media_base:
            return Path(media_base) / self._sub("media")
        return self.project_root / self._sub("media")

    @property
    def deployment_data_dir(self) -> Path:
        return self.project_root / self.base_dir / self._sub("deployment_data")

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
    def archived_models_dir(self) -> Path:
        """Where superseded production models go on promotion.

        `_promote_model_locally()` moves the current `pipeline_model_dir`
        contents here before writing new weights, so there's always a stack
        of prior production models to roll back to. Sub-dir name is defined
        in `config.yaml` under `paths.sub_dirs.archived_models`.
        """
        return self.models_root_dir / self._sub("archived_models")

    @property
    def media_s3_prefix(self) -> str:
        """S3 prefix for raw videos; lives at bucket root, independent of base_dir."""
        return f"{self._sub('media')}/"

    @property
    def s3_deployment_data_dir(self) -> str:
        return f"{self.base_dir}/{self._sub('deployment_data')}"

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

    @property
    def db_path(self) -> Path:
        return self.get_db_path("spyfish_pipeline.db")

    @property
    def s3_db_key(self) -> str:
        return self.db_rel_path("spyfish_pipeline.db")

    @property
    def annotations_db_path(self) -> Path:
        return self.get_db_path("spyfish_annotations.db")

    @property
    def s3_annotations_db_key(self) -> str:
        return self.db_rel_path("spyfish_annotations.db")

    # ── Model paths ─────────────────────────────────────────────────────────

    @staticmethod
    def _first_model_file(directory: Path) -> Optional[Path]:
        """Return the first .pt file in `directory`, or None if the directory
        doesn't exist or contains no .pt file.

        Single-file lookups are fine — production workflows keep exactly one
        promoted model per directory (pipeline_model/ or base_model/).
        """
        if directory.exists():
            pt_files = list(directory.glob("*.pt"))
            if pt_files:
                return pt_files[0]
        return None

    @property
    def pipeline_model_path(self) -> Path:
        """Find the local pipeline model weights (.pt) from pipeline_model_dir."""
        path = self._first_model_file(self.pipeline_model_dir)
        if path is None:
            raise FileNotFoundError(f"No model file found in {self.pipeline_model_dir}")
        return path

    @property
    def base_model_path(self) -> Path:
        """Find the local base model weights (.pt) from base_model_dir."""
        path = self._first_model_file(self.base_model_dir)
        if path is None:
            raise FileNotFoundError(f"No model file found in {self.base_model_dir}")
        return path

    @property
    def s3_model_key(self) -> str:
        return f"{self.base_dir}/models/pipeline_model/{self.pipeline_model_path.name}"

    @property
    def s3_training_base_model_key(self) -> str:
        path = self._first_model_file(self.base_model_dir)
        filename = path.name if path is not None else "model.pt"
        return f"{self.base_dir}/models/base_model/{filename}"

    # ── Orchestrator ────────────────────────────────────────────────────────

    @property
    def log_output(self) -> str:
        return get_required(self.orchestrator, "log_output", "orchestrator").lower()

    # ── Per-drop helpers ────────────────────────────────────────────────────

    def get_survey_id_from_drop(self, drop_id: str) -> str:
        """Derive SurveyID from a validated DropID using the config survey_id pattern."""
        raw = self.get_validation_pattern("survey_id")
        pattern = raw.lstrip("^").rstrip("$")
        if not pattern.startswith("("):
            pattern = f"({pattern})"
        match = re.search(pattern, drop_id)
        return match.group(1) if match else "UNKNOWN_SURVEY"

    def get_site_id_from_drop(self, drop_id: str) -> str:
        """Derive SiteID from a validated DropID.

        DropID format is ^[A-Z]{3}_\\d{8}_BUV_[A-Z]{3}_\\d{3}_\\d{2}$
        SiteID is always parts[3:5] — positional, not regex, because the
        site_id pattern also matches parts of the survey prefix.
        """
        parts = drop_id.split("_")
        if len(parts) >= 5:
            return f"{parts[3]}_{parts[4]}"
        return "UNKNOWN_SITE"

    def get_drop_dir(self, drop_id: str) -> Path:
        validated = self.validate_drop_id(drop_id)
        survey_id = self.get_survey_id_from_drop(validated)
        return self.deployment_data_dir / survey_id / validated

    def get_drop_annotations_dir(self, drop_id: str) -> Path:
        return self.get_drop_dir(drop_id) / "annotations"

    def get_frames_s3_prefix(self, drop_id: str) -> str:
        """S3 prefix for a drop's frames. Mirrors local `get_frames_dir(drop_id)`."""
        validated = self.validate_drop_id(drop_id)
        survey_id = self.get_survey_id_from_drop(validated)
        return f"{self.s3_deployment_data_dir}/{survey_id}/{validated}/frames/"

    def get_video_path(self, drop_id: str) -> Path:
        return self.media_dir / f"{self.validate_drop_id(drop_id)}.mp4"

    def get_maxn_csv_path(self, drop_id: str, model_name: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_ml_{model_name}_maxn.csv"
        )

    def get_zooniverse_maxn_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_zooniverse_maxn.csv"
        )

    def get_selections_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_frames_selection.csv"
        )

    def get_biigle_selections_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_biigle_frames_selection.csv"
        )

    biigle_expert_raw_suffix = "_biigle_expert_raw.csv"
    biigle_expert_maxn_suffix = "_biigle_expert_maxn.csv"

    def get_biigle_expert_raw_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_expert_raw_suffix}"
        )

    def get_biigle_expert_maxn_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_expert_maxn_suffix}"
        )

    def get_raw_csv_path(self, drop_id: str, model_name: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_{model_name}_raw.csv"
        )

    def get_clips_dir(self, drop_id: str, target: str = "") -> Path:
        sub_path = f"{target}_clips" if target else "clips"
        return self.get_drop_dir(drop_id) / sub_path

    def get_frames_dir(self, drop_id: str, target: str = "") -> Path:
        sub_path = f"{target}_frames" if target else "frames"
        return self.get_drop_dir(drop_id) / sub_path
