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

    @property
    def species_labels_csv_path(self) -> Path:
        """Biigle label-tree export (``Common - Scientific`` names + label IDs).

        Shared by BIIGLE upload (species → label_id routing) and Zooniverse
        parsing (choice key → scientific name normalisation).
        """
        return (
            self.project_root
            / self.base_dir
            / "biigle"
            / "labels"
            / "species_labels.csv"
        )

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

    def get_pipeline_model(self, kind: str) -> Path:
        """Find a specific pipeline model variant in pipeline_model_dir by filename prefix.

        Files in pipeline_model_dir are expected to follow the convention
        `{kind}_*.pt`, e.g.:
          - binary_cfd_water_20260301.pt
          - species_cfd_20260426_234352.pt

        If multiple files match, the most recently modified is returned.

        Args:
            kind: Model variant — must be "binary" or "species".

        Raises:
            ValueError: kind is not one of {"binary", "species"}.
            FileNotFoundError: no `{kind}_*.pt` file in pipeline_model_dir.
        """
        if kind not in {"binary", "species"}:
            raise ValueError(f"kind must be 'binary' or 'species', got {kind!r}")
        if not self.pipeline_model_dir.exists():
            raise FileNotFoundError(
                f"Model directory does not exist: {self.pipeline_model_dir}"
            )
        candidates = sorted(
            self.pipeline_model_dir.glob(f"{kind}_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No '{kind}_*.pt' model found in {self.pipeline_model_dir}. "
                f"Expected naming convention: {kind}_*.pt (e.g. {kind}_cfd_20260101.pt)"
            )
        return candidates[0]

    @property
    def pipeline_model_path(self) -> Path:
        """Find the local pipeline model weights (.pt) from pipeline_model_dir.

        Defaults to the species model when both binary and species coexist.
        For explicit selection between variants, use `get_pipeline_model(kind)`.

        Falls back to the first .pt file in the directory if no `species_*.pt`
        is present — preserves backward compatibility with single-model setups
        that pre-date the binary/species naming convention.
        """
        try:
            return self.get_pipeline_model("species")
        except FileNotFoundError:
            path = self._first_model_file(self.pipeline_model_dir)
            if path is None:
                raise FileNotFoundError(
                    f"No model file found in {self.pipeline_model_dir}"
                )
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

    def get_training_frames_s3_prefix(self, survey_id: str) -> str:
        """S3 prefix for the survey-level training-frames Biigle volume.

        All drops in a survey upload their training frames to the same S3
        prefix (filenames carry drop_id, so they're unique). This is the
        URL Biigle's volume points at.
        """
        return f"{self.s3_deployment_data_dir}/{survey_id}/training_frames/"

    def get_video_path(self, drop_id: str) -> Path:
        return self.media_dir / f"{self.validate_drop_id(drop_id)}.mp4"

    def get_video_s3_key(self, drop_id: str) -> str:
        """Canonical S3 key for a drop's source video file.

        Format: ``media/{survey_id}/{drop_id}/{drop_id}.mp4``

        Single source of truth for this convention — previously inlined in
        `orchestrator/ingest.py`, `zooniverse/upload.py`, and the
        training-frames extractor. Use this method instead of constructing
        the string inline.
        """
        validated = self.validate_drop_id(drop_id)
        survey_id = self.get_survey_id_from_drop(validated)
        return f"media/{survey_id}/{validated}/{validated}.mp4"

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

    def get_zooniverse_raw_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_zooniverse_raw.csv"
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
    # Training-frame volume export (download_training_volume_labels). Kept a
    # SEPARATE artifact from the expert MaxN-review export above: it's a
    # training-label source, never a MaxN/ecology source, so it must not share
    # the expert filename (avoids clobbering the expert CSV and stops db_refresh
    # from mistaking a training download for completed expert review).
    biigle_training_raw_suffix = "_biigle_training_raw.csv"

    def get_biigle_expert_raw_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_expert_raw_suffix}"
        )

    def get_biigle_training_raw_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_training_raw_suffix}"
        )

    def get_biigle_expert_maxn_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_expert_maxn_suffix}"
        )

    def get_coco_annotations_path(self, drop_id: str) -> Path:
        """COCO JSON of YOLO detections for a drop's selected frames.

        Single source of truth for this filename — written by the frame
        extractors, rebuilt by the Zooniverse-path inference rerun, and
        read by ``upload_frames_to_biigle`` before upload.
        """
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_coco_annotations_for_biigle.json"
        )

    _ZOONIVERSE_FRAMES_RAW_SUFFIXES = {
        "species": "_zooniverse_frames_species_raw.csv",
        "binary": "_zooniverse_frames_binary_raw.csv",
        "merged": "_zooniverse_frames_raw.csv",
    }

    def get_zooniverse_frames_raw_csv_path(self, drop_id: str, kind: str) -> Path:
        """Raw inference CSV for the Zooniverse-frame rerun ensemble.

        ``kind`` is ``"species"``, ``"binary"``, or ``"merged"`` — written
        by the two-pass species+binary inference + IoU merge that runs
        on Zooniverse-selected frames before BIIGLE upload.
        """
        suffix = self._ZOONIVERSE_FRAMES_RAW_SUFFIXES.get(kind)
        if suffix is None:
            raise ValueError(
                f"kind must be one of {sorted(self._ZOONIVERSE_FRAMES_RAW_SUFFIXES)}, "
                f"got {kind!r}"
            )
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{suffix}"
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
