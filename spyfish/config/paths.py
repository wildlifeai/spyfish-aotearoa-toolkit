import logging
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
    def legacy_zooniverse_s3_prefix(self) -> str:
        """The legacy dir's repo-relative path, reused as its S3 prefix.

        Same convention as `training_results_s3_prefix`: local layout and S3
        layout mirror each other, so one value cannot drift from the other.
        """
        return self.legacy_zooniverse_dir.relative_to(self.project_root).as_posix()

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
        """Return the single .pt file in `directory`, or None if there is none.

        Production workflows keep exactly one promoted model per directory
        (pipeline_model/ or base_model/), so the usual case is unambiguous.

        Sorted, not `glob()` order: `Path.glob` yields in filesystem order,
        which differs between machines, so a directory holding two weights
        files could train against one model locally and a different one on
        NeSI, with nothing in the logs to say so. A second file is a mistake
        rather than a choice, so it is logged.
        """
        if not directory.exists():
            return None
        pt_files = sorted(directory.glob("*.pt"))
        if not pt_files:
            return None
        if len(pt_files) > 1:
            logging.warning(
                "%s holds %d .pt files (%s); using %s. Keep one promoted model "
                "per directory.",
                directory,
                len(pt_files),
                ", ".join(p.name for p in pt_files),
                pt_files[0].name,
            )
        return pt_files[0]

    @property
    def pipeline_model_path(self) -> Path:
        """Find the local pipeline model weights (.pt) from pipeline_model_dir.

        Prefers `species_*.pt` (most recently modified when several match, the
        promotion naming convention); falls back to the first .pt file in the
        directory for single-model setups that pre-date the naming convention.
        The binary model variant was retired 2026-08-21 — the species model IS
        the binary model with more informative labels (see design_doc.md,
        "ML model strategy").
        """
        if self.pipeline_model_dir.exists():
            candidates = sorted(
                self.pipeline_model_dir.glob("species_*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0]
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
        SiteID is always parts[3:5], positional, not regex, because the
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

         This is the URL Biigle's volume points at. It is the SURVEY directory,
         not a shared frames bucket under it, because Biigle resolves an image
         as ``volume.url + "/" + filename``, so putting the per-drop segment
         in the *filename* (``{drop}/frames/{drop}__frame_<secs>s.jpg``) lets
         one survey-pooled volume span per-deployment directories.

         That keeps a single layout everywhere: S3 mirrors local, and
         ``prepare_training_data``'s ``_IMAGE_SOURCE_DIRS`` walk (``frames/``
         and legacy ``training_frames/``) works against either.

         Volumes created before this convention point straight at
         ``{survey}/training_frames`` and hold bare basenames. They keep working
        . Biigle fixes a volume's url at creation and resolves images against
         it, but they cannot be converted in place, since that would mean
         rewriting every image's filename. Read both via
         ``biigle_to_yolo.drop_id_from_frame_filename``.
        """
        return f"{self.s3_deployment_data_dir}/{survey_id}/"

    def get_video_path(self, drop_id: str) -> Path:
        return self.media_dir / f"{self.validate_drop_id(drop_id)}.mp4"

    def get_video_s3_key(self, drop_id: str) -> str:
        """Canonical S3 key for a drop's source video file.

        Format: ``media/{survey_id}/{drop_id}/{drop_id}.mp4``

        Single source of truth for this convention, use this method instead
        of constructing the string inline.
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

    # Expert substrate percent-cover export (process_substrate). Separate from
    # the species MaxN export above: substrate is an area-cover statistic, not a
    # count, and never feeds YOLO/MaxN, so it must not share the MaxN filename.
    biigle_expert_substrate_suffix = "_biigle_expert_substrate.csv"

    def get_biigle_expert_substrate_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_expert_substrate_suffix}"
        )

    def get_coco_annotations_path(self, drop_id: str, target: str = "") -> Path:
        """COCO JSON of YOLO detections for a drop's selected frames.

        `target` scopes the file to the workflow that produced it, mirroring
        ``get_frames_dir(drop_id, target)``:

          - ``""``          → ``{drop}_coco_annotations_for_biigle.json``
            The expert-review path: written by ``extract_frames``, rebuilt by
            the Zooniverse-path inference rerun, read by
            ``upload_frames_to_biigle``. Those three intentionally share one
            file, and each rewrite targets the same per-drop review volume.
          - ``"training"``  → ``{drop}_training_coco_annotations_for_biigle.json``
            The survey-pooled training path (``extract_training_frames``).

        A COCO is only meaningful against the exact image set it was built
        for, and the two workflows select DIFFERENT frames from the same drop
        (review picks ML detection peaks at fractional timestamps; training
        picks blind evenly-spaced ones). Before this split both wrote the same
        filename, so whichever ran last silently destroyed the other's record
        of what the model had predicted, and an upload could push one
        workflow's boxes at the other's images, where every filename join
        misses and the annotations vanish without an error.
        """
        suffix = f"_{target}" if target else ""
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{suffix}_coco_annotations_for_biigle.json"
        )

    def get_zooniverse_frames_raw_csv_path(self, drop_id: str) -> Path:
        """Raw inference CSV for the Zooniverse-frame rerun.

        Written by the pipeline-model pass over Zooniverse-selected frames
        before BIIGLE upload. (Was a species+binary IoU-merged ensemble until
        the binary model's retirement, 2026-08-21; the filename is unchanged so
        historical CSVs stay discoverable.)
        """
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_zooniverse_frames_raw.csv"
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
