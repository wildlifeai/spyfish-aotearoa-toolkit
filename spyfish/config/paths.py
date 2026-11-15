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
    def legacy_help_dir(self) -> Path:
        """Knowledge files for the legacy workbook extraction (fix keys, maps).

        Gitignored and synced to S3 under `legacy_help_s3_prefix`; a fresh
        clone pulls them from the bucket rather than from the repo.
        """
        return self.project_root / get_required(
            self.legacy_paths, "annotations_help", "paths.legacy"
        )

    @property
    def legacy_drop_id_map_path(self) -> Path:
        """`legacy DropID: SharePoint DropID` YAML applied by --legacy-experts (see config.yaml)."""
        return self.legacy_help_dir / get_required(
            self.legacy_paths, "drop_id_map", "paths.legacy"
        )

    @property
    def species_synonyms_path(self) -> Path:
        """`written name: canonical name` YAML applied to every annotation on insert.

        One map for the legacy workbook extraction and the annotations DB gate,
        see the comment on `paths.species_synonyms` in config.yaml.
        """
        return self.project_root / get_required(self.paths, "species_synonyms", "paths")

    @property
    def legacy_help_s3_prefix(self) -> str:
        """Repo-relative path of `legacy_help_dir`, reused as its S3 prefix."""
        return self.legacy_help_dir.relative_to(self.project_root).as_posix()

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
    def exports_dir(self) -> Path:
        """Review exports (per-survey reports, slate review lists). Data, so under process_files/.

        The MaxN frames and slates themselves live in each deployment's folder
        (``get_review_dir``, ``get_slate_frame_path``), where the S3 sync finds them.
        """
        return self.project_root / self.base_dir / self._sub("exports")

    @property
    def review_reports_dir(self) -> Path:
        """``{survey}_review.html``, one self-contained page per survey."""
        return self.exports_dir / "reports"

    @property
    def trash_dir(self) -> Path:
        """Where code moves files instead of deleting them (``spyfish.utils.move_to_trash``).

        One dated folder per day; empty it by hand once you have looked. NeSI has
        no system trash, so a wrong delete there would be unrecoverable.
        """
        return self.exports_dir / "trash"

    @property
    def slates_verified_path(self) -> Path:
        """Drops whose slate a person has checked, one DropID per line."""
        return self.exports_dir / "slates_verified.txt"

    @property
    def slates_none_path(self) -> Path:
        """Drops a reviewer marked as never showing a board, one DropID per line."""
        return self.exports_dir / "slates_none.txt"

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
    def classifier_model_dir(self) -> Path:
        """Fine-tuned BioCLIP 2 checkpoints (second-stage species classifier)."""
        return self.models_root_dir / self._sub("classifier")

    @property
    def bioclip_cache_dir(self) -> Path:
        """open_clip hub cache for the frozen BioCLIP 2 backbone (~1.7 GB, once per machine)."""
        return self.models_root_dir / self._sub("bioclip")

    @property
    def two_stage_dir(self) -> Path:
        """Second-stage classifier training outputs: crops, embeddings, reports,
        fine-tune runs (``spyfish/ml/classifier``, ``scripts/two_stage``)."""
        return self.project_root / self.base_dir / self._sub("two_stage")

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
        """The pipeline model weights (.pt) in pipeline_model_dir.

        Only files named to the standard count (`validation_patterns.model_name`,
        `{type}_{YYYYMMDD}[_{HHMMSS}]`), and the newest TRAINING STAMP wins, not
        the newest mtime: a copy made today of an old model is still the old
        model. A directory with no canonical file raises and names what it
        found, because the stem becomes the model name on every ML annotation
        and a wrong name there is far harder to undo than a rename now. The
        binary model variant was retired 2026-08-21 — the species model IS the
        binary model with more informative labels (see design_doc.md, "ML
        model strategy").
        """
        return self._newest_stamped_model(
            self.pipeline_model_dir, "species_20260822_192536.pt"
        )

    def model_path_for_name(self, model_name: str) -> Path:
        """The weights file of a model named in a raw/MaxN CSV: the production
        copy, else the archived one. Raises when neither exists, because the
        sidecars beside the weights (MaxN cutoff) are what callers are after
        and guessing them is worse than stopping."""
        self.validate_model_name(model_name)
        for directory in (self.pipeline_model_dir, self.archived_models_dir):
            candidate = directory / f"{model_name}.pt"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No weights for model {model_name!r} in {self.pipeline_model_dir} or "
            f"{self.archived_models_dir}."
        )

    @property
    def pipeline_model_name(self) -> str:
        """The pipeline model's name: the stem of its weights file, which is
        what every ML artifact and annotation row is keyed by."""
        return self.pipeline_model_path.stem

    @property
    def classifier_model_path(self) -> Path:
        """The second-stage classifier checkpoint in classifier_model_dir, same
        naming rule and newest-stamp pick as the pipeline model."""
        return self._newest_stamped_model(
            self.classifier_model_dir, "bioclip_20261104_162700.pt"
        )

    def _newest_stamped_model(self, directory: Path, example: str) -> Path:
        found = sorted(directory.glob("*.pt")) if directory.exists() else []
        canonical = [
            (self.model_name_stamp(p.stem), p)
            for p in found
            if self.model_name_stamp(p.stem) is not None
        ]
        if not canonical:
            names = ", ".join(p.name for p in found) or "no .pt files"
            raise FileNotFoundError(
                f"No model named to the standard in {directory} "
                f"({names}). Weights must be named {{type}}_{{YYYYMMDD}}_{{HHMMSS}}.pt, "
                f"e.g. {example}; the stem is the model name "
                "stored on every ML annotation, so rename the file."
            )
        if len(canonical) > 1:
            logging.warning(
                f"{directory} holds {len(canonical)} model files; "
                f"using the newest stamp. Keep one promoted model here."
            )
        return max(canonical)[1]

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

    def get_reserve_code_from_drop(self, drop_id: str) -> Optional[str]:
        """The marine reserve code a DropID was made under (its first segment),
        or None when the name does not start with one.

        Deliberately looser than ``validate_drop_id``: legacy workbook drops
        carry site names instead of site codes
        (``CRP_20030423_BUV_Liitle-waterfall_01``) and fail validation, yet
        their reserve segment is well-formed. Never build a path from the result.
        """
        m = re.match(r"([A-Z]{3})_", drop_id or "")
        return m.group(1) if m else None

    def get_video_path(self, drop_id: str) -> Path:
        return self.media_dir / f"{self.validate_drop_id(drop_id)}.mp4"

    # The handwritten metadata whiteboard rangers hold up before the drop, one
    # JPEG per deployment, for cross-checking against SharePoint metadata.
    slate_frame_suffix = "_slate.jpg"

    def get_slate_frame_path(self, drop_id: str) -> Path:
        return self.get_drop_dir(drop_id) / (
            f"{self.validate_drop_id(drop_id)}{self.slate_frame_suffix}"
        )

    # MaxN frames for the Survey review (one per source and species, boxed),
    # with the manifest that says what each one is: the second, the model's
    # count on the frame, the agreement colour.
    review_dir_name = "review"
    review_frame_marker = "__maxn_"  # in every review frame's file name
    review_manifest_name = "maxn_frames.csv"

    def get_review_dir(self, drop_id: str) -> Path:
        return self.get_drop_dir(drop_id) / self.review_dir_name

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

    def get_clip_uploads_csv_path(self, drop_id: str) -> Path:
        """Every clip sent to Zooniverse for this drop and when, append-only.

        The selections CSV is rewritten by each clips run (a top-up replaces
        the ML selection with the full grid), so it says what the LAST run
        picked, not what volunteers were shown.
        """
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_zooniverse_clip_uploads.csv"
        )

    def get_biigle_selections_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_biigle_frames_selection.csv"
        )

    # Zooniverse validation (--zooniverse-frames / --zooniverse-crops). Clean
    # frames go in the drop's shared frames/ folder like every other frame; the
    # selection and manifests are CSVs in annotations/; only the upload copies
    # (boxes drawn, crops) get their own folder, and those can be regenerated.
    zooniverse_validation_selection_suffix = "_zooniverse_validation_selection.csv"
    VALIDATION_KINDS = ("frames", "crops")

    def get_validation_selection_path(self, drop_id: str) -> Path:
        """Which frames are validation frames, and why. Shared by both stages."""
        return self.get_drop_annotations_dir(drop_id) / (
            f"{self.validate_drop_id(drop_id)}"
            f"{self.zooniverse_validation_selection_suffix}"
        )

    def get_validation_upload_dir(self, drop_id: str) -> Path:
        """Upload copies: frames with boxes drawn and crops, ``{drop}/zooniverse_validation/``."""
        return self.get_drop_dir(drop_id) / "zooniverse_validation"

    def get_validation_manifest_path(self, drop_id: str, kind: str) -> Path:
        """One row per upload image (or not-cropped box) of `kind` ("frames" or "crops")."""
        if kind not in self.VALIDATION_KINDS:
            raise ValueError(
                f"kind must be one of {self.VALIDATION_KINDS}, got {kind!r}"
            )
        return self.get_drop_annotations_dir(drop_id) / (
            f"{self.validate_drop_id(drop_id)}_zooniverse_{kind}_manifest.csv"
        )

    # ── validation sync-back (--zooniverse-validation-sync) ──────────────────

    def _validation_kind_file(self, drop_id: str, kind: str, suffix: str) -> Path:
        if kind not in self.VALIDATION_KINDS:
            raise ValueError(
                f"kind must be one of {self.VALIDATION_KINDS}, got {kind!r}"
            )
        return self.get_drop_annotations_dir(drop_id) / (
            f"{self.validate_drop_id(drop_id)}_zooniverse_{kind}_{suffix}"
        )

    def get_validation_classifications_path(self, drop_id: str, kind: str) -> Path:
        """Cached Panoptes classifications for the drop's ``kind`` set (JSON)."""
        return self._validation_kind_file(drop_id, kind, "classifications.json")

    def get_validation_verdicts_path(self, drop_id: str, kind: str) -> Path:
        """One row per crop (``crops``) or per frame (``frames``) with its verdict."""
        return self._validation_kind_file(drop_id, kind, "verdicts.csv")

    def get_validation_missed_boxes_path(self, drop_id: str) -> Path:
        """Clusters of volunteer-drawn boxes with no ML box, in video pixels."""
        return self.get_drop_annotations_dir(drop_id) / (
            f"{self.validate_drop_id(drop_id)}_zooniverse_missed_boxes.csv"
        )

    @property
    def zooniverse_workflow_versions_dir(self) -> Path:
        """Cached workflow versions (one JSON per workflow) for the validation sync."""
        return self.project_root / get_required(
            self.paths, "zooniverse_workflow_versions", "paths"
        )

    @property
    def zooniverse_classifications_dir(self) -> Path:
        """Per-workflow classification cache and cursors (``panoptes_fetch``)."""
        return self.project_root / get_required(
            self.paths, "zooniverse_classifications", "paths"
        )

    @property
    def site_prior_dir(self) -> Path:
        """Species-presence prior per reserve (``scripts/build_site_prior.py``).

        Sensitive: species presence per reserve, gitignored, never public.
        """
        return self.project_root / get_required(self.paths, "site_prior", "paths")

    @property
    def site_prior_csv_path(self) -> Path:
        """The prior itself, one row per (reserve, species) with ``presence_prob``."""
        return self.site_prior_dir / "species_presence_by_reserve.csv"

    biigle_expert_raw_suffix = "_biigle_expert_raw.csv"
    # Every frame of the drop the expert was SHOWN, annotated or not. A Biigle
    # annotation report lists only frames that HAVE annotations, so on its own it
    # cannot distinguish "expert looked and saw nothing" from "never uploaded".
    # Recording the reviewed-frame list at sync time is what lets biigle_to_yolo write an
    # empty .txt (a YOLO background/negative) for the reviewed-but-empty frames.
    biigle_expert_reviewed_frames_suffix = "_biigle_expert_reviewed_frames.csv"
    biigle_expert_maxn_suffix = "_biigle_expert_maxn.csv"
    # Training-frame volume export (download_training_volume_labels). Kept a
    # SEPARATE artifact from the expert MaxN-review export above: it's a
    # training-label source, never a MaxN/ecology source, so it must not share
    # the expert filename (avoids clobbering the expert CSV and stops db_refresh
    # from mistaking a training download for completed expert review).
    biigle_training_raw_suffix = "_biigle_training_raw.csv"
    # The training volume's counterpart of the expert reviewed-frames list:
    # every frame of the drop in the training volume, annotated or not.
    # biigle_to_yolo rebuilds labels/ from scratch, so the training volume's
    # negatives survive a rebuild only if this list is on disk.
    biigle_training_reviewed_frames_suffix = "_biigle_training_reviewed_frames.csv"

    def get_biigle_expert_raw_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_expert_raw_suffix}"
        )

    def get_biigle_expert_reviewed_frames_csv_path(self, drop_id: str) -> Path:
        """Frames of this drop present in its Biigle volume, annotated or not."""
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_expert_reviewed_frames_suffix}"
        )

    def get_biigle_training_raw_csv_path(self, drop_id: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_training_raw_suffix}"
        )

    def get_biigle_training_reviewed_frames_csv_path(self, drop_id: str) -> Path:
        """Frames of this drop present in its training volume, annotated or not."""
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{self.biigle_training_reviewed_frames_suffix}"
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

    biigle_upload_record_glob = "_biigle_upload_*.json"

    def get_biigle_upload_record_path(
        self, drop_id: str, volume_id: int, stamp: str, target: str = ""
    ) -> Path:
        """Dated, write-once record of one BIIGLE annotation upload for a drop.

        ``{drop}{_target}_biigle_upload_{volume_id}_{YYYYMMDD_HHMMSS}.json``. The
        working COCO keeps its fixed name because extraction and upload hand it
        to each other; this copy is taken at the moment of upload and never
        rewritten, so the S3 sync (additive, but same-name files are replaced)
        can only ever add records, never lose the one the expert reviewed.
        """
        suffix = f"_{target}" if target else ""
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}{suffix}_biigle_upload_{int(volume_id)}_{stamp}.json"
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

    ml_raw_suffix = "_raw.csv"

    def get_raw_csv_path(self, drop_id: str, model_name: str) -> Path:
        return (
            self.get_drop_annotations_dir(drop_id)
            / f"{self.validate_drop_id(drop_id)}_{model_name}{self.ml_raw_suffix}"
        )

    def parse_ml_raw_csv_name(self, filename: "str | Path") -> "tuple[str, str] | None":
        """``(drop_id, model_name)`` from a full-video ML raw CSV's name (or S3
        key), None for anything else.

        The inverse of ``get_raw_csv_path``, and the one place that reads a
        raw CSV name back. Other artifacts share the ``_raw.csv`` suffix
        (``_zooniverse_raw``, ``_zooniverse_frames_raw``, ``_biigle_expert_raw``,
        ``_biigle_training_raw``, an experiment's ``_conf050_raw``); none of
        them is a canonical model name, so all return None.
        """
        name = Path(filename).name
        if not name.endswith(self.ml_raw_suffix):
            return None
        drop_pattern = self.get_validation_pattern("drop_id").lstrip("^").rstrip("$")
        m = re.match(
            f"(?P<drop>{drop_pattern})_(?P<model>.+)$",
            name[: -len(self.ml_raw_suffix)],
        )
        if not m or self.model_name_stamp(m.group("model")) is None:
            return None
        return m.group("drop"), m.group("model")

    def list_ml_raw_csvs(self, drop_id: str) -> "dict[str, Path]":
        """``{model_name: path}`` for every model's raw CSV in the drop's
        annotations dir. Which one to use is the caller's rule (written last,
        newest training stamp); this only says which exist."""
        drop_id = self.validate_drop_id(drop_id)
        found = {}
        for p in sorted(
            self.get_drop_annotations_dir(drop_id).glob(
                f"{drop_id}_*{self.ml_raw_suffix}"
            )
        ):
            parsed = self.parse_ml_raw_csv_name(p.name)
            if parsed and parsed[0] == drop_id:
                found[parsed[1]] = p
        return found

    def resolve_ml_raw_csv(self, drop_id: str) -> "tuple[str, Path] | None":
        """``(model_name, raw CSV)`` a downstream stage should read for this drop.

        The pipeline model's raw CSV when it exists, else the newest model (by
        training stamp) that ran on the drop, else None. Downstream stages
        (clips, BIIGLE upload, Zooniverse validation) used to look only for the
        pipeline model's file, so promoting or renaming a model left every
        ml_complete drop "not ready" forever with its detections on disk.
        """
        found = self.list_ml_raw_csvs(drop_id)
        if not found:
            return None
        pipeline = self.pipeline_model_name
        if pipeline in found:
            return pipeline, found[pipeline]
        newest = max(found, key=lambda m: (self.model_name_stamp(m) or "", m))
        return newest, found[newest]

    def get_clips_dir(self, drop_id: str, target: str = "") -> Path:
        sub_path = f"{target}_clips" if target else "clips"
        return self.get_drop_dir(drop_id) / sub_path

    def get_frames_dir(self, drop_id: str, target: str = "") -> Path:
        sub_path = f"{target}_frames" if target else "frames"
        return self.get_drop_dir(drop_id) / sub_path
