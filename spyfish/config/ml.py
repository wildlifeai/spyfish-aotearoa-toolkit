from pathlib import Path
from typing import List, Optional

from spyfish.config.base import BaseConfig, get_required


class MLConfig(BaseConfig):
    """ML inference settings and training hyperparameters."""

    # ── Inference ────────────────────────────────────────────────────────

    @property
    def ml_inference(self) -> dict:
        return get_required(self._yaml_config, "ml_inference", "")

    @property
    def limit_processing(self):
        return get_required(self.ml_inference, "limit_processing", "ml_inference")

    @property
    def ml_fps(self):
        return float(get_required(self.ml_inference, "ml_fps", "ml_inference"))

    @property
    def log_interval_frames(self) -> int:
        return int(
            get_required(self.ml_inference, "log_interval_frames", "ml_inference")
        )

    @property
    def imgsz(self):
        return get_required(self.ml_inference, "imgsz", "ml_inference")

    @property
    def confidence_threshold(self):
        value = float(
            get_required(self.ml_inference, "confidence_threshold", "ml_inference")
        )
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"ml_inference.confidence_threshold must be in (0, 1]; got {value}. "
                "A value of 0 disables YOLO's confidence filter, so inference returns "
                "its full max_det boxes per frame (e.g. 300), flooding the pipeline "
                "with near-zero-confidence detections. Set a real threshold (e.g. 0.15)."
            )
        return value

    @property
    def ml_max_boxes_per_frame(self) -> int:
        """Per-frame detection ceiling above which a drop's inference is treated as
        degenerate (model saturating max_det, or confidence_threshold set to ~0)."""
        return int(
            get_required(self.ml_inference, "max_boxes_per_frame", "ml_inference")
        )

    @property
    def ml_batch_size(self) -> int:
        """Frames per YOLO predict() call during video inference (GPU batching)."""
        return int(get_required(self.ml_inference, "batch_size", "ml_inference"))

    @property
    def ml_nms_iou(self) -> float:
        """IoU threshold passed to YOLO's NMS at inference (``model.predict(iou=...)``).
        Boxes overlapping more than this collapse to the highest-conf one."""
        return float(get_required(self.ml_inference, "nms_iou", "ml_inference"))

    @property
    def ml_nms_agnostic(self) -> bool:
        """Passed to YOLO's NMS at inference (``model.predict(agnostic_nms=...)``).
        When True, overlapping boxes merge across classes (a species box and a
        generic 'fish' box on one animal); False = same-class only.
        """
        return bool(get_required(self.ml_inference, "nms_agnostic", "ml_inference"))

    @property
    def maxn_confidence_threshold(self):
        return get_required(
            self.ml_inference, "maxn_confidence_threshold", "ml_inference"
        )

    @property
    def maxn_persistence_seconds(self) -> float:
        """Rolling-min window (seconds) a count must hold to set MaxN; 0 disables
        (single-frame MaxN, the pre-2026-08 behaviour). Converted to sampled
        frames at runtime from the raw CSV's own frame spacing."""
        return float(
            get_required(self.ml_inference, "maxn_persistence_seconds", "ml_inference")
        )

    @property
    def maxn_gap_fill_seconds(self) -> float:
        """Zero-gaps up to this long between detections take min(neighbours)
        before the persistence window runs, so detector flicker on a
        continuously present animal isn't taxed as two short visits."""
        return float(
            get_required(self.ml_inference, "maxn_gap_fill_seconds", "ml_inference")
        )

    @property
    def maxn_exclude_classes(self) -> List[str]:
        """Classes whose detections never count toward MaxN (e.g. 'bait').
        They keep their raw-CSV detections for frame selection and BIIGLE."""
        return list(
            get_required(self.ml_inference, "maxn_exclude_classes", "ml_inference")
            or []
        )

    @property
    def interval_seconds(self):
        return get_required(
            get_required(self.ml_inference, "extraction", "ml_inference"),
            "interval_seconds",
            "ml_inference.extraction",
        )

    # ── Training ─────────────────────────────────────────────────────────

    @property
    def training_config(self) -> dict:
        return get_required(self._yaml_config, "training", "")

    @property
    def retrain_min_improvement_pct(self) -> float:
        """Minimum mAP@0.5 gain (in percentage points) a retrained model must
        show over production to be promoted. Shared by the evaluate step and
        the Model Metrics page so the two cannot disagree on the threshold."""
        return float(
            get_required(
                self.training_config, "retrain_min_improvement_pct", "training"
            )
        )

    @property
    def image_extensions(self) -> tuple:
        """Canonical image suffixes the pipeline accepts (e.g. ('.jpg', '.jpeg', '.png'))."""
        return tuple(get_required(self.training_config, "image_extensions", "training"))

    @property
    def training_epochs(self) -> int:
        return int(get_required(self.training_config, "epochs", "training"))

    @property
    def training_patience(self) -> int:
        return int(get_required(self.training_config, "patience", "training"))

    @property
    def training_imgsz(self) -> int:
        return int(get_required(self.training_config, "imgsz", "training"))

    @property
    def training_batch(self) -> int:
        return int(get_required(self.training_config, "batch", "training"))

    @property
    def training_optimizer(self) -> str:
        return get_required(self.training_config, "optimizer", "training")

    @property
    def training_lr0(self) -> float:
        return float(get_required(self.training_config, "lr0", "training"))

    @property
    def training_dropout(self) -> float:
        return float(get_required(self.training_config, "dropout", "training"))

    @property
    def training_floor_min_images(self) -> int:
        """Image-count floor, species appearing in fewer distinct frames get merged into 'fish'."""
        return int(
            get_required(self.training_config, "class_floor_min_images", "training")
        )

    @property
    def training_class_order(self) -> tuple[str, ...]:
        """Frozen YOLO class ordering: index in this tuple IS the class id.

        Append-only by contract, see the comment on `training.class_order` in
        config.yaml. Returned as a tuple so it can key the registry's cache and
        cannot be mutated by a caller. `fish` is required because the class
        floor needs somewhere to redirect merged species to.
        """
        order = get_required(self.training_config, "class_order", "training")
        names = tuple(str(n).strip() for n in order)
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"training.class_order contains duplicate entries {dupes}; each "
                "name must appear exactly once, its position is its class id."
            )
        if "fish" not in names:
            raise ValueError(
                "training.class_order must contain 'fish': the class floor "
                "merges under-represented species into it."
            )
        return names

    @property
    def training_split_seed(self) -> Optional[int]:
        """Random seed for reproducible train/val/test splits and per-drop frame filtering.

        Set to an integer (default 42) for deterministic, reproducible results
        across retrain runs, same drops + same labels always produce the same
        splits and the same frame selections.

        Set to `null` in config.yaml for fresh randomness each run (Python uses
        system entropy when seed is None). Useful for ablation experiments where
        you want to measure result variance, or for exploring multiple
        independent splits of the same dataset.
        """
        val = self.training_config.get("split_seed", 42)
        return int(val) if val is not None else None

    @property
    def training_cap_frames_per_drop(self) -> int:
        """Max frames per drop in the assembled YOLO dataset.

        Applied per-drop in `assemble_yolo_dataset`. When a drop has more than
        this many annotated frames, dominant-species-only frames are dropped
        first (see `training_dominant_species`). Default 60 fits ~20 drops at
        small scale; raise as the dataset grows.

        **Extras (drops under `extra_no_survey_id/`) bypass this cap entirely**,
        they're externally curated bulk imports (BIIGLE volume uploads) where
        every annotated frame is high-signal training data; capping them throws
        away expensive expert annotation work.
        """
        return int(
            get_required(self.training_config, "cap_frames_per_drop", "training")
        )

    @property
    def training_dominant_species(self) -> List[str]:
        """Species whose frames are deprioritized when over the per-drop cap.

        A frame whose only labels are in this list is treated as 'dominant-only'
        and dropped first when the drop exceeds `cap_frames_per_drop`. Frames
        containing at least one species *not* in this list are kept by default.
        Empty list = no deprioritization (cap-only behavior).
        """
        return list(self.training_config.get("dominant_species", []) or [])

    @property
    def training_background_ratio(self) -> float:
        """Target share of background (empty-label) frames in the TRAIN split.

        A background frame is an image paired with an empty .txt (no objects);
        it teaches the detector to suppress false positives. Ultralytics
        recommends ~0–10% of the training set (COCO ≈ 1%). Assembly pools every
        background frame across train drops and subsamples them to hit this
        ratio, `B = r/(1-r) * P` backgrounds for `P` positive train frames.
        0 disables (no backgrounds). The main supplier is the training-frame
        volumes downloaded via `download_training_volume_labels`, where empty
        frames are real no-fish reviews; without this they'd be dropped at
        assembly (`assemble_yolo_dataset` discards empty .txt by default).
        """
        return float(get_required(self.training_config, "background_ratio", "training"))

    @property
    def training_train_pct(self) -> float:
        return float(get_required(self.training_config, "train_pct", "training"))

    @property
    def training_val_pct(self) -> float:
        return float(get_required(self.training_config, "val_pct", "training"))

    @property
    def training_test_pct(self) -> float:
        return float(get_required(self.training_config, "test_pct", "training"))

    @property
    def training_auto_balance_val(self) -> bool:
        """When True, val is chosen by the species-balanced selector instead of
        per-survey donation."""
        return bool(get_required(self.training_config, "auto_balance_val", "training"))

    @property
    def training_val_balance_pct(self) -> float:
        """Target val fraction per multi-source species for the balanced selector."""
        return float(get_required(self.training_config, "val_balance_pct", "training"))

    @property
    def training_val_balance_tolerance(self) -> float:
        """Greedy stops when the worst-covered species is within this of target."""
        return float(
            get_required(self.training_config, "val_balance_tolerance", "training")
        )

    @property
    def training_val_balance_overshoot_weight(self) -> float:
        """Penalty per box a candidate val drop carries beyond any species'
        deficit; makes the selector prefer the smallest drop that covers a need."""
        return float(
            get_required(
                self.training_config, "val_balance_overshoot_weight", "training"
            )
        )

    @property
    def local_training_dir(self) -> Path:
        return self.project_root / get_required(
            self.training_config, "local_training_dir", "training"
        )

    @staticmethod
    def _parse_drop_ids_from_file(path: Path) -> set:
        """One DropID per line; '#' starts a comment. Empty set if file missing."""
        if not path.exists():
            return set()
        return {
            id_part
            for line in path.read_text().splitlines()
            if (id_part := line.split("#", 1)[0].strip())
        }

    @property
    def training_excluded_drops_file(self) -> Path:
        return self.project_root / get_required(
            self.training_config, "excluded_drops_file", "training"
        )

    @property
    def training_excluded_drops(self) -> set:
        """DropIDs to exclude from training entirely."""
        return self._parse_drop_ids_from_file(self.training_excluded_drops_file)

    @property
    def training_excluded_species(self) -> set:
        """Canonical species names removed from training entirely: their boxes
        are dropped at label conversion (never merged into 'fish'). The
        annotations DB and BIIGLE are unaffected."""
        return set(
            get_required(self.training_config, "excluded_species", "training") or []
        )

    @property
    def training_val_balance_max_share(self) -> float:
        """Hard cap on any multi-source species' share of boxes in val; candidate
        drops that would breach it are refused, so concentrated species resolve
        train-heavy instead of val-heavy. 1.0 disables."""
        return float(
            get_required(self.training_config, "val_balance_max_share", "training")
        )

    @property
    def training_cap_exempt_rare_below_frames(self) -> int:
        """Frames holding a species this rare (corpus-wide frame count) skip the
        per-drop cap. 0 disables."""
        return int(
            get_required(
                self.training_config, "cap_exempt_rare_below_frames", "training"
            )
        )

    @property
    def training_val_min_boxes_per_species(self) -> int:
        """Absolute floor on a species' val boxes, on top of val_balance_pct.

        A percentage target alone is useless for rare classes (8% of 71 boxes is
        6). Clamped by val_balance_max_share at the call site so an unreachable
        floor cannot drag extra drops into val.
        """
        return int(
            get_required(self.training_config, "val_min_boxes_per_species", "training")
        )

    @property
    def training_species_canonicalization(self) -> dict:
        """Synonym / hierarchy-split species-name merges, applied before any
        counting, balancing, flooring, or class-list building. Maps a label's
        scientific name to the canonical class name (e.g. 'Chelidonichthys
        kumu' -> 'Triglidae')."""
        mapping = dict(
            get_required(self.training_config, "species_canonicalization", "training")
            or {}
        )
        chained = set(mapping) & set(mapping.values())
        if chained:
            raise ValueError(
                "training.species_canonicalization has chained entries "
                f"(a value is also a key): {sorted(chained)}. "
                "Map every synonym directly to its final canonical name."
            )
        return mapping

    @property
    def training_force_train_biigle_volumes(self) -> set:
        """BIIGLE volume ids whose drops are always placed in train (old-label
        training volumes whose annotations shouldn't be evaluated against).
        Resolved to drop ids via the volume_id column in each drop's
        _biigle_training_raw.csv."""
        return {
            int(v)
            for v in get_required(
                self.training_config, "force_train_biigle_volumes", "training"
            )
        }

    @property
    def training_results_dir(self) -> Path:
        return self.local_training_dir / "results"

    @property
    def class_map_path(self) -> Path:
        return self.local_training_dir / "class_map.json"

    @property
    def training_results_s3_prefix(self) -> str:
        return self.training_results_dir.relative_to(self.project_root).as_posix()

    # ── Training-frame extraction (bootstrap dataset) ────────────────────
    # These settings drive `spyfish.ml.training.extract_training_frames`,
    # the standalone tool that pulls N frames per drop directly from S3
    # (via cv2 + presigned URL) for upload to Biigle as a training-data
    # annotation campaign.

    @property
    def _training_extraction(self) -> dict:
        return get_required(self._yaml_config, "training_extraction", "")

    @property
    def training_extraction_n_frames(self) -> int:
        """How many frames to extract per drop (default 10)."""
        return int(
            get_required(self._training_extraction, "n_frames", "training_extraction")
        )
