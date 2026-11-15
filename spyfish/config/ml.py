from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from spyfish.config.base import BaseConfig, get_required

CLASSIFIER_SOURCES = ("biigle_csv", "yolo_dataset", "uoa")


@dataclass(frozen=True)
class Augmentation:
    """Crop augmentation for the BioCLIP fine-tune (training.classifier.finetune.augmentation)."""

    rotation_degrees: float
    horizontal_flip: bool
    color_jitter: tuple[float, float, float, float]
    blur_probability: float
    blur_kernel: int
    blur_sigma: tuple[float, float]


@dataclass(frozen=True)
class FinetuneRecipe:
    """training.classifier.finetune: the defaults finetune_bioclip.py trains with."""

    fold: int
    epochs: int
    batch: int
    unfreeze_blocks: int
    lr_encoder: float
    lr_head: float
    weight_decay: float
    label_smoothing: float
    workers: int
    augmentation: Augmentation


@dataclass(frozen=True)
class ProbeRecipe:
    """training.classifier.probe: the logistic probe on frozen embeddings."""

    max_iter: int
    c: float


@dataclass(frozen=True)
class HardNegatives:
    """training.classifier.hard_negatives."""

    iou_negative: float
    max_per_drop: int
    n_clusters: int


@dataclass(frozen=True)
class StaticBoxes:
    """training.classifier.static_boxes: boxes that never move are not fish."""

    link_distance_box: float
    link_distance_px: float
    link_size_ratio: float
    max_gap_seconds: float
    min_span_seconds: float
    min_boxes: int
    max_spread_px: float
    max_size_spread: float
    crops_per_track: int
    crop_min_iou: float
    not_a_fish_flag: bool
    label: str
    flag_below: float
    score_conf_mean: float
    score_conf_std: float
    score_rate_mean: float
    score_rate_std: float


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
        """Inference image size: the string ``model`` (use the checkpoint's own
        ``train_args.imgsz``, resolved by ``spyfish.ml.model_meta.resolve_imgsz``)
        or a positive integer that overrides it. Never a silent default: a model
        run at a size it was not trained at loses accuracy with no error."""
        value = get_required(self.ml_inference, "imgsz", "ml_inference")
        if isinstance(value, str):
            if value.strip().lower() == "model":
                return "model"
            raise ValueError(
                f"ml_inference.imgsz must be 'model' or a positive integer; got {value!r}."
            )
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(
                f"ml_inference.imgsz must be 'model' or a positive integer; got {value!r}."
            )
        return int(value)

    @property
    def ml_batch_size_imgsz(self) -> int:
        """The image size ``ml_inference.batch_size`` was sized for. Inference at a
        larger size scales the batch down by the squared ratio (model_meta)."""
        return int(get_required(self.ml_inference, "batch_size_imgsz", "ml_inference"))

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
    def ml_video_reader(self) -> str:
        """How inference decodes video: ``cv2`` or ``ffmpeg`` (video stream only)."""
        value = str(get_required(self.ml_inference, "video_reader", "ml_inference"))
        if value not in ("cv2", "ffmpeg"):
            raise ValueError(
                f"ml_inference.video_reader must be cv2 or ffmpeg; got {value!r}"
            )
        return value

    @property
    def ml_min_frame_coverage(self) -> float:
        """Share of the sampled frames a video must yield before its raw CSV counts
        as finished; less means the capture stopped early (stalled read, decode error).
        """
        return float(
            get_required(self.ml_inference, "min_frame_coverage", "ml_inference")
        )

    @property
    def ml_batch_size(self) -> int:
        """Frames per YOLO predict() call during video inference (GPU batching)."""
        return int(get_required(self.ml_inference, "batch_size", "ml_inference"))

    @property
    def ml_half(self) -> bool:
        """FP16 inference (``model.predict(half=...)``). Ignored by Ultralytics on CPU."""
        return bool(get_required(self.ml_inference, "half", "ml_inference"))

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

    # ── Second stage: species classifier on the YOLO crops ───────────────

    @property
    def _classifier(self) -> dict:
        return get_required(self.ml_inference, "classifier", "ml_inference")

    @property
    def classifier_enabled(self) -> bool:
        """Run BioCLIP 2 on every YOLO box and write its name to the raw CSV."""
        return bool(
            get_required(self._classifier, "enabled", "ml_inference.classifier")
        )

    @property
    def classifier_backbone(self) -> str:
        """open_clip model name of the frozen backbone (``hf-hub:imageomics/bioclip-2``)."""
        return str(
            get_required(self._classifier, "backbone", "ml_inference.classifier")
        )

    @property
    def classifier_crop_margin(self) -> float:
        """Share of the box size added on each side before cropping."""
        return float(
            get_required(self._classifier, "crop_margin", "ml_inference.classifier")
        )

    @property
    def classifier_min_box_px(self) -> int:
        """Boxes under this on either side keep the YOLO name: nothing to classify."""
        return int(
            get_required(self._classifier, "min_box_px", "ml_inference.classifier")
        )

    @property
    def classifier_batch_size(self) -> int:
        return int(
            get_required(self._classifier, "batch_size", "ml_inference.classifier")
        )

    @property
    def classifier_preprocess_workers(self) -> int:
        """Threads that resize the crops of one classifier batch. 0 or 1 = on
        the inference thread, as before 2026-11-11."""
        value = int(
            get_required(
                self._classifier, "preprocess_workers", "ml_inference.classifier"
            )
        )
        if value < 0:
            raise ValueError(
                f"ml_inference.classifier.preprocess_workers must be >= 0, got {value}"
            )
        return value

    @property
    def classifier_p_min(self) -> float:
        """Top probability below which the box is named ``fallback_label``."""
        value = float(
            get_required(self._classifier, "p_min", "ml_inference.classifier")
        )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"ml_inference.classifier.p_min must be in [0, 1], got {value}"
            )
        return value

    @property
    def classifier_fallback_label(self) -> str:
        return str(
            get_required(self._classifier, "fallback_label", "ml_inference.classifier")
        )

    @property
    def classifier_fallback_to_yolo(self) -> bool:
        """Below ``p_min``, write YOLO's species when its detection confidence
        clears ``classifier_fallback_yolo_min_confidence``; else ``fallback_label``."""
        return bool(
            get_required(
                self._classifier, "fallback_to_yolo", "ml_inference.classifier"
            )
        )

    @property
    def classifier_fallback_yolo_min_confidence(self) -> float:
        value = float(
            get_required(
                self._classifier,
                "fallback_yolo_min_confidence",
                "ml_inference.classifier",
            )
        )
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "ml_inference.classifier.fallback_yolo_min_confidence must be in "
                f"[0, 1], got {value}"
            )
        return value

    @property
    def classifier_label_map(self) -> dict:
        """Classifier class → pipeline name, applied before the registry canonicalises."""
        value = get_required(self._classifier, "label_map", "ml_inference.classifier")
        return dict(value or {})

    @property
    def classifier_reject_label(self) -> str:
        """The classifier's `not a fish` class: kept in the raw CSV, dropped by
        every downstream reader (``spyfish/ml/raw_detections.py``). The same name
        the fine-tune trains it under, so there is one key, not two to keep equal."""
        return self.classifier_training_not_fish_class

    @property
    def maxn_confidence_threshold(self):
        """MaxN confidence cutoff: the string ``model`` (use the cutoff picked for
        the checkpoint on val, stored in its ``.thresholds.json`` sidecar and
        resolved by ``spyfish.ml.model_meta.resolve_maxn_threshold``) or a number
        in (0, 1] that overrides it. Never a silent default."""
        value = get_required(
            self.ml_inference, "maxn_confidence_threshold", "ml_inference"
        )
        if isinstance(value, str):
            if value.strip().lower() == "model":
                return "model"
            raise ValueError(
                "ml_inference.maxn_confidence_threshold must be 'model' or a number "
                f"in (0, 1]; got {value!r}."
            )
        if isinstance(value, bool) or not 0.0 < float(value) <= 1.0:
            raise ValueError(
                "ml_inference.maxn_confidence_threshold must be 'model' or a number "
                f"in (0, 1]; got {value!r}."
            )
        return float(value)

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
    def legacy_review(self) -> dict:
        return get_required(self.ml_inference, "legacy_review", "ml_inference")

    @property
    def legacy_review_random_intervals(self) -> int:
        """Non-peak scored intervals per species to pull alongside the peak."""
        return int(
            get_required(
                self.legacy_review, "random_intervals", "ml_inference.legacy_review"
            )
        )

    @property
    def legacy_review_max_time_delta_seconds(self) -> float:
        """How far a raw-CSV frame may sit from the legacy second and still be
        used (with its boxes) as that moment's frame."""
        return float(
            get_required(
                self.legacy_review,
                "max_time_delta_seconds",
                "ml_inference.legacy_review",
            )
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
    def training_augmentation(self) -> dict:
        """``training.augmentation``: YOLO augmentation arguments, passed verbatim to
        every trainer (key = Ultralytics argument name)."""
        value = get_required(self.training_config, "augmentation", "training")
        if not isinstance(value, dict):
            raise ValueError(
                "training.augmentation must be a mapping of YOLO arguments"
            )
        return dict(value)

    @property
    def training_stability(self) -> dict:
        """``training.stability``: AMP, warm-up, loss weight and caching arguments
        every trainer passes verbatim; the recipe that keeps runs comparable."""
        value = get_required(self.training_config, "stability", "training")
        if not isinstance(value, dict):
            raise ValueError("training.stability must be a mapping of YOLO arguments")
        return dict(value)

    @property
    def threshold_beta(self) -> float:
        """F-beta weight the MaxN cutoff is picked at (``spyfish.ml.training.threshold``):
        how many times more recall matters than precision. 1.0 is plain F1."""
        value = float(get_required(self.training_config, "threshold_beta", "training"))
        if value <= 0:
            raise ValueError(f"training.threshold_beta must be positive; got {value}.")
        return value

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

        **Extras (drops under `extra_no_survey_id/`)** are capped per SOURCE
        deployment when `training_extra_frame_deployments` maps their frames
        (volume 30505: 1,887 Zooniverse frames from 29 deployments, up to 328
        from one video). Unmapped extras still bypass the cap.
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
    def training_extra_frame_deployments_file(self) -> Path:
        return self.project_root / get_required(
            self.training_config, "extra_frame_deployments_file", "training"
        )

    @property
    def training_extra_frame_deployments(self) -> Dict[str, Dict[str, str]]:
        """``{extra drop: {frame stem: source deployment}}`` for curated volumes.

        Empty when the file does not exist: no volume is mapped, and every
        extra keeps the uncapped behaviour.
        """
        import csv

        path = self.training_extra_frame_deployments_file
        if not path.exists():
            return {}
        out: Dict[str, Dict[str, str]] = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                out.setdefault(row["drop"], {})[row["frame_stem"]] = row["deployment"]
        return out

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
    def training_test_drops_file(self) -> Path:
        return self.project_root / get_required(
            self.training_config, "test_drops_file", "training"
        )

    @property
    def training_test_drops(self) -> set:
        """Frozen test set: DropIDs that are always test, never train or val."""
        return self._parse_drop_ids_from_file(self.training_test_drops_file)

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
        """Same-animal merges: two labels that denote one fish in this data
        ('Chelidonichthys kumu' -> 'Triglidae'), applied before the rollup
        counts anything. Plain synonyms (one animal, two names) belong in
        `paths.species_synonyms`, which `canonicalize_species` applies first;
        which thin taxa pool into a group is the rollup's decision
        (`spyfish.ml.training.rollup`), not this map's."""
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
    def training_hierarchy_file(self) -> Path:
        """`hierarchy.json`, child -> parent over WoRMS ranks plus the pool
        nodes; the taxonomy the rollup walks."""
        return self.project_root / get_required(
            self.training_config, "hierarchy_file", "training"
        )

    @property
    def training_rollup_pool_nodes(self) -> tuple[str, ...]:
        """Taxa whose every descendant trains as that node (eels, sharks, rays)."""
        return tuple(
            get_required(self.training_config, "rollup_pool_nodes", "training") or ()
        )

    @property
    def training_rollup_keep_separate(self) -> tuple[str, ...]:
        """Species under a pool node that get their own class instead."""
        keep = tuple(
            get_required(self.training_config, "rollup_keep_separate", "training") or ()
        )
        clash = set(keep) & set(self.training_rollup_pool_nodes)
        if clash:
            raise ValueError(
                f"training.rollup_keep_separate and rollup_pool_nodes both list {sorted(clash)}"
            )
        return keep

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
    def training_dataset_dir(self) -> Path:
        """The assembled YOLO dataset (images/, labels/, data.yaml,
        rollup_map.json). A shared workspace: every data prep wipes and
        rebuilds it, so a run's own copy is the snapshot under
        `training_runs_dir`."""
        return self.local_training_dir / "species"

    @property
    def training_data_yaml(self) -> Path:
        """data.yaml of the assembled dataset, the file training and
        evaluation are pointed at."""
        return self.training_dataset_dir / "data.yaml"

    @property
    def training_runs_dir(self) -> Path:
        """One directory per training run, named by its model name
        (`species_{stamp}`): weights/, Ultralytics outputs and the frozen
        dataset/ snapshot."""
        return self.local_training_dir / "runs"

    @property
    def training_labels_staged_dir(self) -> Path:
        """Per-drop YOLO labels remapped to the unified class ids
        (`<drop_id>/*.txt`), written by `flatten_and_remap_labels` and wiped
        at the start of each data prep."""
        return self.local_training_dir / "labels_staged"

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

    # ── Second-stage classifier TRAINING (training.classifier) ───────────
    # Inference-side knobs (crop margin, min box, backbone) are the
    # `classifier_*` properties above; these drive building the crops, picking
    # the classes and the fine-tune (spyfish/ml/classifier/, scripts/two_stage/).

    @property
    def _classifier_training(self) -> dict:
        return get_required(self.training_config, "classifier", "training")

    def _ct(self, key: str):
        return get_required(self._classifier_training, key, "training.classifier")

    @property
    def classifier_training_source(self) -> str:
        """Where the crops come from: biigle_csv, yolo_dataset or uoa.
        ``TWO_STAGE_SOURCE`` in the environment overrides it (resolved in
        ``spyfish.ml.classifier.crops``)."""
        source = str(self._ct("source"))
        if source not in CLASSIFIER_SOURCES:
            raise ValueError(
                f"training.classifier.source={source!r}; expected one of {CLASSIFIER_SOURCES}"
            )
        return source

    @property
    def classifier_training_min_drops(self) -> int:
        return int(self._ct("min_drops"))

    @property
    def classifier_training_min_crops(self) -> int:
        return int(self._ct("min_crops"))

    @property
    def classifier_training_other_class(self) -> str:
        return str(self._ct("other_class"))

    @property
    def classifier_training_not_fish_class(self) -> str:
        return str(self._ct("not_fish_class"))

    @property
    def classifier_training_exclude_classes(self) -> tuple[str, ...]:
        """Labels that are not species (bait, the catch-all): never a crop."""
        return tuple(str(c) for c in self._ct("exclude_classes") or ())

    @property
    def classifier_training_max_crops_per_drop_species(self) -> Optional[int]:
        value = self._ct("max_crops_per_drop_species")
        return None if value is None else int(value)

    @property
    def classifier_training_seed(self) -> int:
        return int(self._ct("seed"))

    @property
    def classifier_training_folds(self) -> int:
        return int(self._ct("folds"))

    @property
    def classifier_training_iou_match(self) -> float:
        return float(self._ct("iou_match"))

    @property
    def classifier_training_prior_alpha(self) -> float:
        return float(self._ct("prior_alpha"))

    @property
    def classifier_finetune(self) -> "FinetuneRecipe":
        ft = self._ct("finetune")
        where = "training.classifier.finetune"
        aug = get_required(ft, "augmentation", where)
        aug_where = where + ".augmentation"
        jitter = tuple(float(v) for v in get_required(aug, "color_jitter", aug_where))
        if len(jitter) != 4:
            raise ValueError(
                f"{aug_where}.color_jitter needs 4 values (brightness, contrast, saturation, hue)"
            )
        sigma = tuple(float(v) for v in get_required(aug, "blur_sigma", aug_where))
        if len(sigma) != 2:
            raise ValueError(f"{aug_where}.blur_sigma needs 2 values (min, max)")
        workers = int(get_required(ft, "workers", where))
        if workers < 0:
            raise ValueError(f"{where}.workers must be >= 0, got {workers}")
        return FinetuneRecipe(
            fold=int(get_required(ft, "fold", where)),
            epochs=int(get_required(ft, "epochs", where)),
            batch=int(get_required(ft, "batch", where)),
            unfreeze_blocks=int(get_required(ft, "unfreeze_blocks", where)),
            lr_encoder=float(get_required(ft, "lr_encoder", where)),
            lr_head=float(get_required(ft, "lr_head", where)),
            weight_decay=float(get_required(ft, "weight_decay", where)),
            label_smoothing=float(get_required(ft, "label_smoothing", where)),
            workers=workers,
            augmentation=Augmentation(
                rotation_degrees=float(
                    get_required(aug, "rotation_degrees", aug_where)
                ),
                horizontal_flip=bool(get_required(aug, "horizontal_flip", aug_where)),
                color_jitter=jitter,
                blur_probability=float(
                    get_required(aug, "blur_probability", aug_where)
                ),
                blur_kernel=int(get_required(aug, "blur_kernel", aug_where)),
                blur_sigma=sigma,
            ),
        )

    @property
    def classifier_probe(self) -> "ProbeRecipe":
        probe = self._ct("probe")
        where = "training.classifier.probe"
        return ProbeRecipe(
            max_iter=int(get_required(probe, "max_iter", where)),
            c=float(get_required(probe, "c", where)),
        )

    @property
    def classifier_hard_negatives(self) -> "HardNegatives":
        hn = self._ct("hard_negatives")
        where = "training.classifier.hard_negatives"
        return HardNegatives(
            iou_negative=float(get_required(hn, "iou_negative", where)),
            max_per_drop=int(get_required(hn, "max_per_drop", where)),
            n_clusters=int(get_required(hn, "n_clusters", where)),
        )

    @property
    def static_box_label(self) -> str:
        """The class static junk boxes get in the raw CSV and the BIIGLE label they
        upload under ("Not a fish?")."""
        return self.classifier_static_boxes.label

    @property
    def classifier_static_boxes(self) -> "StaticBoxes":
        sb = self._ct("static_boxes")
        where = "training.classifier.static_boxes"
        return StaticBoxes(
            link_distance_box=float(get_required(sb, "link_distance_box", where)),
            link_distance_px=float(get_required(sb, "link_distance_px", where)),
            link_size_ratio=float(get_required(sb, "link_size_ratio", where)),
            max_gap_seconds=float(get_required(sb, "max_gap_seconds", where)),
            min_span_seconds=float(get_required(sb, "min_span_seconds", where)),
            min_boxes=int(get_required(sb, "min_boxes", where)),
            max_spread_px=float(get_required(sb, "max_spread_px", where)),
            max_size_spread=float(get_required(sb, "max_size_spread", where)),
            crops_per_track=int(get_required(sb, "crops_per_track", where)),
            crop_min_iou=float(get_required(sb, "crop_min_iou", where)),
            not_a_fish_flag=bool(get_required(sb, "not_a_fish_flag", where)),
            label=str(get_required(sb, "label", where)),
            flag_below=float(get_required(sb, "flag_below", where)),
            score_conf_mean=float(get_required(sb, "score_conf_mean", where)),
            score_conf_std=float(get_required(sb, "score_conf_std", where)),
            score_rate_mean=float(get_required(sb, "score_rate_mean", where)),
            score_rate_std=float(get_required(sb, "score_rate_std", where)),
        )

    # ── Site prior (two_stage_classifier_plan.md §6) ─────────────────────

    @property
    def site_prior(self) -> dict:
        return get_required(self._yaml_config, "site_prior", "")

    @property
    def site_prior_length_scale_km(self) -> float:
        value = float(get_required(self.site_prior, "length_scale_km", "site_prior"))
        if value <= 0:
            raise ValueError(f"site_prior.length_scale_km must be > 0; got {value}")
        return value

    @property
    def site_prior_pseudo_drops(self) -> float:
        value = float(get_required(self.site_prior, "pseudo_drops", "site_prior"))
        if value < 0:
            raise ValueError(f"site_prior.pseudo_drops must be >= 0; got {value}")
        return value

    @property
    def site_prior_donor_min_drops(self) -> int:
        return int(get_required(self.site_prior, "donor_min_drops", "site_prior"))

    @property
    def site_prior_floor(self) -> float:
        value = float(get_required(self.site_prior, "floor", "site_prior"))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"site_prior.floor must be in [0, 1); got {value}")
        return value

    @property
    def site_prior_rarefaction_sample_sizes(self) -> List[int]:
        sizes = get_required(self.site_prior, "rarefaction_sample_sizes", "site_prior")
        return [int(n) for n in sizes]

    @property
    def site_prior_bucket_names(self) -> List[str]:
        return list(get_required(self.site_prior, "bucket_names", "site_prior"))
