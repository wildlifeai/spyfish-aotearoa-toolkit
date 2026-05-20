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
        return get_required(self.ml_inference, "confidence_threshold", "ml_inference")

    @property
    def maxn_confidence_threshold(self):
        return get_required(
            self.ml_inference, "maxn_confidence_threshold", "ml_inference"
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
        """Image-count floor — species appearing in fewer distinct frames get merged into 'fish'."""
        return int(
            get_required(self.training_config, "class_floor_min_images", "training")
        )

    @property
    def training_split_seed(self) -> Optional[int]:
        """Random seed for reproducible train/val/test splits and per-drop frame filtering.

        Set to an integer (default 42) for deterministic, reproducible results
        across retrain runs — same drops + same labels always produce the same
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

        **Extras (drops under `extra_no_survey_id/`) bypass this cap entirely** —
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
    def training_train_pct(self) -> float:
        return float(get_required(self.training_config, "train_pct", "training"))

    @property
    def training_val_pct(self) -> float:
        return float(get_required(self.training_config, "val_pct", "training"))

    @property
    def training_test_pct(self) -> float:
        return float(get_required(self.training_config, "test_pct", "training"))

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
    def training_force_val_drops_file(self) -> Path:
        return self.project_root / get_required(
            self.training_config, "force_val_drops_file", "training"
        )

    @property
    def training_force_val_drops(self) -> set:
        """DropIDs to force into the val split (overrides survey-aware donation)."""
        return self._parse_drop_ids_from_file(self.training_force_val_drops_file)

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

    @property
    def training_extraction_annotation_type(self) -> str:
        """Which model to run on extracted frames — 'binary' or 'species'.

        Resolves to `config.get_pipeline_model(annotation_type)` at runtime.
        """
        kind = str(
            get_required(
                self._training_extraction,
                "annotation_type",
                "training_extraction",
            )
        )
        if kind not in {"binary", "species"}:
            raise ValueError(
                f"training_extraction.annotation_type must be 'binary' or "
                f"'species', got {kind!r}"
            )
        return kind
