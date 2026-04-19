from pathlib import Path

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
    def training_epochs(self) -> int:
        return int(get_required(self.training_config, "epochs", "training"))

    @property
    def training_patience(self) -> int:
        return int(get_required(self.training_config, "patience", "training"))

    @property
    def training_imgsz(self) -> int:
        return int(get_required(self.training_config, "imgsz", "training"))

    @property
    def training_ceiling_pct(self) -> float:
        return float(
            get_required(self.training_config, "class_ceiling_pct", "training")
        )

    @property
    def training_floor_pct(self) -> float:
        return float(get_required(self.training_config, "class_floor_pct", "training"))

    @property
    def training_ceiling_max_iterations(self) -> int:
        return int(
            get_required(self.training_config, "ceiling_max_iterations", "training")
        )

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
    def training_val_min_images(self) -> int:
        return int(get_required(self.training_config, "val_min_images", "training"))

    @property
    def local_training_dir(self) -> Path:
        return self.project_root / get_required(
            self.training_config, "local_training_dir", "training"
        )

    @property
    def training_results_dir(self) -> Path:
        return self.local_training_dir / "results"

    @property
    def class_map_path(self) -> Path:
        return self.local_training_dir / "class_map.json"

    @property
    def training_results_s3_prefix(self) -> str:
        return self.training_results_dir.relative_to(self.project_root).as_posix()
