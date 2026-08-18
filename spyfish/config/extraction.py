from spyfish.config.base import BaseConfig, get_required


class ExtractionConfig(BaseConfig):

    @property
    def _extraction(self) -> dict:
        return get_required(self._yaml_config, "extraction", "")

    @property
    def clip_length(self) -> float:
        return float(get_required(self._extraction, "clip_length", "extraction"))

    @property
    def clip_cap(self) -> int:
        return int(get_required(self._extraction, "clip_cap", "extraction"))

    @property
    def video_start_threshold(self) -> int:
        return int(
            get_required(
                self._extraction, "video_start_threshold_seconds", "extraction"
            )
        )

    @property
    def force_binary_strategy(self) -> bool:
        return bool(
            get_required(self._extraction, "force_binary_strategy", "extraction")
        )

    @property
    def sample_all_clips(self) -> bool:
        return bool(get_required(self._extraction, "sample_all_clips", "extraction"))

    @property
    def min_frames_per_drop(self) -> int:
        """Frame floor per deployment, never a ceiling, peaks may exceed it."""
        return int(get_required(self._extraction, "min_frames_per_drop", "extraction"))

    @property
    def catchall_class(self) -> str:
        """The model's catch-all class, an animal it detected but could not name."""
        return str(
            get_required(
                get_required(self._yaml_config, "reporting", ""),
                "catchall_class",
                "reporting",
            )
        )

    @property
    def frame_strategy(self) -> dict:
        """Per-species bucket quotas for FRAME selection (clips have their own)."""
        return get_required(self._extraction, "frame_strategy", "extraction")

    @property
    def binary_strategy(self) -> dict:
        return get_required(self._extraction, "binary_strategy", "extraction")

    @property
    def multiclass_strategy(self) -> dict:
        return get_required(self._extraction, "multiclass_strategy", "extraction")

    @property
    def _ml_peak_augmentation(self) -> dict:
        return get_required(self._extraction, "ml_peak_augmentation", "extraction")

    @property
    def ml_peak_top_k_per_species(self) -> int:
        return int(
            get_required(
                self._ml_peak_augmentation,
                "top_k_per_species",
                "extraction.ml_peak_augmentation",
            )
        )

    @property
    def ml_peak_min_confidence(self) -> float:
        return float(
            get_required(
                self._ml_peak_augmentation,
                "min_confidence",
                "extraction.ml_peak_augmentation",
            )
        )

    @property
    def ml_peak_citsci_dedupe_tolerance_seconds(self) -> float:
        return float(
            get_required(
                self._ml_peak_augmentation,
                "citsci_dedupe_tolerance_seconds",
                "extraction.ml_peak_augmentation",
            )
        )

    # ── FFmpeg encoding settings ─────────────────────────────────────────

    @property
    def ffmpeg_config(self) -> dict:
        return get_required(self._yaml_config, "ffmpeg", "")

    @property
    def ffmpeg_crf(self) -> str:
        return str(get_required(self.ffmpeg_config, "crf", "ffmpeg"))

    @property
    def ffmpeg_preset(self) -> str:
        return str(get_required(self.ffmpeg_config, "preset", "ffmpeg"))

    @property
    def ffmpeg_codec(self) -> str:
        return str(get_required(self.ffmpeg_config, "codec", "ffmpeg"))
