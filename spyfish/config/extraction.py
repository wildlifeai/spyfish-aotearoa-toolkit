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
    def frame_multiplier(self) -> float:
        return float(get_required(self._extraction, "frame_multiplier", "extraction"))

    @property
    def binary_strategy(self) -> dict:
        return get_required(self._extraction, "binary_strategy", "extraction")

    @property
    def multiclass_strategy(self) -> dict:
        return get_required(self._extraction, "multiclass_strategy", "extraction")

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
