import os

from spyfish.config.base import BaseConfig, get_required


class ZooniverseConfig(BaseConfig):

    @property
    def user(self):
        return os.getenv("ZOONIVERSE_USER")

    @property
    def password(self):
        return os.getenv("ZOONIVERSE_PASSWORD")

    @property
    def extraction(self) -> dict:
        return get_required(self._yaml_config, "zooniverse_extraction", "")

    @property
    def zooniverse_project_id(self) -> int:
        return int(get_required(self.extraction, "project_id", "zooniverse_extraction"))

    @property
    def clip_length(self) -> float:
        return float(
            get_required(self.extraction, "clip_length", "zooniverse_extraction")
        )

    @property
    def health_check_count(self) -> int:
        return int(
            get_required(self.extraction, "health_check_count", "zooniverse_extraction")
        )

    @property
    def video_start_threshold(self) -> int:
        return int(
            get_required(
                self.extraction,
                "video_start_threshold_seconds",
                "zooniverse_extraction",
            )
        )

    @property
    def clip_cap(self) -> int:
        return int(get_required(self.extraction, "clip_cap", "zooniverse_extraction"))

    @property
    def size_limit_mb(self) -> float:
        return float(get_required(self.extraction, "size_limit_mb", "zooniverse_extraction"))

    @property
    def force_binary_strategy(self) -> bool:
        return bool(
            get_required(
                self.extraction, "force_binary_strategy", "zooniverse_extraction"
            )
        )

    @property
    def sample_all_clips(self) -> bool:
        return bool(
            get_required(
                self.extraction, "sample_all_clips", "zooniverse_extraction"
            )
        )

    @property
    def temporal_spacing(self) -> int:
        return int(
            get_required(
                self.extraction, "temporal_spacing_seconds", "zooniverse_extraction"
            )
        )

    @property
    def zooniverse_min_votes(self) -> int:
        return int(get_required(self.extraction, "min_votes", "zooniverse_extraction"))

    @property
    def zooniverse_max_frames_per_run(self) -> int:
        return int(get_required(self.extraction, "max_frames_per_run", "zooniverse_extraction"))

    @property
    def binary_strategy(self) -> dict:
        return get_required(self.extraction, "binary_strategy", "zooniverse_extraction")

    @property
    def multiclass_strategy(self) -> dict:
        return get_required(
            self.extraction, "multiclass_strategy", "zooniverse_extraction"
        )


zooniverse_config = ZooniverseConfig()
