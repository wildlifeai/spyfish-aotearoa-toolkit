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
    def _zooniverse(self) -> dict:
        return get_required(self._yaml_config, "zooniverse", "")

    @property
    def zooniverse_project_id(self) -> int:
        return int(get_required(self._zooniverse, "project_id", "zooniverse"))

    @property
    def size_limit_mb(self) -> float:
        return float(get_required(self._zooniverse, "size_limit_mb", "zooniverse"))

    @property
    def health_check_count(self) -> int:
        return int(get_required(self._zooniverse, "health_check_count", "zooniverse"))

    @property
    def zooniverse_min_votes(self) -> int:
        return int(get_required(self._zooniverse, "min_votes", "zooniverse"))

    @property
    def zooniverse_max_frames_per_run(self) -> int:
        return int(get_required(self._zooniverse, "max_frames_per_run", "zooniverse"))


zooniverse_config = ZooniverseConfig()
