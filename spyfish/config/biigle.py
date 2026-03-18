import os

from spyfish.config.base import BaseConfig, get_required


class BiigleConfig(BaseConfig):

    @property
    def email(self) -> str | None:
        return os.getenv("BIIGLE_API_EMAIL")

    @property
    def token(self) -> str | None:
        return os.getenv("BIIGLE_API_TOKEN")

    @property
    def biigle_section(self) -> dict:
        return get_required(self._yaml_config, "biigle", "")

    @property
    def biigle_project_id(self) -> int:
        val = os.getenv("BIIGLE_PROJECT_ID")
        if val:
            return int(val)
        return int(get_required(self.biigle_section, "project_id", "biigle"))

    @property
    def disk_id(self) -> int:
        val = os.getenv("BIIGLE_DISK_ID")
        if val:
            return int(val)
        return int(get_required(self.biigle_section, "disk_id", "biigle"))

    @property
    def annotation_report_type_video(self) -> int:
        return int(
            get_required(self.biigle_section, "annotation_report_type_video", "biigle")
        )

    @property
    def annotation_report_type_images(self) -> int:
        return int(
            get_required(self.biigle_section, "annotation_report_type_images", "biigle")
        )

    @property
    def volume_report_type(self) -> int:
        return int(
            get_required(self.biigle_section, "volume_report_type_image", "biigle")
        )

    @property
    def done_labels(self) -> list:
        return get_required(self.biigle_section, "done_labels", "biigle")

    @property
    def default_fish_label_id(self) -> int:
        return int(get_required(self.biigle_section, "default_fish_label_id", "biigle"))

    @property
    def label_mapping(self) -> dict:
        mapping = get_required(self.biigle_section, "label_mapping", "biigle")
        return mapping if mapping is not None else {}

    @property
    def default_label_tree_id(self) -> int:
        return int(get_required(self.biigle_section, "default_label_tree_id", "biigle"))

    @property
    def request_timeout_secs(self) -> int:
        return int(get_required(self.biigle_section, "request_timeout_secs", "biigle"))

    @property
    def volume_finalize_max_retries(self) -> int:
        return int(
            get_required(self.biigle_section, "volume_finalize_max_retries", "biigle")
        )

    @property
    def volume_finalize_retry_interval_secs(self) -> float:
        return float(
            get_required(
                self.biigle_section, "volume_finalize_retry_interval_secs", "biigle"
            )
        )

    @property
    def report_download_max_retries(self) -> int:
        return int(
            get_required(self.biigle_section, "report_download_max_retries", "biigle")
        )

    @property
    def report_download_retry_interval_secs(self) -> float:
        return float(
            get_required(
                self.biigle_section, "report_download_retry_interval_secs", "biigle"
            )
        )


biigle_config = BiigleConfig()
