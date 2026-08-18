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
        return int(get_required(self.biigle_section, "project_id", "biigle"))

    @property
    def biigle_projects(self) -> dict:
        """Named BIIGLE projects (in_progress / done / playground)."""
        return get_required(self.biigle_section, "projects", "biigle")

    @property
    def biigle_upload_project_id(self) -> int:
        """Where the pipeline uploads new volumes (in_progress)."""
        return int(get_required(self.biigle_projects, "in_progress", "biigle.projects"))

    @property
    def biigle_done_project_id(self) -> int:
        """Where annotator-finished volumes live; --biigle-sync reads from here."""
        return int(get_required(self.biigle_projects, "done", "biigle.projects"))

    @property
    def disk_id(self) -> int:
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
    def done_labels(self) -> list:
        return get_required(self.biigle_section, "done_labels", "biigle")

    @property
    def biigle_require_done_label(self) -> bool:
        """When True, --biigle-sync gates on the Done-label whole-file check.
        When False, every volume awaiting sync is ingested (project membership
        in `done` is the gate)."""
        return bool(get_required(self.biigle_section, "require_done_label", "biigle"))

    @property
    def default_fish_label_id(self) -> int:
        return int(get_required(self.biigle_section, "default_fish_label_id", "biigle"))

    @property
    def default_bait_label_id(self) -> int:
        """Biigle label ID used when an ML annotation's class resolves to 'bait'.

        Distinct from `default_fish_label_id` so bait predictions land in the
        Biigle 'Bait' label (tree 3375 → 537309) instead of getting lumped with
        fish predictions in the Fish review bucket.
        """
        return int(get_required(self.biigle_section, "default_bait_label_id", "biigle"))

    @property
    def label_mapping(self) -> dict:
        mapping = get_required(self.biigle_section, "label_mapping", "biigle")
        return mapping if mapping is not None else {}

    @property
    def default_label_tree_id(self) -> int:
        return int(get_required(self.biigle_section, "default_label_tree_id", "biigle"))

    @property
    def biigle_substrate_label_tree_id(self) -> int:
        """Label-tree ID whose labels are CMECS substrate / cover categories.

        Membership in this tree (resolved via the report's `label_id` column)
        marks an annotation as substrate, measured for percent-cover. This is
        what separates a substrate LineString from a fish-SIZE LineString, whose
        label lives in the species tree (or is "Scale bar") instead."""
        return int(
            get_required(self.biigle_section, "substrate_label_tree_id", "biigle")
        )

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
