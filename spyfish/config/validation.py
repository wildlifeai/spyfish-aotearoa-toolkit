import math
import re

from spyfish.config.base import BaseConfig, get_required


class ValidationConfig(BaseConfig):
    """Validation patterns, rules, and drop_id format checking."""

    def get_validation_pattern(self, name: str) -> str:
        raw_patterns = get_required(self._yaml_config, "validation_patterns", "")
        return str(get_required(raw_patterns, name, "validation_patterns"))

    @property
    def validation_patterns(self) -> dict:
        patterns = get_required(self._yaml_config, "validation_patterns", "")
        return {
            self.drop_id_column: get_required(
                patterns, "drop_id", "validation_patterns"
            ),
            self.survey_id_column: get_required(
                patterns, "survey_id", "validation_patterns"
            ),
            self.site_id_column: get_required(
                patterns, "site_id", "validation_patterns"
            ),
        }

    @property
    def validation_rules(self) -> dict:
        return get_required(self._yaml_config, "validation_rules", "")

    @property
    def movie_extensions(self) -> list:
        return get_required(self.paths, "movie_extensions", "paths")

    @property
    def file_presence_rules(self) -> dict:
        return {
            "file_presence": {
                "bucket": self.s3_bucket,
                "s3_sharepoint_path": self.sharepoint_root,
                "csv_filename": get_required(
                    self.metadata_files, "deployment_csv", "paths.metadata.files"
                ),
                "csv_column_to_extract": self.csv_video_file_link_column,
                "column_filter": None,
                "column_value": None,
                "valid_extensions": self.movie_extensions,
                "path_prefix": get_required(self.sub_dirs, "media", "paths.sub_dirs"),
            }
        }

    @property
    def _deployment_validation(self) -> dict:
        return get_required(self._yaml_config, "deployment_validation", "")

    @property
    def buv_video_duration_seconds(self) -> int:
        return int(
            get_required(
                self._deployment_validation,
                "buv_video_duration_seconds",
                "deployment_validation",
            )
        )

    @property
    def sampling_end_buffer_seconds(self) -> int:
        return int(
            get_required(
                self._deployment_validation,
                "sampling_end_buffer_seconds",
                "deployment_validation",
            )
        )

    def validate_sampling_window(
        self, drop_id: str, sampling_start: float, sampling_end: float
    ) -> list[str]:
        """Check that the sampling window looks valid for a BUV deployment.

        Returns a list of error messages (empty = valid).

        Rules:
          - sampling_start=0 means the ranger didn't set the window →
            bait-settling footage included.
          - sampling_end shorter than (expected duration - buffer) means the
            video is likely corrupted or incomplete.
          - sampling window (end - start) must not exceed the expected BUV duration.
        """
        errors = []
        expected = self.buv_video_duration_seconds
        buffer = self.sampling_end_buffer_seconds

        # NaN first: every comparison below is False against NaN, so a NaN
        # window would sail through all three rules and be reported valid.
        # Blank SamplingStart/SamplingEnd cells reach here as NaN (float64
        # column), which is exactly the case these rules exist to catch.
        if sampling_start is None or sampling_end is None or math.isnan(
            sampling_start
        ) or math.isnan(sampling_end):
            errors.append(
                f"{drop_id}: sampling window missing "
                f"(start={sampling_start!r}, end={sampling_end!r})."
            )
            return errors

        if sampling_start == 0:
            errors.append(
                f"{drop_id}: sampling_start=0, likely missing sampling window metadata."
            )

        if sampling_end < (expected - buffer):
            errors.append(
                f"{drop_id}: sampling_end={sampling_end:.0f}s is shorter than "
                f"expected ({expected}s - {buffer}s buffer = {expected - buffer}s)."
            )

        if sampling_end > sampling_start + expected:
            errors.append(
                f"{drop_id}: sampling window ({sampling_end - sampling_start:.0f}s) "
                f"exceeds expected BUV duration ({expected}s)."
            )

        return errors

    def validate_drop_id(self, drop_id: str) -> str:
        pattern = self.validation_patterns[self.drop_id_column]
        if not re.match(pattern, drop_id):
            raise ValueError(
                f"Invalid DropID format: '{drop_id}'. Must match {pattern}"
            )
        if ".." in drop_id or "/" in drop_id or "\\" in drop_id:
            raise ValueError(
                f"Security Alert: Malicious DropID detected (potential path traversal): '{drop_id}'"
            )
        return drop_id
