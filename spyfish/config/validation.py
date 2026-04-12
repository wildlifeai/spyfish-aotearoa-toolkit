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
