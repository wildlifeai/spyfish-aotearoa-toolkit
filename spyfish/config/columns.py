from spyfish.config.base import BaseConfig, get_required


class ColumnsConfig(BaseConfig):
    """CSV column name accessors — maps config.yaml `csv_mapping` keys to strings."""

    @property
    def csv_mapping(self) -> dict:
        return get_required(self._yaml_config, "csv_mapping", "")

    def _col(self, key: str) -> str:
        return get_required(self.csv_mapping, key, "csv_mapping")

    @property
    def drop_id_column(self) -> str:
        return self._col("drop_id_column")

    @property
    def pipeline_status_column(self) -> str:
        return self._col("pipeline_status_column")

    @property
    def survey_id_column(self) -> str:
        return self._col("survey_id_column")

    @property
    def site_id_column(self) -> str:
        return self._col("site_id_column")

    @property
    def replicate_column(self) -> str:
        return self._col("replicate_column")

    @property
    def file_name_column(self) -> str:
        return self._col("file_name_column")

    @property
    def link_to_marine_reserve_column(self) -> str:
        return self._col("link_to_marine_reserve_column")

    @property
    def protection_status_column(self) -> str:
        return self._col("protection_status_column")

    @property
    def site_name_column(self) -> str:
        return self._col("site_name_column")

    @property
    def selection_reason_column(self) -> str:
        return self._col("selection_reason_column")

    @property
    def csv_video_file_link_column(self) -> str:
        return self._col("video_file_link_column")

    @property
    def csv_sampling_start_column(self) -> str:
        return self._col("sampling_start_column")

    @property
    def csv_sampling_end_column(self) -> str:
        return self._col("sampling_end_column")

    @property
    def csv_clip_start_absolute_column(self) -> str:
        return self._col("clip_start_absolute_column")

    @property
    def csv_clip_end_absolute_column(self) -> str:
        return self._col("clip_end_absolute_column")

    @property
    def csv_clip_max_time_column(self) -> str:
        return self._col("clip_max_time_column")

    @property
    def csv_confidence_agreement_column(self) -> str:
        return self._col("confidence_agreement_column")

    @property
    def csv_confusion_score_column(self) -> str:
        return self._col("confusion_score_column")

    @property
    def csv_scientific_name_column(self) -> str:
        return self._col("scientific_name_column")

    @property
    def csv_maxn_time_column(self) -> str:
        return self._col("maxn_time_column")

    @property
    def csv_maxn_time_seconds_column(self) -> str:
        return self._col("maxn_time_seconds_column")

    @property
    def csv_max_interval_column(self) -> str:
        return self._col("max_interval_column")

    @property
    def csv_annotated_by_column(self) -> str:
        return self._col("annotated_by_column")

    @property
    def csv_interval_annotation_column(self) -> str:
        return self._col("interval_annotation_column")

    @property
    def csv_time_seconds_column(self) -> str:
        return self._col("time_seconds_column")
