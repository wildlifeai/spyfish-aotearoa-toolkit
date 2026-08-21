from spyfish.config.base import BaseConfig, get_required


class ColumnsConfig(BaseConfig):
    """CSV column name accessors, maps config.yaml `csv_mapping` keys to strings."""

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
    def region_column(self) -> str:
        return self._col("region_column")

    @property
    def reserve_title_column(self) -> str:
        return self._col("reserve_title_column")

    @property
    def reserve_acronym_column(self) -> str:
        return self._col("reserve_acronym_column")

    @property
    def targeted_latitude_column(self) -> str:
        return self._col("targeted_latitude_column")

    @property
    def targeted_longitude_column(self) -> str:
        return self._col("targeted_longitude_column")

    @property
    def latitude_column(self) -> str:
        return self._col("latitude_column")

    @property
    def longitude_column(self) -> str:
        return self._col("longitude_column")

    @property
    def depth_column(self) -> str:
        return self._col("depth_column")

    @property
    def protection_status_aliases(self) -> dict:
        """Lowercased ProtectionStatus → the canonical spelling to store."""
        return get_required(self._yaml_config, "protection_status_aliases", "")

    @property
    def reporting(self) -> dict:
        return get_required(self._yaml_config, "reporting", "")

    @property
    def non_species_classes(self) -> list:
        """Model classes that are not species, excluded from abundance figures."""
        return get_required(self.reporting, "non_species_classes", "reporting")

    @property
    def known_protection_statuses(self) -> list:
        """Every ProtectionStatus spelling the pipeline recognises."""
        return get_required(self._yaml_config, "known_protection_statuses", "")

    @property
    def unprotected_statuses(self) -> list:
        """ProtectionStatus values treated as unprotected in inside/outside splits.

        Named explicitly rather than "everything not protected", so a partial
        regime lands in neither list and is reported as Other.
        """
        return get_required(self.reporting, "unprotected_statuses", "reporting")

    @property
    def unidentified_label(self) -> str:
        """What the report calls the merged non-species classes."""
        return get_required(self.reporting, "unidentified_label", "reporting")

    @property
    def indicator_species(self) -> list:
        """Scientific names used to characterise a reserve's population."""
        return get_required(self.reporting, "indicator_species", "reporting")

    @property
    def protected_statuses(self) -> list:
        """ProtectionStatus values treated as protected in inside/outside splits."""
        return get_required(self.reporting, "protected_statuses", "reporting")

    @property
    def sensitive_columns(self) -> list:
        """DB columns to strip from any export leaving the pipeline."""
        return get_required(self._yaml_config, "sensitive_columns", "")

    def strip_sensitive(self, df):
        """Drop sensitive columns from a dataframe before it is exported.

        Case-insensitive, and a no-op for columns that are not present, so it is
        safe to call on any frame regardless of which table it came from.
        """
        sensitive = {c.lower() for c in self.sensitive_columns}
        return df.drop(
            columns=[c for c in df.columns if c.lower() in sensitive],
            errors="ignore",
        )

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

    @property
    def csv_subject_id_column(self) -> str:
        return self._col("subject_id_column")

    # ML MaxN CSV only — persistence-filter provenance (absent from citsci/expert
    # MaxN CSVs, so consumers must guard on column presence).

    @property
    def csv_raw_max_interval_column(self) -> str:
        return self._col("raw_max_interval_column")

    @property
    def csv_spike_flag_column(self) -> str:
        return self._col("spike_flag_column")

    @property
    def csv_spike_time_seconds_column(self) -> str:
        return self._col("spike_time_seconds_column")
