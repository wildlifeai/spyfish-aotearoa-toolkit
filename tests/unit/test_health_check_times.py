"""
Regression tests for health check frame time computation.

Convention: all times in the pipeline are ABSOLUTE video timestamps
(seconds from the start of the video file). sampling_start is the
beginning of the sampling window within that video.

Health check times must be in [sampling_start, sampling_end].
extract_frames_from_selections uses stored times directly (no sampling_start addition).
"""

from unittest.mock import MagicMock, patch

from spyfish.config.wrapper import config


@patch("spyfish.zooniverse.select_zooniverse_clips.DatabaseManager")
@patch("spyfish.zooniverse.select_zooniverse_clips.config")
def test_health_check_times_are_relative_to_sampling_start(
    mock_config, mock_db_class, tmp_path
):
    """
    For a zero-detection deployment with sampling_start=600, health check selection
    times must be absolute video timestamps: in [sampling_start, sampling_end].

    If times were stored relative (old bug direction), they would be < sampling_start.
    """
    from spyfish.zooniverse.select_zooniverse_clips import process_zooniverse_clips

    # Large sampling_start relative to sampling window: makes the bug clearly detectable.
    # Old (buggy) times would be 610, 620, ... (> sampling_start=600).
    # Fixed times are 10, 20, ... (< sampling_start=600).
    sampling_start = 600  # 10 minutes of pre-deployment video before sampling starts
    sampling_end = 660
    clip_length = 5
    min_clips = 6  # top-up to at least 6 clips

    # Wire up config mock with the column names from the real config and test values
    mock_config.sample_all_clips = False
    mock_config.clip_cap = None
    mock_config.clip_length = clip_length
    mock_config.min_clips_per_video = min_clips
    mock_config.csv_time_seconds_column = config.csv_time_seconds_column
    mock_config.csv_clip_max_time_column = config.csv_clip_max_time_column
    mock_config.csv_clip_start_absolute_column = config.csv_clip_start_absolute_column
    mock_config.csv_clip_end_absolute_column = config.csv_clip_end_absolute_column
    mock_config.csv_sampling_start_column = config.csv_sampling_start_column
    mock_config.csv_scientific_name_column = config.csv_scientific_name_column
    mock_config.csv_max_interval_column = config.csv_max_interval_column
    mock_config.csv_confidence_agreement_column = config.csv_confidence_agreement_column
    mock_config.drop_id_column = config.drop_id_column

    # Stub DB to return known sampling metadata
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.fetchone.return_value = {
        "sampling_start": sampling_start,
        "sampling_end": sampling_end,
    }
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_db_class.return_value.get_connection.return_value = mock_conn

    # Empty MaxN CSV → triggers health check path
    maxn_csv = tmp_path / "maxn.csv"
    maxn_csv.write_text(
        f"{config.drop_id_column},{config.csv_scientific_name_column},"
        f"{config.csv_maxn_time_column},{config.csv_max_interval_column},"
        f"{config.csv_annotated_by_column},{config.csv_interval_annotation_column},"
        f"{config.csv_confidence_agreement_column},{config.csv_maxn_time_seconds_column}\n"
    )
    selections_csv = tmp_path / "selections.csv"
    drop_id = "KSF_20240124_BUV_KSF_085_01"

    result = process_zooniverse_clips(str(maxn_csv), str(selections_csv), drop_id)

    assert not result.empty, "Expected health check clips to be generated"

    times = result[config.csv_clip_max_time_column].tolist()

    # All times must be absolute: within the sampling window [sampling_start, sampling_end]
    assert all(sampling_start <= t <= sampling_end for t in times), (
        f"Health check times must be absolute video timestamps "
        f"in [{sampling_start}, {sampling_end}], got: {times}."
    )

    # Times below sampling_start would indicate relative storage (wrong convention)
    assert all(t >= sampling_start for t in times), (
        f"Times < sampling_start={sampling_start} mean relative times were stored "
        f"instead of absolute. Got: {times}"
    )

    assert len(times) > 0
    assert max(times) <= sampling_end
