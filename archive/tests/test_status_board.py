"""
Status Board tests.

Tests the StatusBoard class which generates deployment and survey status reports.
Focus on integration: test the full flow with minimal mocking.
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from sftk.status_board import StatusBoard, classify_video_status


class TestClassifyVideoStatus:
    """Tests for the classify_video_status helper function."""

    def test_all_status_types(self):
        """Should correctly classify all video status types."""
        missing = {"video1.mp4"}
        paired = {"video2.mp4"}

        assert classify_video_status(None, missing, paired) == "NoLink"
        assert (
            classify_video_status("NO VIDEO BAD DEPLOYMENT", missing, paired)
            == "Excluded"
        )
        assert classify_video_status("video1.mp4", missing, paired) == "Missing"
        assert classify_video_status("video2.mp4", missing, paired) == "Present"
        assert classify_video_status("unknown.mp4", missing, paired) == "Unknown"


class TestStatusBoardIntegration:
    """Integration tests for StatusBoard."""

    @pytest.fixture
    def mock_surveys(self):
        """Sample surveys - includes one with no deployments."""
        return pd.DataFrame(
            {
                "SurveyID": ["SUR_20240101_BUV", "SUR_20240202_BUV"],
            }
        )

    @pytest.fixture
    def mock_deployments(self):
        """Sample deployment data covering all status scenarios."""
        return pd.DataFrame(
            {
                "SurveyID": ["SUR_20240101_BUV"] * 4,
                "DropID": ["DROP_001", "DROP_002", "DROP_003", "DROP_004"],
                "LinkToVideoFile": [
                    "video1.mp4",  # Present
                    "video2.mp4",  # Missing
                    None,  # NoLink
                    "NO VIDEO BAD DEPLOYMENT",  # Excluded (bad deployment)
                ],
                "IsBadDeployment": [False, False, False, True],
            }
        )

    @pytest.fixture
    def mock_annotations(self):
        """Sample annotations - only DROP_001 has annotations."""
        return pd.DataFrame(
            {
                "DropID": ["DROP_001", "DROP_001"],
                "annotation_id": [1, 2],
            }
        )

    @pytest.fixture
    def mock_file_differences(self):
        """File differences: video1 exists, video2 is missing."""
        all_files = {"video1.mp4", "video2.mp4"}
        missing = {"video2.mp4"}
        extra = set()
        return (all_files, missing, extra)

    @pytest.fixture
    def status_board(
        self, mock_surveys, mock_deployments, mock_annotations, mock_file_differences
    ):
        """Create StatusBoard with mocked data sources."""
        mock_validator = Mock()
        mock_validator.get_file_differences.return_value = mock_file_differences
        mock_validator.s3_handler = Mock()

        board = StatusBoard(data_validator=mock_validator)
        board.load_data = Mock(
            return_value=(mock_surveys, mock_deployments, mock_annotations)
        )
        return board

    def test_deployment_status_columns(self, status_board):
        """Deployment status should have all expected columns."""
        deployment_df, _ = status_board.generate_status_report()

        expected_columns = [
            "SurveyID",
            "DropID",
            "VideoStatus",
            "IsBadDeployment",
            "HasAnnotations",
            "TotalAnnotations",
            "ExpertAnnotations",
            "MlAnnotations",
            "BiigleAnnotations",
            "NeedsAction",
        ]
        assert list(deployment_df.columns) == expected_columns

    def test_deployment_status_values(self, status_board):
        """Deployment status should correctly compute all fields."""
        deployment_df, _ = status_board.generate_status_report()
        df = deployment_df.set_index("DropID")

        # DROP_001: Present video + has annotations = no action needed
        assert df.loc["DROP_001", "VideoStatus"] == "Present"
        assert df.loc["DROP_001", "HasAnnotations"] == True
        assert df.loc["DROP_001", "NeedsAction"] == False

        # DROP_002: Missing video = NeedsAction
        assert df.loc["DROP_002", "VideoStatus"] == "Missing"
        assert df.loc["DROP_002", "NeedsAction"] == True

        # DROP_003: NoLink = NeedsAction
        assert df.loc["DROP_003", "VideoStatus"] == "NoLink"
        assert df.loc["DROP_003", "NeedsAction"] == True

        # DROP_004: Bad deployment = excluded, no action needed
        assert df.loc["DROP_004", "VideoStatus"] == "Excluded"
        assert df.loc["DROP_004", "IsBadDeployment"] == True
        assert df.loc["DROP_004", "NeedsAction"] == False

    def test_survey_status_aggregation(self, status_board):
        """Survey status should correctly aggregate deployment data."""
        _, survey_df = status_board.generate_status_report()

        # Should include both surveys (one with deployments, one without)
        assert len(survey_df) == 2
        df = survey_df.set_index("SurveyID")

        # Survey with deployments
        row = df.loc["SUR_20240101_BUV"]
        assert row["TotalDeployments"] == 4
        assert row["CompleteDeployments"] == 2  # DROP_001 + DROP_004
        assert row["VideosNoActionNeeded"] == 2  # Present + Excluded (from VideoStatus)
        assert row["VideosMissing"] == 1
        assert row["VideosNoLink"] == 1
        assert row["BadDeployments"] == 1
        assert row["AnnotatedDeployments"] == 1
        assert row["TotalAnnotations"] == 2
        assert row["CompletionPct"] == 50.0  # 2/4 complete
        assert row["NeedsAction"] == True  # Not 100% complete

        # Survey with no deployments
        empty_row = df.loc["SUR_20240202_BUV"]
        assert empty_row["TotalDeployments"] == 0
        assert empty_row["CompletionPct"] == 0.0
        assert empty_row["NeedsAction"] == True  # 0 deployments = needs action

    def test_survey_status_columns(self, status_board):
        """Survey status should have all expected columns in order."""
        _, survey_df = status_board.generate_status_report()

        expected_columns = [
            "SurveyID",
            "TotalDeployments",
            "CompleteDeployments",
            "AnnotatedDeployments",
            "TotalAnnotations",
            "VideosNoActionNeeded",
            "VideosMissing",
            "VideosNoLink",
            "BadDeployments",
            "CompletionPct",
            "AnnotationPct",
            "BadDeploymentPct",
            "NeedsAction",
        ]
        assert list(survey_df.columns) == expected_columns
