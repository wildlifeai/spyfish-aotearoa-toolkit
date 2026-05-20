"""
Tests for S3Handler class.
"""

from unittest.mock import Mock, patch

import pandas as pd

from sftk.s3_handler import S3FileConfig, S3Handler


class TestS3FileConfig:
    """Test S3FileConfig class."""

    def test_s3_file_config_from_keyword(self):
        """Test that from_keyword creates correct configuration."""
        config = S3FileConfig.from_keyword("survey")

        assert config.keyword == "survey"
        assert config.kso_env_var == "S3_KSO_SURVEY_CSV"
        assert config.sharepoint_env_var == "S3_SHAREPOINT_SURVEY_CSV"
        assert config.kso_filename == "survey_kso_temp.csv"
        assert config.sharepoint_filename == "survey_sharepoint_temp.csv"


class TestS3Handler:
    """Test S3Handler class."""

    @patch("sftk.s3_handler.boto3")
    @patch("sftk.s3_handler.S3_BUCKET", "test-bucket")
    @patch("sftk.s3_handler.AWS_ACCESS_KEY_ID", "test-key")
    @patch("sftk.s3_handler.AWS_SECRET_ACCESS_KEY", "test-secret")
    def test_s3_handler_initialization(self, mock_boto3):
        """Test that S3Handler initializes correctly."""
        mock_s3_client = Mock()
        mock_boto3.client.return_value = mock_s3_client

        # Reset singleton
        S3Handler._instance = None

        handler = S3Handler()

        assert handler.bucket == "test-bucket"
        assert handler.s3 == mock_s3_client
        mock_boto3.client.assert_called_once()

    @patch("sftk.s3_handler.boto3")
    @patch("sftk.s3_handler.S3_BUCKET", "test-bucket")
    @patch("sftk.s3_handler.AWS_ACCESS_KEY_ID", "test-key")
    @patch("sftk.s3_handler.AWS_SECRET_ACCESS_KEY", "test-secret")
    def test_s3_handler_read_and_extract_paths_from_csv(self, mock_boto3):
        """Test that S3Handler can read CSV from S3 and extract filtered paths."""
        # Create test DataFrame with file paths
        test_df = pd.DataFrame(
            {
                "video_path": [
                    "videos/file1.mp4",
                    "videos/file2.mp4",
                    "videos/file3.mov",
                ],
                "status": ["active", "active", "inactive"],
            }
        )
        csv_content = test_df.to_csv(index=False)

        # Mock S3 response
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = csv_content.encode()
        mock_response["Body"].__enter__ = Mock(return_value=mock_response["Body"])
        mock_response["Body"].__exit__ = Mock(return_value=False)

        mock_s3_client = Mock()
        mock_s3_client.get_object.return_value = mock_response
        mock_boto3.client.return_value = mock_s3_client

        # Reset singleton
        S3Handler._instance = None

        handler = S3Handler(s3_client=mock_s3_client, bucket="test-bucket")

        # Test 1: Read CSV from S3
        result_df = handler.read_df_from_s3_csv("test/path.csv")
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="test/path.csv"
        )
        pd.testing.assert_frame_equal(result_df, test_df)

        # Test 2: Extract paths from CSV (uses read_df_from_s3_csv internally)
        # Reset the mock call count for the second operation
        mock_s3_client.get_object.reset_mock()
        mock_s3_client.get_object.return_value = mock_response

        result = handler.get_paths_from_csv(
            csv_s3_path="test.csv",
            csv_column="video_path",
            column_filter="status",
            column_value="active",
        )

        # Verify S3 was called again for get_paths_from_csv
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="test.csv"
        )

        # Verify path extraction results
        assert "all" in result
        assert "filtered" in result
        assert (
            len(result["filtered"]) == 2
        )  # Only files with active status (file1.mp4, file2.mp4)
        assert "videos/file1.mp4" in result["filtered"]
        assert "videos/file2.mp4" in result["filtered"]
        assert len(result["all"]) == 3  # All files from CSV
