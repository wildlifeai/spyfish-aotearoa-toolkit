"""
Tests for clean_data module.
"""

from unittest.mock import Mock, patch

from sftk.clean_data import ScientificNameEntry, ScientificNameProcessing


class TestScientificNameProcessing:
    """Test ScientificNameProcessing class."""

    def test_clean_name_with_valid_string(self):
        """Test that clean_name returns cleaned name for valid string."""
        processor = ScientificNameProcessing(
            scientific_name_to_check="Chrysophrys auratus", common_name="Snapper"
        )

        result = processor.clean_name()

        assert result == "Chrysophrys auratus"
        assert processor.higher_taxon == ""

    def test_clean_name_with_sp_suffix(self):
        """Test that clean_name handles ' sp' suffix correctly."""
        processor = ScientificNameProcessing(
            scientific_name_to_check="Chrysophrys sp", common_name="Snapper"
        )

        result = processor.clean_name()

        assert result == "Chrysophrys"
        assert processor.higher_taxon == " sp"

    def test_clean_name_with_invalid_inputs(self):
        """Test that clean_name returns None for non-string inputs."""
        # Test with None
        processor = ScientificNameProcessing(
            scientific_name_to_check=None, common_name="Snapper"
        )
        assert processor.clean_name() is None

        # Test with integer
        processor = ScientificNameProcessing(
            scientific_name_to_check=123, common_name="Snapper"
        )
        assert processor.clean_name() is None

    @patch("sftk.clean_data.requests.get")
    def test_query_api_success(self, mock_get):
        """Test that query_api successfully queries WoRMS API."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "AphiaID": 123456,
                "valid_name": "Chrysophrys auratus",
                "rank": "Species",
                "status": "accepted",
            }
        ]
        mock_get.return_value = mock_response

        processor = ScientificNameProcessing(
            scientific_name_to_check="Chrysophrys auratus", common_name="Snapper"
        )

        result = processor.query_api()

        assert isinstance(result, ScientificNameEntry)
        assert result.aphia_id == 123456
        assert result.scientific_name == "Chrysophrys auratus"
        assert result.common_name == "Snapper"
        assert result.scientific_name_to_check == "Chrysophrys auratus"
        assert result.scientific_names_match is True
        assert result.status == "accepted"

    @patch("sftk.clean_data.requests.get")
    def test_query_api_with_sp_suffix(self, mock_get):
        """Test that query_api handles ' sp' suffix correctly."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "AphiaID": 123456,
                "valid_name": "Chrysophrys",
                "rank": "Genus",
                "status": "accepted",
            }
        ]
        mock_get.return_value = mock_response

        processor = ScientificNameProcessing(
            scientific_name_to_check="Chrysophrys sp", common_name="Snapper"
        )

        result = processor.query_api()

        assert result.scientific_name == "Chrysophrys sp"
        assert result.scientific_names_match is True

    @patch("sftk.clean_data.requests.get")
    def test_query_api_invalid_name(self, mock_get):
        """Test that query_api returns failed entry for invalid name."""
        processor = ScientificNameProcessing(
            scientific_name_to_check=None, common_name="Snapper"
        )

        result = processor.query_api()

        assert isinstance(result, ScientificNameEntry)
        assert result.status == "invalid name"
        assert result.scientific_name_to_check is None
        mock_get.assert_not_called()

    @patch("sftk.clean_data.requests.get")
    def test_query_api_api_error(self, mock_get):
        """Test that query_api handles API errors gracefully."""
        import requests

        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        processor = ScientificNameProcessing(
            scientific_name_to_check="Chrysophrys auratus", common_name="Snapper"
        )

        result = processor.query_api()

        assert isinstance(result, ScientificNameEntry)
        assert result.status == "api error"
        assert "api error" in result.status.lower()
