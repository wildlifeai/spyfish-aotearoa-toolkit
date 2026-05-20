import logging
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class ScientificNameEntry:
    aphia_id: int = -1
    common_name: Optional[str] = None
    scientific_name: Optional[str] = None
    scientific_name_to_check: Optional[str] = None
    scientific_names_match: bool = False
    taxon_rank: Optional[str] = None
    status: str = "unprocessed"


class ScientificNameProcessing:
    """Class species information from the WoRMS API."""

    def __init__(
        self,
        scientific_name_to_check: Optional[str] = None,
        common_name: Optional[str] = None,
    ):
        """Initialize a ScientificNameProcessing instance.

        Args:
            scientific_name_to_check (Optional[str], optional): Scientific name to validate. Defaults to None.
            common_name (Optional[str], optional): Common name of the species. Defaults to None.
        """
        self.common_name = common_name
        self.scientific_name_to_check = scientific_name_to_check
        self.higher_taxon = ""

    def clean_name(self):
        """Cleans scientific_name_to_check and returns the value ready for query or None if invalid."""
        # Genus values are defined as sp in the DOC data
        if not isinstance(self.scientific_name_to_check, str):
            return None

        cleaned_name = self.scientific_name_to_check
        if self.scientific_name_to_check.lower().endswith(" sp"):
            cleaned_name = cleaned_name.split()[0]
            self.higher_taxon = " sp"
        return cleaned_name

    def query_api(self) -> ScientificNameEntry:
        """Queries the WoRMS API and returns a ScientificNameEntry populated from the response.

        API reference: https://www.marinespecies.org/rest/
        If the resquest is successful it populates all the attributes, if not, only with input info and failure reason.
        """
        cleaned_name = self.clean_name()
        if not cleaned_name:
            return self.failed_to_process(
                f"{self.scientific_name_to_check}, {self.common_name} was not processed, not a valid string.",
                "invalid name",
            )

        api_url = (
            f"https://www.marinespecies.org/rest/AphiaRecordsByName/{cleaned_name}"
        )

        params = {"like": "false", "marine_only": "true", "offset": 1}  # starts at 1
        try:
            response = requests.get(api_url, params=params)
        except requests.exceptions.RequestException as e:
            return self.failed_to_process(
                f"{self.scientific_name_to_check}, {self.common_name}  was not processed, error connecting to API: {e}",
                "api error",
            )

        if response.status_code != 200:
            return self.failed_to_process(
                f"No results found for scientific name {self.scientific_name_to_check}, {self.common_name}  API request failed with status code {response.status_code}.",
                "api error",
            )

        try:
            response_json = response.json()
            if len(response_json) > 1:
                # Currently it selects the first option, WAI, no need to fix yet.
                # Example to use to get multiple results: Chrysophrys auratus
                logging.warning(
                    f"Multiple results found for {self.scientific_name_to_check}, {self.common_name} if that's not expected, check it here: {response_json}"
                )
            response_json = response_json[0]
        except (ValueError, IndexError, TypeError) as e:
            return self.failed_to_process(
                f"No results found for scientific name {self.scientific_name_to_check}, {self.common_name}: {e}",
                "no results",
            )

        accepted_scientific_name = response_json.get("valid_name")
        if not accepted_scientific_name:
            return self.failed_to_process(
                f"{self.scientific_name_to_check}, {self.common_name} was not processed, no valid_name found in API response.",
                "no accepted name",
            )

        return ScientificNameEntry(
            aphia_id=response_json.get("AphiaID"),
            common_name=self.common_name,
            scientific_name=accepted_scientific_name + self.higher_taxon,
            scientific_name_to_check=self.scientific_name_to_check,
            scientific_names_match=accepted_scientific_name == cleaned_name,
            taxon_rank=response_json.get("rank"),
            status=response_json.get("status"),
        )

    def failed_to_process(self, message, status="failure"):
        logging.warning(message)
        return ScientificNameEntry(
            common_name=self.common_name,
            scientific_name_to_check=self.scientific_name_to_check,
            status=status,
        )


class MetadataProcessing:
    def __init__(self):
        pass
