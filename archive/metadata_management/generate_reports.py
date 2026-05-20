"""
Error validation and status report generation script.

This script:
1. Runs data validation using DataValidator (exports error reports)
2. Generates status reports using StatusBoard (deployment & survey status)

Results are exported either locally or to S3.
File differences are cached in DataValidator to avoid duplicate S3 calls.
"""

from sftk.data_validator import DataValidator
from sftk.status_board import StatusBoard

if __name__ == "__main__":
    # Run data validation (file_presence=True exports missing/extra file lists)
    # File differences are cached internally to avoid duplicate S3 calls
    dv = DataValidator()
    dv.run_validation(
        remove_duplicates=True,
        extract_clean_dataframes=True,
        file_presence=True,
    )

    # Generate and export status reports
    # Reuses the same DataValidator instance (and its cached file differences)
    sb = StatusBoard(data_validator=dv)
    sb.run()
