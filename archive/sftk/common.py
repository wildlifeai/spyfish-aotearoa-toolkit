import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from sftk.utils import str_to_bool


def load_env_wrapper() -> None:
    """
    Wrapper for loading environment variables from a .env file.

    Guard clause added to prevent loading environment variables when running in
    a GitHub Actions environment.
    """
    if os.getenv("GITHUB_ACTIONS") == "true":
        return

    env_path = find_dotenv()
    # Check if the file exists before trying to load it
    if env_path:
        logging.info(f"Loading .env file from: {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        logging.warning(
            f".env file not found at '{env_path}'. Environment variables might not be loaded."
        )


load_env_wrapper()


# General settings
DEV_MODE = os.getenv("DEV_MODE")
EXPORT_LOCAL = str_to_bool(os.getenv("EXPORT_LOCAL"))

LOCAL_DATA_FOLDER_PATH = os.getenv("LOCAL_DATA_FOLDER_PATH", str(Path.cwd() / "data"))


# Email configuration
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993
EMAIL_ARCHIVE_FOLDER_CANDIDATES = [
    "[Gmail]/All Mail",  # Gmail
    "Archive",  # Common IMAP
    "Archived",  # Alternative naming
    "INBOX.Archive",  # Some email systems
]
EMAIL_ARCHIVE_FOLDER = "[Gmail]/All Mail"


# Biigle credentials and configuration
BIIGLE_API_EMAIL = os.getenv("BIIGLE_API_EMAIL")
BIIGLE_API_TOKEN = os.getenv("BIIGLE_API_TOKEN")
BIIGLE_PROJECT_ID = int(
    os.getenv("BIIGLE_PROJECT_ID", "3711")
)  # Spyfish Aotearoa project
BIIGLE_DISK_ID = int(os.getenv("BIIGLE_DISK_ID", "134"))  # S3 bucket reference


# BIIGLE report type IDs
BIIGLE_ANNOTATION_REPORT_TYPE = 8
BIIGLE_VOLUME_REPORT_TYPE = 10


# S3 configuration.
# TODO check ways to set variables, if there are issues reading the .env file
# For example ask for user input if the env variables are not found.
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

S3_SPYFISH_METADATA = os.path.join("spyfish_metadata")
S3_SHAREPOINT_PATH = os.path.join(S3_SPYFISH_METADATA, "sharepoint_lists")
# Both 'Deployment' and 'Movies' exist because 'Deployment' is the term used in SharePoint,
# while 'Movies' is the equivalent term used in KSO (for example, accessed via keyword in
# the Sharepoint to kso copy workflow)
S3_SHAREPOINT_DEPLOYMENT_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Deployment.csv")
S3_SHAREPOINT_MOVIE_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Deployment.csv")
S3_SHAREPOINT_DEFINITIONS_CSV = os.path.join(
    S3_SHAREPOINT_PATH, "BUV Metadata Definitions.csv"
)
S3_SHAREPOINT_SITE_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Survey Sites.csv")
S3_SHAREPOINT_SPECIES_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Species.csv")
S3_SHAREPOINT_SURVEY_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Survey Metadata.csv")
S3_SHAREPOINT_RESERVES_CSV = os.path.join(S3_SHAREPOINT_PATH, "Marine Reserves.csv")
S3_SHAREPOINT_TEST_CSV = os.path.join(S3_SHAREPOINT_PATH, "Test.csv")

# S3 KSO Files
S3_KSO_PATH = os.path.join(S3_SPYFISH_METADATA, "kso_csvs")
S3_KSO_ANNOTATIONS_CSV = os.path.join(S3_KSO_PATH, "annotations_buv_doc.csv")
S3_KSO_MOVIE_CSV = os.path.join(S3_KSO_PATH, "movies_buv_doc.csv")
S3_KSO_SITE_CSV = os.path.join(S3_KSO_PATH, "sites_buv_doc.csv")
S3_KSO_SPECIES_CSV = os.path.join(S3_KSO_PATH, "species_buv_doc.csv")
S3_KSO_SURVEY_CSV = os.path.join(S3_KSO_PATH, "surveys_buv_doc.csv")
S3_KSO_TEST_CSV = os.path.join(S3_KSO_PATH, "test_buv_doc.csv")


# Status board output files
S3_STATUS_PATH = os.path.join(S3_SPYFISH_METADATA, "status")

# Filenames (shared between local and S3)
DEPLOYMENT_STATUS_FILENAME = "deployments_status.csv"
SURVEY_STATUS_FILENAME = "surveys_status.csv"
ERRORS_FILENAME = "data_errors.csv"
MISSING_FILES_FILENAME = "missing_files_in_aws.txt"
EXTRA_FILES_FILENAME = "extra_files_in_aws.txt"

# S3 paths
S3_DEPLOYMENT_STATUS_CSV = os.path.join(S3_STATUS_PATH, DEPLOYMENT_STATUS_FILENAME)
S3_SURVEY_STATUS_CSV = os.path.join(S3_STATUS_PATH, SURVEY_STATUS_FILENAME)
S3_ERRORS_CSV = os.path.join(S3_STATUS_PATH, ERRORS_FILENAME)
S3_MISSING_FILES = os.path.join(S3_STATUS_PATH, MISSING_FILES_FILENAME)
S3_EXTRA_FILES = os.path.join(S3_STATUS_PATH, EXTRA_FILES_FILENAME)


# Specific Column names used in Sharepoint
# TODO create variables for all columns used below
# TODO check templates that use these column names hardcoded
DROP_ID_COLUMN = "DropID"
SURVEY_ID_COLUMN = "SurveyID"
SITE_ID_COLUMN = "SiteID"
REPLICATE_COLUMN = "ReplicateWithinSite"
FILE_NAME_COLUMN = "FileName"
LINK_TO_MARINE_RESERVE_COLUMN = "LinkToMarineReserve"
SITE_NAME_COLUMN = "SiteName"


# Keywords to monitor in email
KEYWORDS = [
    "survey",
    "site",
    "movie",
    "species",
]
KEYWORD_LOOKUP = {
    "survey": SURVEY_ID_COLUMN,
    "site": SITE_ID_COLUMN,
    "movie": DROP_ID_COLUMN,
    "species": "ScientificName",
}

MOVIE_EXTENSIONS = [
    "avi",
    "mov",
    "mp4",
    "mpg",
    "wmv",
]


VALIDATION_RULES = {
    "deployments": {
        "file_name": S3_SHAREPOINT_DEPLOYMENT_CSV,
        # TODO add fps, sampling start and end etc.
        "required": [
            DROP_ID_COLUMN,
            SURVEY_ID_COLUMN,
            SITE_ID_COLUMN,
            # TODO remove? Should be covered by file presence check
            # FILE_NAME_COLUMN,
            # "LinkToVideoFile",
        ],
        "unique": [DROP_ID_COLUMN],
        "info_columns": [SURVEY_ID_COLUMN, SITE_ID_COLUMN],
        "foreign_keys": {"surveys": SURVEY_ID_COLUMN, "sites": SITE_ID_COLUMN},
        "relationships": [
            {
                "column": DROP_ID_COLUMN,
                "rule": "equals",
                "template": "{SurveyID}_{SiteID}_{ReplicateWithinSite:02}",
            },
            {
                "column": FILE_NAME_COLUMN,
                "rule": "equals",
                "template": "{DropID}.mp4",
                "allowed_values": ["NO VIDEO BAD DEPLOYMENT"],
                # TODO: Remove null allowed
                "allow_null": True,
            },
            {
                "column": "LinkToVideoFile",
                "rule": "equals",
                "template": "media/{SurveyID}/{DropID}/{DropID}.mp4",
                "allowed_values": ["NO VIDEO BAD DEPLOYMENT"],
                # TODO: Remove null allowed
                "allow_null": True,
            },
        ],
        "values": [
            # TODO the Lat and Long ranges are approximate,correct area
            {
                "column": "Latitude",
                "rule": "value_range",
                "range": [-46, -36],
                "allowed_values": [0],
            },
            {
                "column": "Longitude",
                "rule": "value_range",
                "range": [170, 178.5],
                "allowed_values": [0],
            },
        ],
    },
    "surveys": {
        "file_name": S3_SHAREPOINT_SURVEY_CSV,
        "required": [SURVEY_ID_COLUMN],
        "unique": [SURVEY_ID_COLUMN],
        "info_columns": ["SurveyName"],
        "formats": [SURVEY_ID_COLUMN],
        # TODO it flags the missing surveys in Deployments,
        # maybe ok even tho it technically isn't a foreign key
        "foreign_keys": {
            # "deployments" : "SurveyID",
        },
        "relationships": [],
    },
    "sites": {
        "file_name": S3_SHAREPOINT_SITE_CSV,
        "required": [SITE_ID_COLUMN, LINK_TO_MARINE_RESERVE_COLUMN],
        "unique": [SITE_ID_COLUMN],
        "info_columns": [SITE_NAME_COLUMN, LINK_TO_MARINE_RESERVE_COLUMN],
        "formats": [SITE_ID_COLUMN],
        "foreign_keys": {},
        "relationships": [],
    },
    "species": {
        "file_name": S3_SHAREPOINT_SPECIES_CSV,
        "required": ["AphiaID", "CommonName", "ScientificName"],
        "unique": [
            "AphiaID",
            "ScientificName",
            "CommonName",
        ],
        "info_columns": ["AphiaID", "CommonName", "ScientificName"],
        "foreign_keys": {},
        "relationships": [],
    },
    "reserves": {
        "file_name": S3_SHAREPOINT_RESERVES_CSV,
        "required": [],
        "unique": [],
        "info_columns": [],
        "foreign_keys": {},
        "relationships": [],
    },
}

# File presence validation rules configuration
# Dictionary containing configuration for validating file presence in S3 against CSV references
FILE_PRESENCE_RULES = {
    "file_presence": {
        "bucket": S3_BUCKET,
        "s3_sharepoint_path": S3_SHAREPOINT_PATH,
        "csv_filename": "BUV Deployment.csv",
        "csv_column_to_extract": "LinkToVideoFile",
        # "column_filter": "IsBadDeployment",
        # "column_value": False,
        "column_filter": None,
        "column_value": None,
        "valid_extensions": MOVIE_EXTENSIONS,
        "path_prefix": "media",
    }
}


VALIDATION_PATTERNS = {
    DROP_ID_COLUMN: r"^[A-Z]{3}_\d{8}_BUV_[A-Z]{3}_\d{3}_\d{2}$",
    SURVEY_ID_COLUMN: r"^[A-Z]{3}_\d{8}_BUV$",
    SITE_ID_COLUMN: r"^[A-Z]{3}_\d+$",
}
