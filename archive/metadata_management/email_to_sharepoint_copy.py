"""
This module processes incoming emails, checks for specific keywords,
and stores relevant data in an AWS S3 bucket.

Functions:
    - process_emails: Fetch and process emails for given keywords.
    - append_new_obs_to_csv: Append new observations to a CSV file in S3.
"""

import os
import re
from io import StringIO
import logging
from typing import List, Dict, Tuple
import imaplib
import email
import json

import pandas as pd
import boto3

from tqdm import tqdm
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Load environment variables from .env file only in local environment
if os.environ.get("GITHUB_ACTIONS") != "true":
    load_dotenv()

# Email configuration
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

# S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

S3_SHAREPOINT_MOVIE_CSV = os.getenv("S3_SHAREPOINT_MOVIE_CSV")
S3_SHAREPOINT_SURVEY_CSV = os.getenv("S3_SHAREPOINT_SURVEY_CSV")
S3_SHAREPOINT_SITE_CSV = os.getenv("S3_SHAREPOINT_SITE_CSV")
S3_SHAREPOINT_SPECIES_CSV = os.getenv("S3_SHAREPOINT_SPECIES_CSV")


def get_inbox(keywords: List[str]) -> Tuple[imaplib.IMAP4_SSL, Dict[str, List[str]]]:
    """
    Fetches unread emails from the inbox that contain any of the specified keywords in the subject.

    Args:
        keywords (list): A list of keywords to search for in email subjects.

    Returns:
        A tuple containing:
            - IMAP connection object (imaplib.IMAP4_SSL)
            - Dictionary mapping keywords to lists of corresponding unread email IDs (str)
                (None if no unread emails found for any keyword)
    """
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        keyword_emails = {keyword: [] for keyword in keywords}

        for keyword in keywords:
            status, data = mail.search(None, f'(UNSEEN SUBJECT "{keyword}")')

            # Check for None or empty result
            if status != "OK" or not data or not data[0]:
                logging.info(
                    "No emails found with the keyword '%s'.",
                    keyword,
                )
                continue  # Skip this keyword if no emails found or error

            unread_email_ids = data[0].split()
            keyword_emails[keyword].extend(unread_email_ids)

        # Check if there are any unseen messages
        if all(
            len(unread_email_ids) == 0 for unread_email_ids in keyword_emails.values()
        ):
            logging.info("No unseen messages found for the specified keywords.")
            return mail, None

        return mail, keyword_emails

    except imaplib.IMAP4.error as e:
        logging.error("IMAP error occurred: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while fetching inbox: %s", e)
        raise


def get_email_body(mail: imaplib.IMAP4_SSL, mail_id: str) -> str:
    """
    Retrieves the body of an email and attempts to extract JSON content from it.

    Args:
        mail (imaplib.IMAP4_SSL): The IMAP connection object.
        mail_id (str): The ID of the email.

    Returns:
        str: The extracted JSON content from the email body, if available.
        Raises a ValueError if no body is found or no JSON content is extracted.
    """
    try:
        _, data = mail.fetch(mail_id, "(RFC822)")

        # Check if the data is None or empty
        if not data or not data[0]:
            logging.error("No data returned for email ID: %s", mail_id)
            raise ValueError(f"Failed to retrieve email content for mail ID: {mail_id}")

        raw_email = data[0][1]
        msg = email.message_from_bytes(raw_email)
        body = ""

        # Check if the email is multipart or not
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                # Extract body from text/plain parts
                if (
                    content_type == "text/plain"
                    and "attachment" not in content_disposition
                ):
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        # Ensure the body is not None or empty
        if body is None:
            logging.error("No body found in email ID: %s", mail_id)
            raise ValueError(f"No body found in email ID: {mail_id}")

        # Attempt to extract JSON content from the email body
        json_match = re.search(r"({.*})", body, re.DOTALL)

        # If no JSON match, log the error and raise a specific exception
        if json_match:
            json_str = json_match.group(1)
            return json_str
        else:
            logging.error(
                "No JSON content found in email body for mail ID: %s", mail_id
            )
            raise ValueError(
                f"No JSON content found in email body for mail ID: {mail_id}"
            )

    except Exception as e:
        logging.error("Error retrieving email body for mail ID %s: %s", mail_id, e)
        raise


def clean_sharepoint_data(json_str: str) -> pd.DataFrame:
    """
    Cleans and parses JSON data from SharePoint.

    Args:
        json_str (str): The JSON string to parse.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned data.
    """
    try:
        # Parse JSON string
        data = json.loads(json_str)

        # Get the first item from the 'value' list
        item = data["value"][0]

        # Function to extract value from SharePoint field
        def extract_value(field):
            if isinstance(field, dict):
                if "Value" in field:
                    return field["Value"]
                elif "DisplayName" in field:
                    return field["DisplayName"]
                else:
                    return str(field)
            elif isinstance(field, list):
                # Handle list fields (like Reser)
                if field and isinstance(field[0], dict) and "Value" in field[0]:
                    return field[0]["Value"]
            return field

        # Create cleaned dictionary
        cleaned_data = {}

        for key, value in item.items():
            # Skip system fields and metadata
            if (
                not key.startswith("{")
                and not key.startswith("@")
                and "#" not in key
                and not key.endswith("@odata.type")
            ):
                cleaned_data[key] = extract_value(value)

        # Convert to DataFrame
        df = pd.DataFrame([cleaned_data])
        return df

    except json.JSONDecodeError as e:
        logging.error("JSON parsing error: %s", e)
        raise


def get_s3_client() -> boto3.client:
    """
    Creates and returns an S3 client.

    Returns:
        boto3.client: The S3 client.
    """
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def download_object_from_s3(
    client: boto3.client, bucket: str, key: str, filename: str, version_id: str = None
) -> None:
    """
    Downloads an object from S3.

    Args:
        client (boto3.client): The S3 client.
        bucket (str): The S3 bucket name.
        key (str): The S3 object key.
        filename (str): The local filename to save the object to.
        version_id (str, optional): The version ID of the object. Defaults to None.
    """
    try:
        kwargs = {"Bucket": bucket, "Key": key}
        if version_id:
            kwargs["VersionId"] = version_id

        object_size = client.head_object(**kwargs)["ContentLength"]

        def progress_update(bytes_transferred):
            pbar.update(bytes_transferred)

        with tqdm(total=object_size, unit="B", unit_scale=True, desc=filename) as pbar:
            client.download_file(
                Bucket=bucket,
                Key=key,
                Filename=filename,
                Callback=progress_update,
                Config=boto3.s3.transfer.TransferConfig(use_threads=False),
            )
    except Exception as e:
        logging.error("Failed to download %s from S3: %s", key, e)
        raise


def standarise_new_obs_df(
    new_obs_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    keyword: str,
) -> pd.DataFrame:
    """
    Compare a DataFrame of new observations to match the format of the
    sharepoint copy uploaded to S3.

    Args:
        new_obs_df (pd.DataFrame): The DataFrame with new observations.
        s3_df (pd.DataFrame): The standardised DataFrame from S3.
        keyword (str): Identifier for the type of data in new_obs_df.

    Returns:
        pd.DataFrame: A standardised DataFrame.
    """

    if s3_df is None or new_obs_df is None:
        logging.warning(
            "Cannot compare DataFrames for %s: One or both DataFrames are None", keyword
        )
        return s3_df

    # Define mappings based on keyword
    if keyword == "survey":
        dict_lookup = {
            "Title": "SurveyName",
            "Encoder": "EncoderName",
            "Reser": "LinkToMarineReserve",
            "Reser_x003a_Reserve_x0020_Code": "Reserve Code",
            "BaitSpecies": "Bait Species",
            "BaitAmount": "Bait Amount",
        }
    elif keyword == "site":
        dict_lookup = {
            "Title": "SiteID",
            "field_5": "SiteName",
            "field_6": "SiteCode",
            "field_7": "SiteExposure",
            "field_8": "ProtectionStatus",
            "field_14": "Latitude",
            "field_15": "Longitude",
            "field_16": "geodeticDatum",
            "Reserve1": "Reserve",
            "ControlToMR": "ControlToMR1",
            "Country": "countryCode",
        }
    elif keyword == "movie":
        dict_lookup = {
            "Title": "DropID",
            "Notes": "NotesDeployment",
            "Survey": "SurveyID",
            "Site": "SiteID",
            "Author": "Created By",
        }
    elif keyword == "species":
        dict_lookup = {"Title": "DOC_TaxonID"}
    else:
        logging.error("keyword: %s not found", keyword)
        raise ValueError(f"Unknown keyword: {keyword}")

    # Rename the column names in new_obs_df based on the dictionary
    new_obs_df = new_obs_df.rename(columns=dict_lookup)

    # Check both datasets have the same column names
    missing_columns = set(s3_df.columns) - set(new_obs_df.columns)
    extra_columns = set(new_obs_df.columns) - set(s3_df.columns)

    if extra_columns:
        logging.info(
            "THe new observation has the following extra columns and will be dropped: %s",
            extra_columns,
        )
        # Only drop columns that actually exist in new_obs_df
        new_obs_df = new_obs_df.drop(columns=extra_columns)

    if missing_columns:
        logging.info(
            "Sharepoint list is missing the following columns and empty values will be added %s.",
            missing_columns,
        )
        # Add any missing columns from target_columns, setting them to None or NaN
        for col in missing_columns:
            new_obs_df[col] = None

    else:
        logging.info("Both s3_df and new_obs_df have the same column names")

    # Ensure both DataFrames have the same columns
    common_columns = list(set(s3_df.columns) & set(new_obs_df.columns))
    if len(common_columns) != len(new_obs_df.columns):
        raise ValueError("s3_df and new_obs_df still have different columns.")

    # Return updated DataFrame (new_obs version)
    return new_obs_df.copy()


def append_new_obs_to_csv(
    new_obs_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    keyword: str,
    s3_client: boto3.client,
    s3_file_name: str,
):
    """
    Append new observations to an existing CSV file in S3 bucket.

    Args:
        new_obs_df (pd.DataFrame): DataFrame containing new observations.
        keyword (str): Keyword associated with the CSV file.
    """

    try:
        if not new_obs_df.empty and not s3_df.empty:
            # Replace NaN or NA values with empty strings in new_obs_df and s3_df
            new_obs_df = new_obs_df.fillna("")
            s3_df = s3_df.fillna("")

            # Update existing rows and append new ones
            s3_df = pd.concat(
                [s3_df[~s3_df[keyword].isin(new_obs_df[keyword])], new_obs_df]
            )

            # Save the updated DataFrame to a CSV file in-memory
            csv_buffer = StringIO()
            s3_df.to_csv(csv_buffer, index=False)

            # Upload the updated CSV file to the S3 bucket
            s3_client.put_object(
                Bucket=S3_BUCKET, Key=s3_file_name, Body=csv_buffer.getvalue()
            )
            logging.info("Successfully updated %s file in S3.", keyword)
        else:
            logging.warning("One of the DataFrames is empty; skipping concatenation.")

    except Exception as e:
        logging.error("Error appending new observations to CSV for %s: %s", keyword, e)
        raise


def mark_as_read_and_archive(mail, mail_id):
    """
    Mark an email as read and move it to the archive folder.

    Args:
        mail (imaplib.IMAP4_SSL): The IMAP connection object.
        mail_id (bytes or str): The ID of the email to process (bytes from IMAP response, str otherwise).
    """
    try:
        # Ensure mail_id is in the correct string format
        mail_id = mail_id.decode() if isinstance(mail_id, bytes) else str(mail_id)

        # List potential archive folder names
        archive_folders = [
            "[Gmail]/All Mail",  # Gmail
            "Archive",  # Common IMAP
            "Archived",  # Alternative naming
            "INBOX.Archive",  # Some email systems
        ]

        # Try to find a valid archive folder
        valid_archive_folder = None
        for folder in archive_folders:
            try:
                # Check if the folder exists
                status, folder_list = mail.list(f'"{folder}"')
                if status == "OK" and folder_list:
                    valid_archive_folder = folder
                    break
            except Exception as folder_check_error:
                logging.warning(
                    f"Could not check folder {folder}: {folder_check_error}"
                )

        if not valid_archive_folder:
            logging.error("No valid archive folder found")
            raise ValueError("Cannot find a suitable archive folder")
        # Mark the email as read
        mail.store(mail_id, "+FLAGS", "\\Seen")

        # Specify the Gmail archive folder explicitly, or set this according to
        # your IMAP folder structure
        archive_folder = "[Gmail]/All Mail"

        # Perform the copy to the archive folder
        result = mail.copy(mail_id, archive_folder)
        if result[0] != "OK":
            logging.error("Error moving email to %s: %s", archive_folder, result[1])
            raise RuntimeError("Error copying email to %s", archive_folder)

        # Mark the original email for deletion and expunge
        mail.store(mail_id, "+FLAGS", "\\Deleted")
        mail.expunge()

        logging.info(
            "Email %s marked as read and archived to %s", mail_id, archive_folder
        )

    except Exception as e:
        logging.error(
            "Email %s couldn't be marked as read and archived: %s", mail_id, e
        )
        raise


def main():
    """Main function to process emails and update S3."""
    logging.info("Starting main function")
    keywords = ["survey", "site", "movie", "species"]
    keyword_lookup = {
        "survey": "SurveyID",
        "site": "SiteID",
        "movie": "DropID",
        "species": "ScientificName",
    }
    logging.info(f"Processing keywords: {keywords}")

    try:
        logging.info("Initializing S3 client...")
        s3_client = get_s3_client()
        logging.info("S3 client initialized successfully")

        logging.info("Processing inbox mails...")
        mail, keyword_emails = get_inbox(keywords)
        logging.info("Inbox processed")

        # Check if there are unseen messages
        if keyword_emails is None:
            logging.info("No new emails found. Stopping further processing.")
        else:
            for keyword, unread_email_ids in keyword_emails.items():
                logging.info(
                    f"\nProcessing data from {len(unread_email_ids)} mails from {keyword} data..."
                )

                df_list = []

                for mail_id in unread_email_ids:
                    try:
                        body = get_email_body(mail, mail_id)
                        print(body)
                        new_obs_df = clean_sharepoint_data(body)
                        print(body)
                        df_list.append(new_obs_df)
                        # mark_as_read_and_archive(mail, mail_id)
                    except (ValueError, TypeError) as e:
                        logging.error(
                            "Error processing email %s for keyword %s: %s",
                            mail_id,
                            keyword,
                            e,
                        )
                        continue
                    except Exception as e:
                        logging.error(
                            "Unexpected error processing email %s for keyword %s: %s",
                            mail_id,
                            keyword,
                            e,
                        )
                        continue

                if df_list:
                    logging.info(f"\nProcessing new_obs_df for {keyword} data...")
                    # Concatenate all DataFrames in the list
                    new_obs_df_from_list = pd.concat(df_list, axis=0, ignore_index=True)

                    # Select the latest changes if multiple emails were generated per observation
                    new_obs_df = new_obs_df_from_list.loc[
                        new_obs_df_from_list.groupby("ItemInternalId")[
                            "Modified"
                        ].idxmax()
                    ]
                    logging.info(
                        f"\n{new_obs_df.shape[0]} observations processed for {keyword}",
                        f" from {new_obs_df_from_list.shape[0]} entries...",
                    )
                else:
                    logging.error(
                        "Skipping the emails for keyword: %s due to an empty df",
                        keyword,
                    )
                    continue

                try:
                    logging.info(
                        f"\nRetrieving sharepoint copy for {keyword} keyword from S3..."
                    )
                    s3_file_name = os.getenv(f"S3_SHAREPOINT_{keyword.upper()}_CSV")
                    if not s3_file_name:
                        raise ValueError(
                            f"Environment variable '{s3_file_name}' not set."
                        )

                    try:
                        download_object_from_s3(
                            client=s3_client,
                            bucket=S3_BUCKET,
                            key=s3_file_name,
                            filename="temp.csv",
                        )
                        s3_df = pd.read_csv("temp.csv")

                    except Exception as e:
                        logging.warning("CSV file not found in S3: %s", e)
                except Exception as e:
                    logging.warning(
                        "Found an error when trying to download the CSV from S3: %s",
                        e,
                    )

                logging.info(
                    "Updating column names of new observations to match those"
                    " from the sharepoint copy uploaded to S3..."
                )
                logging.info(f"new_obs_df DataFrame shape: {new_obs_df.shape}")
                logging.info(f"s3_df DataFrame shape: {s3_df.shape}")
                # Format the new_obs_df to match the s3_df colnames
                obs_df = standarise_new_obs_df(new_obs_df, s3_df, keyword)
                logging.info("Column names of new observations updated...")

                logging.info("Adding new observations for %s:%s", keyword, obs_df)
                # Map the keyword id
                keyword = keyword_lookup[keyword]

                append_new_obs_to_csv(obs_df, s3_df, keyword, s3_client, s3_file_name)
                logging.info(
                    "New observations for %s added to the sharepoint copy in S3",
                    keyword,
                )

        mail.logout()
        logging.info("Email processing completed successfully")
    except Exception as e:
        logging.error("An error occurred during email processing: %s", e)


if __name__ == "__main__":
    # Ensure logging is configured at the start
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,  # This will override any existing logging configuration
    )
    logging.info("Script started")
    main()
    logging.info("Script completed")
