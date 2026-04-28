import os
import logging
from typing import Dict, List
import boto3
from dotenv import load_dotenv
import pandas as pd
import re


# S3 configuration
def load_aws_credentials(env_path):
    """
    Load AWS credentials from .env file.

    Args:
        env_path (str, optional): Path to .env file.
                                  If None, uses default dotenv behavior.

    Returns:
        tuple: AWS access key and secret key
    """
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    return (
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("S3_BUCKET"),
    )


def create_s3_client(access_key, secret_key):
    """
    Create and return an S3 client.

    Args:
        access_key (str): AWS access key
        secret_key (str): AWS secret key

    Returns:
        boto3.client: S3 client
    """
    try:
        return boto3.client(
            "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
    except Exception as e:
        logging.error(f"Failed to create S3 client: {e}")
        raise ValueError("Unable to create S3 client. Check AWS credentials.")


def list_csv_files_in_prefix(client, bucket, prefix):
    """
    List all CSV files in a given S3 prefix.

    Args:
        client (boto3.client): S3 client
        bucket (str): S3 bucket name
        prefix (str): S3 prefix path

    Returns:
        List[str]: List of CSV file keys
    """
    try:
        csv_files = []
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.csv'):
                        csv_files.append(key)
        
        logging.info(f"Found {len(csv_files)} CSV files in prefix '{prefix}'")
        return csv_files
    
    except Exception as e:
        logging.error(f"Failed to list CSV files in prefix '{prefix}': {e}")
        raise


def sanitize_dataframe_name(file_key):
    """
    Convert S3 file key to a PowerBI-friendly dataframe name.
    
    Args:
        file_key (str): S3 file key (e.g., 'spyfish_metadata/status/file_name.csv')
    
    Returns:
        str: Sanitized name suitable for PowerBI (e.g., 'status_file_name')
    """
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(file_key))[0]
    
    # Get the parent directory name
    parent_dir = os.path.basename(os.path.dirname(file_key))
    
    # Combine parent directory and filename
    # Replace any non-alphanumeric characters with underscores
    combined_name = f"{parent_dir}_{filename}"
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', combined_name)
    
    # Remove consecutive underscores and trailing/leading underscores
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    
    return sanitized


def read_csv_from_s3(client, bucket, key):
    """
    Read a CSV file from S3 and return as a pandas DataFrame.

    Args:
        client (boto3.client): S3 client
        bucket (str): S3 bucket name
        key (str): S3 object key

    Returns:
        pd.DataFrame: DataFrame from CSV file
    """
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"], low_memory=False)

    except Exception as e:
        logging.error(f"Failed to process S3 file {key}: {e}")
        raise IOError(f"Could not download or read file {key}")


def main(env_path=None):
    """
    Main function to orchestrate S3 file download and processing.
    Dynamically loads all CSV files from specified S3 prefixes.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes keyed by sanitized names
    """
    # S3 prefixes to scan for CSV files
    csv_prefixes = [
        "spyfish_metadata/status/",
    ]

    # Load AWS credentials and create S3 client
    try:
        access_key, secret_key, bucket_name = load_aws_credentials(env_path)
        logging.info(f"AWS Credentials loaded. Bucket: {bucket_name}")
    except Exception as cred_error:
        logging.error(f"Credential loading failed: {cred_error}")
        raise

    # Create S3 client
    try:
        s3_client = create_s3_client(access_key, secret_key)
    except Exception as client_error:
        logging.error(f"S3 client creation failed: {client_error}")
        raise

    # Discover all CSV files in the specified prefixes
    all_csv_files = []
    for prefix in csv_prefixes:
        try:
            csv_files = list_csv_files_in_prefix(s3_client, bucket_name, prefix)
            all_csv_files.extend(csv_files)
        except Exception as list_error:
            logging.warning(f"Failed to list files in prefix '{prefix}': {list_error}")
            # Continue with other prefixes even if one fails

    if not all_csv_files:
        logging.warning("No CSV files found in any of the specified prefixes")
        return {}

    # Read all CSV files and store in a dictionary of DataFrames
    dataframes = {}
    for csv_key in all_csv_files:
        try:
            df_name = sanitize_dataframe_name(csv_key)
            logging.info(f"Reading '{csv_key}' as '{df_name}'")
            dataframes[df_name] = read_csv_from_s3(s3_client, bucket_name, csv_key)
            logging.info(f"Successfully read '{df_name}'. Shape: {dataframes[df_name].shape}")
        except Exception as read_error:
            logging.error(f"Failed to read {csv_key}: {read_error}")
            # Continue with other files even if one fails

    logging.info(f"Successfully loaded {len(dataframes)} dataframes")
    return dataframes
