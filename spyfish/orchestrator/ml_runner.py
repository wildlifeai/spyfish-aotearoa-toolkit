import logging
import sqlite3
import pandas as pd
import subprocess
import os
import argparse
from typing import List
from pathlib import Path

from spyfish.config import config, PipelineStatus, get_required
from spyfish.storage.s3_handler import S3Handler
from spyfish.database.manager import DatabaseManager
from spyfish.storage.db_sync import download_db, upload_db

class MLRunner:
    def __init__(self):
        self.is_test_run = config.is_test_run
        self.is_local = config.is_local

        # S3 properties
        self.bucket = get_required(config.storage, "bucket_name", "storage")
        self.s3_db_key = get_required(config.orchestrator, "s3_db_key", "orchestrator")
        self.s3 = S3Handler(bucket=self.bucket)
        self.db = DatabaseManager()

        # Local state properties
        self.local_db_path = config.db_path
        self.manifest_dir = config.local_manifest_dir_path
        self.manifest_path = os.path.join(self.manifest_dir, config.local_manifest_name)
        self.nesi_video_dir = get_required(config.orchestrator, "nesi_video_dir", "orchestrator")

        # Logic properties
        self.limit = get_required(config.ml_inference, "limit_processing", "ml_inference")
        self.frame_skip = get_required(config.ml_inference, "frame_skip", "ml_inference")
        self.confidence = get_required(config.ml_inference, "confidence_threshold", "ml_inference")
        self.model = config.mock_model_path if self.is_local else get_required(config.ml_inference, "model_path", "ml_inference")

    def sync_down(self):
        """Downloads the master pipeline database from S3."""
        if self.is_test_run:
            return
        download_db()

    def generate_manifest(self) -> List[str]:
        """Queries the local sqlite DB for videos READY_FOR_ML and creates the target CSV manifest."""
        logging.info(f"Querying local state database for READY_FOR_ML videos (LIMIT: {self.limit})...")

        # Dummy behavior during dry run if we don't have the DB
        if self.is_test_run and not os.path.exists(self.local_db_path):
            logging.info("Test run: Skipping DB query because DB file was not downloaded.")
            return ["DRY_RUN_ID_1"]

        conn = sqlite3.connect(self.local_db_path)

        drop_id_col = config.csv_mapping.get('drop_id_column', 'DropID')
        video_link_col = config.csv_mapping.get('video_file_link_column', 'LinkToVideoFile')

        test_drops = config.test_drops
        test_drop_ids = [t[0] for t in test_drops]
        if test_drop_ids:
            placeholders = ','.join(['?'] * len(test_drop_ids))
            logging.info(f"TEST MODE ACTIVE: Forcing manifest to include {drop_id_col}: {test_drop_ids}")
            query = f"""
                SELECT drop_id as {drop_id_col},
                       video_path as {video_link_col},
                       sampling_start as SamplingStart,
                       sampling_end as SamplingEnd
                FROM deployments
                WHERE drop_id IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=test_drop_ids)
        else:
            df = pd.DataFrame(self.db.get_deployments_by_status(PipelineStatus.READY_FOR_ML))
            if not df.empty:
                df = df.head(self.limit)

        if df.empty:
            logging.info("No drops available for ML processing. Exiting.")
            return []

        # Rename columns to match what the rest of the method expects (BUV Deployment CSV style)
        df = df.rename(columns={
            "drop_id": drop_id_col,
            "video_path": video_link_col,
            "sampling_start": "SamplingStart",
            "sampling_end": "SamplingEnd"
        })

        # On NeSI, it is significantly faster and more robust to rsync the files natively
        # using the aws cli instead of generating presigned URLs (since Slurm jobs may sit in queue for days)
        # TODO: This downloads the videos *synchronously* in the Python orchestrator loop.
        # As tech debt cleanup, this should eventually be moved OUT of the orchestrator Python logic
        # into a native Snakemake Rule (e.g. `rule download_videos`) so that Slurm can
        # parallelize the file transfer step across multiple nodes instead of the head node blocking.
        logging.info("Generating target paths and downloading media via aws s3...")

        # In a local run, '/nesi/' doesn't exist. Safely mock it to prevent MacOS read-only errors.
        actual_video_dir = self.nesi_video_dir if not (self.is_test_run or self.is_local) else "mock_media"
        os.makedirs(actual_video_dir, exist_ok=True)

        local_filepaths = []
        for path in df[video_link_col]:
            filename = os.path.basename(path)
            local_path = os.path.join(actual_video_dir, filename)

            s3_uri = f"s3://{self.bucket}/{path}"

            # Skip if the file already exists locally
            if os.path.exists(local_path):
                logging.info(f"Video {filename} already exists at {local_path}. Skipping download.")
                local_filepaths.append(local_path)
                continue

            logging.info(f"Downloading {s3_uri} to {local_path}...")

            try:
                # Execute aws s3 cp natively
                # --no-progress prevents log bloat
                subprocess.run(
                    ["aws", "s3", "cp", s3_uri, local_path, "--only-show-errors"],
                    check=True
                )
                local_filepaths.append(local_path)
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to download video {s3_uri}: {e}")
                # If download fails, we shouldn't pass it to snakemake
                continue

        df["VideoURL"] = local_filepaths

        # Save manifest
        logging.info(f"Writing {len(df)} target records to {self.manifest_path}")
        # Make sure directory exists
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        df.to_csv(self.manifest_path, index=False)

        return df[drop_id_col].tolist()

    def run_snakemake(self, targets: List[str]) -> List[str]:
        """Executes the Snakemake workflow based on the targets"""
        if not targets:
            return []

        logging.info(f"Setting {len(targets)} targets to {PipelineStatus.PROCESSING_ML}...")
        logging.info(f"Setting {len(targets)} targets to {PipelineStatus.PROCESSING_ML}...")
        if not self.is_test_run:
            # Batch update status to PROCESSING_ML
            download_db()
            for target in targets:
                self.db.update_status(target, PipelineStatus.PROCESSING_ML, auto_sync=False)
            upload_db()

        logging.info(f"Starting Snakemake orchestrator for {len(targets)} drops...")

        # Automatically download the YOLO weights from S3 if they don't exist locally
        if not os.path.exists(self.model):
            model_s3_key = config.model_s3_key
            if model_s3_key:
                s3_uri = f"s3://{self.bucket}/{model_s3_key}"
                logging.info(f"Model weights not found at {self.model}. Downloading from {s3_uri} via aws s3 cp...")
                os.makedirs(os.path.dirname(self.model), exist_ok=True)
                try:
                    subprocess.run(
                        ["aws", "s3", "cp", s3_uri, self.model, "--no-progress"],
                        check=True
                    )
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to download model weights {s3_uri}: {e}")
            else:
                logging.warning(f"Model missing locally and 'model_s3_key' not configured in yaml. Inference will likely fail.")

        # Since we use `--directory nesi_pipeline`, relative paths inside Snakemake
        # (like where to find the manifest) must be adjusted if they were passed from the root.
        # We convert the manifest path to absolute so Snakemake finds it regardless of its working directory.
        abs_manifest = os.path.abspath(self.manifest_path)

        cmd = [
            "snakemake",
            "--cores", "1",
            "--snakefile", "nesi_pipeline/Snakefile",
            "--config",
            f"frame_skip={self.frame_skip}",
            f"confidence_threshold={self.confidence}",
            f"model_path={self.model}",
            f"manifest={abs_manifest}",
            f"output_dir={self.manifest_dir}",
            f"log_dir={config.local_logs_dir}"
        ]

        logging.info(f"Executing: {' '.join(cmd)}")

        success_targets = []
        try:
            # We use run to block until the workflow completes
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                logging.info("Snakemake pipeline completed successfully.")
                # For now, we assume all targets in the manifest succeeded if sm returned 0
                # In robust production, we would parse the Snakemake output logs or output files
                # to strictly verify row-by-row success.
                success_targets = targets
        except subprocess.CalledProcessError as e:
            logging.error(f"Snakemake pipeline failed with error code {e.returncode}, {e}")
            # If Snakemake fails, we return an empty success list to ensure no jobs are marked ML_COMPLETED.
            # In a production setting we might want to flag them as ERROR in the DB here so they don't get stuck in PROCESSING_ML
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred during Snakemake execution: {e}")
            return []

        return success_targets

    def sync_up(self, successful_drops: List[str]):
        """Redownloads the DB to avoid race conditions, applies UPDATES, and uploads back to S3"""
        if not successful_drops:
            logging.info("No successful drops to sync. Exiting.")
            return

        logging.info(f"Syncing state for {len(successful_drops)} successfully processed drops...")

        if self.is_test_run:
            logging.info(f"Test run: Would update status to ML_COMPLETED for DropIDs: {successful_drops}")
            # Simulate CSV mock output creation so it tests
            model_name = Path(self.model).stem
            for drop_id in successful_drops:
                local_csv = os.path.join(self.manifest_dir, f"{drop_id}_{model_name}_raw.csv")
                logging.info(f"Test run: Would upload {local_csv} to S3...")
            return

        # Upload generated ML CSVs back up to S3 natively
        model_name = Path(self.model).stem
        for drop_id in successful_drops:
            local_csv = os.path.join(self.manifest_dir, f"{drop_id}_{model_name}_raw.csv")
            if os.path.exists(local_csv):
                s3_key = os.path.join(config.s3_annotations_dir, f"{drop_id}_{model_name}_raw.csv")
                logging.info(f"Uploading ML inferences {local_csv} -> s3://{self.bucket}/{s3_key}")
                self.s3.upload_file_to_s3(local_csv, s3_key)
            else:
                logging.warning(f"Expected to find ML output at {local_csv} but file is missing.")

        # Re-download the DB to ensure we don't overwrite UI changes while Snakemake ran
        download_db()

        # Apply the updates, BUT ONLY if the video is STILL marked as PROCESSING_ML.
        # This protects against cases where a user might have manually excluded or deleted
        # the run from the dashboard while the long ML queue was executing!
        updated_count = 0
        for drop_id in successful_drops:
            record = self.db.get_deployment(drop_id)
            if record and record['status'] == PipelineStatus.PROCESSING_ML:
                self.db.update_status(drop_id, PipelineStatus.ML_COMPLETE, auto_sync=False)
                updated_count += 1

        # Upload back up
        upload_db()
        logging.info(f"Successfully updated {updated_count} rows in local DB to ML_COMPLETE.")
        logging.info("State Sync Complete.")

def main():
    parser = argparse.ArgumentParser(description='Spyfish ML Pipeline Orchestrator')
    args = parser.parse_args()

    runner = MLRunner()
    runner.sync_down()
    targets = runner.generate_manifest()
    successes = runner.run_snakemake(targets)
    runner.sync_up(successes)

if __name__ == "__main__":
    main()
