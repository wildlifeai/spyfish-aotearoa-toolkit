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
from spyfish.ml.run_inference import main as run_inference_main

class MLRunner:
    def __init__(self):
        self.is_test_run = config.is_test_run
        self.is_local = config.is_local

        # S3 properties
        self.bucket = config.s3_bucket
        self.s3_db_key = config.s3_db_key
        self.s3 = S3Handler(bucket=self.bucket)
        self.db = DatabaseManager()

        self.video_storage_dir = config.media_dir
        self.local_db_path = config.db_path

        # Logic properties
        self.limit = get_required(config.ml_inference, "limit_processing", "ml_inference")
        self.frame_skip = get_required(config.ml_inference, "frame_skip", "ml_inference")
        self.imgsz = int(config.imgsz)
        self.confidence = get_required(config.ml_inference, "confidence_threshold", "ml_inference")
        # TODO what is happening here
        self.model = config.pipeline_model_path if self.is_local else get_required(config.ml_inference, "model_path", "ml_inference")


    def get_inference_targets(self) -> List[dict]:
        """Queries the local sqlite DB for videos READY_FOR_ML and returns a list of target dictionaries."""
        logging.debug(f"Querying local state database for {PipelineStatus.READY_FOR_ML} videos (LIMIT: {self.limit})...")

        if not os.path.exists(self.local_db_path):
            logging.warning(f"Database not found at {self.local_db_path}. Ingestion must be run first.")
            return []

        conn = sqlite3.connect(self.local_db_path)

        drop_id_col = config.csv_mapping.get('drop_id_column', 'DropID')
        video_link_col = config.csv_mapping.get('video_file_link_column', 'LinkToVideoFile')

        df = pd.DataFrame(self.db.get_deployments_by_status(PipelineStatus.READY_FOR_ML))

        df = df.head(self.limit)

        if df.empty:
            return []

        # Rename columns to match what the rest of the method expects (BUV Deployment CSV style)
        df = df.rename(columns={
            "drop_id": drop_id_col,
            "video_path": video_link_col,
            "sampling_start": config.csv_sampling_start_column,
            "sampling_end": config.csv_sampling_end_column
        })

        logging.debug("Generating target paths and downloading media via aws s3...")

        actual_video_dir = self.video_storage_dir
        os.makedirs(actual_video_dir, exist_ok=True)

        valid_indices = []
        local_filepaths = []
        for idx, path in df[video_link_col].items():
            filename = os.path.basename(path)
            local_path = os.path.join(actual_video_dir, filename)

            s3_uri = f"s3://{self.bucket}/{path}"

            # Skip if the file already exists locally
            if os.path.exists(local_path):
                logging.debug(f"Video {filename} already exists at {local_path}. Skipping download.")
                local_filepaths.append(local_path)
                valid_indices.append(idx)
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
                valid_indices.append(idx)
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to download video {s3_uri}: {e}")
                continue

        # Keep only the rows where the video is actually available locally
        df = df.loc[valid_indices].copy()

        if df.empty:
            logging.warning("No media was successfully downloaded. Exiting.")
            return []

        df["VideoURL"] = local_filepaths

        df["VideoURL"] = local_filepaths

        # Return list of records for inference
        return df.to_dict('records')

    def run_inference_loop(self, targets: List[dict]) -> List[str]:
        """Executes the YOLO inference directly in a Python loop for each target dictionary."""
        if not targets:
            return []

        drop_id_col = config.csv_mapping.get('drop_id_column', 'DropID')
        drop_ids = [t[drop_id_col] for t in targets]

        logging.info(f"Setting {len(drop_ids)} targets to {PipelineStatus.PROCESSING_ML}...")
        # Batch update status to PROCESSING_ML
        for drop_id in drop_ids:
            self.db.update_status(drop_id, PipelineStatus.PROCESSING_ML)

        logging.info(f"Starting inference loop for {len(drop_ids)} drops...")

        # Automatically download the YOLO weights from S3 if they don't exist locally
        if not os.path.exists(self.model):
            model_s3_key = config.model_s3_key
            if model_s3_key:
                s3_uri = f"s3://{self.bucket}/{model_s3_key}"
                logging.debug(f"Model weights not found at {self.model}. Downloading from {s3_uri} via aws s3 cp...")
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


        success_targets = []
        for row in targets:
            try:
                drop_id = row[drop_id_col]

                drop_annotations_dir = config.get_drop_annotations_dir(drop_id)
                inference_args = {
                    'drop_id': drop_id,
                    'video_url': row['VideoURL'],
                    'sampling_start': row[config.csv_sampling_start_column],
                    'sampling_end': row[config.csv_sampling_end_column],
                    'model_path': self.model,
                    'frame_skip': self.frame_skip,
                    'imgsz': self.imgsz,
                    'confidence_threshold': self.confidence,
                    'output_csv': os.path.join(drop_annotations_dir, f"{drop_id}_{model_name}_raw.csv")
                }

                logging.info(f"  → Running ML inference: {drop_id}")
                run_inference_main(inference_args)
                success_targets.append(drop_id)

            except Exception as e:
                logging.error(f"Inference failed for {drop_id}: {e}", exc_info=True)
                self.db.update_status(drop_id, PipelineStatus.ERROR)
                self.db.add_validation_errors([{
                        "SurveyID": "_".join(drop_id.split("_")[:3]),
                        "DropID": drop_id,
                        "ErrorType": "PIPELINE_ERROR",
                        "FileName": "",
                        "ColumnName": "ml_inference",
                        "ErrorMessage": f"Inference failed: {type(e).__name__}: {e}",
                        "InvalidValue": "",
                    }])
                # drop_id is not added to success_targets — it will not proceed

        return success_targets

    def finalize_batch_results(self, successful_drops: List[str]):
        """Uploads generated ML CSVs to S3 and marks deployments as ML_COMPLETE locally."""
        if not successful_drops:
            logging.info("No successful drops to sync. Exiting.")
            return

        logging.info(f"Syncing state for {len(successful_drops)} successfully processed drops...")

        for drop_id in successful_drops:
            self.db.update_status(drop_id, PipelineStatus.ML_COMPLETE)
        logging.info(f"Successfully updated {successful_drops} rows in local DB to {PipelineStatus.ML_COMPLETE}.")

def main():
    parser = argparse.ArgumentParser(description='Spyfish ML Pipeline Orchestrator')
    args = parser.parse_args()

    runner = MLRunner()
    targets = runner.get_inference_targets()
    successes = runner.run_inference_loop(targets)
    runner.finalize_batch_results(successes)

if __name__ == "__main__":
    main()
