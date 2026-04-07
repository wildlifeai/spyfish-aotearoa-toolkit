import logging
import os
from pathlib import Path
from typing import List

import pandas as pd

from spyfish.config.base import PipelineStatus, get_required
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.ml.run_inference import main as run_inference_main
from spyfish.storage.s3_handler import S3Handler
from spyfish.utils import validate_model_path


class MLRunner:
    def __init__(self):
        # S3 properties
        self.bucket = config.s3_bucket
        self.s3_db_key = config.s3_db_key
        self.s3 = S3Handler(bucket=self.bucket)
        self.db = DatabaseManager()

        self.video_storage_dir = config.media_dir
        self.local_db_path = config.db_path

        # Logic properties
        self.limit = get_required(
            config.ml_inference, "limit_processing", "ml_inference"
        )
        self.ml_fps = config.ml_fps
        self.imgsz = int(config.imgsz)
        self.confidence = get_required(
            config.ml_inference, "confidence_threshold", "ml_inference"
        )
        # Use the standardized pipeline model path
        self.model = str(validate_model_path(config.pipeline_model_path))

    def get_inference_targets(self) -> List[dict]:
        """Queries the local sqlite DB for videos READY_FOR_ML and returns a list of target dictionaries."""
        logging.debug(
            f"Querying local state database for {PipelineStatus.READY_FOR_ML} videos (LIMIT: {self.limit})..."
        )

        if not os.path.exists(self.local_db_path):
            logging.warning(
                f"Database not found at {self.local_db_path}. Ingestion must be run first."
            )
            return []

        drop_id_col = config.drop_id_column
        video_link_col = config.csv_video_file_link_column

        df = pd.DataFrame(
            self.db.get_deployments_by_status(PipelineStatus.READY_FOR_ML)
        )

        # Float any drops listed in pipeline_targets.csv to the top (in CSV order)
        targets_csv = config.pipeline_targets_csv
        if targets_csv and os.path.exists(targets_csv):
            try:
                priority_ids = (
                    pd.read_csv(targets_csv)[config.drop_id_column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                priority_order = {d: i for i, d in enumerate(priority_ids)}
                df["_priority"] = df["drop_id"].map(priority_order).fillna(len(priority_ids))
                df = df.sort_values("_priority").drop(columns=["_priority"])
                logging.debug(f"Priority order applied from {targets_csv}: {priority_ids}")
            except Exception as e:
                logging.warning(f"Could not apply priority ordering from {targets_csv}: {e}")

        df = df.head(self.limit)

        if df.empty:
            return []

        # Rename columns to match what the rest of the method expects (BUV Deployment CSV style)
        df = df.rename(
            columns={
                "drop_id": drop_id_col,
                "video_path": video_link_col,
                "sampling_start": config.csv_sampling_start_column,
                "sampling_end": config.csv_sampling_end_column,
            }
        )

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
                logging.debug(
                    f"Video {filename} already exists at {local_path}. Skipping download."
                )
                local_filepaths.append(local_path)
                valid_indices.append(idx)
                continue

            logging.info(f"Downloading {s3_uri} to {local_path}...")

            try:
                success = self.s3.download_object_from_s3(key=path, filename=local_path)
                if not success:
                    raise RuntimeError("S3Handler returned False")
                local_filepaths.append(local_path)
                valid_indices.append(idx)
            except Exception as e:
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
        return df.to_dict("records")

    def run_inference_loop(self, targets: List[dict]) -> List[str]:
        """Executes the YOLO inference directly in a Python loop for each target dictionary."""
        if not targets:
            return []

        drop_id_col = config.drop_id_column
        drop_ids = [t[drop_id_col] for t in targets]

        # Model must exist before any drop status is changed.
        # Checking here keeps all drops in READY_FOR_ML so the batch can be
        # retried once the model file is placed — no manual DB repair needed.
        if not os.path.exists(self.model):
            logging.error(
                f"Model weights not found at {self.model}. Automatic download from S3 is disabled for security."
            )
            logging.info(
                "Please manually place the production model at the expected path."
            )
            raise FileNotFoundError(f"Model missing: {self.model}")

        logging.info(
            f"Setting {len(drop_ids)} targets to {PipelineStatus.PROCESSING_ML}..."
        )
        # Batch update status to PROCESSING_ML
        for drop_id in drop_ids:
            self.db.advance_status(drop_id, PipelineStatus.PROCESSING_ML)

        logging.info(f"Starting inference loop for {len(drop_ids)} drops...")

        success_targets = []
        for row in targets:
            try:
                drop_id = row[drop_id_col]

                drop_annotations_dir = config.get_drop_annotations_dir(drop_id)
                model_name = Path(self.model).stem
                inference_args = {
                    "drop_id": drop_id,
                    "video_url": row["VideoURL"],
                    "sampling_start": row[config.csv_sampling_start_column],
                    "sampling_end": row[config.csv_sampling_end_column],
                    "model_path": self.model,
                    "ml_fps": self.ml_fps,
                    "imgsz": self.imgsz,
                    "confidence_threshold": self.confidence,
                    "output_csv": os.path.join(
                        drop_annotations_dir, f"{drop_id}_{model_name}_raw.csv"
                    ),
                }

                logging.info(f"  → Running ML inference: {drop_id}")
                run_inference_main(inference_args)
                success_targets.append(drop_id)

            except Exception as e:
                logging.error(f"Inference failed for {drop_id}: {e}", exc_info=True)
                self.db.advance_status(drop_id, PipelineStatus.ERROR)
                self.db.add_validation_errors(
                    [
                        {
                            "SurveyID": "_".join(drop_id.split("_")[:3]),
                            "DropID": drop_id,
                            "ErrorType": "PIPELINE_ERROR",
                            "FileName": "",
                            "ColumnName": "ml_inference",
                            "ErrorMessage": f"Inference failed: {type(e).__name__}: {e}",
                            "InvalidValue": "",
                        }
                    ]
                )
                # drop_id is not added to success_targets — it will not proceed

        return success_targets

    def finalize_batch_results(
        self, successful_drops: List[str], all_drop_ids: List[str] | None = None
    ):
        """Marks successful drops ML_COMPLETE and recovers any stuck PROCESSING_ML drops.

        Args:
            successful_drops: Drop IDs that completed inference without error.
            all_drop_ids: Every drop ID that was set to PROCESSING_ML at batch start.
                          Used as a safety net: any drop still in PROCESSING_ML after
                          the loop (e.g. process killed mid-batch) is advanced to ERROR
                          so it can be retried rather than stuck permanently.
        """
        # Safety net: advance any drop that is still PROCESSING_ML to ERROR.
        # Under normal operation run_inference_loop already does this per-drop,
        # but if the process was interrupted the per-drop handler may not have run.
        if all_drop_ids:
            for drop_id in set(all_drop_ids) - set(successful_drops):
                dep = self.db.get_deployment(drop_id)
                if dep and dep["status"] == PipelineStatus.PROCESSING_ML:
                    logging.error(
                        f"Drop {drop_id} is still {PipelineStatus.PROCESSING_ML} after batch "
                        "— advancing to ERROR to allow retry."
                    )
                    self.db.update_status(drop_id, PipelineStatus.ERROR)

        if not successful_drops:
            logging.info("No successful drops to finalize.")
            return

        logging.info(
            f"Syncing state for {len(successful_drops)} successfully processed drops..."
        )
        for drop_id in successful_drops:
            self.db.advance_status(drop_id, PipelineStatus.ML_COMPLETE)
        logging.info(
            f"Updated {len(successful_drops)} drops to {PipelineStatus.ML_COMPLETE}."
        )


def main():
    runner = MLRunner()
    targets = runner.get_inference_targets()
    all_drop_ids = [t[config.drop_id_column] for t in targets]
    successes = runner.run_inference_loop(targets)
    runner.finalize_batch_results(successes, all_drop_ids=all_drop_ids)


if __name__ == "__main__":
    main()
