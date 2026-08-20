import logging
import os
from pathlib import Path
from typing import List

import pandas as pd

from spyfish.config.base import MlStatus, VideoPresence, get_required
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager
from spyfish.ml.process_ml_annotations import process_one_drop
from spyfish.ml.run_inference import main as run_inference_main
from spyfish.storage.s3_handler import S3Handler
from spyfish.utils import validate_model_path


class MLRunner:
    def __init__(self):
        self.bucket = config.s3_bucket
        self.s3 = S3Handler(bucket=self.bucket)
        self.db = DatabaseManager()

        self.video_storage_dir = config.media_dir
        self.local_db_path = config.db_path

        self.limit = get_required(
            config.ml_inference, "limit_processing", "ml_inference"
        )
        self.ml_fps = config.ml_fps
        self.imgsz = int(config.imgsz)
        # Via the property (not raw get_required) so the (0, 1] validation fires,
        # a stray confidence_threshold=0 floods inference with max_det garbage.
        self.confidence = config.confidence_threshold
        self.model = str(validate_model_path(config.pipeline_model_path))

    def get_inference_targets(self) -> List[dict]:
        """Queries the DB for drops with ml_status='ready' and downloads their videos."""
        logging.debug(
            f"Querying local state database for ml_status={MlStatus.READY!r} (LIMIT: {self.limit})..."
        )

        if not os.path.exists(self.local_db_path):
            logging.warning(
                f"Database not found at {self.local_db_path}. Ingestion must be run first."
            )
            return []

        records = self.db.get_deployments_eligible("ml_status", [MlStatus.READY])
        df = pd.DataFrame(records)

        if df.empty:
            return []

        logging.debug("Generating target paths and downloading media via aws s3...")

        actual_video_dir = self.video_storage_dir
        os.makedirs(actual_video_dir, exist_ok=True)

        # Download in priority order, stopping once we have `self.limit` successes.
        valid_indices = []
        local_filepaths = []
        for idx, path in df["video_path"].items():
            if len(valid_indices) >= self.limit:
                break

            if df.at[idx, "video_presence"] == VideoPresence.ARCHIVED:
                drop_id = df.at[idx, "drop_id"]
                logging.warning(
                    f"Skipping {drop_id}: video in DEEP_ARCHIVE, "
                    f"restore with `aws s3api restore-object` before ML can run."
                )
                continue

            filename = os.path.basename(path)
            local_path = os.path.join(actual_video_dir, filename)
            s3_uri = f"s3://{self.bucket}/{path}"

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

        # Return list of records for inference (using DB key names: drop_id, sampling_start, sampling_end)
        return df.to_dict("records")

    def run_inference_loop(self, targets: List[dict]) -> List[str]:
        """Executes YOLO inference for each target. Returns drop_ids that succeeded."""
        if not targets:
            return []

        drop_ids = [t["drop_id"] for t in targets]

        if not os.path.exists(self.model):
            logging.error(
                f"Model weights not found at {self.model}. Automatic download from S3 is disabled for security."
            )
            logging.info(
                "Please manually place the production model at the expected path."
            )
            raise FileNotFoundError(f"Model missing: {self.model}")

        logging.info(
            f"Setting {len(drop_ids)} targets to ml_status={MlStatus.RUNNING!r}..."
        )
        for drop_id in drop_ids:
            self.db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.RUNNING)

        logging.info(f"Starting inference loop for {len(drop_ids)} drops...")

        # Shared resources for per-drop post-ML processing. Initialised once
        # outside the loop so each iteration reuses the same DB connections.
        ann_db = AnnotationDatabaseManager()
        model_name = Path(self.model).stem
        interval = config.interval_seconds
        base_conf = config.confidence_threshold
        maxn_conf = config.maxn_confidence_threshold

        success_targets = []
        for row in targets:
            drop_id = row["drop_id"]
            try:
                drop_annotations_dir = config.get_drop_annotations_dir(drop_id)
                inference_args = {
                    "drop_id": drop_id,
                    "video_url": row["VideoURL"],
                    "sampling_start": row["sampling_start"],
                    "sampling_end": row["sampling_end"],
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

                # Per-drop post-ML: write MaxN + QA frames BEFORE marking complete,
                # so a crash here leaves the drop in `ml_running` for retry rather
                # than `ml_complete` with no artifacts on disk.
                process_one_drop(
                    drop_id=drop_id,
                    video_dir=Path(self.video_storage_dir),
                    ann_db=ann_db,
                    model_name=model_name,
                    interval=interval,
                    base_conf=base_conf,
                    maxn_conf=maxn_conf,
                )
                self.db.sync_annotation_counts([drop_id])
                # sync_annotation_counts already advances ml_status → ml_complete for
                # any drop that gained annotations; only advance here if it didn't
                # (e.g. a zero-detection drop still in ml_running), so we never trip
                # the ml_complete → ml_complete guard.
                dep = self.db.get_deployment(drop_id)
                if not dep or dep["ml_status"] != MlStatus.COMPLETE:
                    self.db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.COMPLETE)
                success_targets.append(drop_id)

            except Exception as e:
                logging.error(f"ML processing failed for {drop_id}: {e}", exc_info=True)
                # Only a drop still in ml_running can legally move to ml_error. If it
                # already reached ml_complete (inference succeeded but a later step
                # raised), forcing ml_error would itself be an invalid transition and
                # mask the real success, so skip it.
                dep = self.db.get_deployment(drop_id)
                if dep and dep["ml_status"] == MlStatus.RUNNING:
                    self.db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.ERROR)
                    self.db.add_validation_error(
                        survey_id=config.get_survey_id_from_drop(drop_id),
                        drop_id=drop_id,
                        error_type=MlStatus.ERROR,
                        column_name="ml_inference",
                        error_message=f"ML processing failed: {type(e).__name__}: {e}",
                    )
                else:
                    current = dep["ml_status"] if dep else "unknown"
                    logging.error(
                        f"{drop_id}: exception after status reached {current!r}, not "
                        "forcing ml_error (would be an invalid transition)."
                    )

        return success_targets

    def finalize_batch_results(
        self, successful_drops: List[str], all_drop_ids: List[str] | None = None
    ):
        """Safety net for drops left stuck in `ml_running` after the inference loop.

        Status advancement to `ml_complete` now happens inside `run_inference_loop`
        immediately after each drop's MaxN + QA frames are written, so this only
        needs to catch drops where the loop's try/except didn't fire (e.g. process
        killed mid-iteration).
        """
        if not all_drop_ids:
            return
        for drop_id in set(all_drop_ids) - set(successful_drops):
            dep = self.db.get_deployment(drop_id)
            if dep and dep["ml_status"] == MlStatus.RUNNING:
                logging.error(
                    f"Drop {drop_id} is still ml_status=running after batch, advancing to error."
                )
                self.db.update_section_status(drop_id, MlStatus.COLUMN, MlStatus.ERROR)


def main():
    runner = MLRunner()
    targets = runner.get_inference_targets()
    all_drop_ids = [t["drop_id"] for t in targets]
    successes = runner.run_inference_loop(targets)
    runner.finalize_batch_results(successes, all_drop_ids=all_drop_ids)


if __name__ == "__main__":
    main()
