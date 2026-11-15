import logging
import os
from pathlib import Path
from time import perf_counter
from typing import List

from spyfish.config.base import (
    InvalidTransitionError,
    MlStatus,
    VideoPresence,
    get_required,
)
from spyfish.config.wrapper import config
from spyfish.database.annotation_manager import AnnotationDatabaseManager
from spyfish.database.manager import DatabaseManager
from spyfish.extraction.slate import extract_slate_frame
from spyfish.ml.model_meta import resolve_imgsz, resolve_maxn_threshold
from spyfish.ml.process_ml_annotations import (
    DegenerateInferenceError,
    process_one_drop,
)
from spyfish.ml.run_inference import main as run_inference_main
from spyfish.orchestrator.download import download_videos
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
        # Via the property (not raw get_required) so the (0, 1] validation fires,
        # a stray confidence_threshold=0 floods inference with max_det garbage.
        self.confidence = config.confidence_threshold
        self.model = str(validate_model_path(config.pipeline_model_path))
        # The checkpoint knows what size it was trained at; config only overrides.
        self.imgsz = resolve_imgsz(self.model)
        # Drops this runner successfully claimed (advanced to ml_running).
        # Populated by run_inference_loop; finalize_batch_results must only
        # ever touch these, never a target another job claimed first.
        self.claimed_drop_ids: List[str] = []

    def get_inference_targets(
        self, survey_id: str | None = None, force: bool = False
    ) -> List[dict]:
        """Queries the DB for drops with ml_status='ready' and downloads their videos.

        With ``survey_id``, targets every ready drop in that survey (drop_id
        prefix match) and bypasses ``limit_processing`` — the survey is the
        batch. Survey drops in other ml states are reported, not advanced;
        run --check-arrivals first for drops still awaiting video confirmation.

        ``force`` (survey mode only) additionally resets the survey's
        ml_complete / ml_error / ml_running drops back to ml_ready so inference
        re-runs on them. Outputs are keyed by model name, so a re-run with a new
        model writes alongside the old model's CSVs; only a same-model re-run
        overwrites (the intended refresh). ml_running is included because
        nothing else clears it: a killed job (SLURM timeout, Ctrl-C) strands
        drops there forever. The caveat is that --force cannot tell a stranded
        drop from a live one, so do not run it while another ML job is working
        on the same survey.
        """
        logging.debug(
            f"Querying local state database for ml_status={MlStatus.READY!r} (LIMIT: {self.limit})..."
        )

        if not os.path.exists(self.local_db_path):
            logging.warning(
                f"Database not found at {self.local_db_path}. Ingestion must be run first."
            )
            return []

        if survey_id and force:
            # Direct status set (same path as --set-targets), not advance_status:
            # ml_complete -> ml_ready is not a forward transition, it is an
            # operator-requested re-run.
            summary = self.db.get_survey_ml_status_summary(survey_id)
            # ml_inferred is in here too: its raw CSV is from the model being
            # redone, so post-processing it would complete a stale result.
            redo = (
                summary.get(MlStatus.COMPLETE, [])
                + summary.get(MlStatus.ERROR, [])
                + summary.get(MlStatus.INFERRED, [])
            )
            # ml_running has no other way out: it is set before inference and
            # only advances on completion, so a killed job leaves drops stuck.
            stranded = summary.get(MlStatus.RUNNING, [])
            for drop_id in redo + stranded:
                self.db.update_section_status(drop_id, MlStatus.COLUMN, MlStatus.READY)
            if redo:
                logging.info(
                    f"--force: reset {len(redo)} {survey_id} drop(s) from "
                    f"ml_complete/ml_error/ml_inferred back to ml_ready: {sorted(redo)}"
                )
            if stranded:
                logging.warning(
                    f"--force: reset {len(stranded)} {survey_id} drop(s) from "
                    f"ml_running back to ml_ready: {sorted(stranded)}. If another "
                    f"ML job is currently processing this survey, it will now be "
                    f"processed twice - check for a live job before continuing."
                )

        records = self.db.get_deployments_eligible(
            MlStatus.COLUMN, [MlStatus.READY], survey_id=survey_id
        )
        # With a survey, the survey is the batch.
        limit = len(records) if survey_id else self.limit

        if survey_id:
            summary = self.db.get_survey_ml_status_summary(survey_id)
            not_ready = {s: d for s, d in summary.items() if s != MlStatus.READY}
            logging.info(
                f"Survey {survey_id}: {len(records)} drop(s) ready for ML "
                f"(limit_processing bypassed)."
            )
            for status, drop_ids in sorted(not_ready.items()):
                logging.info(
                    f"  not targeted ({status}): {len(drop_ids)} drop(s), "
                    f"e.g. {drop_ids[:3]}"
                )
            if not records and not summary:
                logging.warning(
                    f"Survey {survey_id}: no deployments found — check the survey id."
                )

        if not records:
            return []

        # The batch is the TOP `limit` drops by priority, not "walk the queue
        # until `limit` succeed". An earlier loop kept going past a failed
        # drop, so on 2026-10-14 a run meant for 9 targeted SLI drops walked
        # all 800+ ml_ready drops trying to download each one; with a real
        # bucket that is the whole backlog's egress bill for a 9-drop job. A
        # drop that fails here is reported, not replaced.
        # Archived drops are set aside BEFORE the cut: the transfer below skips
        # every DEEP_ARCHIVE video, so nine archived drops at the top of the
        # queue made every run download nothing while the rest of ml_ready
        # never moved. This reads only the DB and the local disk (no S3 call).
        # A video already in media_dir is kept whatever the column says, and
        # a restored object keeps `archived` until --check-arrivals refreshes
        # it, so that is the step that brings these back.
        if not survey_id:
            archived = [
                r["drop_id"]
                for r in records
                if r.get("video_presence") == VideoPresence.ARCHIVED
                and not (
                    Path(self.video_storage_dir) / Path(r["video_path"] or "").name
                ).is_file()
            ]
            if archived:
                logging.warning(
                    f"{len(archived)} ml_ready drop(s) have video_presence="
                    f"{VideoPresence.ARCHIVED} and no local copy; left out of this "
                    f"batch (restore, then --check-arrivals): {archived[:5]}"
                    + (" ..." if len(archived) > 5 else "")
                )
                skip = set(archived)
                records = [r for r in records if r["drop_id"] not in skip]

        if len(records) > limit:
            logging.info(
                f"{len(records)} drops at ml_ready; taking the top {limit} by "
                f"priority (limit_processing)."
            )
            records = records[:limit]

        # The same transfer --download runs (orchestrator/download.py): a file
        # already in media_dir is used whatever video_presence says, an
        # unrestored DEEP_ARCHIVE object is reported and skipped. After a
        # --download job this finds everything present; its TIMING
        # phase=download line then reports n_videos=0, which is what a GPU job
        # should show.
        result = download_videos(
            [r["drop_id"] for r in records], db=self.db, s3=self.s3
        )
        on_disk = set(result["downloaded"]) | set(result["present"])
        targets = [
            {
                **r,
                "VideoURL": str(
                    Path(self.video_storage_dir) / Path(r["video_path"]).name
                ),
            }
            for r in records
            if r["drop_id"] in on_disk
        ]
        if not targets:
            logging.warning("No media was successfully downloaded. Exiting.")
        # DB key names: drop_id, sampling_start, sampling_end
        return targets

    def run_inference_loop(
        self, targets: List[dict], post_process: bool = True
    ) -> List[str]:
        """Executes YOLO inference for each target. Returns drop_ids that succeeded.

        Each drop goes ``ml_ready → ml_running → ml_inferred`` once its raw
        detections CSV is on disk. With ``post_process`` (the default, what
        ``--ml`` does) the drop is then post-processed in the same loop and
        ends at ``ml_complete``. ``--ml-infer`` passes ``post_process=False``
        so the GPU job stops at ``ml_inferred`` and ``--ml-post`` finishes the
        CPU-only part on a CPU node (`run_post_processing`).
        """
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
        # Claim each drop independently. The ml_ready read in
        # get_inference_targets happens minutes-to-hours before this claim (the
        # video download sits in between), so a concurrently running ML job can
        # have taken some of these drops already, leaving them in ml_running.
        # That drop is not ours to process: log it, drop it from the batch, and
        # carry on with the rest rather than failing the whole ml stage.
        claimed = []
        stolen = []
        for drop_id in drop_ids:
            try:
                self.db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.RUNNING)
            except InvalidTransitionError as e:
                stolen.append(drop_id)
                logging.warning(f"Skipping {drop_id}, could not claim it: {e}")
                continue
            claimed.append(drop_id)

        if stolen:
            logging.warning(
                f"{len(stolen)} of {len(drop_ids)} target(s) were claimed by another "
                f"process between target selection and now, skipping them: "
                f"{sorted(stolen)}. Check for a concurrent ML job on this survey."
            )

        self.claimed_drop_ids = claimed
        if not claimed:
            logging.error(
                "No targets could be claimed, every drop in this batch is already "
                "owned by another process. Nothing to do."
            )
            return []

        claimed_set = set(claimed)
        targets = [t for t in targets if t["drop_id"] in claimed_set]

        logging.info(f"Starting inference loop for {len(targets)} drops...")

        # One annotations DB handle for the whole batch; only opened when this
        # loop also post-processes, the GPU-only job never touches it.
        ann_db = AnnotationDatabaseManager() if post_process else None
        model_name = Path(self.model).stem

        success_targets = []
        for row in targets:
            drop_id = row["drop_id"]
            try:
                inference_args = {
                    "drop_id": drop_id,
                    "video_url": row["VideoURL"],
                    "sampling_start": row["sampling_start"],
                    "sampling_end": row["sampling_end"],
                    "model_path": self.model,
                    "ml_fps": self.ml_fps,
                    "imgsz": self.imgsz,
                    "confidence_threshold": self.confidence,
                    "output_csv": str(config.get_raw_csv_path(drop_id, model_name)),
                }
                # Slate first: the handwritten metadata board at the start of the
                # video, while it is on local disk. The rig is on the bottom by
                # sampling_start, so the board cannot be later than that. Its own
                # try so a miss or a bug here never costs the ML run. It stays
                # with inference (seconds of CPU) rather than post-processing so
                # the ordering guarantee "slate before ML" holds in every mode.
                # In the three-job chain --download has already saved it (on CPU),
                # so this returns at once: extract_slate_frame skips a saved slate.
                try:
                    extract_slate_frame(
                        drop_id,
                        Path(row["VideoURL"]),
                        max_seconds=row["sampling_start"],
                    )
                except Exception as e:
                    logging.warning(
                        f"{drop_id}: slate extraction failed, continuing: {e}"
                    )

                logging.info(f"  → Running ML inference: {drop_id}")
                t_inference = perf_counter()
                run_inference_main(inference_args)
                # The raw CSV is on disk (atomic rename inside run_inference), so
                # the GPU's part is done whatever happens next.
                self.db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.INFERRED)
                logging.info(
                    f"TIMING drop={drop_id} phase=inference "
                    f"seconds={perf_counter() - t_inference:.0f}"
                )

                if post_process:
                    self.post_process_drop(drop_id, ann_db, model_name)
                success_targets.append(drop_id)

            except Exception as e:
                logging.error(f"ML processing failed for {drop_id}: {e}", exc_info=True)
                self._mark_ml_error(drop_id, e, column_name="ml_inference")

        return success_targets

    def post_process_drop(
        self,
        drop_id: str,
        ann_db: AnnotationDatabaseManager,
        model_name: str,
    ) -> None:
        """CPU-only tail of the ML stage for one ``ml_inferred`` drop.

        MaxN + annotation ingest + QA frames from the raw CSV already on disk,
        then ``ml_inferred → ml_complete``. Runs before the drop is marked
        complete so a crash leaves it at ``ml_inferred`` for a retry rather
        than complete with no artifacts. Raises on failure; the caller decides
        how to record it.
        """
        t_post = perf_counter()
        process_one_drop(
            drop_id=drop_id,
            video_dir=Path(self.video_storage_dir),
            ann_db=ann_db,
            model_name=model_name,
            interval=config.interval_seconds,
            base_conf=config.confidence_threshold,
            maxn_conf=resolve_maxn_threshold(config.model_path_for_name(model_name)),
            deployment=self.db.get_deployment(drop_id),
        )
        self.db.sync_annotation_counts([drop_id])
        # sync_annotation_counts already sets ml_complete for any drop that
        # gained annotations; only advance here if it didn't (e.g. a
        # zero-detection drop still at ml_inferred), so we never trip the
        # ml_complete → ml_complete guard.
        dep = self.db.get_deployment(drop_id)
        if not dep or dep["ml_status"] != MlStatus.COMPLETE:
            self.db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.COMPLETE)
        logging.info(
            f"TIMING drop={drop_id} phase=post seconds={perf_counter() - t_post:.0f}"
        )

    def run_post_processing(self, survey_id: str | None = None) -> List[str]:
        """``--ml-post``: finish every ``ml_inferred`` drop (CPU node).

        No ``limit_processing`` here: these drops were already bounded when
        their inference batch was picked, and a backlog of inferred drops from
        several GPU jobs should all be finished. ``survey_id`` narrows to one
        survey (DropID prefix), same as every other stage. Needs the video on
        local disk at the runner's path (QA and legacy review frames seek into
        it), which on NeSI is the shared project filesystem the GPU job wrote
        to. Returns the drop_ids that reached ``ml_complete``.
        """
        records = self.db.get_deployments_eligible(
            MlStatus.COLUMN, [MlStatus.INFERRED], survey_id=survey_id
        )
        drop_ids = [r["drop_id"] for r in records]
        if not drop_ids:
            logging.info(f"No drops at {MlStatus.INFERRED}. Nothing to post-process.")
            return []

        logging.info(f"Post-processing {len(drop_ids)} {MlStatus.INFERRED} drop(s)...")
        ann_db = AnnotationDatabaseManager()
        done = []
        for drop_id in drop_ids:
            try:
                self.post_process_drop(
                    drop_id, ann_db, self._inferred_model_name(drop_id)
                )
                done.append(drop_id)
            except Exception as e:
                logging.error(
                    f"ML post-processing failed for {drop_id}: {e}", exc_info=True
                )
                self._mark_ml_error(drop_id, e, column_name="ml_post")
        return done

    def _inferred_model_name(self, drop_id: str) -> str:
        """The model whose raw CSV ``--ml-post`` should finish for ``drop_id``.

        The GPU job names its raw CSV after the model it ran; by the time the
        CPU job runs, ``--retrain`` may have promoted a newer model, so the
        configured model is not necessarily the one that produced the drop's
        detections. Read the name back from disk: of the ``{drop}_{model}_raw.csv``
        files present, take the one written last. That is the run that put the
        drop at ``ml_inferred``; the newest training stamp is not, after a
        rollback to an older checkpoint or when a trial model once ran on the
        drop. Falls back to the configured model when no raw CSV is found, so
        the error raised downstream names the file that was expected.
        """
        configured = Path(self.model).stem
        # Canonical model names only: the zooniverse / biigle raw CSVs share
        # the suffix but are not listed.
        candidates = config.list_ml_raw_csvs(drop_id)
        if not candidates:
            return configured
        # Stamp only breaks an mtime tie (files restored in one copy).
        newest = max(
            candidates,
            key=lambda m: (
                candidates[m].stat().st_mtime,
                config.model_name_stamp(m),
                m,
            ),
        )
        if newest != configured:
            logging.info(
                f"  {drop_id}: raw CSV is from {newest}, not the configured "
                f"{configured}; post-processing the model that ran."
            )
        return newest

    def _mark_ml_error(self, drop_id: str, exc: Exception, column_name: str) -> None:
        """Record a failed drop in ``validation_errors``, moving it to ``ml_error``
        only when a GPU re-run is what fixes it.

        ``ml_running`` → ``ml_error``: inference itself failed; the only exit
        from ``ml_error`` is ``ml_ready``, which re-infers.
        ``ml_inferred`` stays ``ml_inferred``: the raw CSV is on disk and the
        failure is CPU-side (video evicted before the QA frame seek, DB write).
        The next ``--ml-post`` retries it; sending it to ``ml_error`` would
        force a GPU re-inference for a CPU problem. The error is still recorded.
        If the drop already reached ``ml_complete`` (work succeeded but a later
        step raised), forcing ``ml_error`` would itself be an invalid transition
        and mask the real success, so skip it.
        """
        dep = self.db.get_deployment(drop_id)
        current = dep["ml_status"] if dep else None
        if current in (MlStatus.RUNNING, MlStatus.INFERRED):
            # A degenerate raw CSV fails the same way on every retry, so it
            # goes to ml_error for a person even from ml_inferred.
            if current == MlStatus.RUNNING or isinstance(exc, DegenerateInferenceError):
                self.db.advance_status(drop_id, MlStatus.COLUMN, MlStatus.ERROR)
            else:
                logging.warning(
                    f"{drop_id}: left at {MlStatus.INFERRED}; the raw CSV is on "
                    "disk, the next --ml-post retries it."
                )
            self.db.add_validation_error(
                survey_id=config.get_survey_id_from_drop(drop_id),
                drop_id=drop_id,
                error_type=MlStatus.ERROR,
                column_name=column_name,
                error_message=f"ML processing failed: {type(exc).__name__}: {exc}",
            )
        else:
            logging.error(
                f"{drop_id}: exception after status reached {current or 'unknown'!r}, "
                "not forcing ml_error (would be an invalid transition)."
            )

    def finalize_batch_results(
        self, successful_drops: List[str], all_drop_ids: List[str] | None = None
    ):
        """Safety net for drops left stuck in `ml_running` after the inference loop.

        Status advancement to `ml_complete` now happens inside `run_inference_loop`
        immediately after each drop's MaxN + QA frames are written, so this only
        needs to catch drops where the loop's try/except didn't fire (e.g. process
        killed mid-iteration).

        `all_drop_ids` defaults to the drops this runner actually claimed. Do not
        pass the full target list: a target lost to a concurrent job is sitting in
        ml_running under *that* job's ownership, and flipping it to ml_error here
        would sabotage a live run.
        """
        if all_drop_ids is None:
            all_drop_ids = self.claimed_drop_ids
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
    successes = runner.run_inference_loop(targets)
    runner.finalize_batch_results(successes)


if __name__ == "__main__":
    main()
