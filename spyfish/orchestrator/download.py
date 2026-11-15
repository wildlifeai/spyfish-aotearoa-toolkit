"""Fetch deployment videos from S3 into ``media_dir`` ahead of the stages that read them.

Every stage that touches pixels (``--ml-infer``, the QA frames in ``--ml-post``,
``--zooniverse-frames`` / ``--zooniverse-crops``, ``--biigle-upload``) expects
the video on local disk at ``config.media_dir / <basename of video_path>``;
none streams from S3. ``--download`` is that transfer as its own CPU job, so a
GPU job never idles through tens of GB (scripts/slurm/submit_ml.sh). The
``MLRunner.get_inference_targets`` calls the same ``download_videos``, so the
GPU job then finds every file "already present" and transfers nothing.
"""

import logging
from pathlib import Path
from time import perf_counter

from spyfish.config.base import IngestStatus, MlStatus, VideoPresence
from spyfish.config.wrapper import config
from spyfish.database.manager import DatabaseManager
from spyfish.storage.s3_handler import S3Handler


def select_download_drops(
    db: DatabaseManager, survey_id: str | None = None
) -> list[str]:
    """Which drops ``--download`` fetches.

    With a survey: every drop of that survey, whatever its pipeline state,
    because the stages after ML (QA frames, Zooniverse validation, BIIGLE
    upload) need ``ml_complete`` drops' videos as much as inference needs
    ``ml_ready`` ones. Files already present cost nothing, so re-running on a
    survey is cheap. Without a survey: the next ``--ml`` batch, the top
    ``limit_processing`` ``ml_ready`` drops by priority, the same frame
    ``MLRunner.get_inference_targets`` picks.
    """
    if survey_id:
        summary = db.get_survey_ml_status_summary(survey_id)
        return sorted(d for drops in summary.values() for d in drops)
    records = db.get_deployments_eligible(MlStatus.COLUMN, [MlStatus.READY])
    return [r["drop_id"] for r in records[: config.limit_processing]]


def download_videos(
    drop_ids: list[str],
    *,
    db: DatabaseManager | None = None,
    s3: S3Handler | None = None,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Download each drop's video unless it is already in ``media_dir``.

    Returns ``{"downloaded": [...], "present": [...], "no_video": [...],
    "failed": [...]}``.

    Every drop is attempted whatever its ``ingest_status``: a download is a
    copy, not a stage, and the video of an excluded or errored drop is exactly
    what you need on disk to work out what is wrong with it. The DB's
    ``video_presence`` is not trusted either way (a restored object keeps
    ``archived`` until the next ``--check-arrivals``, a late upload keeps
    ``absent``): one HEAD per drop asks S3 itself.

    ``no_video``: no ``video_path`` in the DB, or no object at it. That is a
    fact about the survey, not a transfer error, so it does not fail the run.
    ``failed``: not in the table, still in DEEP_ARCHIVE without a readable
    restore, or the transfer itself failed. Both are returned, not raised:
    the GPU job downloads whatever is still missing itself, so one failed
    transfer here must not leave a dependent SLURM job waiting forever.

    Two passes: every drop is checked first (local file, DB path, one S3 HEAD
    each, seconds for a whole survey) and the plan is logged, so a missing
    video shows up at the top of the log, not after the hours of transfers
    that sort before it. ``dry_run`` stops after that pass: it still asks S3,
    which is read-only, so it answers "is anything missing?" for real.
    """
    db = db or DatabaseManager()
    drop_ids = [config.validate_drop_id(d) for d in drop_ids]
    records = db.get_deployments_by_ids(drop_ids)

    media_dir = config.media_dir
    logging.info(f"media dir: {media_dir}  bucket: {config.s3_bucket}")

    s3 = s3 or S3Handler()
    queue: list[tuple[str, str, Path]] = []  # (drop_id, key, local path)
    present: list[str] = []
    no_video: list[str] = []
    failed: list[str] = []
    for drop_id in drop_ids:
        rec = records.get(drop_id)
        if rec is None:
            logging.error(f"{drop_id}: not in the deployments table")
            failed.append(drop_id)
            continue
        presence = rec.get("video_presence")
        # A blank SharePoint cell reaches the DB as the text "nan".
        key = str(rec.get("video_path") or "").strip()
        if not key or key.lower() == "nan":
            logging.info(f"{drop_id}: no video_path in the DB ({presence=})")
            no_video.append(drop_id)
            continue
        local = media_dir / Path(key).name
        if local.exists():
            present.append(drop_id)
            continue
        storage = s3.get_object_storage(key)
        if storage is None:
            # Expected for drops the DB already knows have no video; a
            # WARNING only when the DB thought there was one.
            expected = presence in (
                VideoPresence.ABSENT,
                VideoPresence.NO_VIDEO_BAD_DEP,
            )
            (logging.info if expected else logging.warning)(
                f"{drop_id}: no object at s3://{config.s3_bucket}/{key} "
                f"({presence=})"
            )
            no_video.append(drop_id)
            continue
        if storage["storage_class"] == "DEEP_ARCHIVE" and not s3.is_restored(key):
            logging.warning(f"{drop_id}: in DEEP_ARCHIVE, restore it first")
            failed.append(drop_id)
            continue
        if rec.get("ingest_status") != IngestStatus.OK:
            logging.info(
                f"{drop_id}: ingest_status={rec.get('ingest_status')!r}, "
                "downloading anyway (no stage will process it)"
            )
        queue.append((drop_id, key, local))

    logging.info(
        f"plan: download {len(queue)}, already present {len(present)}, "
        f"no video in S3 {len(no_video)}"
        + (f": {no_video}" if no_video else "")
        + f", failed {len(failed)}"
        + (f": {failed}" if failed else "")
    )

    done: list[str] = []
    if dry_run:
        for drop_id, key, local in queue:
            logging.info(f"{drop_id}: s3://{config.s3_bucket}/{key} → {local}")
        done = [drop_id for drop_id, _, _ in queue]
        queue = []
    elif queue:
        media_dir.mkdir(parents=True, exist_ok=True)

    t_download = perf_counter()
    for i, (drop_id, key, local) in enumerate(queue, 1):
        logging.info(
            f"[{i}/{len(queue)}] {drop_id}: s3://{config.s3_bucket}/{key} → {local}"
        )
        t_file = perf_counter()
        try:
            ok = s3.download_object_from_s3(key=key, filename=str(local))
        except Exception as e:
            logging.error(f"[{i}/{len(queue)}] {drop_id}: {e}")
            ok = False
        if ok:
            seconds = perf_counter() - t_file
            gb = local.stat().st_size / 1024**3
            logging.info(
                f"[{i}/{len(queue)}] {drop_id}: finished, {gb:.1f} GB in "
                f"{seconds:.0f}s ({gb * 1024 / max(seconds, 1):.0f} MB/s)"
            )
            done.append(drop_id)
        else:
            logging.error(f"[{i}/{len(queue)}] {drop_id}: download failed")
            failed.append(drop_id)

    verb = "would download" if dry_run else "downloaded"
    logging.info(
        f"{verb} {len(done)}, already present {len(present)}, "
        f"no video in S3 {len(no_video)}, failed {len(failed)}"
        + (f": {failed}" if failed else "")
    )
    logging.info(
        f"TIMING phase=download n_videos={0 if dry_run else len(done)} "
        f"seconds={perf_counter() - t_download:.0f}"
    )
    return {
        "downloaded": done,
        "present": present,
        "no_video": no_video,
        "failed": failed,
    }
