import csv
import logging
import math
import threading
from functools import lru_cache
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
from ultralytics import YOLO

from spyfish.config.wrapper import config

# The video readers moved to spyfish.extraction.video_io. ffmpeg_frame_at is
# re-exported only for the scripts that still import it from here.
from spyfish.extraction.video_io import (  # noqa: F401
    ffmpeg_frame_at,
    ffmpeg_sampled_frames,
    get_video_fps,
)
from spyfish.ml.model_meta import inference_batch_size, resolve_imgsz


@lru_cache(maxsize=1)
def get_cached_pipeline_model():
    """Return a process-cached YOLO instance for the pipeline model.

    First call loads weights from disk; subsequent calls reuse the same
    instance, important on multi-drop batch runs where reloading per
    drop adds 1–3s of disk + GPU init overhead.
    """
    model_path = config.pipeline_model_path
    logging.info(f"Loading pipeline model: {model_path.name}")
    return YOLO(str(model_path))


@lru_cache(maxsize=2)
def _load_video_model(model_path: str, mtime_ns: int, size: int, half: bool):
    logging.info(f"Loading model for video inference: {Path(model_path).name}")
    return YOLO(model_path)


def get_cached_video_model(model_path, half: bool):
    """The YOLO model video inference runs, loaded once per process per weights
    file instead of once per drop.

    Keyed on the resolved path plus the file's mtime and size, so another
    weights file (``--model``, a UoA detector) gets its own model and weights
    replaced on disk under the same name are reloaded. ``half`` is in the key
    because Ultralytics fixes the precision when the model's predictor is first
    set up; every other ``predict`` setting (conf, imgsz, iou, agnostic_nms) is
    re-applied on each call. Deliberately NOT the instance
    ``get_cached_pipeline_model`` hands to the frame-file paths: those predict
    without ``half``, and one shared instance would run them at whichever
    precision came first in the process. Two models are kept at most.
    """
    path = Path(model_path).resolve()
    stat = path.stat()
    return _load_video_model(str(path), stat.st_mtime_ns, stat.st_size, bool(half))


# Raw CSV schema. The first eight columns are the contract every consumer reads
# (MaxN, frame selection, BIIGLE COCO, validation crops). With the second-stage
# classifier on, `class` is ITS name and four columns keep both opinions.
RAW_COLUMNS = ["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"]
CLASSIFIER_COLUMNS = ["yolo_class", "yolo_confidence", "clf_class", "clf_confidence"]


@lru_cache(maxsize=1)
def get_cached_classifier():
    """The second-stage species classifier, loaded once per process; None when
    ``ml_inference.classifier.enabled`` is false. Imported lazily: open_clip and
    the backbone weights are only needed when it runs."""
    if not config.classifier_enabled:
        return None
    from spyfish.ml.classifier.species_classifier import SpeciesClassifier

    return SpeciesClassifier.from_config()


def _raw_header(classifier) -> list:
    return RAW_COLUMNS + (CLASSIFIER_COLUMNS if classifier is not None else [])


def _detection_rows(results, frames, frame_indices, times, names, classifier) -> list:
    """One raw CSV row per box across a batch of YOLO results.

    With a classifier, every box of the batch is cropped from its frame and
    named in one call, so the classifier batches across frames too. A box it
    declines (too small) keeps the YOLO name with blank classifier columns.
    """
    dets = []
    for i, (res, frame_idx, t) in enumerate(zip(results, frame_indices, times)):
        for box in res.boxes:
            x, y, w, h = box.xywh[0].tolist()
            dets.append(
                (
                    i,
                    frame_idx,
                    t,
                    names[int(box.cls[0])],
                    float(box.conf[0]),
                    x,
                    y,
                    w,
                    h,
                )
            )
    if classifier is None:
        return [
            [fi, t, c, conf, x, y, w, h] for (_, fi, t, c, conf, x, y, w, h) in dets
        ]
    named = classifier.name_boxes(
        frames, [(i, x, y, w, h) for (i, *_r, x, y, w, h) in dets]
    )
    rows = []
    for (_, fi, t, c, conf, x, y, w, h), naming in zip(dets, named):
        if naming is None:
            rows.append([fi, t, c, conf, x, y, w, h, c, conf, "", ""])
        else:
            label = classifier.final_label(naming, c, conf)
            rows.append(
                [
                    fi,
                    t,
                    label,
                    conf,
                    x,
                    y,
                    w,
                    h,
                    c,
                    conf,
                    naming.raw_label,
                    round(naming.prob, 4),
                ]
            )
    return rows


def inference_stride(fps: float, ml_fps: float | None = None) -> int:
    """Frames skipped between inference samples: one sample every `stride` frames.

    The only definition. Anything that needs to know which frames inference
    looked at (e.g. the Zooniverse validation grid) must use this, not its own
    copy, or a change here would silently desynchronise them.
    """
    return max(1, round(fps / (ml_fps if ml_fps is not None else config.ml_fps)))


def seek_to_sampling_start(cap, sampling_start: float) -> int:
    """Seek `cap` to `sampling_start` and return the frame index it landed on.

    The first frame inference samples is wherever this seek lands, not
    ``round(start * fps)``. Shared so that rebuilding the sampled grid later
    (Zooniverse validation) lands on the same frame inference did.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, sampling_start * 1000.0)
    return int(cap.get(cv2.CAP_PROP_POS_FRAMES))


def _predict_kwargs(conf, imgsz, half=None) -> dict:
    """The ``model.predict`` settings every inference path shares.

    ``half`` is passed on only when given: video inference runs FP16 from
    ``ml_inference.half``, prediction on frame files leaves it to Ultralytics.
    """
    kwargs = dict(
        conf=conf,
        imgsz=imgsz,
        iou=config.ml_nms_iou,
        agnostic_nms=config.ml_nms_agnostic,
        verbose=False,
        project=None,
        name="ul_predict",
        exist_ok=True,
        save=False,
    )
    if half is not None:
        kwargs["half"] = half
    return kwargs


def run_yolo_inference(
    video_url,
    model_path,
    conf,
    imgsz,
    output_csv,
    true_fps,
    vid_stride,
    drop_id,
    sampling_start,
    sampling_end,
    second_stage=None,
    reader=None,
):
    """Executes YOLO inference correctly, processing the video stream and writing CSV.

    `second_stage` overrides ``ml_inference.classifier.enabled`` for this call
    (None = follow config). Off keeps YOLO's own class in `class`: a binary
    detector run whose counts must not be split by another project's species.
    `reader` overrides ``ml_inference.video_reader`` (``cv2`` / ``ffmpeg``);
    ffmpeg is for raw multi-stream camera files (see ffmpeg_sampled_frames).
    """
    reader = reader or config.ml_video_reader

    # Ensure output directory exists
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    try:
        if not Path(model_path).exists():
            logging.error(
                f"Model weights not found at {model_path}. Please check configuration."
            )
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        half = config.ml_half
        model = get_cached_video_model(model_path, half)

        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_url} during inference.")

        # cv2 lands wherever its seek lands; ffmpeg counts from frame 0, so it
        # starts at the frame the start time names (0 for a whole video).
        if reader == "ffmpeg":
            current_frame = int(round(sampling_start * true_fps))
        else:
            current_frame = seek_to_sampling_start(cap, sampling_start)

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_duration_seconds = total_video_frames / true_fps
        if sampling_end is not None and sampling_end > actual_duration_seconds:
            raise ValueError(
                f"{drop_id}: sampling_end={sampling_end:.0f}s exceeds actual video "
                f"duration ({actual_duration_seconds:.0f}s). Video may be truncated "
                f"or sampling window is wrong."
            )
        # `if sampling_end` is NOT enough: NaN is truthy, so a missing window
        # (blank in SharePoint -> NaN through pandas) passed this guard and
        # killed the drop on int(nan). Ingest now rejects such deployments, so
        # reaching here means something upstream regressed - fail with a
        # message that says which field, not a bare ValueError.
        if sampling_end is not None and math.isnan(sampling_end):
            raise ValueError(
                f"{drop_id}: sampling_end is NaN (no sampling window in the "
                f"deployment metadata). This drop should have been caught at "
                f"ingest as a validation_error."
            )
        end_frame = int(sampling_end * true_fps) if sampling_end else total_video_frames
        end_frame = min(end_frame, total_video_frames)
        total_frames_to_process = max(1, (end_frame - current_frame) // vid_stride)

        frames_processed = 0
        batch_size = inference_batch_size(int(imgsz))
        classifier = get_cached_classifier() if second_stage is not False else None

        # Write to a temp file and atomically rename only after the loop finishes
        # cleanly, so an interrupted run never leaves a truncated CSV that the
        # skip-existing check would mistake for a completed one.
        tmp_csv = f"{output_csv}.partial"
        with open(tmp_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(_raw_header(classifier))

            # Decode on a producer thread, predict on this one. cv2 decodes
            # EVERY frame on the CPU (grab() skips only the colour conversion),
            # so at stride 10 the decoder does ten frames of work per inferred
            # frame and was the bulk of a GPU job's wall time while the GPU sat
            # idle waiting for the next batch. cv2 releases the GIL inside
            # read()/grab(), so one plain thread overlaps the two. The queue is
            # bounded so a slow GPU never lets decoded frames pile up in RAM
            # (two batches of 1080p is ~400 MB). Frame indexes stay the exact
            # absolute `current_frame` so downstream seeks remain pixel-perfect.
            # YOLO runs in batches: single-frame predicts pay full Python/
            # pre-process/NMS overhead per call and leave the GPU at ~0% util.
            frames: Queue = Queue(maxsize=batch_size * 2)
            end_marker = object()
            stop = threading.Event()
            decode_error: list = []

            def _put(item) -> None:
                while not stop.is_set():
                    try:
                        frames.put(item, timeout=0.5)
                        return
                    except Full:
                        continue

            def _decode_ffmpeg() -> None:
                try:
                    n_frames = len(range(current_frame, end_frame, vid_stride))
                    for item in ffmpeg_sampled_frames(
                        video_url, current_frame, vid_stride, n_frames
                    ):
                        if stop.is_set():
                            break
                        _put(item)
                except BaseException as e:  # surfaced on the main thread below
                    decode_error.append(e)
                finally:
                    _put(end_marker)

            def _decode() -> None:
                if reader == "ffmpeg":
                    _decode_ffmpeg()
                    return
                frame_idx = current_frame
                try:
                    while not stop.is_set():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if (
                            sampling_end is not None
                            and frame_idx / true_fps > sampling_end
                        ):
                            break
                        _put((frame_idx, frame))
                        # Fast forward vid_stride frames using grab()
                        end_of_video = False
                        for _ in range(vid_stride - 1):
                            if not cap.grab():
                                end_of_video = True
                                break
                            frame_idx += 1
                        if end_of_video:
                            break
                        frame_idx += 1
                except BaseException as e:  # surfaced on the main thread below
                    decode_error.append(e)
                finally:
                    _put(end_marker)

            batch_frames: list = []
            batch_indices: list = []

            def _flush_batch() -> None:
                nonlocal frames_processed
                if not batch_frames:
                    return
                results = model.predict(
                    source=batch_frames,
                    **_predict_kwargs(float(conf), int(imgsz), half=half),
                )
                # IMPORTANT: rows store the absolute frame_idx, NOT index // stride,
                # extraction tools seek back to this exact frame.
                writer.writerows(
                    _detection_rows(
                        results,
                        batch_frames,
                        batch_indices,
                        [idx / true_fps for idx in batch_indices],
                        model.names,
                        classifier,
                    )
                )
                frames_processed += len(batch_frames)
                percent = min(100.0, (frames_processed / total_frames_to_process) * 100)
                logging.info(
                    f"Inference progress for {drop_id}: {frames_processed}/"
                    f"{total_frames_to_process} frames ({percent:.1f}%)"
                )
                batch_frames.clear()
                batch_indices.clear()

            decoder = threading.Thread(
                target=_decode, name=f"decode-{drop_id}", daemon=True
            )
            decoder.start()
            try:
                while True:
                    try:
                        item = frames.get(timeout=0.5)
                    except Empty:
                        if not decoder.is_alive():
                            break  # decoder died before posting its end marker
                        continue
                    if item is end_marker:
                        break
                    frame_idx, frame = item
                    batch_frames.append(frame)
                    batch_indices.append(frame_idx)
                    if len(batch_frames) >= batch_size:
                        _flush_batch()
                # Flush any frames left in the final partial batch.
                _flush_batch()
            finally:
                # A predict() failure must not leave the decoder blocked on a
                # full queue holding the capture open for the rest of the batch.
                stop.set()
                decoder.join()
            if decode_error:
                raise RuntimeError(
                    f"{drop_id}: video decode failed mid-run: {decode_error[0]}"
                ) from decode_error[0]

        cap.release()
        # A failed read()/grab() ends the decode loop exactly like the end of the
        # video, so a stalled read promoted a near-empty CSV as finished (UoA
        # Vault on NeSI, 2026-11-11: 3 of 2,121 frames, every chapter). Leave it
        # .partial and fail the drop instead.
        if frames_processed < config.ml_min_frame_coverage * total_frames_to_process:
            raise RuntimeError(
                f"{drop_id}: read {frames_processed} of {total_frames_to_process} "
                f"sampled frames before the capture stopped; left {tmp_csv} "
                "unpromoted. Check the file reads end to end (ffmpeg -f null)."
            )
        # Atomic promote, only reached if the loop above completed without error.
        Path(tmp_csv).replace(output_csv)

        logging.info(f"Inference complete. Output saved to {output_csv}")

    except Exception as e:
        logging.error(f"Inference failed for {drop_id}: {e}")
        raise


def predict_on_frame_paths(
    frame_paths: list,
    timestamps: list,
    output_csv: Path,
    *,
    model=None,
    model_path=None,
    confidence: float = None,
    imgsz: int = None,
    fps: float = None,
    second_stage=None,
) -> Path:
    """Run batched YOLO inference on a list of JPEG paths and write a raw CSV.

    Output schema matches `run_yolo_inference`:
    ``frame, time_seconds, class, confidence, x, y, w, h``.
    Output is consumed downstream by ``build_coco_from_raw_csv`` without
    modification, same shape as the video-inference path.

    Batched predict (single call across the whole frame list) amortises
    GPU/CPU dispatch overhead vs. per-frame calls.

    Args:
        frame_paths: JPEG paths to predict on.
        timestamps: Parallel list of absolute video timestamps in seconds.
            Written to the CSV's ``time_seconds`` column.
        output_csv: Where to write the raw CSV.
        model: Preloaded YOLO instance. If None, loads from ``model_path``.
        model_path: Path to a YOLO weights file (used when ``model`` is None).
        confidence: Override ``config.confidence_threshold``.
        imgsz: Override the size resolved from the checkpoint (``model_meta.resolve_imgsz``).
        fps: Source video fps for synthesising the ``frame`` index column.
            None → frame=-1 (unknown, used when video isn't re-opened).
        second_stage: Overrides ``ml_inference.classifier.enabled`` for this
            call, as in `run_yolo_inference` (None = follow config). False
            keeps YOLO's own class in ``class``.
    """
    if len(frame_paths) != len(timestamps):
        raise ValueError(
            f"frame_paths ({len(frame_paths)}) and timestamps ({len(timestamps)}) "
            "must be parallel lists of equal length."
        )

    if model is None:
        if model_path is None:
            raise ValueError(
                "predict_on_frame_paths: pass either `model` or `model_path`."
            )
        logging.info(f"Loading model for inference: {Path(model_path).name}")
        model = YOLO(str(model_path))

    conf = (
        float(confidence)
        if confidence is not None
        else float(config.confidence_threshold)
    )
    img_size = (
        int(imgsz)
        if imgsz is not None
        else resolve_imgsz(model_path or "", model=model)
    )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    sources = [str(p) for p in frame_paths]
    logging.info(f"Running batched inference on {len(sources)} frame(s)")
    all_results = model.predict(source=sources, **_predict_kwargs(conf, img_size))

    classifier = get_cached_classifier() if second_stage is not False else None
    # The classifier crops from the frame pixels; YOLO was given paths, so read
    # the images only when there is a classifier to feed.
    frames = [cv2.imread(p) for p in sources] if classifier is not None else None
    frame_indices = [int(round(t * fps)) if fps else -1 for t in timestamps]
    rows = _detection_rows(
        all_results,
        frames,
        frame_indices,
        [float(t) for t in timestamps],
        model.names,
        classifier,
    )
    n_detections = len(rows)
    with open(output_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_raw_header(classifier))
        writer.writerows(rows)

    logging.info(
        f"{n_detections} detection(s) across {len(frame_paths)} frame(s) → {output_csv}"
    )
    return Path(output_csv)


def rerun_inference_on_extracted_frames(drop_id: str, frames_df) -> None:
    """Re-run YOLO on already-extracted JPGs and write the COCO JSON.

    Used on the Zooniverse → BIIGLE path: frames are extracted at
    volunteer-identified peak timestamps where the ML raw CSV has no
    detections, so the upstream extractor is invoked with
    ``write_coco=False`` and this function writes the COCO from a fresh
    inference pass instead. The extractor must be called with
    ``write_coco=False`` on this path, otherwise this function would be
    overwriting whatever it wrote.

    Single pipeline-model pass. (Was a species+binary IoU-merged ensemble
    until the binary model's retirement, 2026-08-21 — measured
    class-agnostically, the species model finds strictly more fish than
    the binary model did, so the ensemble's second pass added nothing;
    the species model's own ``fish`` catch-all covers species it can't
    identify.)

    Lives in ``run_inference.py`` because the dominant work is inference;
    the COCO build/write tail reuses ``build_coco_from_raw_csv`` from
    ``spyfish.extraction.extract_frames``.
    """
    import json

    from spyfish.extraction.extract_frames import build_coco_from_raw_csv

    extracted = (
        frames_df[frames_df["FramePath"].notna()] if frames_df is not None else None
    )
    if extracted is None or extracted.empty:
        logging.warning(f"{drop_id}: no extracted frames to run inference on.")
        return

    frame_paths = [str(path) for path in extracted["FramePath"].tolist()]
    timestamps = [
        float(timestamp)
        for timestamp in extracted[config.csv_clip_max_time_column].tolist()
    ]

    fresh_raw_csv = config.get_zooniverse_frames_raw_csv_path(drop_id)

    predict_on_frame_paths(
        frame_paths=frame_paths,
        timestamps=timestamps,
        output_csv=fresh_raw_csv,
        model=get_cached_pipeline_model(),
    )

    # Image dimensions: read from the first JPG via cv2, the extractor
    # already baked rotation into the pixels, so this is what BIIGLE will
    # display against. Saves us threading the dimensions through the caller.
    image_width, image_height = 0, 0
    if frame_paths:
        sample = cv2.imread(frame_paths[0])
        if sample is not None:
            image_height, image_width = sample.shape[:2]

    frame_records = [
        {
            "image_id": image_id,
            "file_name": Path(frame_path).name,
            "time_of_max": float(timestamp),
            "drop_id": drop_id,
            "selection_reason": "",
            "img_w": image_width,
            "img_h": image_height,
        }
        for image_id, (frame_path, timestamp) in enumerate(
            zip(frame_paths, timestamps), start=1
        )
    ]
    coco_path = config.get_coco_annotations_path(drop_id)
    coco_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_coco = build_coco_from_raw_csv(str(fresh_raw_csv), frame_records)
    coco_path.write_text(json.dumps(fresh_coco, indent=2))
    logging.info(
        f"{drop_id}: wrote COCO from fresh inference, "
        f"{len(fresh_coco.get('annotations', []))} annotations on "
        f"{len(fresh_coco.get('images', []))} frames."
    )


def main(args):
    """Entry point called by the orchestrator (ml_runner.py) with an args dict."""
    drop_id = args.get("drop_id")
    video_url = args.get("video_url")
    sampling_start = args.get("sampling_start")
    sampling_end = args.get("sampling_end")

    if sampling_start is None or sampling_end is None:
        raise ValueError(
            f"Missing mandatory sampling metadata for {drop_id}. Both start and end times must be provided."
        )

    sampling_start = float(sampling_start)
    sampling_end = float(sampling_end)
    model_path = args.get("model_path")
    ml_fps = float(args.get("ml_fps", config.ml_fps))
    imgsz = (
        int(args["imgsz"])
        if args.get("imgsz") is not None
        else resolve_imgsz(model_path)
    )
    conf = float(args.get("confidence_threshold", config.confidence_threshold))
    output_csv = args.get("output_csv")

    logging.info(f"Starting YOLO inference on {drop_id}")
    logging.info(f"Video Source: {video_url}")
    logging.info(f"Model: {model_path}")

    true_fps = get_video_fps(video_url)
    vid_stride = inference_stride(true_fps, ml_fps)
    logging.info(
        f"Video FPS: {true_fps:.2f}, target ML FPS: {ml_fps}, stride: {vid_stride}, Confidence: {conf}"
    )

    # Launch modularized inference logic
    run_yolo_inference(
        video_url,
        model_path,
        conf,
        imgsz,
        output_csv,
        true_fps,
        vid_stride,
        drop_id,
        sampling_start,
        sampling_end,
    )
