import csv
import logging
from functools import lru_cache
from pathlib import Path

import cv2
from ultralytics import YOLO

from spyfish.config.wrapper import config


@lru_cache(maxsize=2)
def get_cached_pipeline_model(kind: str):
    """Return a process-cached YOLO instance for the pipeline ``kind`` model.

    ``kind`` is ``"species"`` or ``"binary"`` (matches ``config.get_pipeline_model``).
    First call loads weights from disk; subsequent calls reuse the same
    instance, important on multi-drop batch runs where reloading per
    drop adds 1–3s of disk + GPU init overhead.
    """
    model_path = config.get_pipeline_model(kind)
    logging.info(f"Loading pipeline model '{kind}': {model_path.name}")
    return YOLO(str(model_path))


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(
            f"Could not open video {video_path} to read FPS. File may be missing or corrupt."
        )
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Video {video_path} reported an invalid FPS of {fps}.")
    return fps


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
):
    """Executes YOLO inference correctly, processing the video stream and writing CSV."""

    # Ensure output directory exists
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    try:
        if not Path(model_path).exists():
            logging.error(
                f"Model weights not found at {model_path}. Please check configuration."
            )
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        model = YOLO(model_path)

        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_url} during inference.")

        cap.set(cv2.CAP_PROP_POS_MSEC, sampling_start * 1000.0)

        # align current frame after precise seek
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_duration_seconds = total_video_frames / true_fps
        if sampling_end is not None and sampling_end > actual_duration_seconds:
            raise ValueError(
                f"{drop_id}: sampling_end={sampling_end:.0f}s exceeds actual video "
                f"duration ({actual_duration_seconds:.0f}s). Video may be truncated "
                f"or sampling window is wrong."
            )
        end_frame = int(sampling_end * true_fps) if sampling_end else total_video_frames
        end_frame = min(end_frame, total_video_frames)
        total_frames_to_process = max(1, (end_frame - current_frame) // vid_stride)

        frames_processed = 0
        batch_size = config.ml_batch_size

        # Write to a temp file and atomically rename only after the loop finishes
        # cleanly, so an interrupted run never leaves a truncated CSV that the
        # skip-existing check would mistake for a completed one.
        tmp_csv = f"{output_csv}.partial"
        with open(tmp_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"]
            )

            # Decode frames with cv2 (exact, unambiguous absolute indexing) but run
            # YOLO in batches. Single-frame predicts pay full Python/pre-process/NMS
            # overhead per call and leave the GPU idle (~0% util) between them, the
            # dominant cost on long videos. Batching keeps the GPU fed while the
            # frame index stays the exact absolute `current_frame`, so downstream
            # frame extraction still seeks back pixel-perfectly.
            batch_frames: list = []
            batch_indices: list = []

            def _flush_batch() -> None:
                nonlocal frames_processed
                if not batch_frames:
                    return
                results = model.predict(
                    source=batch_frames,
                    conf=float(conf),
                    imgsz=int(imgsz),
                    iou=config.ml_nms_iou,
                    agnostic_nms=config.ml_nms_agnostic,
                    verbose=False,
                    project=None,
                    name="ul_predict",
                    exist_ok=True,
                    save=False,
                )
                for res, frame_idx in zip(results, batch_indices):
                    t_seconds = frame_idx / true_fps
                    for box in res.boxes:
                        x, y, w, h = box.xywh[0].tolist()
                        # IMPORTANT: store absolute frame_idx, NOT index // stride,
                        # extraction tools seek back to this exact frame.
                        writer.writerow(
                            [
                                frame_idx,
                                t_seconds,
                                model.names[int(box.cls[0])],
                                float(box.conf[0]),
                                x,
                                y,
                                w,
                                h,
                            ]
                        )
                frames_processed += len(batch_frames)
                percent = min(100.0, (frames_processed / total_frames_to_process) * 100)
                logging.info(
                    f"Inference progress for {drop_id}: {frames_processed}/"
                    f"{total_frames_to_process} frames ({percent:.1f}%)"
                )
                batch_frames.clear()
                batch_indices.clear()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                real_video_seconds = current_frame / true_fps
                if sampling_end is not None and real_video_seconds > sampling_end:
                    break

                batch_frames.append(frame)
                batch_indices.append(current_frame)
                if len(batch_frames) >= batch_size:
                    _flush_batch()

                # Fast forward vid_stride frames using grab()
                end_of_video = False
                for _ in range(vid_stride - 1):
                    if not cap.grab():
                        end_of_video = True
                        break
                    current_frame += 1

                if end_of_video:
                    break

                current_frame += 1

            # Flush any frames left in the final partial batch.
            _flush_batch()

        cap.release()
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
        imgsz: Override ``config.imgsz``.
        fps: Source video fps for synthesising the ``frame`` index column.
            None → frame=-1 (unknown, used when video isn't re-opened).
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
    img_size = int(imgsz) if imgsz is not None else int(config.imgsz)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    sources = [str(p) for p in frame_paths]
    logging.info(f"Running batched inference on {len(sources)} frame(s)")
    all_results = model.predict(
        source=sources,
        conf=conf,
        imgsz=img_size,
        iou=config.ml_nms_iou,
        agnostic_nms=config.ml_nms_agnostic,
        verbose=False,
        project=None,
        name="ul_predict",
        exist_ok=True,
        save=False,
    )

    n_detections = 0
    with open(output_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"]
        )
        for result, t in zip(all_results, timestamps):
            frame_idx = int(round(t * fps)) if fps else -1
            for box in result.boxes:
                x, y, w, h = box.xywh[0].tolist()
                cls_id = int(box.cls[0])
                writer.writerow(
                    [
                        frame_idx,
                        float(t),
                        model.names[cls_id],
                        float(box.conf[0]),
                        x,
                        y,
                        w,
                        h,
                    ]
                )
                n_detections += 1

    logging.info(
        f"{n_detections} detection(s) across {len(frame_paths)} frame(s) → {output_csv}"
    )
    return Path(output_csv)


def _iou_xywh(a: tuple, b: tuple) -> float:
    """IoU for two boxes in YOLO center format (cx, cy, w, h)."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def merge_raw_csvs_by_iou(
    primary_csv: Path,
    fallback_csv: Path,
    output_csv: Path,
    iou_threshold: float = 0.5,
) -> Path:
    """Merge two raw YOLO CSVs by spatial IoU per frame; primary labels win.

    Used to combine species-model detections (specific labels, lower recall)
    with binary-model detections (generic 'fish' label, higher recall). For
    each frame:
      - All ``primary`` boxes are kept as-is.
      - Each ``fallback`` box is kept only if it does NOT overlap any primary
        box at IoU >= ``iou_threshold``, overlap means the primary already
        described the same fish more specifically.

    Result: union of detections, with the more specific label on overlap.
    """
    import pandas as pd

    primary_df = (
        pd.read_csv(primary_csv) if Path(primary_csv).exists() else pd.DataFrame()
    )
    fallback_df = (
        pd.read_csv(fallback_csv) if Path(fallback_csv).exists() else pd.DataFrame()
    )

    if primary_df.empty and fallback_df.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"]
            )
        return Path(output_csv)

    # Group fallback boxes by time so we can do the IoU check per-frame.
    rows = []
    rows.extend(primary_df.to_dict("records"))

    n_dropped = 0
    if not fallback_df.empty:
        primary_by_time = (
            primary_df.groupby("time_seconds") if not primary_df.empty else None
        )
        for _, row in fallback_df.iterrows():
            t = row["time_seconds"]
            box = (row["x"], row["y"], row["w"], row["h"])
            primary_frame_boxes = (
                primary_by_time.get_group(t)
                if primary_by_time is not None and t in primary_by_time.groups
                else None
            )
            overlapped = False
            if primary_frame_boxes is not None:
                for _, p in primary_frame_boxes.iterrows():
                    if (
                        _iou_xywh(box, (p["x"], p["y"], p["w"], p["h"]))
                        >= iou_threshold
                    ):
                        overlapped = True
                        break
            if overlapped:
                n_dropped += 1
            else:
                rows.append(row.to_dict())

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    logging.info(
        f"Merged inference CSVs → {output_csv}: {len(primary_df)} primary + "
        f"{len(fallback_df)} fallback - {n_dropped} overlapped = {len(rows)} total."
    )
    return Path(output_csv)


def rerun_inference_on_extracted_frames(drop_id: str, frames_df) -> None:
    """Re-run YOLO on already-extracted JPGs and write the COCO JSON.

    Used on the Zooniverse → BIIGLE path: frames are extracted at
    volunteer-identified peak timestamps where the ML raw CSV has no
    detections, so the upstream extractor is invoked with
    ``write_coco=False`` and this function writes the COCO from a fresh
    ensemble pass instead.

    Two-pass ensemble: species first (specific labels), binary second
    (high-recall "fish here"). Merged by IoU, species labels win on
    overlap; binary-only boxes stay as ``fish``. This catches fish in
    species the model can't identify (e.g. tarakihi).

    Reads ``frames_df`` to discover which JPGs were extracted (paths +
    timestamps), runs the two-pass inference, merges by IoU, then writes
    the COCO once. The extractor must be called with ``write_coco=False``
    on this path, otherwise this function would be overwriting whatever
    it wrote.

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

    species_raw = config.get_zooniverse_frames_raw_csv_path(drop_id, "species")
    binary_raw = config.get_zooniverse_frames_raw_csv_path(drop_id, "binary")
    fresh_raw_csv = config.get_zooniverse_frames_raw_csv_path(drop_id, "merged")

    predict_on_frame_paths(
        frame_paths=frame_paths,
        timestamps=timestamps,
        output_csv=species_raw,
        model=get_cached_pipeline_model("species"),
    )
    predict_on_frame_paths(
        frame_paths=frame_paths,
        timestamps=timestamps,
        output_csv=binary_raw,
        model=get_cached_pipeline_model("binary"),
    )
    merge_raw_csvs_by_iou(
        primary_csv=species_raw,
        fallback_csv=binary_raw,
        output_csv=fresh_raw_csv,
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
    imgsz = int(args.get("imgsz", config.imgsz))
    conf = float(args.get("confidence_threshold", config.confidence_threshold))
    output_csv = args.get("output_csv")

    logging.info(f"Starting YOLO inference on {drop_id}")
    logging.info(f"Video Source: {video_url}")
    logging.info(f"Model: {model_path}")

    true_fps = get_video_fps(video_url)
    vid_stride = max(1, round(true_fps / ml_fps))
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
