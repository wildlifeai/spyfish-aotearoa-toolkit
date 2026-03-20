import csv
import logging
from pathlib import Path

import cv2
from ultralytics import YOLO

from spyfish.config.wrapper import config


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
        end_frame = int(sampling_end * true_fps) if sampling_end else total_video_frames
        end_frame = min(end_frame, total_video_frames)
        total_frames_to_process = max(1, (end_frame - current_frame) // vid_stride)

        frames_processed = 0

        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["frame", "time_seconds", "class", "confidence", "x", "y", "w", "h"]
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                real_video_seconds = current_frame / true_fps
                if sampling_end is not None and real_video_seconds > sampling_end:
                    break

                # Absolute video timestamp — seconds from the start of the video file.
                ml_timeline_seconds = real_video_seconds

                # Run prediction on single frame
                # TODO checkif we need this, project=None prevents the creation of the 'runs' directory
                results = model.predict(
                    source=frame,
                    conf=float(conf),
                    imgsz=int(imgsz),
                    verbose=False,
                    project=None,
                    save=False,
                )
                r = results[0]

                frames_processed += 1
                if (
                    frames_processed % config.log_interval_frames == 0
                    or frames_processed == total_frames_to_process
                ):
                    percent = (frames_processed / total_frames_to_process) * 100
                    logging.info(
                        f"Inference progress for {drop_id}: {frames_processed}/{total_frames_to_process} frames ({percent:.1f}%) at {real_video_seconds:.1f}s"
                    )

                boxes = r.boxes
                for box in boxes:
                    x, y, w, h = box.xywh[0].tolist()
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]

                    # IMPORTANT: Store absolute current_frame, NOT index // stride
                    # This ensures extraction tools can seek back pixel-perfectly.
                    writer.writerow(
                        [
                            current_frame,
                            ml_timeline_seconds,
                            class_name,
                            confidence,
                            x,
                            y,
                            w,
                            h,
                        ]
                    )

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

        cap.release()

        logging.info(f"Inference complete. Output saved to {output_csv}")

    except Exception as e:
        logging.error(f"Inference failed for {drop_id}: {e}")
        raise


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO inference on a video.")
    parser.add_hide = True  # Internal flag for orchestrator

    # If running standalone via CLI, parse args.
    # If called from orchestrator, args will be passed as a dict or object.
    if args is None:
        import argparse

        parser = argparse.ArgumentParser(description="Run YOLO inference on a video.")
        parser.add_argument("drop_id", type=str, help="The Drop ID to process.")
        cli_args = parser.parse_args()

        # Standalone manual run
        logging.info(f"Running in standalone manual mode for drop {cli_args.drop_id}.")
        drop_id = cli_args.drop_id

        # We don't need sampling_start/end for a full manual inference test right now
        # the user can provide it via arguments if needed in the future
        video_url = str(config.get_video_path(drop_id))
        # TODO make required if worried about dialing
        model_path = config.pipeline_model_path
        vid_stride = int(config.frame_skip)
        imgsz = int(config.imgsz)
        conf = float(config.confidence_threshold)
        model_name = Path(model_path).stem
        output_csv = str(config.get_raw_csv_path(drop_id, model_name))
    else:
        # Called from Orchestrator (ml_runner.py)
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
        vid_stride = int(args.get("frame_skip", config.frame_skip))
        imgsz = int(args.get("imgsz", config.imgsz))
        conf = float(args.get("confidence_threshold", config.confidence_threshold))
        output_csv = args.get("output_csv")

    logging.info(f"Starting YOLO inference on {drop_id}")
    logging.info(f"Video Source: {video_url}")
    logging.info(f"Model: {model_path}")
    logging.info(f"Frame Skip: {vid_stride}, Confidence: {conf}")

    true_fps = get_video_fps(video_url)
    logging.info(f"Actual Video FPS: {true_fps:.2f}, Stride: {vid_stride}")

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


if __name__ == "__main__":
    main()
