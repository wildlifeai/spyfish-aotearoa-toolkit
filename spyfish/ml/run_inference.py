import sys
import time
import csv
import logging
from pathlib import Path
import cv2

from ultralytics import YOLO


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path} to read FPS. File may be missing or corrupt.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Video {video_path} reported an invalid FPS of {fps}.")
    return fps


def run_yolo_inference(video_url, model_path, conf, output_csv, true_fps, vid_stride, drop_id, sampling_start, sampling_end):
    """Executes YOLO inference correctly, processing the video stream and writing CSV."""

    # Ensure output directory exists
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)


    try:
        if not Path(model_path).exists():
            logging.error(f"Model weights not found at {model_path}. Please check configuration.")
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        model = YOLO(model_path)

        # Run prediction
        results = model.predict(source=video_url, conf=float(conf), stream=True, vid_stride=vid_stride)

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'time_seconds', 'class', 'confidence', 'x', 'y', 'w', 'h'])

            for idx, r in enumerate(results):
                # Calculate exact second for this physical frame
                physical_frame = idx * vid_stride
                real_video_seconds = physical_frame / true_fps

                # If a sampling offset is provided, we skip inference outside the permitted box
                if sampling_start > 0 and real_video_seconds < sampling_start:
                    continue
                if sampling_end is not None and real_video_seconds > sampling_end:
                    # Video is finished based on metadata limits
                    break

                # In order for Zooniverse selections to exactly match the metadata time,
                # we record the ML timeline starting at T=0 from the SamplingStart value forward.
                ml_timeline_seconds = real_video_seconds - sampling_start if sampling_start > 0 else real_video_seconds

                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates (center x, center y, width, height format)
                    x, y, w, h = box.xywh[0].tolist()
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]

                    writer.writerow([idx, ml_timeline_seconds, class_name, confidence, x, y, w, h])

        logging.info(f"Inference complete. Output saved to {output_csv}")

    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Exception during inference: {e}")
        sys.exit(1)


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Run YOLO inference on a video.')
    parser.add_hide = True # Internal flag for orchestrator

    # If running standalone via CLI, parse args.
    # If called from orchestrator, args will be passed as a dict or object.

    from spyfish.config import config

    # Default values from config
    repo_root = Path(__file__).parent.parent.parent

    if args is None:
        # Standalone manual run
        logging.info("Running in standalone manual mode.")
        drop_id, _, sampling_start, sampling_end = config.test_drops[0]
        video_url = str(repo_root / config.mock_video_dir / f"{drop_id}.mp4")
        model_path = config.mock_model_path if config.mock_model_path else "mock_model.pt"
        vid_stride = int(config.frame_skip)
        conf = float(config.confidence_threshold)
        output_dir = repo_root / config.local_manifest_dir_path
        model_name = Path(model_path).stem
        output_csv = str(output_dir / f"{drop_id}_{model_name}_raw.csv")
    else:
        # Called from Orchestrator (ml_runner.py)
        drop_id = args.get('drop_id')
        video_url = args.get('video_url')
        sampling_start = args.get('sampling_start', 0)
        sampling_end = args.get('sampling_end')
        model_path = args.get('model_path')
        vid_stride = int(args.get('frame_skip', config.frame_skip))
        conf = float(args.get('confidence_threshold', config.confidence_threshold))
        output_csv = args.get('output_csv')

    logging.info(f"Starting YOLO inference on {drop_id}")
    logging.info(f"Video Source: {video_url}")
    logging.info(f"Model: {model_path}")
    logging.info(f"Frame Skip: {vid_stride}, Confidence: {conf}")

    true_fps = get_video_fps(video_url)
    logging.info(f"Actual Video FPS: {true_fps:.2f}, Stride: {vid_stride}")

    # Launch modularized inference logic
    run_yolo_inference(video_url, model_path, conf, output_csv, true_fps, vid_stride, drop_id, sampling_start, sampling_end)

if __name__ == "__main__":
    main()
