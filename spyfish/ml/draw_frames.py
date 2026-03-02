"""
Draw bounding boxes from a raw ML CSV onto video frames and save as JPEGs.

The raw CSV `time_seconds` column is ML-relative (starts at 0 from SamplingStart).
To seek the correct video frame, we add sampling_start back to get the absolute video time.
The output filename uses the absolute video time for easy cross-referencing.
"""
import os
import cv2
import pandas as pd
import logging


def draw_boxes_on_frames(video_path, raw_csv_path, output_dir, frame_list,
                         confidence_threshold, sampling_start):
    """
    Draws ML bounding boxes on specific frames and saves them as JPEGs.

    Args:
        video_path: Path to the source video file.
        raw_csv_path: Path to the raw ML CSV (frame, time_seconds, class, confidence, x, y, w, h).
        output_dir: Directory to save the annotated JPEG frames.
        frame_list: List of CSV 'frame' indices to draw (from the raw CSV 'frame' column).
        confidence_threshold: Only draw boxes above this confidence.
        sampling_start: Seconds offset where ML analysis began in the video.
    """
    logging.info(f"Drawing {len(frame_list)} frames from {video_path}")

    if not os.path.exists(video_path):
        logging.error(f"Video not found: {video_path}")
        return []

    if not os.path.exists(raw_csv_path):
        logging.error(f"Raw CSV not found: {raw_csv_path}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_csv_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    saved_frames = []

    for csv_frame in frame_list:
        frame_rows = df[df['frame'] == csv_frame]
        if frame_rows.empty:
            logging.warning(f"No CSV data for frame {csv_frame}")
            continue

        # ML time is relative to sampling_start. Add it back for the real video position.
        ml_time = frame_rows['time_seconds'].iloc[0]
        video_time = ml_time + sampling_start
        video_frame_num = int(video_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_num)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Could not read video frame {video_frame_num}")
            continue

        # Draw boxes — YOLO xywh format: x,y are CENTER, w,h are width/height
        for _, row in frame_rows.iterrows():
            if row['confidence'] < confidence_threshold:
                continue

            cx, cy, w, h = row['x'], row['y'], row['w'], row['h']
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            conf = row['confidence']
            cls = row['class']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Filename uses absolute video time
        h_part = int(video_time // 3600)
        m_part = int((video_time % 3600) // 60)
        s_part = int(video_time % 60)
        ms_part = int((video_time % 1) * 1000)
        time_str = f"{h_part:02d}h{m_part:02d}m{s_part:02d}s{ms_part:03d}ms"

        out_path = os.path.join(output_dir, f"frame_{time_str}_f{video_frame_num}.jpg")
        cv2.imwrite(out_path, frame)
        saved_frames.append(out_path)
        logging.info(f"Saved {out_path} (csv_frame={csv_frame}, video_frame={video_frame_num})")

    cap.release()
    return saved_frames
