"""
Draw bounding boxes from a raw ML CSV onto video frames and save as JPEGs.

The raw CSV `time_seconds` column stores absolute video timestamps (seconds from video start).
The output filename uses this absolute video time for easy cross-referencing.
"""

import hashlib
import logging
import os
from typing import Callable

import cv2
import pandas as pd

from spyfish.config.species import species_registry
from spyfish.extraction.extract_frames import read_frame_from_cap, read_video_rotation
from spyfish.utils import generate_frame_filename, seconds_to_time

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Turns a raw-CSV class name into the name a person reads. Every species name
# shown to a person is the common name (`SpeciesRegistry.label_for`: "Snapper",
# "fish" for the bucket class, the input echoed back when unmapped); the class
# name itself stays the key for colours and for the CSV. Defaults to identity
# so the pure helpers stay testable without a class map.
Labeller = Callable[[str], str]


def common_name_labeller() -> Labeller:
    """The project convention: common names on anything a person looks at."""
    return species_registry().label_for


def class_colour(name: str) -> tuple:
    """A stable BGR colour per class name.

    Hashed, not a palette, so the same species is the same colour in every
    QA frame however many classes a model has, and a new model's classes
    need no table edit. Several species share most frames, which is why the
    old single-green drawing was hard to read.
    """
    h = int(hashlib.md5(str(name).encode()).hexdigest()[:6], 16)  # nosec B324
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255)


def draw_banner(frame, text: str) -> None:
    """Burn a one-line banner into the top-left corner: white on a black box.

    The banner is what makes a QA frame readable on its own, away from the
    folder it came from: which model drew the boxes, at what confidence, at
    what second, and what it counted. Shared by the normal QA frames and the
    legacy review frames so the two read the same way side by side.
    """
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.8, 2)
    cv2.rectangle(frame, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, th + 10), _FONT, 0.8, (255, 255, 255), 2)


def count_summary(
    rows: pd.DataFrame, confidence_threshold: float, label: Labeller = str
) -> str:
    """``"Snapper 3, fish 1"`` for the boxes above threshold, or ``"nothing"``.

    ``label`` names each class for the reader (see ``common_name_labeller``);
    two classes that map to one label are counted together.
    """
    kept = rows[rows["confidence"] >= confidence_threshold]
    counts = kept["class"].map(label).value_counts().to_dict()
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing"


def draw_boxes(
    frame, rows: pd.DataFrame, confidence_threshold: float, label: Labeller = str
) -> None:
    """Draw raw-CSV detection rows (YOLO centre xywh, pixels) onto ``frame`` in place.

    The box colour is keyed on the class name, the tab text on ``label(class)``,
    so renaming what a person sees never changes which colour a species has.
    """
    for _, row in rows.iterrows():
        if row["confidence"] < confidence_threshold:
            continue
        cx, cy, w, h = row["x"], row["y"], row["w"], row["h"]
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        colour = class_colour(row["class"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        text = f"{label(row['class'])} {row['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.7, 2)
        top = max(y1 - th - 10, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + 10), colour, -1)
        cv2.putText(frame, text, (x1 + 3, top + th + 3), _FONT, 0.7, (255, 255, 255), 2)


def draw_boxes_on_video_frames(
    video_path,
    raw_csv_path,
    output_dir,
    frame_list,
    confidence_threshold,
    drop_id="UNKNOWN",
    model_name=None,
):
    """
    Draws ML bounding boxes on specific frames and saves them as JPEGs.

    Args:
        video_path: Path to the source video file.
        raw_csv_path: Path to the raw ML CSV (frame, time_seconds, class, confidence, x, y, w, h).
        output_dir: Directory to save the annotated JPEG frames.
        frame_list: List of CSV 'frame' indices to draw (from the raw CSV 'frame' column).
        confidence_threshold: Only draw boxes above this confidence.
        model_name: When given, each frame carries a banner naming the model,
            the confidence threshold, the video second and what was counted,
            same shape as the legacy review frames, so a QA frame says which
            model drew it without the folder name.
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
    # Container rotation is applied to the pixels, as the clean frames and the
    # legacy review frames do, so a box drawn here lands where inference saw it.
    rotation = read_video_rotation(cap)

    saved_frames = []
    label = common_name_labeller()

    for csv_frame in frame_list:
        frame_rows = df[df["frame"] == csv_frame]
        if frame_rows.empty:
            logging.warning(f"No CSV data for frame {csv_frame}")
            continue

        # The 'frame' column in the raw CSV is already an absolute frame index from the source video
        video_frame_num = csv_frame

        # time_seconds is already an absolute video timestamp
        video_time = frame_rows["time_seconds"].iloc[0]

        frame = read_frame_from_cap(
            cap,
            float(video_time),
            frame_index=int(video_frame_num),
            rotation=rotation,
        )
        if frame is None:
            continue

        # One colour per class (was a single green for everything), common
        # names on the tabs and in the banner.
        draw_boxes(frame, frame_rows, confidence_threshold, label)
        if model_name:
            draw_banner(
                frame,
                f"{model_name} conf>={confidence_threshold:g} "
                f"@ {seconds_to_time(float(video_time))}  |  "
                f"{count_summary(frame_rows, confidence_threshold, label)}",
            )

        # Filename uses absolute video time
        out_filename = generate_frame_filename(drop_id, video_time)

        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, frame)
        saved_frames.append(out_path)
        logging.info(
            f"Saved {out_path} (csv_frame={csv_frame}, video_frame={video_frame_num})"
        )

    cap.release()
    return saved_frames
