import logging
import pandas as pd
from pathlib import Path

from sftk.common import LOCAL_DATA_FOLDER_PATH, CONFIDENCE_THRESHOLD, FRAME_RATE


class MLParser:
    """Parser for ML-generated annotations."""

    @staticmethod
    def process_annotations(
            ml_annotations_filename: str,
            extract_dir: str = LOCAL_DATA_FOLDER_PATH,
            confidence_threshold: float = CONFIDENCE_THRESHOLD,
            frame_rate: int = FRAME_RATE,
            window_seconds: int = 30
    ):
        """Processes the ML annotations and saves the parsed data.

        Args:
            extract_dir: Path to save the data folder to which the parsed data will be saved.
            ml_annotations_filename: Name of annotation file.
            confidence_threshold: Minimum confidence score to include an annotation.
            frame_rate: The frame rate of the video to calculate seconds from frame number.
            window_seconds: Time in seconds of window over which max count has to be checked
        """
        input_file_name = Path(extract_dir) / ml_annotations_filename
        output_file_name = Path(extract_dir) / (ml_annotations_filename.split(".")[0] + "_parsed.csv")
        annotations_df = pd.read_csv(input_file_name)
        logging.info(f"Processing ML annotations from {annotations_df.shape[0]} rows.")

        # 1. Filter by confidence score
        confident_annotations_df = annotations_df[annotations_df["conf"] >= confidence_threshold].copy()
        logging.info(f"{annotations_df.shape[0] - confident_annotations_df.shape[0]} rows filtered out due to low confidence.")

        # 2. Extract video_id and calculate seconds
        confident_annotations_df["video_id"] = confident_annotations_df["filename"].str.rsplit("_", n=1).str[0]
        if confident_annotations_df["video_id"].isna().any():
            raise ValueError("Some rows have invalid filename format for video_id extraction")

        confident_annotations_df["seconds"] = confident_annotations_df["frame_no"] / frame_rate

        # 3. Create 30-second window
        confident_annotations_df["window_30s"] = (confident_annotations_df["frame_no"] //
                                                  (frame_rate * window_seconds)).astype(int)

        # 4. Count detections per frame for each species
        frame_counts = (
            confident_annotations_df
            .groupby(["video_id", "class_id", "frame_no", "window_30s"], as_index=False)
            .size()
            .rename(columns={"size": "count_in_frame"})
        )

        # 5. Deterministic tie-break (highest count, then earliest frame)
        max_n_df = (
            frame_counts
            .sort_values(
                ["video_id", "class_id", "window_30s", "count_in_frame", "frame_no"],
                ascending=[True, True, True, False, True],
            )
            .drop_duplicates(subset=["video_id", "class_id", "window_30s"], keep="first")
            .reset_index(drop=True)
        )

        # 6. Rename columns to BIIGLE-compatible output schema
        max_n_df = max_n_df.rename(
            columns={
                "video_id": "DropID",
                "class_id": "ScientificName",
                "count_in_frame": "MaxInterval",
            }
        )

        # Convert frame to HH:MM:SS in BIIGLE-compatible time column
        max_n_df["TimeOfMax"] = pd.to_datetime(
            max_n_df["frame_no"] / frame_rate, unit="s"
        ).dt.strftime("%H:%M:%S")

        # 7. Add metadata columns
        max_n_df["AnnotatedBy"] = "ml_model"
        max_n_df["IntervalAnnotation"] = window_seconds
        max_n_df["ConfidenceAgreement"] = "NA"

        # 8. Select and reorder final columns to match biigle_parser format
        final_df = max_n_df[
            [
                "DropID",
                "ScientificName",
                "TimeOfMax",
                "MaxInterval",
                "AnnotatedBy",
                "IntervalAnnotation",
                "ConfidenceAgreement",
            ]
        ]

        logging.info(f"Saving parsed annotations to {output_file_name}")
        final_df.to_csv(output_file_name, index=False)

        return final_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        MLParser.process_annotations(ml_annotations_filename="ML_Annotations_Examples.csv")
    except FileNotFoundError:
        logging.error("Please provide the correct path to your ML annotations file.")
