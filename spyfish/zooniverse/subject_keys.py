"""
Zooniverse subject metadata keys.

`upload.py` writes these keys onto every new subject. The live parser
(`parse_classifications.py`) and the legacy normalizer (`legacy_extract.py`)
read from this single source of truth, so producer and consumer can't drift.

If you change a key here, change it in upload.py at the same time.
"""

# Validation manifest columns naming the rendered image per row. Here, not in
# `validation_frames.py`, so `upload.py` can read them without importing the
# renderer (which pulls in cv2 and the YOLO inference module). The renderer
# writes these columns; the uploader reads them.
FRAME_PATH_COLUMN = "FramePath"
CROP_PATH_COLUMN = "CropPath"


class SubjectKeys:
    VIDEO_FILENAME = "#VideoFilename"
    UPL_SECONDS = "UplAbsSeconds"
    SUBJECT_TYPE = "#SubjectType"
    TIME_OF_MAX = "#TimeOfMaxAbsSeconds"
    SITE_NAME = "#siteName"
    LINK_TO_RESERVE = "LinkToMarineReserve"
    EVENT_DATE = "#EventDate"

    # Subset that must be present on every classification for the parser to do
    # anything useful. Subjects missing any of these will be flagged.
    REQUIRED = (VIDEO_FILENAME, SUBJECT_TYPE, UPL_SECONDS)
