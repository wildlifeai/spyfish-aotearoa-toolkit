"""File names of the two-stage outputs that one script writes and another reads.

All of them live under ``crops.OUT_DIR`` (``paths.sub_dirs.two_stage``, or its
``uoa/`` subfolder) or in a run folder below it. Only names shared between
modules are here; a file a single script writes and nobody reads back keeps
its name in that script.
"""

# build_embeddings.py writes these; the probe, the fine-tune and the demo read them.
CROPS_META = "crops_meta.csv"  # one row per crop; read it through crops.read_crops_meta
CROPS_DIR = "crops"  # 224 px thumbnails, {crop_id}.jpg
EMBEDDINGS = "embeddings.npy"  # image embeddings, same row order as CROPS_META
TEXT_EMBEDDINGS = "text_embeddings.npy"  # one per line of SPECIES_LIST
SPECIES_LIST = "species.txt"

# run_yolo_on_frames.py: the current detector on the expert frames
# (read it through crops.read_predictions).
YOLO_PREDICTIONS = "yolo_predictions.csv"

# hard_negatives.py, one folder per source (negatives/, negatives_upload/, ...).
NEGATIVES_META = "negatives_meta.csv"
NEGATIVES_EMBEDDINGS = "negatives_embeddings.npy"

# The model-only review round trip: upload_model_only_for_review.py writes a
# manifest per upload, review_outcome.py turns the re-pulled volumes into outcomes.
REVIEW_UPLOAD_PREFIX = "review_upload_"
REVIEW_UPLOAD_GLOB = f"{REVIEW_UPLOAD_PREFIX}*.csv"
REVIEW_OUTCOME = "review_outcome.csv"

# evaluate.py and demo_two_stage.py outputs that other scripts read back.
OOF_PREDICTIONS = "oof_predictions.csv"
DEMO_DIR = "demo"
DEMO_BOXES = "demo_boxes.csv"

# finetune.py, inside its run folder.
BEST_CHECKPOINT = "best.pt"  # held-out mode: the best epoch on the held-out fold
FINAL_CHECKPOINT = "final.pt"  # all-train mode: what inference loads
FINETUNE_REPORT = "finetune_report.md"
FINETUNE_PREDICTIONS = "finetune_predictions.csv"
CLASS_TABLE = "classes.json"  # the class decision log beside a checkpoint

# uoa/scripts/two_stage/detector_dataset.py: which deployments the detector held out as val.
DETECTOR_SPLIT = "detector_split.csv"
