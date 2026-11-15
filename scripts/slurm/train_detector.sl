#!/bin/bash -e
#SBATCH --job-name=train_detector
#SBATCH --account=wildlife03546
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# Same GPU block as spyfish/ml/training/train_job.sl (h100 in genoa queued hours
# sooner than milan's a100 and runs faster; 2026-08-24). The first milan a100
# attempt died with "CUDA-capable device(s) is/are busy or unavailable" at
# model.to(device) after the GPU check had passed (job 10478735), so stay with
# the block that trains the species model.
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=h100:1
##SBATCH --partition=milan
##SBATCH --gpus-per-node=a100:1
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/train_detector_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/train_detector_%j.err

# WIP three-class detector (fish / Jasus edwardsii / bait) from the CFD base,
# plan §5.2. Trains on the dataset make_detector_dataset.py writes; does NOT
# touch pipeline_model/ or the species workspace, and nothing promotes it.
#
#   python scripts/two_stage/make_detector_dataset.py          # once, on a login node
#   sbatch scripts/slurm/train_detector.sl                  # 640, as the species model
#   sbatch --partition=milan --gpus-per-node=a100:1 --exclude=mg15 scripts/slurm/train_detector.sl
#   IMGSZ=1024 BATCH=8 sbatch scripts/slurm/train_detector.sl   # if small fish are the problem
#   DATA=process_files/training/detector_fish1/data.yaml sbatch scripts/slurm/train_detector.sl   # one-class
#
# l4 OOMs at batch 16 and silently halves it (slower and worse), so not l4.
# Same hyperparameters as the species training in config.yaml unless
# overridden on the sbatch line.

DATA="${DATA:-process_files/training/detector/data.yaml}"
MODEL="${MODEL:-process_files/models/base_model/cfd-yolov12x-1.00.pt}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-25}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
LR0="${LR0:-0.001}"
DROPOUT="${DROPOUT:-0.1}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"   # keep every 10th epoch too: best.pt is picked by mean mAP over classes
# Layers to keep fixed. 0 = all train (the Spyfish default). 10 freezes CFD's
# backbone exactly (YOLOv12x layers 0-8, 56% of the weights; 9 is a weightless
# Upsample), so DOC frames tune only the neck and head. Worked for UoA's 351
# frames (2026-11-11: val recall 0.70 -> 0.88 where the unfrozen run collapsed).
FREEZE="${FREEZE:-0}"
# Anything else, appended last so it wins: TRAIN_ARGS="box=7.5" sbatch ...
TRAIN_ARGS="${TRAIN_ARGS:-}"
DATA_NAME="$(basename "$(dirname "${DATA}")")"   # detector -> detector_..., detector_fish1 -> detector_fish1_...
RUN_NAME="${RUN_NAME:-${DATA_NAME}_$(date +%Y%m%d_%H%M%S)_img${IMGSZ}}"

# Shared environment (modules, venv, PROJECT_DIR, YOLO_CONFIG_DIR, cd into the
# repo); submit from the repo root like cpu.sl / gpu.sl.
# No CUDA module: the venv's torch 2.5.1+cu124 bundles its runtime (scripts/slurm/_env.sh),
# and the inference and fine-tune jobs run without one. train_job.sl still loads
# CUDA/11.0.2; it was dropped here after three milan failures (2026-11-04).
source scripts/slurm/_env.sh
yolo settings mlflow=False comet=False wandb=False dvc=False neptune=False clearml=False || true

spyfish_stamp_start "run=${RUN_NAME} data=${DATA} imgsz=${IMGSZ} batch=${BATCH}"
# No torch pre-check here: on milan a100s two jobs (10478735, 10481379) died at
# model.to(device) with "CUDA-capable device(s) is/are busy or unavailable"
# right after such a check had passed, which is what an exclusive-process or
# MIG card does to a second CUDA context. `yolo` must be the first and only
# CUDA user in this job. nvidia-smi only reads the driver and shows the mode.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,compute_mode,mig.mode.current,memory.used,memory.total --format=csv || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true

RUN_DIR="${PROJECT_DIR}/process_files/training/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
# Freeze what was trained on, like train.py does, so test contamination can be checked later.
python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from spyfish.ml.training.train import freeze_dataset_snapshot
print('snapshot:', freeze_dataset_snapshot(Path('${DATA}'), Path('${RUN_DIR}')))
" || echo "snapshot helper unavailable on this checkout; continuing" >&2

# Same augmentation and stability recipe as the species model, read from
# config.yaml (training.augmentation + training.stability: underwater HSV
# shifts, rotation, mixup, BGR swap, AMP off, box=5.0, warm-up, disk cache).
# The first detector run (detector_fish1_20261109) went without them, i.e.
# Ultralytics defaults with AMP on, so its 53% val recall against the species
# model's 74% was two recipes being compared, not two class lists.
RECIPE=$(python -c "
from spyfish.config.wrapper import config
print(' '.join(f'{k}={v}' for k, v in {**config.training_augmentation, **config.training_stability}.items()))
")
echo "recipe: ${RECIPE} freeze=${FREEZE} ${TRAIN_ARGS}"

yolo detect train \
    data="${DATA}" model="${MODEL}" epochs="${EPOCHS}" patience="${PATIENCE}" \
    imgsz="${IMGSZ}" batch="${BATCH}" optimizer="${OPTIMIZER}" lr0="${LR0}" dropout="${DROPOUT}" \
    save_period="${SAVE_PERIOD}" ${RECIPE} freeze="${FREEZE}" ${TRAIN_ARGS} \
    device=0 project="${PROJECT_DIR}/process_files/training/runs" name="${RUN_NAME}" exist_ok=True

echo "best weights: ${RUN_DIR}/weights/best.pt"
echo "Score it against the expert boxes on the test drops (an l4 job, NOT on the login node: 6k frames take 20 h on CPU):"
echo "  YOLO_ARGS=\"--model ${RUN_DIR}/weights/best.pt --out ${RUN_NAME}.csv --imgsz ${IMGSZ}\" RECALL_FILE=${RUN_NAME}.csv sbatch scripts/slurm/two_stage.sl yolo recall"
spyfish_end
