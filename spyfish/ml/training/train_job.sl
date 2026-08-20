#!/bin/bash -e
#SBATCH --job-name=spyfish_train
#SBATCH --account=wildlife03546
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# GPU choice (billing weight per GPU-hour in brackets — sinfo TRESBillingWeights):
#   genoa: l4 [20], pro_6000 [130], h100 [200]; milan: a100 [90]. No a100 in genoa.
# l4 (24GB) is ample for imgsz=640 batch=16 and the cheapest option.
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=l4:1
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_train_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_train_%j.err

# Spyfish Aotearoa training job wrapper.
#
# Runs the retraining pipeline end-to-end: data prep + binary + species training.
# Optimizer / lr / dropout come from config.yaml's training section.
# Auto-promotion is on — models that beat production by `retrain_min_improvement_pct`
# are copied into pipeline_model/ at the end of the run.
#
# To scope the run, pass any subset of --data-prep / --binary / --species:
#   python run_pipeline.py --retrain --species              # just species training
#   python run_pipeline.py --retrain --data-prep            # rebuild dataset only
#   python run_pipeline.py --retrain --data-prep --binary   # data prep + binary
#
# Before first use, update the three placeholders marked TODO:
#   1. SBATCH --account above
#   2. VENV path below
#   3. PROJECT_DIR below

module purge
# module load Python/3.11.6-foss-2023a
module load Python/3.10.5-gimkl-2022a
module load CUDA/11.0.2

# TODO update to your venv
VENV=/nesi/project/wildlife03546/kso_venv_0627/bin/activate
# TODO update to where this repo is checked out on NeSI
PROJECT_DIR=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit

source "${VENV}"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs

# Ultralytics keeps per-user defaults (runs_dir and friends) in
# ~/.config/Ultralytics/settings.json, shared by every project on this
# account. A YOLO call that forgets `project=` writes wherever that file
# points, which is whatever project first imported ultralytics. A repo-local
# config dir makes any such fallback land inside this repo instead.
# The dir must exist BEFORE ultralytics imports: with a missing dir the env
# var is silently ignored and everything falls back to /tmp.
export YOLO_CONFIG_DIR="${PROJECT_DIR}/.ultralytics"
mkdir -p "${YOLO_CONFIG_DIR}"

echo "Starting Spyfish retraining on $(hostname)"
nvidia-smi || true

python run_pipeline.py --retrain

echo "Retraining job complete. Outputs in process_files/training/runs/ and process_files/training/results/."
