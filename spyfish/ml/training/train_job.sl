#!/bin/bash -e
#SBATCH --job-name=spyfish_train
#SBATCH --account=wildlife03546
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/spyfish_train_%j.out
#SBATCH --error=logs/spyfish_train_%j.err

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
module load Python/3.11.6-foss-2023a
module load CUDA/11.0.2

# TODO update to your venv
VENV=/nesi/project/uoa04631/mussels-0115/bin/activate
# TODO update to where this repo is checked out on NeSI
PROJECT_DIR=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit

source "${VENV}"
cd "${PROJECT_DIR}"
mkdir -p logs

echo "Starting Spyfish retraining on $(hostname)"
nvidia-smi || true

python run_pipeline.py --retrain

echo "Retraining job complete. Outputs in process_files/training/runs/ and process_files/training/results/."
