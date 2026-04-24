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
# By default runs a full sweep over SWEEP_RUNS from sweep.py.
# To run a single binary+species train instead, swap the python command below.
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
# Path to the species data.yaml produced by prepare_training_data + assemble_yolo_dataset
DATA_YAML=${PROJECT_DIR}/process_files/training/species/data.yaml

source "${VENV}"
cd "${PROJECT_DIR}"
mkdir -p logs

echo "Starting Spyfish training sweep on $(hostname)"
nvidia-smi || true

SWEEP_NAME="sweep_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
python -m spyfish.ml.training.sweep --data "${DATA_YAML}" --sweep-name "${SWEEP_NAME}"

# Generate a Markdown report (tables + plots + example images) for the sweep
python -m spyfish.ml.training.sweep_report \
    --sweep-dir "process_files/training/runs/${SWEEP_NAME}"

echo "Training job complete. Report: process_files/training/runs/${SWEEP_NAME}/report.md"
