#!/bin/bash -e
#SBATCH --job-name=spyfish_train
#SBATCH --account=wildlife03546
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=1
#SBATCH --output=/nesi/project/wildlife03546/spyfish-play-new/slurm_logs/spyfish_train_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-play-new/slurm_logs/spyfish_train_%j.err

# Spyfish Aotearoa training job wrapper.
#
# Runs a full sweep over SWEEP_RUNS from sweep.py, then auto-generates the
# Markdown comparison report. Assumes prepare+split+assemble has already run
# (i.e. process_files/training/{binary,species}/data.yaml exist).
#
# To also run the data-prep steps on NeSI, swap the python command for:
#   python run_pipeline.py --retrain --sweep
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
PROJECT_DIR=/nesi/project/wildlife03546/spyfish-play-new

source "${VENV}"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs

echo "Starting Spyfish training sweep on $(hostname)"
nvidia-smi || true

# python -m spyfish.ml.training.sweep
# python run_pipeline.py --retrain --sweep
python -m spyfish.ml.training.sweep --species-only            

echo "Training job complete. Reports in process_files/training/runs/sweep_*/report.md"
