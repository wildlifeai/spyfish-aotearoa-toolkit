#!/bin/bash -e
#SBATCH --job-name=spyfish_train
#SBATCH --account=wildlife03546
#SBATCH --time=7:00:00

# GPU choice (billing weight per GPU-hour in brackets — sinfo TRESBillingWeights):
#   genoa: l4 [20], pro_6000 [130], h100 [200]; milan: a100 [90]. No a100 in genoa.
# l4 (22.5GB) OOMs at imgsz=640 batch=16 AND batch=32 and silently falls back to
# batch 8 (~3.4x slower per epoch, worse convergence) - do not train on it.
# pro_6000 is Blackwell/sm_120 and cannot run this venv's torch 2.5.1+cu124.
# h100 chosen 2026-08-24: sbatch --test-only put its start 6.5 h earlier than
# a100 (02:07 vs 08:45) - milan had 218 a100 jobs queued on 16 GPUs, genoa 19
# h100 jobs on 6 - and it is faster once running, so 200 vs 90 billing is close
# to cost-neutral. Swap the blocks below to move between GPUs.
##SBATCH --partition=genoa
##SBATCH --gpus-per-node=h100:1
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_train_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_train_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8


##SBATCH --partition=genoa
##SBATCH --gpus-per-node=l4:1

#SBATCH --partition=milan
#SBATCH --gpus-per-node=a100:1

# Spyfish Aotearoa training job wrapper.
#
# Runs the retraining pipeline end-to-end: data prep + species training.
# Optimizer / lr / dropout come from config.yaml's training section.
# Auto-promotion is on — models that beat production by `retrain_min_improvement_pct`
# are copied into pipeline_model/ at the end of the run.
#
# To scope the run, pass any subset of --data-prep / --species:
#   python run_pipeline.py --retrain --species              # just species training
#   python run_pipeline.py --retrain --data-prep            # rebuild dataset only
#


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
# mkdir -p slurm_logs

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
# python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
# nvidia-smi || true

# python run_pipeline.py --retrain
# python -m spyfish.biigle.biigle_to_yolo download-project --project-id 4626 --force
# python -m spyfish.biigle.biigle_to_yolo download-project --project-id 3711 --force
python run_pipeline.py --retrain --data-prep --species
# python run_pipeline.py --retrain --data-prep --dry-run

# python run_pipeline.py --biigle-sync   # pull latest annotations from BIIGLE API → raw CSVs on disk
# python run_pipeline.py --retrain --no-upload

echo "Retraining job complete. Outputs in process_files/training/runs/ and process_files/training/results/."
