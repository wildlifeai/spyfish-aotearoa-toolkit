#!/bin/bash -e
#SBATCH --job-name=spyfish_train
#SBATCH --account=wildlife03546
#SBATCH --time=10:00:00

# GPU choice (billing weight per GPU-hour in brackets — sinfo TRESBillingWeights):
#   genoa: l4 [20], pro_6000 [130], h100 [200]; milan: a100 [90]. No a100 in genoa.
# l4 (22.5GB) OOMs at imgsz=640 batch=16 AND batch=32 and silently falls back to
# batch 8 (~3.4x slower per epoch, worse convergence) - do not train on it.
# pro_6000 is Blackwell/sm_120 and cannot run this venv's torch 2.5.1+cu124.
# h100 chosen 2026-08-24: sbatch --test-only put its start 6.5 h earlier than
# a100 (02:07 vs 08:45) - milan had 218 a100 jobs queued on 16 GPUs, genoa 19
# h100 jobs on 6 - and it is faster once running, so 200 vs 90 billing is close
# to cost-neutral. Swap the blocks below to move between GPUs.
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_train_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_train_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8


##SBATCH --partition=genoa
##SBATCH --gpus-per-node=l4:1

##SBATCH --partition=milan
##SBATCH --gpus-per-node=a100:1

# Spyfish Aotearoa training job wrapper.
#
# Trains the species model, evaluates it and promotes it on improvement
# (python run_pipeline.py --retrain --species). It does NOT run data prep:
# that is CPU work (balance, flatten, split, assemble) and has no business
# holding an h100. Data prep goes through the three-job chain, or on its own:
#   scripts/slurm/submit_retrain.sh                         # cpu.sl data prep, this job, then cpu.sl --sync-only (afterok)
#   sbatch scripts/slurm/cpu.sl --retrain --data-prep       # rebuild the dataset only
#   sbatch spyfish/ml/training/train_job.sl                 # train on the dataset already in species/
# Until 2026-11-11 a bare `sbatch train_job.sl` ran data prep on the GPU first.
# Data prep rmtrees the shared species/ workspace, so only one retrain at a
# time: never submit data prep while a training job is queued or running.
#
# Optimizer / lr / dropout come from config.yaml's training section.
# Auto-promotion is on: models that beat production by `retrain_min_improvement_pct`
# are copied into pipeline_model/ at the end of the run.
#
# Flags on the sbatch line replace the default command below (this is how
# submit_retrain.sh passes --retrain --species --no-upload, leaving the S3
# sync to its CPU job):
#   sbatch spyfish/ml/training/train_job.sl --retrain --species
#


# Modules (with the milan `module use` fix), venv, PROJECT_DIR, YOLO_CONFIG_DIR,
# /opt/nesi/bin on PATH for the end-of-run S3 sync, cd into the repo and the
# START/END stamps all come from the shared env (venv and repo paths are set
# there); submit from the repo root like cpu.sl / gpu.sl.
# This job is the one that still loads a CUDA module on top of it.
SPYFISH_CUDA_MODULE=CUDA/11.0.2
source scripts/slurm/_env.sh

# Start/end stamps go to stderr so the .err file shows the run window even
# when it holds nothing else (INFO logging goes to .out, WARNING and above
# to .err, see log_config.py; an .err with only these two lines is a clean run).
spyfish_stamp_start
echo "Starting Spyfish retraining on $(hostname)"
# python -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"
# nvidia-smi || true

# python run_pipeline.py --retrain
# python -m spyfish.biigle.biigle_to_yolo download-project --project-id 4626 --force
# python -m spyfish.biigle.biigle_to_yolo download-project --project-id 3711 --force
if [ $# -gt 0 ]; then
    echo "Running: python run_pipeline.py $*"
    python run_pipeline.py "$@"
else
    # Train only. Data prep is a CPU job (see the header), never the default here.
    echo "Running: python run_pipeline.py --retrain --species"
    python run_pipeline.py --retrain --species
fi
# python run_pipeline.py --retrain --data-prep --dry-run

# python run_pipeline.py --biigle-sync   # pull latest annotations from BIIGLE API → raw CSVs on disk
# python run_pipeline.py --retrain --no-upload

echo "Retraining job complete. Outputs in process_files/training/runs/ and process_files/training/results/."
spyfish_end
