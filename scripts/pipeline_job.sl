#!/bin/bash -e
#SBATCH --job-name=spyfish
#SBATCH --account=wildlife03546
#SBATCH --partition=genoa
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_%j.err

# ── Resources: keep ONE block active ─────────────────────────────────────────
# A doubled hash (##SBATCH) is how SLURM comments a directive out; flip which
# block has the single hash. CPU block suits ingest/sync/backfill/db-refresh;
# GPU block is for --ml and --retrain. l4 in genoa is the cheapest GPU and
# ample at imgsz=640; a100s exist only in milan (change the partition too).

# CPU (default)
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# GPU
##SBATCH --time=24:00:00
##SBATCH --mem=32G
##SBATCH --cpus-per-task=8
##SBATCH --gpus-per-node=l4:1

# ── Command: uncomment exactly one ───────────────────────────────────────────
CMD="python run_pipeline.py --db-refresh --no-upload"
# CMD="python run_pipeline.py --legacy-zooniverse --db-refresh --no-upload"
# CMD="python run_pipeline.py --zooniverse-sync --no-upload"
# CMD="python run_pipeline.py --ml --survey NOI_20250226_BUV --no-upload"
# CMD="python run_pipeline.py --ml --survey NOI_20250226_BUV --force --no-upload"
# CMD="python run_pipeline.py --biigle-upload --survey NOI_20250226_BUV --survey-volume --no-upload"
# CMD="python run_pipeline.py --retrain"
# CMD="python scripts/fetch_zooniverse_exports.py --generate --to-s3"

# ── Setup (same for every job) ───────────────────────────────────────────────
module purge
module load Python/3.10.5-gimkl-2022a
module load FFmpeg/5.1.1-GCC-11.3.0
module load CUDA/11.8.0

VENV=/nesi/project/wildlife03546/kso_venv_0627/bin/activate
PROJECT_DIR=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit

source "${VENV}"
cd "${PROJECT_DIR}"
mkdir -p slurm_logs

# Repo-local Ultralytics config: a YOLO call without `project=` must not
# write into another project's runs_dir. The dir must exist before
# ultralytics imports, or the env var is silently ignored.
export YOLO_CONFIG_DIR="${PROJECT_DIR}/.ultralytics"
mkdir -p "${YOLO_CONFIG_DIR}"

echo "Running: ${CMD}"
eval "${CMD}"
