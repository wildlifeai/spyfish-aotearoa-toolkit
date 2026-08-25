#!/bin/bash -e

#SBATCH --job-name=run-spyfish-pipeline
#SBATCH --account=wildlife03546
#SBATCH --time=06:00:00

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/run_pipeline_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/run_pipeline_%j.err

#£SBATCH --partition=genoa
#£SBATCH --gpus-per-node=a100:1   # lowercase a100 to match sinfo's GRES type exactl

#SBATCH --partition=genoa
#SBATCH --gpus-per-node=l4:1

# ── Per-survey ML runs ──────────────────────────────────────────────────────
# Pass a survey id as the first argument and this file runs ML for just that
# survey, so one file covers every survey instead of a wrapper each:
#
#   sbatch --time=4:00:00 --job-name=ml_ORA26 run_pipeline.sl ORA_20260306_BUV --force
#
# Command-line --time/--job-name override the #SBATCH lines above, so size each
# job to its survey. Everything after the survey id is passed straight to
# run_pipeline.py, which is where --force belongs: it is deliberately NOT
# hardcoded here, because it resets the survey's ml_complete drops and would
# silently redo finished work on every later run. Add it when you mean it:
#
#   ... run_pipeline.sl OKA_20250121_BUV --force   # redo complete + recover stranded
#   ... run_pipeline.sl OKA_20250121_BUV           # only drops sitting at ml_ready
#
# Run surveys in PARALLEL freely, but never two jobs on the SAME survey: --force
# resets that survey's ml_running drops back to ml_ready, which would yank the
# drops a sibling job has already claimed out from under it.
#
# With no argument, the script falls through to the commented CMD block below.
SURVEY="${1:-}"
shift || true

# Load modules
module purge
# On milan/zen3 nodes `module purge` leaves NeSI/zen3 loaded, whose tree has only
# Python 3.14/foss-2026, so the 3.10.5 module kso_venv_0627 was built against drops
# off MODULEPATH and the load below fails outright (2026-08-22). No-op on genoa.
# module use /opt/nesi/lmod/mahuika
module load Python/3.10.5-gimkl-2022a
module load FFmpeg/5.1.1-GCC-11.3.0
# module load Python/3.11.6-foss-2023a
# module load CUDA/11.8.0

# Activate your virtual environment
source /nesi/project/wildlife03546/kso_venv_0627/bin/activate


# `aws` lives at /opt/nesi/bin and is on PATH in LOGIN shells only. Without this
# the end-of-run S3 sync dies with "No such file or directory: 'aws'" after all
# the work is done (run 8598080, 2026-08-23).
export PATH=/opt/nesi/bin:$PATH

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# A job that silently loses its GPU takes ~40 min per video instead of ~6, and
# only shows up as a timeout hours later. Fail fast and loudly instead.
python -c "import torch, sys; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NONE'); sys.exit(0 if ok else 1)"

# Change to script directory
cd /nesi/project/wildlife03546/spyfish-aotearoa-toolkit



if [ -n "${SURVEY}" ]; then
    echo "=== ML run for survey ${SURVEY} ==="
    echo "    extra args: $*"
    python run_pipeline.py --ml --survey "${SURVEY}" "$@"
else

# Run video preprocessing with YOLO detection and frame selection
# python run_pipeline.py --set-targets --ml --biigle-upload
# python run_pipeline.py --set-targets --ml --zooniverse-clip
# python run_pipeline.py --set-targets --zooniverse-clip
# python run_pipeline.py --set-targets --ingest --ml --zooniverse-clip
# python run_pipeline.py --biigle-upload
# python run_pipeline.py --set-targets --ml --biigle-upload
# python run_pipeline.py --set-targets --biigle-upload
# python run_pipeline.py --set-targets --zooniverse-clip

# python scripts/fetch_zooniverse_exports.py --to-s3
# python run_pipeline.py --legacy-zooniverse --db-refresh --no-upload

python run_pipeline.py --check-arrivals

fi

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo ""
