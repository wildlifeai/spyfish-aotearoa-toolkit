#!/bin/bash -e

#SBATCH --job-name=run-spyfish-pipeline
#SBATCH --account=wildlife03546
#SBATCH --time=06:00:00

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/run_pipeline_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/run_pipeline_%j.err

# GPU choice (billing weight per GPU-hour in brackets - sinfo TRESBillingWeights):
#   genoa: l4 [20], pro_6000 [130], h100 [200];  milan: a100 [90].
# There is NO a100 in genoa - asking for one there fails at submit with
# "Requested node configuration is not available" (2026-08-25). a100 means milan,
# which the `module use` line below already handles.
# pro_6000 is Blackwell/sm_120 and cannot run this venv's torch 2.5.1+cu124.
# Inference is far lighter than training, so l4 is the right default here; swap to
# h100 (same partition, no module change) only when the l4 queue is long.
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=l4:1

##SBATCH --partition=genoa
##SBATCH --gpus-per-node=h100:1

##SBATCH --partition=milan
##SBATCH --gpus-per-node=a100:1

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
# off MODULEPATH and the load below fails outright (2026-08-22). Re-adding the
# mahuika tree fixes milan and is a NO-OP on genoa, so it stays in regardless of
# which GPU block is active - same as train_job.sl, which has always had it on.
module use /opt/nesi/lmod/mahuika
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
    # python run_pipeline.py --biigle-upload --survey OKA_20250121_BUV --survey-volume
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

# Targeted batch, then clips for whole surveys (from NeSI, 2026-10-14): --ml
# takes the top limit_processing drops by priority, and --survey bypasses that
# limit so each clips step covers its whole survey. Chained with && so a failed
# ML run does not go on to upload clips.
# python run_pipeline.py --set-targets --ml \
#   && python run_pipeline.py --zooniverse-clips --survey SLI_20220228_BUV \
#   && python run_pipeline.py --zooniverse-clips --survey SLI_20240124_BUV

python run_pipeline.py --check-arrivals

fi

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo ""
