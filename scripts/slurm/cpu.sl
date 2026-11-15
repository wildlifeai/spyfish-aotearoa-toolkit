#!/bin/bash -e
#SBATCH --job-name=spyfish_cpu
#SBATCH --account=wildlife03546
#SBATCH --partition=genoa
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_cpu_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_cpu_%j.err

# CPU-only pipeline job. This file describes RESOURCES; the action is whatever
# run_pipeline.py flags you pass on the sbatch line, so nothing in here is
# ever uncommented or edited to run something:
#
#   sbatch scripts/slurm/cpu.sl --zooniverse-sync
#   sbatch scripts/slurm/cpu.sl --download --survey NOI_20250226_BUV
#   sbatch scripts/slurm/cpu.sl --ml-post --survey NOI_20250226_BUV
#   sbatch scripts/slurm/cpu.sl --biigle-upload --survey NOI_20250226_BUV --survey-volume
#   sbatch --time=12:00:00 --mem=16G scripts/slurm/cpu.sl --legacy-zooniverse --db-refresh --no-upload
#
# --time / --mem / --job-name on the sbatch line override the #SBATCH lines above.
# Everything that needs the GPU (--ml, --ml-infer, --retrain) goes through gpu.sl
# or spyfish/ml/training/train_job.sl instead.

source scripts/slurm/_env.sh

spyfish_require_args cpu.sl "--zooniverse-sync" "$@"

spyfish_start "$@"
python run_pipeline.py "$@"
spyfish_end
