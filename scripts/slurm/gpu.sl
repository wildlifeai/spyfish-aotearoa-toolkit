#!/bin/bash -e
#SBATCH --job-name=spyfish_gpu
#SBATCH --account=wildlife03546
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_gpu_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_gpu_%j.err

# GPU pipeline job, for inference only. Resources here, action on the sbatch line:
#
#   sbatch scripts/slurm/gpu.sl --ml-infer --survey NOI_20250226_BUV
#   sbatch scripts/slurm/gpu.sl --ml-infer --survey NOI_20250226_BUV --force
#   sbatch scripts/slurm/gpu.sl --ml            # all-in-one, when no CPU job will follow
#
# Prefer scripts/slurm/submit_ml.sh, which runs --download (CPU) before this
# and --ml-post (CPU) after it, so the GPU job holds the GPU only while YOLO
# runs. Billing weight per GPU-hour (sinfo TRESBillingWeights):
#   genoa: l4 [20], pro_6000 [130], h100 [200];  milan: a100 [90].
# Inference is far lighter than training, so l4 is the default; swap to h100
# (same partition) only when the l4 queue is long. There is NO a100 in genoa.
# pro_6000 is Blackwell/sm_120 and cannot run this venv's torch 2.5.1+cu124.
# Training does not belong here: see spyfish/ml/training/train_job.sl.
#
# --cpus-per-task=8 matters: cv2 decodes the video on the CPU side of the same
# process, and ffmpeg's decoder threads are what keep the GPU fed.
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=l4:1

##SBATCH --partition=genoa
##SBATCH --gpus-per-node=h100:1

##SBATCH --partition=milan
##SBATCH --gpus-per-node=a100:1

source scripts/slurm/_env.sh

spyfish_require_args gpu.sl "--ml-infer --survey X" "$@"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# Fail fast if the job got no GPU (reason in _env.sh).
spyfish_require_gpu

spyfish_start "$@"
python run_pipeline.py "$@"
spyfish_end
