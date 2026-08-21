#!/bin/bash -e

#SBATCH --job-name=run-spyfish-pipeline
#SBATCH --account=wildlife03546
#SBATCH --time=12:00:00
#SBATCH --partition=milan
#SBATCH --gpus-per-node=a100:1   # lowercase a100 to match sinfo's GRES type exactl
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/run_pipeline_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/run_pipeline_%j.err


# Load modules
module purge
module load Python/3.10.5-gimkl-2022a
module load FFmpeg/5.1.1-GCC-11.3.0
# module load Python/3.11.6-foss-2023a
# module load CUDA/11.8.0

# Activate your virtual environment
source /nesi/project/wildlife03546/kso_venv_0627/bin/activate

echo "yo"
# nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"


# Change to script directory
cd /nesi/project/wildlife03546/spyfish-aotearoa-toolkit

# Run video preprocessing with YOLO detection and frame selection
# python run_pipeline.py --set-targets --ml --biigle-upload
# python run_pipeline.py --set-targets --ml --zooniverse-clip
# python run_pipeline.py --set-targets --zooniverse-clip
# python run_pipeline.py --set-targets --ingest --ml --zooniverse-clip
# python run_pipeline.py --biigle-upload
# python run_pipeline.py --set-targets --ml --biigle-upload
# python run_pipeline.py --set-targets --biigle-upload
# python run_pipeline.py --set-targets --zooniverse-clip

# python run_pipeline.py --ml --survey  ORA_20250121_BUV --force
# python run_pipeline.py --ml --survey  OKA_20250121_BUV --force
# python scripts/fetch_zooniverse_exports.py --generate --to-s3

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo ""
