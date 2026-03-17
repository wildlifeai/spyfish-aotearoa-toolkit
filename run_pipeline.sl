#!/bin/bash -e

PROJECT_DIR="/nesi/project/wildlife03546/spyfish-play"

#SBATCH --job-name=run-spyfish-pipeline
#SBATCH --account=wildlife03546
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=${PROJECT_DIR}/slurm_logs/run_pipeline_%j.out
#SBATCH --error=${PROJECT_DIR}/slurm_logs/run_pipeline_%j.err



# Load modules
module purge
# module load Python/3.11.6-foss-2023a
module load Python/3.10.5-gimkl-2022a
module load FFmpeg/5.1.1-GCC-11.3.0
module load CUDA/11.8.0

# Activate your virtual environment
source /nesi/project/wildlife03546/kso_venv_0627/bin/activate

# Change to script directory
cd /nesi/project/wildlife03546/spyfish-play

# Create logs directory
mkdir -p slurm_logs


# Run video preprocessing with YOLO detection and frame selection
python run_pipeline.py --step0

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo ""


# to create a cron job, you can add the following line to your crontab file
# by running `crontab -e`:

# #SCRON -A wildlife03546
# #SCRON -t 00:02:00
# #SCRON -o /nesi/project/wildlife03546/spyfish-play/scron_test.log

# * * * * * sbatch /nesi/project/wildlife03546/spyfish-play/run_pipeline.sl
