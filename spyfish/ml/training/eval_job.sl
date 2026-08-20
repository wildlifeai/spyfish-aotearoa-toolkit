#!/bin/bash -e
#SBATCH --job-name=spyfish_eval
#SBATCH --account=wildlife03546
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=l4:1
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_eval_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/spyfish_eval_%j.err

# Evaluate the newest species training run's best.pt against production.
# Use after a training job was walltime-killed before its evaluation stage ran.
# Prints metrics + should_promote; promotion itself stays a manual decision —
# read the per-class AP table and the production model's score on this val
# split before promoting:
#   python -c "from spyfish.orchestrator.retrain_runner import \
#     _promote_model_locally; _promote_model_locally('<best.pt>', 'species')"

module purge
# module load ...   # mirror whatever train_job.sl loads on this cluster

VENV=/nesi/project/wildlife03546/kso_venv_0627/bin/activate
PROJECT_DIR=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit

source "${VENV}"
cd "${PROJECT_DIR}"

# Repo-local Ultralytics config, same reason as in train_job.sl: a val call
# without `project=` must not write into another project's runs_dir. The dir
# must exist before ultralytics imports, or the env var is silently ignored.
export YOLO_CONFIG_DIR="${PROJECT_DIR}/.ultralytics"
mkdir -p "${YOLO_CONFIG_DIR}"

BEST=$(ls -d process_files/training/runs/*_species | tail -1)/weights/best.pt
echo "Evaluating: ${BEST}"

python -c "
from spyfish.ml.training.evaluate import run_evaluation_pipeline
r = run_evaluation_pipeline(
    model_path='${BEST}',
    data_yaml='process_files/training/species/data.yaml',
    model_type='species',
)
print(r)
"
