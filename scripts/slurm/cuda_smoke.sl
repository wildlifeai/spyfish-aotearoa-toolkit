#!/bin/bash -e
#SBATCH --job-name=cuda_smoke
#SBATCH --account=wildlife03546
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=milan
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/cuda_smoke_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/cuda_smoke_%j.err

# Does this venv's torch get a CUDA context on a milan a100 at all? Written
# after three train_detector.sl jobs died at the first allocation (2026-11-04).
#   sbatch scripts/slurm/cuda_smoke.sl
#   sbatch --exclude=mg15 scripts/slurm/cuda_smoke.sl
# One .out with "cuda ok" means the node and driver are fine and the fault is
# in the training job; a failure here is a NeSI ticket, not a script change.

# Same environment as the jobs it stands in for; submit from the repo root.
source scripts/slurm/_env.sh
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,compute_mode,mig.mode.current,memory.used --format=csv
python - <<'PY'
import torch
print("torch", torch.__version__, "available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0))
x = torch.ones(4, device="cuda")
print("cuda ok", x.sum().item())
import ultralytics
from ultralytics import YOLO
m = YOLO("process_files/models/base_model/cfd-yolov12x-1.00.pt")
m.model.to("cuda")
print("yolov12x on cuda ok, ultralytics", ultralytics.__version__)
PY
