#!/bin/bash -e
#SBATCH --job-name=two_stage_cpu
#SBATCH --account=wildlife03546
#SBATCH --partition=genoa
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/two_stage_cpu_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/two_stage_cpu_%j.err

# The two-stage steps that need no GPU, on a CPU job: `recall` (IoU matching of
# a predictions file against the expert boxes) and `evaluate` (logistic
# regression on the saved embeddings). Same steps, same env vars and same
# commands as two_stage.sl, which this file sources; only the resources differ.
# Steps on the sbatch line, in order; there is no default:
#
#   sbatch scripts/slurm/two_stage_cpu.sl evaluate
#   RECALL_FILE=detector_predictions.csv sbatch scripts/slurm/two_stage_cpu.sl recall
#
# A list that mixes GPU and CPU steps goes through scripts/slurm/submit_two_stage.sh,
# which reads CPU_STEPS below to decide which job file each step runs on.
# Any other step is refused here: on a CPU node `yolo` over 6k frames is 20 h.

CPU_STEPS="recall evaluate"

if [ $# -eq 0 ]; then
    echo "two_stage_cpu.sl: pass steps on the sbatch line (CPU steps: ${CPU_STEPS})" >&2
    exit 2
fi
for step in "$@"; do
    case " ${CPU_STEPS} " in
        *" ${step} "*) ;;
        *) echo "two_stage_cpu.sl: '${step}' is not a CPU step (${CPU_STEPS}); use two_stage.sl or submit_two_stage.sh" >&2
           exit 2 ;;
    esac
done

# Submit from the repo root, like every job file (two_stage.sl sources _env.sh
# by the same relative path).
export TWO_STAGE_NO_GPU=1
source scripts/slurm/two_stage.sl "$@"
