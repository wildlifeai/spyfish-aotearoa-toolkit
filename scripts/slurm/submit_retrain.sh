#!/bin/bash -e
# Submit a retrain as a three-job SLURM chain so the GPU is held only for training:
#
#   1. cpu.sl        --retrain --data-prep             balance, flatten, split, assemble   (afterok →)
#   2. train_job.sl  --retrain --species --no-upload   train, evaluate, promote on improvement   (afterok →)
#   3. cpu.sl        --sync-only                       both DBs + deployment data (promoted model) → S3
#
# Usage:
#   scripts/slurm/submit_retrain.sh [--dry-run]
#
#   scripts/slurm/submit_retrain.sh
#   scripts/slurm/submit_retrain.sh --dry-run          # print the sbatch lines, submit nothing
#   PREP_TIME=04:00:00 PREP_MEM=32G scripts/slurm/submit_retrain.sh
#   SYNC_TIME=01:00:00 scripts/slurm/submit_retrain.sh
#
# Data prep rmtrees the shared species/ workspace, which kills any training job
# that is mid-epoch on it, so never submit this while a training job is queued
# or running (squeue -u $USER). `afterok` means a failed prep never starts a
# GPU job on a half-built dataset. The chain is strictly sequential: each job
# waits for the one before it, nothing runs alongside the training.
#
# The training job runs --no-upload so h100 minutes are not spent on
# `aws s3 sync`; during --retrain --species that flag skips the end-of-run sync
# and nothing else (training, evaluation and promotion write local files only).
# Job 3 does that sync. It waits with `afterok`, so a training job that fails
# or times out syncs nothing: once it is sorted out, run the sync by hand with
#   sbatch scripts/slurm/cpu.sl --sync-only
#
# Pull fresh BIIGLE annotations first if you need them (login shell, CPU):
#   python -m spyfish.biigle.biigle_to_yolo download-project --project-id 4626
# then `python run_pipeline.py --retrain --data-prep --dry-run` is the cheap
# way to check the val balance before paying for the assembly and the GPU.

usage() { sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

DRY_RUN=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        *) usage ;;
    esac
    shift
done

PREP_TIME="${PREP_TIME:-03:00:00}"
PREP_MEM="${PREP_MEM:-16G}"
SYNC_TIME="${SYNC_TIME:-}"                 # empty: cpu.sl's own walltime
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SL="${SLURM_DIR}/../../spyfish/ml/training/train_job.sl"

# submit <label> <job number> <sbatch args...>: sets JOB to the job id, or to a
# <jobN> placeholder on --dry-run, where the sbatch line is printed instead.
submit() {
    local label="$1" n="$2"
    shift 2
    if [ -n "${DRY_RUN}" ]; then
        JOB="<job${n}>"
        echo "sbatch $*"
    else
        JOB=$(sbatch "$@")
        echo "${label} job ${JOB}"
    fi
}

submit "data prep" 1 --parsable --job-name=retrain_prep --time="${PREP_TIME}" --mem="${PREP_MEM}" \
    "${SLURM_DIR}/cpu.sl" --retrain --data-prep
J1="${JOB}"

submit "training " 2 --parsable --dependency="afterok:${J1}" \
    "${TRAIN_SL}" --retrain --species --no-upload
J2="${JOB}"

SYNC_ARGS=(--parsable --dependency="afterok:${J2}" --job-name=retrain_sync)
[ -n "${SYNC_TIME}" ] && SYNC_ARGS+=(--time="${SYNC_TIME}")
submit "S3 sync  " 3 "${SYNC_ARGS[@]}" "${SLURM_DIR}/cpu.sl" --sync-only
J3="${JOB}"

echo
if [ -n "${DRY_RUN}" ]; then
    echo "dry run: nothing submitted"
else
    echo "1. ${J1}  cpu.sl --retrain --data-prep"
    echo "2. ${J2}  train_job.sl --retrain --species --no-upload   (after ${J1} ok)"
    echo "3. ${J3}  cpu.sl --sync-only   (after ${J2} ok)"
    echo
    echo "squeue -u \$USER -o '%.10i %.14j %.8T %.10M %.20R %E'   # %E shows the dependency"
fi
