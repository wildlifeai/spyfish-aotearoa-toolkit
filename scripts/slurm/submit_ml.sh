#!/bin/bash -e
# Submit one survey's ML run as a three-job SLURM chain so the GPU is held only
# for inference:
#
#   1. cpu.sl  --download      S3 → media_dir, every video of the survey (afterok →)
#   2. gpu.sl  --ml-infer      ml_ready → ml_inferred, raw CSVs      (afterany →)
#   3. cpu.sl  --ml-post       ml_inferred → ml_complete, MaxN/QA, S3 sync
#
# Usage:
#   scripts/slurm/submit_ml.sh SURVEY_ID [--force] [-- <extra flags for the post job>]
#
#   scripts/slurm/submit_ml.sh NOI_20250226_BUV
#   scripts/slurm/submit_ml.sh NOI_20250226_BUV --force
#   scripts/slurm/submit_ml.sh NOI_20250226_BUV -- --zooniverse-frames --zooniverse-crops
#   scripts/slurm/submit_ml.sh NOI_20250226_BUV -- --biigle-upload --survey-volume
#
# Job 3 waits with `afterany`, not `afterok`: if the GPU job hits its walltime
# with some drops already at ml_inferred, those still get finished. Drops it
# left at ml_running need a `--force` rerun (see run_pipeline.py --help).
# Job 2 waits with `afterok`: no point holding a GPU if the download failed.
#
# --force goes to the GPU job only: it resets the survey's finished drops to
# ml_ready, and the download job fetches every video of the survey regardless.
# Only the last job syncs to S3; the first two run --no-upload so GPU minutes
# are not spent on `aws s3 sync`. Never submit two chains for the same survey
# at once: --force on the second resets the drops the first has claimed.
#
# Walltimes below suit a ~10-drop survey; override per survey with env vars:
#   DL_TIME=01:00:00 ML_TIME=06:00:00 POST_TIME=02:00:00 scripts/slurm/submit_ml.sh ...

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

SURVEY="${1:-}"
[ -z "${SURVEY}" ] && usage
shift

FORCE=""
POST_EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE="--force"; shift ;;
        --) shift; POST_EXTRA=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

DL_TIME="${DL_TIME:-02:00:00}"
ML_TIME="${ML_TIME:-04:00:00}"
POST_TIME="${POST_TIME:-03:00:00}"
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
SHORT="${SURVEY%_BUV}"

J1=$(sbatch --parsable --job-name="dl_${SHORT}" --time="${DL_TIME}" \
    "${SLURM_DIR}/cpu.sl" --download --survey "${SURVEY}" --no-upload)
echo "download  job ${J1}  cpu.sl --download --survey ${SURVEY}"

J2=$(sbatch --parsable --dependency="afterok:${J1}" --job-name="ml_${SHORT}" --time="${ML_TIME}" \
    "${SLURM_DIR}/gpu.sl" --ml-infer --survey "${SURVEY}" ${FORCE} --no-upload)
echo "inference job ${J2}  gpu.sl --ml-infer --survey ${SURVEY} ${FORCE}   (after ${J1} ok)"

J3=$(sbatch --parsable --dependency="afterany:${J2}" --job-name="post_${SHORT}" --time="${POST_TIME}" \
    "${SLURM_DIR}/cpu.sl" --ml-post --survey "${SURVEY}" "${POST_EXTRA[@]}")
echo "post      job ${J3}  cpu.sl --ml-post --survey ${SURVEY} ${POST_EXTRA[*]}   (after ${J2} ends)"

echo
echo "squeue -u \$USER -o '%.10i %.14j %.8T %.10M %.20R %E'   # %E shows the dependency"
