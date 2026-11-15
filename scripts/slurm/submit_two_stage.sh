#!/bin/bash -e
# Submit two-stage steps as a SLURM chain so the GPU is held only for the steps
# that use it. The steps are split, in the order given, into runs of GPU steps
# (two_stage.sl, l4) and runs of CPU steps (two_stage_cpu.sl), and each job
# waits for the one before it with afterok:
#
#   yolo embed evaluate negatives   ->   1. two_stage.sl      yolo embed   (afterok →)
#                                        2. two_stage_cpu.sl  evaluate     (afterok →)
#                                        3. two_stage.sl      negatives
#
# Usage:
#   scripts/slurm/submit_two_stage.sh [--dry-run] [--pack] [step ...]
#
#   scripts/slurm/submit_two_stage.sh                     # the default steps of two_stage.sl
#   scripts/slurm/submit_two_stage.sh --dry-run           # print the sbatch lines, submit nothing
#   scripts/slurm/submit_two_stage.sh yolo recall
#   scripts/slurm/submit_two_stage.sh --pack              # yolo embed negatives on ONE GPU job, then evaluate
#   YOLO_ARGS="--model <run>/weights/best.pt --out detector_predictions.csv" \
#   RECALL_FILE=detector_predictions.csv scripts/slurm/submit_two_stage.sh yolo recall
#
# Env vars (YOLO_ARGS, RECALL_FILE, FINETUNE_ARGS, NEGATIVES_ARGS, TWO_STAGE_SOURCE,
# ...) reach every job, as they do with a plain sbatch. GPU_TIME / CPU_TIME
# override the job files' walltimes.
#
# --pack: one GPU queue wait instead of one per run of GPU steps. CPU steps
# before the first GPU step stay first; every GPU step then runs in one job,
# in order; the remaining CPU steps run after it. Safe because no GPU step
# reads a CPU step's output (checked 2026-11-11: recall, evaluate, fit, report
# only write reports and the probe), but a failing CPU step no longer stops
# the GPU steps that were listed after it.
#
# Which steps are CPU-only is CPU_STEPS in the CPU job file; the default and
# the known steps are read from the GPU job file. The UoA wrapper
# (uoa/scripts/two_stage/submit_uoa_two_stage.sh) points TWO_STAGE_GPU_SL /
# TWO_STAGE_CPU_SL at the UoA pair and reuses everything else.

usage() { sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# Job files source scripts/slurm/_env.sh by its path from the repo root.
cd "$(dirname "$0")/../.."

GPU_SL="${TWO_STAGE_GPU_SL:-scripts/slurm/two_stage.sl}"
CPU_SL="${TWO_STAGE_CPU_SL:-scripts/slurm/two_stage_cpu.sl}"

DRY_RUN=""
PACK=""
STEPS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --pack) PACK=1 ;;
        -h|--help) usage ;;
        -*) echo "unknown argument: $1" >&2; usage ;;
        *) STEPS+=("$1") ;;
    esac
    shift
done

CPU_STEPS=$(sed -n 's/^CPU_STEPS="\(.*\)"$/\1/p' "${CPU_SL}")
DEFAULT_STEPS=$(sed -n 's/^STEPS="\${\*:-\(.*\)}"$/\1/p' "${GPU_SL}")
KNOWN_STEPS=$(sed -n 's/^ *\([a-z_][a-z_]*\)).*/\1/p' "${GPU_SL}" | tr '\n' ' ')
if [ -z "${CPU_STEPS}" ] || [ -z "${DEFAULT_STEPS}" ] || [ -z "${KNOWN_STEPS}" ]; then
    echo "could not read the step lists from ${GPU_SL} / ${CPU_SL}" >&2
    exit 2
fi
if [ ${#STEPS[@]} -eq 0 ]; then
    read -r -a STEPS <<< "${DEFAULT_STEPS}"
fi

is_cpu() { case " ${CPU_STEPS} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

for step in "${STEPS[@]}"; do
    case " ${KNOWN_STEPS} " in
        *" ${step} "*) ;;
        *) echo "unknown step: ${step} (known: ${KNOWN_STEPS})" >&2; exit 2 ;;
    esac
done

# Groups, as parallel arrays: kind (gpu|cpu) and the steps of that job.
KINDS=()
GROUPS_=()
add_step() {  # kind step: start a new job unless the last one is of this kind
    local n=${#KINDS[@]}
    if [ "${n}" -gt 0 ] && [ "${KINDS[$((n - 1))]}" = "$1" ]; then
        GROUPS_[$((n - 1))]="${GROUPS_[$((n - 1))]} $2"
    else
        KINDS+=("$1")
        GROUPS_+=("$2")
    fi
}

if [ -n "${PACK}" ]; then
    LEAD="" GPU="" TAIL=""
    for step in "${STEPS[@]}"; do
        if ! is_cpu "${step}"; then
            GPU="${GPU} ${step}"
        elif [ -z "${GPU}" ]; then
            LEAD="${LEAD} ${step}"
        else
            TAIL="${TAIL} ${step}"
        fi
    done
    for step in ${LEAD}; do add_step cpu "${step}"; done
    for step in ${GPU}; do add_step gpu "${step}"; done
    for step in ${TAIL}; do add_step cpu "${step}"; done
else
    for step in "${STEPS[@]}"; do
        if is_cpu "${step}"; then add_step cpu "${step}"; else add_step gpu "${step}"; fi
    done
fi

PREV=""
for i in "${!KINDS[@]}"; do
    kind="${KINDS[$i]}"
    if [ "${kind}" = "gpu" ]; then SL="${GPU_SL}"; TIME="${GPU_TIME:-}"; else SL="${CPU_SL}"; TIME="${CPU_TIME:-}"; fi
    CMD=(sbatch --parsable)
    [ -n "${PREV}" ] && CMD+=("--dependency=afterok:${PREV}")
    [ -n "${TIME}" ] && CMD+=("--time=${TIME}")
    # shellcheck disable=SC2206  # the steps of one job, split on spaces on purpose
    CMD+=("${SL}" ${GROUPS_[$i]})
    if [ -n "${DRY_RUN}" ]; then
        JOB="<job$((i + 1))>"
        echo "${CMD[*]}"
    else
        JOB=$("${CMD[@]}")
        echo "${kind} job ${JOB}  $(basename "${SL}") ${GROUPS_[$i]}${PREV:+   (after ${PREV} ok)}"
    fi
    PREV="${JOB}"
done

echo
if [ -n "${DRY_RUN}" ]; then
    echo "dry run: nothing submitted (from $(pwd))"
else
    echo "squeue -u \$USER -o '%.10i %.14j %.8T %.10M %.20R %E'   # %E shows the dependency"
fi
