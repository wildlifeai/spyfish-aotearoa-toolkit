#!/bin/bash -e
#SBATCH --job-name=two_stage
#SBATCH --account=wildlife03546
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=l4:1
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/two_stage_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/two_stage_%j.err

# Two-stage classifier training (spyfish/ml/classifier, scripts/two_stage). Not a pipeline
# action, so it does not go through cpu.sl/gpu.sl + run_pipeline.py. Steps on
# the sbatch line, in order; default is the whole chain:
#
#   sbatch scripts/slurm/two_stage.sl                 # yolo embed evaluate negatives
#   sbatch scripts/slurm/two_stage.sl embed evaluate  # a subset
#   FINETUNE_ARGS="--epochs 15 --unfreeze 6" sbatch scripts/slurm/two_stage.sl finetune   # heavier fine-tune
#
# Score another detector on the same frames (YOLO_ARGS goes to run_yolo_on_frames.py),
# then the recall table against the expert boxes (RECALL_FILE is the predictions file):
#   YOLO_ARGS="--model process_files/training/runs/<run>/weights/best.pt --out detector_predictions.csv" \
#   RECALL_FILE=detector_predictions.csv sbatch scripts/slurm/two_stage.sl yolo recall
#
# Hard negatives from the boxes BIIGLE experts deleted (needs no detector run), or the union:
#   NEGATIVES_ARGS="--source upload" sbatch scripts/slurm/two_stage.sl negatives
#
# Source of boxes: TWO_STAGE_SOURCE=yolo_dataset reads the assembled YOLO
# dataset under training.local_training_dir (every BIIGLE drop, frozen test
# drops included). Unset, it reads the per-drop BIIGLE CSVs + frames like on
# the Mac. An l4 is plenty: embedding is inference-sized work.
#
# `recall` and `evaluate` need no GPU (IoU matching and a logistic regression on
# saved embeddings). Run alone they belong on a CPU job, and a mixed list goes
# through the submit script, which puts each run of GPU steps on this file and
# each run of CPU steps on two_stage_cpu.sl, chained with afterok:
#   sbatch scripts/slurm/two_stage_cpu.sl evaluate
#   scripts/slurm/submit_two_stage.sh                   # the default steps, split
#   scripts/slurm/submit_two_stage.sh --dry-run yolo recall
# Submitted directly, this file still runs every step it is given on the l4.
#
# One-time, on a LOGIN node (compute nodes may have no internet):
#   source scripts/slurm/_env.sh
#   python -m pip install open_clip_torch scikit-learn tabulate
#   python -c "from spyfish.ml.classifier.crops import load_bioclip; load_bioclip('cpu')"
# That puts the BioCLIP 2 weights in process_files/models/bioclip/, next to the
# YOLO models, where anything else that uses BioCLIP finds them too.

# _env.sh also sets HF_HUB_OFFLINE=1: the BioCLIP weights are in
# process_files/models/bioclip/, never download in a job.
source scripts/slurm/_env.sh
export TWO_STAGE_SOURCE="${TWO_STAGE_SOURCE:-yolo_dataset}"

# two_stage_cpu.sl sources this file with TWO_STAGE_NO_GPU=1 to run the
# CPU-only steps (it refuses the others), so the step list below stays in one place.
if [ -z "${TWO_STAGE_NO_GPU:-}" ]; then
    spyfish_require_gpu
fi

STEPS="${*:-yolo embed evaluate negatives}"
spyfish_stamp_start "steps=${STEPS} source=${TWO_STAGE_SOURCE}"
for step in ${STEPS}; do
    case "${step}" in
        yolo)      python scripts/two_stage/run_yolo_on_frames.py ${YOLO_ARGS:-} ;;
        recall)    python scripts/two_stage/detector_recall.py "${RECALL_FILE:-yolo_predictions.csv}" ;;
        embed)     python scripts/two_stage/build_embeddings.py ;;
        evaluate)  python scripts/two_stage/evaluate.py ;;
        negatives) python scripts/two_stage/hard_negatives.py ${NEGATIVES_ARGS:-} ;;
        finetune)  python scripts/two_stage/finetune_bioclip.py ${FINETUNE_ARGS:-} ;;
        prior)     python scripts/two_stage/yolo_with_prior.py ;;
        *) echo "unknown step ${step}" >&2; exit 2 ;;
    esac
done
spyfish_end
