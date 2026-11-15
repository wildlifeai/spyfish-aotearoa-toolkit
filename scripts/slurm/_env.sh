# Shared environment for every Spyfish SLURM job. Sourced by every job file
# (cpu.sl, gpu.sl, two_stage.sl, two_stage_cpu.sl, train_detector.sl,
# cuda_smoke.sl, train_job.sl, eval_job.sl, and the UoA job files under
# uoa/scripts/) by its path from the repo root, so submit from the repo root.
# Not a job script itself. Keep cluster facts here, once.

PROJECT_DIR=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit
VENV=/nesi/project/wildlife03546/kso_venv_0627/bin/activate

module purge
# On milan/zen3 nodes `module purge` leaves NeSI/zen3 loaded, whose tree has only
# Python 3.14/foss-2026, so the 3.10.5 module kso_venv_0627 was built against drops
# off MODULEPATH and the load below fails outright (2026-08-22). Re-adding the
# mahuika tree fixes milan and is a no-op on genoa.
module use /opt/nesi/lmod/mahuika
module load Python/3.10.5-gimkl-2022a
# No CUDA module by default: the venv's torch (2.5.1+cu124) bundles its own
# runtime. A job that still wants one names it in SPYFISH_CUDA_MODULE before
# sourcing this file (train_job.sl: CUDA/11.0.2); it loads right after Python,
# where that job has always loaded it.
if [ -n "${SPYFISH_CUDA_MODULE:-}" ]; then
    module load "${SPYFISH_CUDA_MODULE}"
fi
module load FFmpeg/5.1.1-GCC-11.3.0

# The venv symlinks to the module's interpreter: activate AFTER the module load.
source "${VENV}"

# `aws` lives at /opt/nesi/bin and is on PATH in LOGIN shells only. Without this
# the end-of-run S3 sync dies with "No such file or directory: 'aws'" after all
# the work is done (run 8598080, 2026-08-23).
export PATH=/opt/nesi/bin:$PATH

# Repo-local Ultralytics config: a YOLO call without `project=` must not write
# into another project's runs_dir. The dir must exist before ultralytics
# imports, or the env var is silently ignored.
export YOLO_CONFIG_DIR="${PROJECT_DIR}/.ultralytics"
mkdir -p "${YOLO_CONFIG_DIR}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

# Start/end stamps go to stderr: INFO logging goes to .out, WARNING and above
# to .err (log_config.py), so an .err holding only these two lines is a clean run.
#   spyfish_stamp_start ["key=value ..."]   START stamp only; the optional label
#                                           is appended after host= (jobs that
#                                           do not run run_pipeline.py)
#   spyfish_start "$@"                      START stamp with cmd=<flags>, plus the
#                                           "Running: python run_pipeline.py" line
#   spyfish_end                             END stamp
spyfish_stamp_start() {
    echo "START $(date '+%F %T') job=${SLURM_JOB_ID:-none} host=$(hostname)${1:+ $1}" >&2
}
spyfish_start() {
    spyfish_stamp_start "cmd=$*"
    echo "Running: python run_pipeline.py $*"
}
spyfish_end() {
    echo "END   $(date '+%F %T') job=${SLURM_JOB_ID:-none}" >&2
}

# Guard for the job files whose action is the run_pipeline.py flags on the
# sbatch line: spyfish_require_args <job file> "<example flags>" "$@"
spyfish_require_args() {
    if [ $# -le 2 ]; then
        echo "$1: pass run_pipeline.py flags on the sbatch line, e.g. $2" >&2
        exit 2
    fi
}

# A job that silently loses its GPU takes ~40 min per video instead of ~6, and
# only shows up as a timeout hours later. Fail fast and loudly instead. Opens a
# CUDA context, so NOT for train_detector.sl (see the note there).
spyfish_require_gpu() {
    python -c "import torch, sys; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NONE'); sys.exit(0 if ok else 1)"
}

# Compute nodes have no internet. The second-stage classifier reads the BioCLIP 2
# backbone from its hub cache under process_files/models/bioclip/ (filled once,
# in a login shell: `python -c "from spyfish.ml.classifier.species_classifier import *;
# import open_clip; open_clip.create_model_and_transforms(config.classifier_backbone,
# cache_dir=str(config.bioclip_cache_dir))"`). Offline mode makes a missing cache
# fail at once instead of hanging on a connection attempt.
export HF_HUB_OFFLINE=1

# GoPro files carry four data streams beside the video (audio, tmcd, gpmd, fdsc).
# On NeSI, cv2 4.11 (FFmpeg 5.1 underneath) hits its default 4096-packet read
# limit after ~45 frames ("grabFrame packet read max attempts exceeded") and
# reports end of video, so every UoA chapter read 3 of 2,121 sampled frames
# (2026-11-11; the same files read whole on a Mac, and with this set on NeSI).
# The limit only bounds a retry loop, so a high value costs nothing on good files.
export OPENCV_FFMPEG_READ_ATTEMPTS=1000000
