#!/bin/bash -e

#SBATCH --job-name=annotate-video
#SBATCH --account=wildlife03546
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/annotate_video_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/annotate_video_%j.err

# ─── Parameters (override at submit, e.g. `DROP=... START=... sbatch annotate_video.sl`) ───
DROP="${DROP:-SLI_20260114_BUV_SLI_073_01}"   # deployment to annotate
START="${START:-250}"                          # segment start, seconds
END="${END:-600}"                              # segment end, seconds
STRIDE="${STRIDE:-1}"                          # 1 = every frame (full/smooth/real-time); 3-5 = lighter, fewer frames (output fps scaled so playback stays real-time)
# ───────────────────────────────────────────────────────────────────────────────────────────

module purge
module load Python/3.10.5-gimkl-2022a
module load FFmpeg/5.1.1-GCC-11.3.0
module load CUDA/11.8.0
source /nesi/project/wildlife03546/kso_venv_0627/bin/activate
cd /nesi/project/wildlife03546/spyfish-aotearoa-toolkit

echo "Annotating ${DROP} [${START}s–${END}s], stride=${STRIDE}"

python - "$DROP" "$START" "$END" "$STRIDE" <<'PY'
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from spyfish.config.wrapper import config

drop, start, end, stride = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])

video = Path(config.media_dir) / f"{drop}.mp4"
print(f"source video: {video} | exists: {video.exists()}", flush=True)
if not video.exists():
    raise SystemExit(
        f"video for {drop} not in media_dir ({config.media_dir}) — nobackup may have "
        "purged it; re-download from S3 or re-run --ml first."
    )

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise SystemExit(f"could not open {video}")
fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
start_f, end_f = int(start * fps), int(end * fps)
out_fps = fps / stride            # scale so the clip plays at real-time regardless of stride
print(f"{fps:.2f} fps, {W}x{H}, frames {start_f}–{end_f}, output {out_fps:.1f} fps", flush=True)

out = Path("annotated"); out.mkdir(exist_ok=True)
out_path = out / f"{drop}_{int(start)}-{int(end)}_boxes.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

model = YOLO(str(config.pipeline_model_path))
conf, iou, agn = config.confidence_threshold, config.ml_nms_iou, config.ml_nms_agnostic
batch_size = config.ml_batch_size

written = [0]  # mutable for the closure

def flush(frames):
    if not frames:
        return
    # Annotate with the LIVE pipeline config (conf + NMS iou/agnostic) so the video
    # shows exactly what the pipeline produces. result.plot() returns the BGR frame
    # with boxes drawn — written straight to the video (no Ultralytics save path).
    for res in model.predict(frames, conf=conf, iou=iou, agnostic_nms=agn, verbose=False):
        writer.write(res.plot())
        written[0] += 1

cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
idx, batch = start_f, []
while idx <= end_f:
    if not cap.grab():
        break
    if (idx - start_f) % stride == 0:
        ok, frame = cap.retrieve()
        if not ok:
            break
        batch.append(frame)
        if len(batch) >= batch_size:
            flush(batch); batch = []
            print(f"  …{written[0]} frames annotated", flush=True)
    idx += 1
flush(batch)
cap.release(); writer.release()

size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
print(f"DONE: {written[0]} annotated frames -> {out_path.resolve()} ({size_mb:.1f} MB)", flush=True)
PY

echo "Annotation complete"
