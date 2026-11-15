"""Reading frames and stream facts straight from a video file.

The ffmpeg readers are for raw multi-stream camera files, where cv2 can stall
(see ffmpeg_sampled_frames). Inference and the UoA scripts share them, which is
why they live here and not in the inference module.
"""

import io
import subprocess
import tempfile

import cv2
import numpy as np


def _ffmpeg_ppm_cmd(video_path, input_args=(), output_args=()) -> list:
    """The ffmpeg command that writes frames of the first video stream to
    stdout as PPM. ``input_args`` go before ``-i`` (a seek), ``output_args``
    after the stream map (filters, frame limits)."""
    return [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        *input_args,
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        *output_args,
        "-f",
        "image2pipe",
        "-vcodec",
        "ppm",
        "-",
    ]


def _read_ppm(stream):
    """The next binary PPM (P6) image on ``stream`` as a BGR array.

    None when the stream has ended, or ends part way through the pixels.
    Raises if what comes next is not a PPM header. The header carries the
    size, so the caller needs to know nothing about the video.
    """
    magic = stream.readline()
    if not magic:
        return None
    if magic.strip() != b"P6":
        raise RuntimeError(f"ffmpeg pipe out of step: {magic[:20]!r}")
    width, height = map(int, stream.readline().split())
    stream.readline()  # maxval, always 255 for rgb24
    size = width * height * 3
    buf = stream.read(size)
    if len(buf) < size:
        return None
    rgb = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def ffmpeg_sampled_frames(video_path, start_frame: int, stride: int, n_frames: int):
    """Yield ``(frame_index, BGR array)`` for frames ``start_frame + k * stride``,
    ``k < n_frames``, decoded by the ffmpeg CLI with every stream but the first
    video one discarded.

    cv2 reads every packet of every stream and skips the non-video ones itself.
    A raw GoPro chapter carries audio, tmcd, gpmd and ~76k fdsc packets beside
    42k video frames, and on some chapters cv2 spun inside one frame past its
    30 s read timeout and reported end of video (UoA, NeSI, 2026-11-11).
    ``-map 0:v:0`` has the demuxer skip the other streams instead.

    ``n`` in the select filter counts decoded frames from 0 in presentation
    order, the same index cv2 gives, so raw-CSV frame numbers stay seekable
    (checked against cv2 on three videos, 2026-11-11: every sampled frame
    matched at offset 0). Frames come out as PPM, whose header carries the
    size, so rotation needs no probing. Raises if ffmpeg exits non-zero; a
    short clean exit is left to the caller's frame-coverage check.
    """
    if n_frames < 1:
        return
    select = f"select='gte(n\\,{start_frame})*not(mod(n-{start_frame}\\,{stride}))'"
    cmd = _ffmpeg_ppm_cmd(
        video_path,
        output_args=[
            "-vf",
            select,
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(n_frames),
        ],
    )
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        finished = False
        try:
            for k in range(n_frames):
                frame = _read_ppm(proc.stdout)
                if frame is None:
                    break
                yield start_frame + k * stride, frame
            # All frames read (or the pipe ended): let ffmpeg exit by itself so
            # its return code means something. A consumer that stops early
            # never gets here; the finally below kills ffmpeg instead.
            proc.stdout.read()
            proc.wait(timeout=120)
            finished = True
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if finished and proc.returncode != 0:
            err.seek(0)
            tail = err.read()[-2000:].decode(errors="replace")
            raise RuntimeError(
                f"ffmpeg failed on {video_path} (exit {proc.returncode}): {tail}"
            )


def ffmpeg_frame_at(video_path, seconds: float, fps: float):
    """The BGR frame nearest ``seconds`` (frame k sits at k / fps), or None past the end.

    For pulling single frames out of a raw GoPro chapter, where a cv2 seek can
    stall like its reads do (see ffmpeg_sampled_frames). ``-ss`` before ``-i``
    seeks to the keyframe before the target and decodes forward, keeping the
    first frame at or after the time asked for, so asking half a frame early
    returns the nearest frame rather than the next one. Video stream only.
    """
    start = max(0.0, seconds - 0.5 / fps)
    cmd = _ffmpeg_ppm_cmd(
        video_path,
        input_args=["-ss", f"{start:.6f}"],
        output_args=["-frames:v", "1"],
    )
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        tail = proc.stderr[-2000:].decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed on {video_path} at {seconds:.2f}s: {tail}")
    if not proc.stdout:
        return None
    # Output that is there but is not one whole PPM is a failure, never "past
    # the end": that case is the empty output handled above.
    try:
        frame = _read_ppm(io.BytesIO(proc.stdout))
    except RuntimeError:
        frame = None
    if frame is None:
        raise RuntimeError(f"ffmpeg returned no PPM for {video_path} at {seconds:.2f}s")
    return frame


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(
            f"Could not open video {video_path} to read FPS. File may be missing or corrupt."
        )
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Video {video_path} reported an invalid FPS of {fps}.")
    return fps
