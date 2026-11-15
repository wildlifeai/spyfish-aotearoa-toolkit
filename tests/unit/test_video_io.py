"""spyfish.extraction.video_io: the PPM reader and the ffmpeg command both
ffmpeg readers share, checked on byte buffers (no ffmpeg, no video)."""

import io
from types import SimpleNamespace

import numpy as np
import pytest

from spyfish.extraction import video_io


def _ppm(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    return b"P6\n%d %d\n255\n" % (w, h) + rgb.tobytes()


def _rgb(seed: int, w: int = 4, h: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_read_ppm_returns_bgr_with_the_header_size():
    rgb = _rgb(1)
    frame = video_io._read_ppm(io.BytesIO(_ppm(rgb)))
    assert frame.shape == (3, 4, 3)
    assert np.array_equal(frame, rgb[:, :, ::-1])
    assert frame.flags["C_CONTIGUOUS"]


def test_read_ppm_reads_images_back_to_back_then_none():
    a, b = _rgb(1), _rgb(2, w=2, h=5)
    stream = io.BytesIO(_ppm(a) + _ppm(b))
    assert np.array_equal(video_io._read_ppm(stream), a[:, :, ::-1])
    assert np.array_equal(video_io._read_ppm(stream), b[:, :, ::-1])
    assert video_io._read_ppm(stream) is None


def test_read_ppm_returns_none_when_the_pixels_are_cut_short():
    assert video_io._read_ppm(io.BytesIO(_ppm(_rgb(1))[:-5])) is None


def test_read_ppm_raises_on_anything_but_a_ppm_header():
    with pytest.raises(RuntimeError, match="out of step"):
        video_io._read_ppm(io.BytesIO(b"P5\n4 3\n255\n" + bytes(36)))


def test_ffmpeg_ppm_cmd_puts_the_seek_before_the_input_and_the_rest_after_the_map():
    cmd = video_io._ffmpeg_ppm_cmd(
        "v.mp4", input_args=["-ss", "1.5"], output_args=["-frames:v", "1"]
    )
    assert cmd == [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        "1.5",
        "-i",
        "v.mp4",
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "ppm",
        "-",
    ]


def _fake_run(stdout=b"", returncode=0, stderr=b""):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

    run.calls = calls
    return run


def test_ffmpeg_frame_at_asks_half_a_frame_early_and_decodes_the_ppm(monkeypatch):
    rgb = _rgb(3)
    run = _fake_run(stdout=_ppm(rgb))
    monkeypatch.setattr(video_io.subprocess, "run", run)
    frame = video_io.ffmpeg_frame_at("v.mp4", seconds=2.0, fps=25.0)
    assert np.array_equal(frame, rgb[:, :, ::-1])
    cmd = run.calls[0]
    assert cmd[cmd.index("-ss") + 1] == "1.980000"
    assert cmd.index("-ss") < cmd.index("-i")


def test_ffmpeg_frame_at_is_none_past_the_end(monkeypatch):
    monkeypatch.setattr(video_io.subprocess, "run", _fake_run(stdout=b""))
    assert video_io.ffmpeg_frame_at("v.mp4", 9999.0, 25.0) is None


@pytest.mark.parametrize("stdout", [b"garbage\n1 1\n255\nxyz", _ppm(_rgb(4))[:-5]])
def test_ffmpeg_frame_at_raises_on_output_that_is_not_one_whole_ppm(
    monkeypatch, stdout
):
    monkeypatch.setattr(video_io.subprocess, "run", _fake_run(stdout=stdout))
    with pytest.raises(RuntimeError, match="returned no PPM"):
        video_io.ffmpeg_frame_at("v.mp4", 1.0, 25.0)


def test_ffmpeg_frame_at_raises_with_ffmpegs_own_error(monkeypatch):
    monkeypatch.setattr(
        video_io.subprocess, "run", _fake_run(returncode=1, stderr=b"moov not found")
    )
    with pytest.raises(RuntimeError, match="moov not found"):
        video_io.ffmpeg_frame_at("v.mp4", 1.0, 25.0)
