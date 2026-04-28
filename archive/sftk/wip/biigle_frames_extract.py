import logging
import re
import subprocess
from pathlib import Path

from sftk.biigle_parser import BiigleParser


def _tstamp(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    tot = int(t)
    return f"{tot//3600:02d}{(tot//60) % 60:02d}{tot % 60:02d}{ms:03d}"


def _avg_fps(inp: str) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=nw=1:nk=1",
                inp,
            ],
            text=True,
        ).strip()
        if "/" in out:
            a, b = out.split("/", 1)
            a = float(a)
            b = float(b)
            return a / b if b else None
        return float(out)
    except Exception:
        return None


_clip_pat = re.compile(r"_clip_(\d+)_([0-9]+)\.", re.IGNORECASE)


def _time_in_file(
    video_path: Path, start_s: float, frame_s: float
) -> tuple[float, float | None]:
    """
    Return (t_in_file, clip_duration_if_known).
    If filename matches *_clip_<start>_<dur>.mp4, we treat the file as a 0..dur clip and use frame_seconds.
    Otherwise we assume full video and use start_seconds + frame_seconds.
    """
    m = _clip_pat.search(video_path.name)
    if m:
        dur = float(m.group(2))
        return float(frame_s), dur
    return float(start_s + frame_s), None


def save_grabs_from_df(df, video_root, out_dir, *, fast=True, limit=None):
    required = {"video_filename", "start_seconds", "frame_seconds"}
    if not required.issubset(df.columns):
        raise ValueError(f"df missing: {sorted(required - set(df.columns))}")

    video_root = Path(video_root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Video root: %s", video_root)
    logging.info("Output dir: %s", out_dir)

    ok = fail = 0
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        if limit and i > limit:
            break
        try:
            fname = str(r["video_filename"]).strip()
            start_s = float(r["start_seconds"])
            frame_s = float(r["frame_seconds"])
            label = (str(r.get("label_name") or "frame")).replace(" ", "_")

            in_path = (video_root / fname).resolve()
            if not in_path.exists():
                logging.info(f"[{i}] MISSING: {in_path}")
                fail += 1
                continue

            t_in, clip_dur = _time_in_file(in_path, start_s, frame_s)
            if clip_dur is not None and not (0.0 <= t_in <= clip_dur):
                logging.info(
                    f"[{i}] WARN: t={t_in:.3f}s outside clip duration {clip_dur:.3f}s; clamping"
                )
                t_in = max(0.0, min(clip_dur - 1e-3, t_in))

            fps = _avg_fps(str(in_path))
            fidx = int(round(t_in * fps)) if fps else None

            out_img = (
                out_dir
                / f"{in_path.stem}__t{_tstamp(t_in)}__{'f'+str(fidx) if fidx is not None else 'fNA'}__{label}.jpg"
            )

            # SPEED: fast seek by default (-ss before -i). Slightly less accurate, but 10–100x faster.
            cmd = (
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{t_in:.6f}",
                    "-i",
                    str(in_path),
                ]
                if fast
                else [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(in_path),
                    "-ss",
                    f"{t_in:.6f}",
                ]
            )
            cmd += ["-frames:v", "1", "-q:v", "2", str(out_img)]

            logging.info("[%d] RUN: %s", i, " ".join(map(str, cmd)))
            subprocess.run(cmd, check=True)

            if not out_img.exists() or out_img.stat().st_size == 0:
                raise RuntimeError(f"Encoded file missing/empty: {out_img}")

            logging.info(f"[{i}] OK  -> {out_img}")
            ok += 1
        except subprocess.CalledProcessError as e:
            logging.info(f"[{i}] FFMPEG ERROR ({e.returncode})")
            fail += 1
        except Exception as e:
            logging.info(f"[{i}] ERROR: {e}")
            fail += 1

    logging.info(f"Done. ok={ok}, fail={fail}")
    return ok, fail


if __name__ == "__main__":
    biigle_parser = BiigleParser()
    # processed_annotations_df = biigle_parser.process_video_annotations(25516, resource="volumes")
    processed_annotations_df = biigle_parser.process_video_annotations(
        26577, resource="volumes"
    )
    print(processed_annotations_df)

    # processed_dfs = {
    #         "max_n_30s_df": max_n_30s_df,
    #         "max_n_df": max_n_df,
    #         "sizes_df": sizes_df,
    #     }

    df_for_ffmpg = processed_annotations_df["max_n_30s_df"]
    print(df_for_ffmpg)
    # df columns needed: video_filename, start_seconds, frame_seconds, (optional) label_name
    save_grabs_from_df(df_for_ffmpg, ".", "frames_out")
