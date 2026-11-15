"""Data for the Survey review view. No Streamlit in here.

What both the page (`survey_review.py`) and its one-file HTML export
(`review_report.py`) say: each deployment's slate and the MaxN frames
`export_maxn_frames.py` wrote to its `review/` folder, its status in words, and
MaxN per deployment from its best source. Frame in, frame out, so it can be tested without a running app.

Sources use the dashboard's keys (`ecology_data.SOURCE_PRIORITY`): the export's
file names say `legacy`, which is `expert_legacy` here.
"""

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image, ImageOps

from spyfish.config.base import IngestStatus, VideoPresence
from spyfish.config.wrapper import config

# {drop}__frame_{secs}s__maxn_{source}_{n}_{Species}.jpg, parsed back when the
# manifest is missing a file (frames exported in an earlier run).
_NAME_RE = re.compile(
    r"^(?P<drop>.+?)__frame_(?P<secs>[\d.]+)s__maxn_(?P<source>[a-z]+)_(?P<n>\d+)_(?P<species>.+)\.jpg$"
)
# The word in an exported file name -> the dashboard's source key.
FRAME_SOURCES = {
    "expert": "expert",
    "legacy": "expert_legacy",
    "citsci": "citsci",
    "ml": "ml",
}
RANDOM = "random"
AGREEMENTS = ("match", "off", "none")
GOOD_VIDEO = (VideoPresence.PRESENT, VideoPresence.ARCHIVED)
# Counts written on the log(MaxN + 1) axis of the species boxplot.
LOG_TICKS = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500)

# What a reader must know about each source's counts: (name, the rest of the
# sentence). The page and the HTML export set the name in bold their own way.
SOURCE_NOTES = {
    "expert_legacy": (
        "Legacy expert",
        "counts were made by an expert watching the video, so the fish counted "
        "may be just before or after the frame. Any boxes on these frames were "
        "drawn by the model, not the expert: unverified, and they may miss or "
        "misname fish.",
    ),
    "expert": (
        "BIIGLE expert",
        "counts go through the ML, citizen science and expert review process. "
        "Fish can be missed early in the pipeline; checks are in place to "
        "reduce this, and retraining improves the models over time.",
    ),
    "ml": ("ML", "counts are the model's own, checked by no one."),
    "citsci": (
        "Citizen science",
        "counts are the agreed count of several volunteers, not checked by an "
        "expert.",
    ),
}
ABOUT_INTRO = (
    "This page is for reviewing surveys. It is a draft and a demonstration, "
    "not the reserve analysis. Each MaxN frame is cut from the video at the "
    "second of its MaxN annotation."
)
RANDOM_NOTE = (
    "Where no MaxN was recorded, a random frame from the last 5 minutes is "
    "shown instead."
)
BORDER = "border"  # caveat key: some frame on the page carries an agreement colour
SKIPPED = "skipped"  # a MaxN the export could not cut a frame for; listed, not drawn


def hms(seconds: Optional[float]) -> str:
    """00:21:52, no milliseconds: the reader wants the moment, not the frame."""
    if seconds is None or pd.isna(seconds):
        return "?"
    s = int(round(float(seconds)))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ── slates and MaxN frames ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Frame:
    drop_id: str
    kind: str  # "slate", "maxn" or "random"
    path: Path
    mtime: float
    source: str = ""  # dashboard source key; "" for slates and random frames
    maxn: int = 0
    scientific: str = ""  # current scientific name: the key for duplicates
    seconds: Optional[float] = None
    agreement: str = ""  # model count vs recorded MaxN: match / off / none / ""
    current: bool = False  # listed in the latest export's manifest
    window: tuple = ()  # random frames: (start, end) of the stretch, video seconds
    flag: str = (
        ""  # why the timing looks wrong (outside the sampling window), or why it was skipped
    )

    def caption(self, label: Callable[[str], str]) -> str:
        """One line, said once: the image itself carries the model's output."""
        if self.kind == "slate":
            return "Slate"
        if self.kind == RANDOM:
            at = f"No MaxN recorded: random frame at {hms(self.seconds)}"
            if self.window:  # exports since 2026-11-09 record the stretch
                start, end = self.window
                return (
                    f"{at}, from the {(end - start) / 60:.0f} min before "
                    f"sampling end ({hms(start)} to {hms(end)})"
                )
            return f"{at}, near the end of the video"
        note = " (model count, unverified)" if self.source == "ml" else ""
        if self.kind == SKIPPED:
            return f"{label(self.scientific)} MaxN {self.maxn} at {hms(self.seconds)}: not shown, {self.flag}"
        flag = f" ⚠ {self.flag}" if self.flag else ""
        return f"{label(self.scientific)} MaxN {self.maxn} at {hms(self.seconds)}{note}{flag}"


def collect_frames(
    root: Path, canonical: Callable[[str], str]
) -> dict[str, list[Frame]]:
    """``{drop_id: [slate, then MaxN frames]}`` for every deployment under ``root``.

    ``root`` is the deployment data folder: ``{survey}/{drop}/{drop}_slate.jpg``
    and ``{survey}/{drop}/review/`` (the frames and their ``maxn_frames.csv``).
    ``canonical`` maps a scientific name to its current one, so a frame saved
    as ``Pagrus_auratus`` and a later ``Chrysophrys_auratus`` are one species.
    """
    review = f"*/*/{config.review_dir_name}"
    manifest: dict[str, dict] = {}  # file names carry the DropID, so one dict
    skipped: list[Frame] = []  # rows with no file: peaks the export could not cut
    for csv_path in sorted(root.glob(f"{review}/{config.review_manifest_name}")):
        for r in pd.read_csv(csv_path).to_dict("records"):
            if pd.isna(r.get("file")) or not str(r["file"]).strip():
                if r.get("source") in FRAME_SOURCES:
                    skipped.append(
                        Frame(
                            drop_id=str(r["drop_id"]),
                            kind=SKIPPED,
                            path=csv_path,
                            mtime=0.0,
                            source=FRAME_SOURCES[r["source"]],
                            maxn=int(r["maxn"]),
                            scientific=canonical(str(r["scientific_name"])),
                            seconds=(
                                float(r["seconds"])
                                if pd.notna(r.get("seconds"))
                                else None
                            ),
                            flag=str(r.get("skipped") or "no frame"),
                        )
                    )
                continue
            manifest[str(r["file"])] = r

    suffix = config.slate_frame_suffix
    frames = [
        Frame(p.name[: -len(suffix)], "slate", p, p.stat().st_mtime)
        for p in sorted(root.glob(f"*/*/*{suffix}"))
    ]
    for p in sorted(root.glob(f"{review}/*.jpg")):
        m = _NAME_RE.match(p.name)
        if not m or m["source"] not in (*FRAME_SOURCES, RANDOM):
            continue
        row = manifest.get(p.name, {})
        agreement = row.get("agreement")
        has_window = pd.notna(row.get("window_start")) and pd.notna(
            row.get("window_end")
        )
        frames.append(
            Frame(
                drop_id=m["drop"],
                kind=RANDOM if m["source"] == RANDOM else "maxn",
                path=p,
                mtime=p.stat().st_mtime,
                source=FRAME_SOURCES.get(m["source"], ""),
                maxn=int(m["n"]),
                scientific=canonical(m["species"].replace("_", " ")),
                seconds=float(row["seconds"]) if "seconds" in row else float(m["secs"]),
                agreement=agreement if agreement in AGREEMENTS else "",
                current=bool(row),
                window=(
                    (float(row["window_start"]), float(row["window_end"]))
                    if has_window
                    else ()
                ),
                flag=str(row["timing"]) if pd.notna(row.get("timing")) else "",
            )
        )

    by_drop: dict[str, list[Frame]] = {}
    for f in _latest_per_species(frames) + skipped:
        by_drop.setdefault(f.drop_id, []).append(f)
    order = list(FRAME_SOURCES.values())
    for v in by_drop.values():
        v.sort(
            key=lambda f: (
                f.kind != "slate",
                order.index(f.source) if f.source in order else len(order),
                -f.maxn,
                f.scientific,
            )
        )
    return by_drop


def _latest_per_species(frames: list[Frame]) -> list[Frame]:
    """One frame per deployment, source and species.

    Exports overwrite by file name, so a renamed species or a moved second
    leaves the old file behind: keep the one in the latest manifest, else the
    newest file.
    """
    best: dict[tuple, Frame] = {}
    out: list[Frame] = []
    for f in frames:
        if f.kind == "slate":
            out.append(f)
            continue
        key = (f.drop_id, f.kind, f.source, f.scientific if f.kind == "maxn" else "")
        old = best.get(key)
        if old is None or (f.current, f.mtime) > (old.current, old.mtime):
            best[key] = f
    return out + list(best.values())


def pick_frames(
    mine: list[Frame], sources: list[str]
) -> tuple[list[Frame], list[Frame]]:
    """(slate, frames to show) for one deployment's exported frames.

    Nothing from the chosen sources: the model's peaks, else the random
    stand-in.
    """
    slate = [f for f in mine if f.kind == "slate"][:1]
    picks = [f for f in mine if f.kind == "maxn" and f.source in sources]
    if not picks:
        picks = [f for f in mine if f.kind == "maxn" and f.source == "ml"] or [
            f for f in mine if f.kind == RANDOM
        ]
    return slate, picks


def skipped_notes(
    mine: list[Frame], sources: list[str], label: Callable[[str], str]
) -> list[str]:
    """One line per MaxN of the shown sources that has no frame, so a gap is never silent."""
    return [f.caption(label) for f in mine if f.kind == SKIPPED and f.source in sources]


def caveats(picks: list[Frame]) -> set[str]:
    """Which notes these frames need under them: source keys, `BORDER`, `RANDOM`."""
    told = {f.source for f in picks if f.kind == "maxn"}
    if any(f.agreement for f in picks):
        told.add(BORDER)
    if any(f.kind == RANDOM for f in picks):
        told.add(RANDOM)
    return told


def survey_of(drop_id: str) -> Optional[str]:
    """SurveyID of an exported file's drop, None when the name is not a DropID."""
    try:
        return config.get_survey_id_from_drop(config.validate_drop_id(drop_id))
    except ValueError:
        return None


def read_drop_list(path: Path) -> set[str]:
    """DropIDs from a one-per-line review file (`#` comments, extra words ignored)."""
    if not path.exists():
        return set()
    return {
        line.split()[0]
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


# ── status in words ──────────────────────────────────────────────────────────


def short_reason(msg: str) -> str:
    """Ingest's validation messages in a few words for the status table."""
    # before the next test: this message also says "missing sampling window"
    if "sampling_start=0" in msg:
        return "no sampling start"
    if "missing" in msg and "sampling window" in msg:
        return "no sampling window"
    m = re.search(r"is (\d+)s shorter", msg)
    if m:
        return f"window {m.group(1)} s short"
    m = re.search(r"sampling_end=(\d+)s is shorter than expected \((\d+)s", msg)
    if m:
        return f"window {int(m.group(2)) - int(m.group(1))} s short"
    return msg


def deployment_status(row: dict, reasons: list[str]) -> tuple[str, str]:
    """(short status, why) for one deployment row, in a reader's words."""
    ingest = row.get(IngestStatus.COLUMN)
    if ingest == IngestStatus.OK:
        if row.get("video_presence") not in GOOD_VIDEO:
            return "no video", f"video {row.get('video_presence')}"
        return "ok", ""
    if ingest == IngestStatus.EXCLUDED and row.get("is_bad_deployment"):
        return "bad deployment", "not counted"
    why = "; ".join(short_reason(m) for m in reasons) or "no reason recorded"
    if ingest == IngestStatus.VALIDATION_ERROR:
        return "error", why
    return str(ingest or "unknown").replace("_", " "), why


def slate_state(drop_id: str, on_disk: set, verified: set, no_board: set) -> str:
    if drop_id in on_disk:
        return "confirmed" if drop_id in verified else "not confirmed"
    return "no board shown" if drop_id in no_board else "missing"


# ── MaxN per deployment, best source ─────────────────────────────────────────


def best_source_maxn(
    ann: pd.DataFrame, drops: list[str], sources: list[str]
) -> tuple[pd.Series, pd.DataFrame]:
    """Each deployment's MaxN per species, from its best source only.

    ``ann`` has ``drop_id``, ``scientific_name`` (NaN = reviewed, nothing
    seen), ``max_interval`` and ``source``; ``sources`` is the allowed ones,
    best first. Returns ``(drop -> source used, drop x species MaxN)``. A
    deployment reviewed with nothing seen is a row of zeros, so it still
    counts in a mean.
    """
    df = ann[ann["drop_id"].isin(drops) & ann["source"].isin(sources)]
    if df.empty:
        return pd.Series(dtype=object), pd.DataFrame()
    rank = df["source"].map(sources.index)
    best = rank.groupby(df["drop_id"]).transform("min")
    df = df[rank == best]
    chosen = df.groupby("drop_id")["source"].first()
    per_drop = (
        df.dropna(subset=["scientific_name"])
        .groupby(["drop_id", "scientific_name"])["max_interval"]
        .max()
        .unstack(fill_value=0)
        .reindex(chosen.index, fill_value=0)
        .fillna(0)
    )
    return chosen, per_drop


# ── images ───────────────────────────────────────────────────────────────────


def jpeg_bytes(
    path: Path,
    width: Optional[int],
    quality: int,
    border: Optional[str] = None,
    border_px: int = 0,
) -> bytes:
    """``path`` as a JPEG no wider than ``width``, with an optional colour frame.

    The frame is added around the picture, never drawn over it. `draft` makes
    the JPEG decoder itself produce a half / quarter / eighth size image, so a
    thumbnail of a 1920 px frame never decodes all of its pixels.
    """
    im = Image.open(path)
    if width and im.width > width:
        im.draft("RGB", (width, round(im.height * width / im.width)))
    im = im.convert("RGB")
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    if border and border_px:
        im = ImageOps.expand(im, border=border_px, fill=border)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()
