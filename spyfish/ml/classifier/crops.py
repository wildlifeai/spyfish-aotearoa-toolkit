"""Expert boxes as classifier crops: where they come from, how they are cut.

Everything here reads data already on disk. Three sources, chosen by
``TWO_STAGE_SOURCE`` in the environment or ``training.classifier.source``:

* ``biigle_csv``: every drop's BIIGLE raw CSV and the frames it names under
  ``deployment_data/``. Species-level labels, what the classifier trains on.
* ``yolo_dataset``: the assembled ``species/`` dataset ``--retrain --data-prep``
  builds (what NeSI has, frozen test drops included). Its labels have been
  through the training rollup, so eels arrive as ``Anguilliformes``.
* ``uoa``: the University of Auckland mussel-farm expert boxes (BIIGLE
  project 4510, re-hosted with deployment-ID names under
  ``uoa/kutai/process_files/deployment_data/<deployment>/``), grouped by
  deployment (``uoa/docs/uoa_two_stage_brief.md``).

Species names go through ``canonicalize_species``: the registry (synonyms,
``Common - Scientific`` spellings, workflow labels) and then the same-animal
pairs in ``training.species_canonicalization``, so a red gurnard crop and a
``Triglidae`` crop are one class here as they are for YOLO. Which thin taxa the
detector pools (the rollup) is NOT applied: the classifier's job is to name
what the detector only found.

Knobs come from ``config.yaml``: ``ml_inference.classifier`` for what is shared
with inference (crop margin, min box size, backbone) and ``training.classifier``
for the rest. Moved out of ``scripts/wip/two_stage/common.py`` 2026-11-10.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from spyfish.config.ml import CLASSIFIER_SOURCES
from spyfish.config.wrapper import config
from spyfish.extraction.render_annotations import center_to_corners
from spyfish.ml.classifier import artifacts
from spyfish.ml.classifier.species_classifier import pick_device  # noqa: F401
from spyfish.ml.training.yolo_labels import SPLITS, read_class_names, read_label_rows

ROOT = config.project_root
DEPLOYMENT_DATA = config.deployment_data_dir
RAW_SUFFIX = config.biigle_expert_raw_suffix


def resolve_source() -> str:
    """``TWO_STAGE_SOURCE`` from the environment, else ``training.classifier.source``."""
    source = os.environ.get("TWO_STAGE_SOURCE") or config.classifier_training_source
    if source not in CLASSIFIER_SOURCES:
        raise ValueError(
            f"TWO_STAGE_SOURCE={source!r}; expected one of {CLASSIFIER_SOURCES}"
        )
    return source


# ── names that depend on the environment, resolved on first use ─────────────
# SOURCE, OUT_DIR, UOA_DETECTOR_SPLIT and MAX_PER_DROP_SPECIES read
# TWO_STAGE_SOURCE / TWO_STAGE_UOA_RUN. They are still module attributes
# (`crops.OUT_DIR`, `from ...crops import OUT_DIR`), but importing this module
# no longer reads the environment or fails on a bad TWO_STAGE_SOURCE: the first
# access resolves the value and stores it as a plain global, so it is read
# once per process as before. Code in this module goes through `_lazy(name)`.


def _resolve_out_dir() -> Path:
    # UoA outputs go to two_stage_dir/uoa; TWO_STAGE_UOA_RUN=<name> gives a comparison
    # run (e.g. without the snapper stills) its own folder instead of overwriting it.
    if _lazy("SOURCE") == "uoa":
        return config.two_stage_dir / os.environ.get("TWO_STAGE_UOA_RUN", "uoa")
    return config.two_stage_dir


_LAZY = {
    "SOURCE": resolve_source,
    "OUT_DIR": _resolve_out_dir,
    # Written by uoa/detector_dataset.py: which train-day deployments the detector
    # held out as `val`. load_uoa_boxes reads it so detector_recall can report val.
    "UOA_DETECTOR_SPLIT": lambda: _lazy("OUT_DIR") / artifacts.DETECTOR_SPLIT,
    # Crops per (drop, species); no cap for UoA, where every box counts.
    "MAX_PER_DROP_SPECIES": lambda: (
        None
        if _lazy("SOURCE") == "uoa"
        else config.classifier_training_max_crops_per_drop_species
    ),
}


def _lazy(name: str):
    if name not in globals():
        globals()[name] = _LAZY[name]()
    return globals()[name]


def __getattr__(name: str):
    if name in _LAZY:
        return _lazy(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── UoA (Underwood & Jeffs, Jan 2022) ──────────────────────────────────────────
# BIIGLE project 4510's expert boxes, re-hosted with deployment-ID frame names
# by uoa/scripts/rehost_biigle_4510.py (2026-11-10; the old volume_<id>
# downloads carried the 18 Jan camera-folder names and a duplicated chapter).
# Laid out like a pipeline drop: <dir>/frames/ + <dir>/annotations/
# <dir>_biigle_expert_raw.csv, one dir per deployment (deployment_data/
# <deployment>/, 2026-11-11; there is no day level), plus the snapper stills
# (frames cut from other people's videos, not UoA footage): one `snapper_stills`
# group with no day, which trains in every fold and is never scored
# (classes.grouped_folds). TWO_STAGE_UOA_STILLS=0 leaves them out. These
# describe one external dataset, so they stay here rather than in config.yaml
# (uoa/docs/uoa_design.md).
UOA_DATA = ROOT / "uoa" / "kutai" / "process_files" / "deployment_data"
UOA_BOX_DAYS = ("20220118", "20220119")
# One JSON per BIIGLE volume ({volume_id, name, reviewed, files}). A frame trains
# only once a volume holding it is marked reviewed: until then its missing boxes
# mean "not looked at yet", not "no fish" (uoa/scripts/upload_review_volume.py).
UOA_VOLUMES = ROOT / "uoa" / "kutai" / "metadata" / "biigle_volumes"
UOA_STILLS = "snapper_stills"
UOA_STILLS_DAY = "extra"
UOA_HOLDOUT_DAY = "20220120"  # enters nothing: not training, tuning or any prior
UOA_TRAIN_DAYS = ("20220118", "20220119", "20220121")
# The paper's five classes: four named species, everything else is `fish`.
# Registry spellings (WoRMS-accepted): the paper says `Pagrus auratus`, every
# label is resolved to `Chrysophrys auratus` before it meets this list, and
# `read_crops_meta` / `read_predictions` do the same to files written before
# the rename. A test asserts every entry here is already canonical.
UOA_CLASSES = (
    "Girella tricuspidata",  # parore
    "Scorpis lineolata",  # silver sweep
    "Chrysophrys auratus",  # snapper (`Pagrus auratus` in the paper)
    "Trachurus novaezelandiae",  # jack mackerel
)
UOA_OTHER = "fish"
UOA_IMGSZ = 1280  # small fish in this footage; the DOC pipeline runs 640

# Classifier knobs (config.yaml)
CROP_MARGIN = config.classifier_crop_margin
MIN_BOX_PX = config.classifier_min_box_px
SEED = config.classifier_training_seed
# Labels that are not species (bait, the catch-all): never a crop, never a class.
NOT_CLASSES = set(config.classifier_training_exclude_classes)
# Species parked for every model (kina): a crop of one must not train the classifier.
EXCLUDED_SPECIES = set(config.training_excluded_species)
NOT_FISH = config.classifier_training_not_fish_class
NOT_SHAPES = {"LineString"}  # BIIGLE shapes that are not boxes

_VOLUME_DIR = re.compile(r"^volume_\d+$")
NO_RESERVE = "NONE"  # no site prior applies; the crop still trains and scores


def reserve_of(drop_dir_name: str) -> str:
    """Reserve code for a drop directory, or ``NONE`` when there is none.

    Pipeline drops give their prefix. Legacy drops with site names instead of
    codes (``ANG_20210313_BUV_NorthRDeep_1_01``) still carry the reserve in
    the prefix. Every ``volume_<id>`` directory gets ``UOA`` and anything else
    ``NONE``; both mean "no site prior applies", so the mislabel of
    ``volume_30505`` (a DOC training volume, not UoA footage) changes nothing
    here. The UoA source mode reads only ``uoa_volume_dirs()``, not by this tag.
    """
    if _VOLUME_DIR.match(drop_dir_name):
        return "UOA"
    return config.get_reserve_code_from_drop(drop_dir_name) or NO_RESERVE


def find_biigle_raw_csvs(suffixes=(RAW_SUFFIX,), any_depth: bool = False) -> list[Path]:
    """Every BIIGLE raw CSV under ``deployment_data`` ending in one of
    ``suffixes`` (``config.biigle_expert_raw_suffix`` and / or
    ``config.biigle_training_raw_suffix``), sorted.

    The ``legacy_video_`` variants are left out: they describe the same drops'
    frames under another name, and the plain CSV is the one whose filenames
    exist on disk. ``any_depth`` also finds an ``annotations/`` folder that is
    not exactly ``<survey>/<drop>/annotations``.
    """
    pattern = "**/annotations/*" if any_depth else "*/*/annotations/*"
    csvs = {p for suffix in suffixes for p in DEPLOYMENT_DATA.glob(pattern + suffix)}
    return sorted(c for c in csvs if not c.name.startswith("legacy_video_"))


def find_expert_csvs() -> list[Path]:
    return find_biigle_raw_csvs()


def frames_on_disk() -> dict[str, Path]:
    if _lazy("SOURCE") == "yolo_dataset":
        # where --retrain --data-prep assembles images/, labels/, data.yaml
        return {p.name: p for p in config.training_dataset_dir.glob("images/*/*.jpg")}
    if _lazy("SOURCE") == "uoa":
        reviewed = uoa_reviewed_frames()
        return {
            p.name: p
            for vol in uoa_volume_dirs()
            for p in (vol / "frames").iterdir()
            # the snapper stills are PNG
            if p.suffix.lower() in (".jpg", ".png") and p.name in reviewed
        }
    return {p.name: p for p in DEPLOYMENT_DATA.glob("*/*/frames/*.jpg")}


def rect_from_points(points: str) -> tuple[float, float, float, float]:
    """BIIGLE rectangle points ``[x1,y1,...,x4,y4]`` → ``x_min, y_min, x_max, y_max``."""
    vals = [float(v) for v in points.strip("[]").split(",")]
    xs, ys = vals[0::2], vals[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def species_name(label: str) -> str:
    """The training name of a BIIGLE label: registry resolution (synonyms,
    spellings, workflow labels) then the same-animal pairs."""
    from spyfish.ml.training.prepare_training_data import canonicalize_species

    return canonicalize_species(label)


def canonical_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return ``frame`` with ``column`` passed through ``species_name``, so a
    file written before a rename (``Pagrus auratus``) reads as today's name.
    Non-species values (``fish``, ``bait``, ``other fish``) pass unchanged."""
    if column not in frame.columns or frame.empty:
        return frame
    names = {n: species_name(n) for n in frame[column].dropna().unique()}
    changed = {a: b for a, b in names.items() if a != b}
    if changed:
        logging.info(
            f"{column}: {len(changed)} name(s) canonicalised on read: {changed}"
        )
    return frame.assign(**{column: frame[column].map(lambda n: names.get(n, n))})


def read_crops_meta(path: Path | None = None) -> pd.DataFrame:
    """``crops_meta.csv`` with ``species`` canonicalised. Every reader of the
    crop table goes through here, never ``pd.read_csv`` directly."""
    path = path or _lazy("OUT_DIR") / artifacts.CROPS_META
    return canonical_column(pd.read_csv(path, dtype={"crop_id": str}), "species")


def read_predictions(path: Path) -> pd.DataFrame:
    """A detector predictions file (``yolo_class`` column) with the class
    names canonicalised: a checkpoint trained before 2026-11-04 still says
    ``Pagrus auratus``."""
    return canonical_column(pd.read_csv(path), "yolo_class")


def load_expert_boxes() -> pd.DataFrame:
    """One row per expert box with a frame on disk, species canonicalised.

    Columns: drop, reserve, filename, frame_path, species, x_min, y_min, x_max, y_max.
    """
    frames = frames_on_disk()
    parts = []
    for csv in find_expert_csvs():
        df = pd.read_csv(csv)
        df = df[df["filename"].isin(frames) & ~df["shape_name"].isin(NOT_SHAPES)]
        if df.empty:
            continue
        df = df.assign(drop=csv.parents[1].name)
        parts.append(df[["drop", "filename", "label_name", "points"]])
    boxes = pd.concat(parts, ignore_index=True)
    boxes["reserve"] = boxes["drop"].map(reserve_of)
    boxes["frame_path"] = boxes["filename"].map(frames)
    names = {n: species_name(n) for n in boxes["label_name"].unique()}
    boxes["species"] = boxes["label_name"].map(names)
    return _finish_boxes(boxes)


def _finish_boxes(boxes: pd.DataFrame, log_as: str | None = None) -> pd.DataFrame:
    """The shared tail of the box loaders.

    Turns BIIGLE ``points`` into ``x_min, y_min, x_max, y_max`` when the frame
    has that column, adds ``w`` / ``h``, drops boxes with a side under
    ``MIN_BOX_PX`` (logged when ``log_as`` names the source), drops the BIIGLE
    columns that have been consumed (``points``, ``label_name``) and returns
    the rest with a fresh index.
    """
    if "points" in boxes.columns:
        rect = boxes["points"].map(rect_from_points)
        boxes[["x_min", "y_min", "x_max", "y_max"]] = pd.DataFrame(
            rect.tolist(), index=boxes.index
        )
    boxes["w"] = boxes["x_max"] - boxes["x_min"]
    boxes["h"] = boxes["y_max"] - boxes["y_min"]
    small = ~((boxes["w"] >= MIN_BOX_PX) & (boxes["h"] >= MIN_BOX_PX))
    if log_as and small.any():
        logging.info(
            f"{log_as}: {int(small.sum())} boxes under {MIN_BOX_PX} px dropped"
        )
    consumed = [c for c in ("points", "label_name") if c in boxes.columns]
    return boxes[~small].drop(columns=consumed).reset_index(drop=True)


def crop_with_margin(img, x_min, y_min, x_max, y_max, margin=CROP_MARGIN):
    w, h = x_max - x_min, y_max - y_min
    W, H = img.size
    box = (
        max(0, int(x_min - margin * w)),
        max(0, int(y_min - margin * h)),
        min(W, int(x_max + margin * w)),
        min(H, int(y_max + margin * h)),
    )
    return img.crop(box)


# ── alternative source: the assembled YOLO dataset ──────────────────────────
# TWO_STAGE_SOURCE=yolo_dataset reads `training.local_training_dir`/species
# (images/{split}/*.jpg + labels/{split}/*.txt + data.yaml) instead of the
# per-drop BIIGLE CSVs. That is what exists on Mahuika after --retrain --data-prep,
# and it carries every BIIGLE drop, including the frozen test drops.


def drop_of_dataset_image(img: Path) -> str:
    """The drop an assembled-dataset image belongs to.

    Data prep symlinks each frame into images/{split}/ under its ORIGINAL name,
    and the names are not uniform: pipeline frames are ``{drop}__frame_…``, but
    BIIGLE training volumes use UUIDs and UoA chapters ``ESK_X_S_…``. The link
    target is always ``…/{drop_dir}/frames/{name}``, so the drop is its
    grandparent. A plain copy falls back to the ``{drop}__`` prefix.
    """
    if img.is_symlink():
        target = Path(os.readlink(img))
        if not target.is_absolute():
            target = (img.parent / target).resolve()
        if target.parent.name == "frames":
            return target.parents[1].name
    stem = img.stem
    if "__" in stem:
        return stem.split("__")[0]
    raise ValueError(
        f"Cannot tell which drop {img} belongs to: not a symlink into a drop's frames/ "
        f"and no '{{drop}}__' prefix. Assemble the dataset with symlinks."
    )


def load_yolo_dataset_boxes() -> pd.DataFrame:
    """Same columns as ``load_expert_boxes`` plus ``split``, from the YOLO dataset."""
    from PIL import Image

    # where --retrain --data-prep assembles images/, labels/, data.yaml
    root = config.training_dataset_dir
    names = read_class_names(config.training_data_yaml, required=True)
    rows = []
    for split in SPLITS:
        for img in sorted((root / "images" / split).glob("*.jpg")):
            label = root / "labels" / split / f"{img.stem}.txt"
            if not label.exists():
                continue
            # every row, so a row that is not `class cx cy w h` fails the
            # unpacking below instead of being passed over
            lines = read_label_rows(label, boxes_only=False)
            if not lines:
                continue
            W, H = Image.open(img).size
            drop = drop_of_dataset_image(img)
            for cid, cx, cy, w, h in (
                (int(a), float(b), float(c), float(d), float(e))
                for a, b, c, d, e in lines
            ):
                x1, y1, x2, y2 = center_to_corners(cx, cy, w, h)
                rows.append(
                    (
                        drop,
                        img.name,
                        img,
                        names[cid],
                        split,
                        x1 * W,
                        y1 * H,
                        x2 * W,
                        y2 * H,
                    )
                )
    boxes = pd.DataFrame(
        rows,
        columns=[
            "drop",
            "filename",
            "frame_path",
            "species",
            "split",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ],
    )
    boxes["reserve"] = boxes["drop"].map(reserve_of)
    per_reserve = (
        boxes.drop_duplicates("drop").groupby("reserve")["drop"].size().to_dict()
    )
    logging.info(
        f"YOLO dataset: {len(boxes)} boxes, {boxes['filename'].nunique()} frames, "
        f"{boxes['drop'].nunique()} drops; drops per reserve {per_reserve}"
    )
    odd = sorted(d for d in boxes["drop"].unique() if reserve_of(d) == NO_RESERVE)
    if odd:
        logging.warning(
            f"YOLO dataset: {len(odd)} drops with no reserve, e.g. {odd[:5]}"
        )
    return _finish_boxes(boxes)


# ── UoA source: the mussel-farm volumes, grouped by deployment ─────────────


def uoa_volume_dirs() -> list[Path]:
    """The UoA box directories (frames/ + annotations/): every deployment of the box
    days that has an expert CSV (header only when its frames carry no box), then the stills.
    """
    dirs = sorted(
        csv.parent.parent
        for day in UOA_BOX_DAYS
        for csv in UOA_DATA.glob(f"*_{day}_*/annotations/*{RAW_SUFFIX}")
    )
    if os.environ.get("TWO_STAGE_UOA_STILLS", "1") != "0":
        dirs.append(UOA_DATA / "extra_videos" / UOA_STILLS)
    return [d for d in dirs if d.is_dir()] if _lazy("SOURCE") == "uoa" else dirs


def uoa_reviewed_frames() -> frozenset[str]:
    """Names of the frames in BIIGLE volumes marked reviewed (``UOA_VOLUMES``)."""
    names: set[str] = set()
    for f in UOA_VOLUMES.glob("*.json"):
        vol = json.loads(f.read_text())
        if vol["reviewed"]:
            names.update(Path(n).name for n in vol["files"])
    if not names:
        raise FileNotFoundError(f"no reviewed BIIGLE volume listed in {UOA_VOLUMES}")
    return frozenset(names)


def uoa_source_name(vol: Path) -> str:
    """The day (``20220118``) of a deployment folder, ``snapper_stills`` for the stills."""
    return vol.name if vol.name == UOA_STILLS else uoa_day_of(vol.name)


def uoa_deployment_of(filename: str) -> str:
    """The deployment a UoA chapter frame belongs to.

    Two naming schemes exist on disk (see uoa/scripts/build_volume_dataset.py,
    whose ``deployment_id`` this reuses): toolkit frames
    ``ESK_X_S_20220118_PM_v008_f040200_t000670.66s.jpg`` and UoA-script
    uploads ``ESK_B_20220119_MM_M_v08__t000574.57.jpg``. Both put the chapter
    in ``_vNN`` and the deployment before it. The two schemes spell a
    deployment differently (18 Jan has ``X``/``C`` and the folder's time label,
    19 Jan has the canonical ``SITE_HAB_DATE_BUCKET_TREAT``), which is fine for
    grouping: no deployment spans two schemes.

    A frame from the snapper stills folder is no deployment: it returns
    ``UOA_STILLS``, the one group all stills share. Recognised by folder, not by
    name shape, so a mis-named chapter frame still fails in ``uoa_day_of``.
    """
    from uoa.scripts.build_volume_dataset import deployment_id

    if Path(filename).name in uoa_stills_names():
        return UOA_STILLS
    return deployment_id(Path(filename).stem)


def uoa_stills_names() -> frozenset[str]:
    """File names in the snapper stills folder (empty when it is absent)."""
    folder = UOA_DATA / "extra_videos" / UOA_STILLS / "frames"
    if not hasattr(uoa_stills_names, "_cache"):
        uoa_stills_names._cache = (
            frozenset(p.name for p in folder.iterdir())
            if folder.is_dir()
            else frozenset()
        )
    return uoa_stills_names._cache


_DAY_TOKEN = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


def uoa_day_of(deployment: str) -> str:
    """The YYYYMMDD token in a UoA deployment id (``UOA_STILLS_DAY`` for the stills
    group); raises when there is none."""
    if deployment == UOA_STILLS:
        return UOA_STILLS_DAY
    m = _DAY_TOKEN.search(deployment)
    if not m:
        raise ValueError(f"No date in UoA deployment id {deployment!r}")
    return m.group(1)


def load_uoa_boxes() -> pd.DataFrame:
    """Expert boxes on the UoA volumes only, with ``drop`` = deployment and ``split``.

    Same columns as ``load_expert_boxes`` plus ``volume``, ``day`` and ``split``:
    ``test`` for every frame of ``UOA_HOLDOUT_DAY``, ``val`` for the deployments
    ``UOA_DETECTOR_SPLIT`` names (once the detector dataset exists), ``train``
    otherwise. A volume whose frames are not on disk is logged and skipped:
    locally only the 18 Jan volumes have frames, 19 Jan's are on NeSI.

    Names go through the registry only (no same-animal pairs): the paper's
    class list is fixed in ``UOA_CLASSES`` and spelled as the paper spells it.
    """
    from spyfish.config.species import species_registry

    reg = species_registry()
    frames = frames_on_disk()
    parts = []
    for vol in uoa_volume_dirs():
        csvs = list((vol / "annotations").glob(f"*{RAW_SUFFIX}"))
        if not csvs:
            logging.warning(f"UoA volume {vol.name}: no expert CSV on disk, skipped")
            continue
        df = pd.read_csv(csvs[0])
        if df.empty:  # a deployment whose frames the experts found empty
            continue
        n_frames_on_disk = df["filename"].isin(frames).sum()
        if n_frames_on_disk == 0:
            logging.warning(
                f"UoA volume {vol.name}: {len(df)} boxes but none of its "
                f"{df['filename'].nunique()} frames are on disk, skipped"
            )
            continue
        df = df[df["filename"].isin(frames) & ~df["shape_name"].isin(NOT_SHAPES)]
        parts.append(
            df[["filename", "label_name", "points"]].assign(volume=uoa_source_name(vol))
        )
    if not parts:
        raise FileNotFoundError(
            f"No UoA expert boxes with frames on disk under {UOA_DATA} "
            f"(run uoa/scripts/rehost_biigle_4510.py prepare)"
        )
    boxes = pd.concat(parts, ignore_index=True)
    boxes["drop"] = boxes["filename"].map(uoa_deployment_of)
    boxes["day"] = boxes["drop"].map(uoa_day_of)
    boxes["reserve"] = "UOA"
    boxes["frame_path"] = boxes["filename"].map(frames)
    boxes["species"] = boxes["label_name"].map(lambda n: reg.resolve_name(n)[0])
    boxes["split"] = np.where(boxes["day"] == UOA_HOLDOUT_DAY, "test", "train")
    detector_split = _lazy("UOA_DETECTOR_SPLIT")
    if detector_split.exists():
        held = pd.read_csv(detector_split)
        val = set(held.loc[held["split"] == "val", "deployment"])
        boxes.loc[boxes["drop"].isin(val) & (boxes["split"] == "train"), "split"] = (
            "val"
        )
    unknown_days = (
        set(boxes["day"]) - set(UOA_TRAIN_DAYS) - {UOA_HOLDOUT_DAY, UOA_STILLS_DAY}
    )
    if unknown_days:
        raise ValueError(f"UoA frames from unexpected days {sorted(unknown_days)}")
    boxes = _finish_boxes(boxes, log_as="UoA")
    per = boxes.groupby(["day", "split"])["drop"].agg(["size", "nunique"])
    logging.info(f"UoA boxes by day and split (boxes, deployments):\n{per}")
    return boxes


def load_boxes() -> pd.DataFrame:
    """Expert boxes from whichever source ``SOURCE`` names."""
    if _lazy("SOURCE") == "yolo_dataset":
        return load_yolo_dataset_boxes()
    if _lazy("SOURCE") == "uoa":
        return load_uoa_boxes()
    return load_expert_boxes()


def classifier_boxes(boxes: pd.DataFrame) -> pd.DataFrame:
    """The boxes the classifier may train on: no bait, no catch-all, no species
    parked in ``training.excluded_species``."""
    skip = NOT_CLASSES | EXCLUDED_SPECIES
    dropped = boxes["species"].isin(skip)
    if dropped.any():
        logging.info(
            f"classifier crops: {int(dropped.sum())} box(es) left out as "
            f"{sorted(boxes.loc[dropped, 'species'].unique())}"
        )
    return boxes[~dropped]


# ── BioCLIP 2 weights live with the other models, not in a per-user HF cache ──
# `open_clip` downloads the hub repo into `cache_dir`; pointing every call at
# this one directory means one download per machine, in a place that is backed
# up with process_files/ and reused by anything else that wants BioCLIP.
BIOCLIP_BACKBONE = config.classifier_backbone
BIOCLIP_DIR = config.bioclip_cache_dir


def load_bioclip(device: str | None = None):
    """BioCLIP 2 ``(model, preprocess, tokenizer)`` from ``BIOCLIP_DIR``, in eval mode.

    First call on a machine with internet fills ``BIOCLIP_DIR`` (~1.7 GB);
    with ``HF_HUB_OFFLINE=1`` set, as the SLURM job does, it only ever reads.
    """
    import open_clip

    BIOCLIP_DIR.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        BIOCLIP_BACKBONE, cache_dir=str(BIOCLIP_DIR)
    )
    tokenizer = open_clip.get_tokenizer(BIOCLIP_BACKBONE, cache_dir=str(BIOCLIP_DIR))
    model = model.to(device or pick_device()).eval()
    return model, preprocess, tokenizer


def embed_batch(model, tensors: list, device: str) -> np.ndarray:
    """L2-normalised image embeddings of preprocessed crops, one row each.

    ``tensors`` are ``preprocess(crop)`` outputs; the encoder runs in FP16
    autocast and the result comes back as float32 on the CPU."""
    import torch

    with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
        e = model.encode_image(torch.stack(tensors).to(device))
    return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


def embed_crops(
    boxes: pd.DataFrame,
    frame_paths,
    model,
    preprocess,
    device: str,
    save_thumbnail=None,
    batch_size: int = 64,
    progress=None,
) -> np.ndarray:
    """Cut every box of ``boxes`` from its frame and embed the crops.

    ``boxes`` needs ``x_min, y_min, x_max, y_max`` and a default
    ``RangeIndex`` (row i of the result is row i of ``boxes``). ``frame_paths``
    is either the name of the column holding each row's frame path or a
    ``{filename: path}`` mapping keyed by the ``filename`` column. Each frame
    is opened once. ``save_thumbnail(crop, row)`` is called for every crop
    before it is embedded (the callers save their review thumbnails there);
    ``progress(done, total)`` after every full batch.
    """
    from PIL import Image

    by_column = isinstance(frame_paths, str)
    key = frame_paths if by_column else "filename"
    out = None
    tensors, rows, done = [], [], 0

    def flush():
        nonlocal out, done
        if not tensors:
            return
        e = embed_batch(model, tensors, device)
        if out is None:
            out = np.zeros((len(boxes), e.shape[1]), dtype=np.float32)
        out[rows] = e
        done += len(rows)
        tensors.clear()
        rows.clear()

    for frame_key, g in boxes.groupby(key, sort=False):
        frame = frame_key if by_column else frame_paths[frame_key]
        img = Image.open(frame).convert("RGB")
        for i, r in g.iterrows():
            crop = crop_with_margin(img, r.x_min, r.y_min, r.x_max, r.y_max)
            if save_thumbnail is not None:
                save_thumbnail(crop, r)
            tensors.append(preprocess(crop))
            rows.append(i)
            if len(tensors) >= batch_size:
                flush()
                if progress is not None:
                    progress(done, len(boxes))
    flush()
    return out if out is not None else np.zeros((len(boxes), 0), dtype=np.float32)
