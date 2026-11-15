"""Second stage of ML inference: BioCLIP 2 names each YOLO box from its crop.

YOLO keeps finding the fish; this names them. On the frozen test drops
(2026-11-04, the same 1,235 detected boxes) the fine-tuned BioCLIP 2 named
0.951 / macro F1 0.768 against the species head's 0.930 / 0.599, better on
every species except silver sweep. The crop is cut from the full-resolution
frame, so a fish YOLO saw at 93 px arrives here at its real size.

What the raw CSV gets (``run_inference``): ``class`` becomes the classifier's
name, ``confidence`` stays the DETECTION score (every downstream threshold
gates on it), and four extra columns keep both opinions:
``yolo_class, yolo_confidence, clf_class, clf_confidence``. A box too small to
classify keeps the YOLO name and blank classifier columns.

Checkpoint format is ``scripts/two_stage/finetune_bioclip.py``'s: the
trainable tail of the image encoder (``encoder_tail``, last ``unfreeze``
transformer blocks + ``ln_post`` + ``proj``), the linear ``head`` and the
``classes`` list. The frozen BioCLIP 2 backbone is read from
``paths.sub_dirs.bioclip`` (an open_clip hub cache; ``HF_HUB_OFFLINE=1`` on
compute nodes, which have no internet).

Class names are resolved once at load: ``ml_inference.classifier.label_map``
first (``other fish`` → ``fish``), then the species registry's synonyms, so a
checkpoint still saying ``Pagrus auratus`` writes ``Chrysophrys auratus``.
"""

from __future__ import annotations

import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image

from spyfish.config.wrapper import config

# (frame index within the batch, x_center, y_center, w, h) in frame pixels, as
# YOLO's ``box.xywh`` reports them.
Box = tuple[int, float, float, float, float]


@dataclass(frozen=True)
class Naming:
    label: str  # the classifier's answer, mapped and canonicalised
    raw_label: str  # the classifier's own class name, before mapping
    prob: float  # top softmax probability
    confident: bool = (
        True  # prob >= p_min; when False `final_label` decides what is written
    )


def crop_box(
    frame_bgr: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    margin: float,
    min_px: int = 1,
) -> Optional[Image.Image]:
    """The box plus ``margin`` of its size on each side, clipped to the frame, as RGB.

    Same geometry as the fine-tune's training crops (``common.crop_with_margin``),
    so the classifier sees at inference what it saw in training. ``None`` when
    what is left after clipping is under ``min_px`` on a side: a box hanging
    off the frame edge must not reach the model as a zero-size image (the
    resize divides by it) and take the whole drop down.
    """
    height, width = frame_bgr.shape[:2]
    x1 = max(0, int(x - w / 2 - margin * w))
    y1 = max(0, int(y - h / 2 - margin * h))
    x2 = min(width, int(x + w / 2 + margin * w))
    y2 = min(height, int(y + h / 2 + margin * h))
    if x2 - x1 < min_px or y2 - y1 < min_px:
        return None
    return Image.fromarray(np.ascontiguousarray(frame_bgr[y1:y2, x1:x2, ::-1]))


def pick_device() -> str:
    """cuda on NeSI, mps on the Mac, cpu otherwise."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpeciesClassifier:
    """Crops in, species names out. The model is behind ``predict_probs`` so the
    cropping, thresholds and name mapping are testable without open_clip."""

    def __init__(
        self,
        predict_probs: Callable[[list[Image.Image]], np.ndarray],
        classes: Sequence[str],
        *,
        crop_margin: float,
        min_box_px: int,
        batch_size: int,
        p_min: float,
        fallback_label: str,
        label_map: dict,
        canonical: Optional[Callable[[str], str]] = None,
        fallback_to_yolo: bool = False,
        fallback_yolo_min_confidence: float = 1.0,
        reject_label: Optional[str] = None,
    ):
        self.predict_probs = predict_probs
        self.fallback_to_yolo = bool(fallback_to_yolo)
        self.fallback_yolo_min_confidence = float(fallback_yolo_min_confidence)
        self.classes = list(classes)
        self.crop_margin = float(crop_margin)
        self.min_box_px = int(min_box_px)
        self.batch_size = int(batch_size)
        self.p_min = float(p_min)
        self.fallback_label = fallback_label
        canonical = canonical or (lambda name: name)
        # `not a fish` is not a species: written as is, never looked up.
        self.labels = [
            name if name == reject_label else canonical(name)
            for name in (label_map.get(c, c) for c in self.classes)
        ]

    @classmethod
    def from_config(cls, weights: Optional[Path] = None) -> "SpeciesClassifier":
        """The configured fine-tuned BioCLIP 2, names resolved through the registry.

        ``weights`` overrides ``config.classifier_model_path`` (a candidate
        checkpoint under test); every other knob comes from config.
        """
        from spyfish.config.species import species_registry

        predict_probs, classes = load_finetuned_bioclip(
            Path(weights) if weights else config.classifier_model_path,
            config.classifier_backbone,
            config.bioclip_cache_dir,
            preprocess_workers=config.classifier_preprocess_workers,
        )
        return cls(
            predict_probs,
            classes,
            crop_margin=config.classifier_crop_margin,
            min_box_px=config.classifier_min_box_px,
            batch_size=config.classifier_batch_size,
            p_min=config.classifier_p_min,
            fallback_label=config.classifier_fallback_label,
            label_map=config.classifier_label_map,
            canonical=species_registry().canonical_name,
            fallback_to_yolo=config.classifier_fallback_to_yolo,
            fallback_yolo_min_confidence=config.classifier_fallback_yolo_min_confidence,
            reject_label=config.classifier_reject_label,
        )

    def final_label(self, naming: Naming, yolo_class: str, yolo_conf: float) -> str:
        """What the raw CSV `class` column gets for one box.

        The classifier's name when it is confident. Below ``p_min``: YOLO's own
        name if ``fallback_to_yolo`` is on and YOLO's detection confidence
        clears ``fallback_yolo_min_confidence`` (the species head is right on
        93% of the boxes it finds, and a box it is sure of is better evidence
        than a generic bucket), otherwise ``fallback_label``. The classifier
        columns keep its answer either way, so these cases can be pulled out
        and looked at (``scripts/compare_two_stage_names.py``).
        """
        if naming.confident:
            return naming.label
        if self.fallback_to_yolo and yolo_conf >= self.fallback_yolo_min_confidence:
            return yolo_class
        return self.fallback_label

    def name_crops(self, crops: Sequence[Image.Image]) -> list[Naming]:
        """One ``Naming`` per crop, in order; ``confident`` is False below ``p_min``."""
        out: list[Naming] = []
        for start in range(0, len(crops), self.batch_size):
            probs = np.asarray(
                self.predict_probs(list(crops[start : start + self.batch_size]))
            )
            for row in probs:
                i = int(row.argmax())
                p = float(row[i])
                out.append(
                    Naming(
                        label=self.labels[i],
                        raw_label=self.classes[i],
                        prob=p,
                        confident=p >= self.p_min,
                    )
                )
        return out

    def name_boxes(
        self, frames_bgr: Sequence[np.ndarray], boxes: Sequence[Box]
    ) -> list[Optional[Naming]]:
        """Name every box, in order; ``None`` for a box under ``min_box_px`` on a
        side, before or after clipping to the frame."""
        crops, kept = [], []
        for j, (frame_i, x, y, w, h) in enumerate(boxes):
            if min(w, h) < self.min_box_px:
                continue
            crop = crop_box(
                frames_bgr[frame_i], x, y, w, h, self.crop_margin, self.min_box_px
            )
            if crop is None:
                continue
            crops.append(crop)
            kept.append(j)
        named = self.name_crops(crops) if crops else []
        out: list[Optional[Naming]] = [None] * len(boxes)
        for j, naming in zip(kept, named):
            out[j] = naming
        return out


def tail_parameters(visual, unfreeze: int) -> list:
    """The image-encoder parameters the fine-tune trains, in checkpoint order:
    the last ``unfreeze`` transformer blocks, then ``ln_post``, then ``proj``.

    ``encoder_tail`` in a checkpoint is exactly this list, so training
    (``finetune.py``) and loading (``load_finetuned_bioclip``) must take it
    from here: a different order would copy tensors onto the wrong parameters.
    """
    params = [
        p
        for block in visual.transformer.resblocks[-unfreeze:]
        for p in block.parameters()
    ]
    params += list(visual.ln_post.parameters())
    if getattr(visual, "proj", None) is not None:
        params.append(visual.proj)
    return params


def _infer_unfreeze(visual, saved: list) -> int:
    """How many blocks an ``encoder_tail`` covers, for checkpoints that predate
    the ``unfreeze`` key (the wip fine-tune, 2026-11-04 and before).

    The tail is the last N blocks' parameters, then ``ln_post``, then ``proj``,
    and every block has the same parameter count, so N falls out of the length.
    """
    per_block = len(list(visual.transformer.resblocks[-1].parameters()))
    after = len(list(visual.ln_post.parameters())) + (
        1 if getattr(visual, "proj", None) is not None else 0
    )
    n, rest = divmod(len(saved) - after, per_block)
    if rest or not 0 < n <= len(visual.transformer.resblocks):
        raise ValueError(
            f"{len(saved)} saved tensors is not N blocks of {per_block} + {after}; "
            "not a BioCLIP fine-tune tail for this backbone."
        )
    logging.info(f"Checkpoint has no `unfreeze`; its tail covers the last {n} blocks.")
    return n


def preprocess_crops(
    preprocess: Callable, crops: Sequence[Image.Image], pool=None
) -> list:
    """``preprocess(crop)`` for every crop, in order.

    With a thread ``pool`` the crops are resized side by side: the bicubic
    resize is most of the cost and PIL releases the GIL inside it, so the GPU
    waits for a batch a fraction as long. Each crop still goes through the same
    ``preprocess`` on its own, and ``map`` keeps the order, so the tensors are
    the serial ones exactly.
    """
    if pool is None or len(crops) < 2:
        return [preprocess(c) for c in crops]
    return list(pool.map(preprocess, crops))


def load_finetuned_bioclip(
    weights: Path,
    backbone: str,
    cache_dir: Path,
    device: Optional[str] = None,
    preprocess_workers: int = 0,
) -> tuple[Callable[[list[Image.Image]], np.ndarray], list[str]]:
    """``(predict_probs, classes)`` from a ``finetune_bioclip.py`` checkpoint.

    Rebuilds the frozen backbone from the hub cache, copies the saved tail
    over the same parameters in the same order the fine-tune trained them,
    and puts the saved head on top. ``preprocess_workers`` threads resize the
    crops of a batch (``preprocess_crops``); 0 or 1 does it on the calling
    thread.
    """
    import open_clip
    import torch

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    missing = [k for k in ("classes", "head", "encoder_tail") if k not in ckpt]
    if missing:
        raise ValueError(
            f"{weights}: not a finetune_bioclip.py checkpoint, missing {missing}."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, cache_dir=str(cache_dir)
    )
    visual = model.visual
    saved = ckpt["encoder_tail"]
    unfreeze = (
        int(ckpt["unfreeze"]) if "unfreeze" in ckpt else _infer_unfreeze(visual, saved)
    )
    trainable = tail_parameters(visual, unfreeze)
    mismatched = len(saved) != len(trainable) or any(
        p.shape != s.shape for p, s in zip(trainable, saved)
    )
    if mismatched:
        raise RuntimeError(
            f"{weights}: {len(saved)} saved tensors do not match the backbone's last "
            f"{unfreeze} blocks ({len(trainable)} parameters); wrong backbone?"
        )
    with torch.no_grad():
        for p, s in zip(trainable, saved):
            p.copy_(s.to(p.dtype))
    classes = list(ckpt["classes"])
    n_out, n_in = ckpt["head"]["weight"].shape
    if n_out != len(classes):
        raise RuntimeError(
            f"{weights}: head has {n_out} outputs for {len(classes)} classes"
        )
    head = torch.nn.Linear(n_in, n_out)
    head.load_state_dict(ckpt["head"])

    device = device or pick_device()
    model = model.to(device).eval()
    head = head.to(device).eval()
    autocast = (
        contextlib.nullcontext
        if device == "cpu"
        else lambda: torch.autocast(device_type=device, dtype=torch.float16)
    )

    # One pool for the life of the classifier (it is loaded once per process).
    pool = (
        ThreadPoolExecutor(
            max_workers=preprocess_workers, thread_name_prefix="clf-preprocess"
        )
        if preprocess_workers > 1
        else None
    )

    def predict_probs(crops: list[Image.Image]) -> np.ndarray:
        x = torch.stack(preprocess_crops(preprocess, crops, pool)).to(device)
        with torch.no_grad(), autocast():
            embeddings = model.encode_image(x).float()
        with torch.no_grad():
            return head(embeddings).softmax(-1).cpu().numpy()

    logging.info(
        f"Species classifier {weights.name}: {len(classes)} classes, backbone "
        f"{backbone}, last {unfreeze} blocks fine-tuned, device {device}, "
        f"{preprocess_workers} preprocess thread(s)."
    )
    return predict_probs, classes
