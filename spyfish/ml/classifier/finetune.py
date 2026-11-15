"""Fine-tune the tail of BioCLIP 2 on the crops, held out by drop.

The frozen part of the encoder is run under autograd anyway but has no
trainable parameters, so backward only touches the last ``unfreeze_blocks``
transformer blocks, the final layer norm, the projection and a new linear head.

Split: with a ``split`` column (the ``yolo_dataset`` source) the frozen test
drops are held out; otherwise the drop-grouped fold ``fold`` of ``N_FOLDS``, so
the numbers compare with the probe's out-of-fold ones. A probe is fitted on the
same training crops for a like-for-like row. ``--all-train`` trains on every
non-test crop and saves ``final.pt`` for the classifier that runs on unseen
footage.

Writes to ``<two_stage_dir>/<out>/``: ``best.pt`` or ``final.pt`` (trainable
params + classes), ``classes.json`` (which species became classes and why),
``finetune_report.md`` and ``finetune_predictions.csv``.

Recipe and augmentation come from ``training.classifier.finetune`` in
config.yaml; the CLI flags override per run. Augmentation: rotation through the
circle (a top-down fish faces any way), flips, colour jitter and blur for
turbidity. Runs on MPS; ~6 min per epoch on the Mac.

Crops are loaded and augmented by ``training.classifier.finetune.workers``
DataLoader processes (``--workers`` per run), so the GPU is not waiting on the
training process to open and rotate JPEGs. A run is reproducible for a given
worker count; changing the count changes which random draws each crop gets.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from spyfish.config.ml import Augmentation
from spyfish.config.wrapper import config
from spyfish.ml.classifier import artifacts
from spyfish.ml.classifier.classes import (
    CLASS_TABLE_FILENAME,
    OTHER,
    classes_for,
    fit_probe,
    grouped_folds,
    write_class_table,
)
from spyfish.ml.classifier.crops import (
    NOT_FISH,
    OUT_DIR,
    SOURCE,
    load_bioclip,
    pick_device,
    read_crops_meta,
    read_predictions,
)
from spyfish.ml.classifier.metrics import (
    PRIOR_CSV,
    apply_prior,
    load_site_prior,
    match_yolo,
    per_class_table,
)
from spyfish.ml.classifier.species_classifier import tail_parameters

RECIPE = config.classifier_finetune
SEED = config.classifier_training_seed


def build_train_transform(aug: Augmentation, mean, std):
    """torchvision transform for the training crops, from the config recipe.
    Evaluation crops use BioCLIP's own ``preprocess`` instead."""
    from torchvision import transforms as T

    steps = [T.RandomRotation(aug.rotation_degrees, expand=False)]
    if aug.horizontal_flip:
        steps.append(T.RandomHorizontalFlip())
    steps += [
        T.ColorJitter(*aug.color_jitter),
        T.RandomApply(
            [T.GaussianBlur(aug.blur_kernel, sigma=aug.blur_sigma)],
            p=aug.blur_probability,
        ),
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
    return T.Compose(steps)


class CropDataset:
    """Crop files and their class ids, as a map-style dataset for a DataLoader.

    At module level and holding only paths, an int array and the transform, so
    it pickles into worker processes (macOS starts them with spawn, which
    cannot pickle a class defined inside ``main``). No file stays open between
    items.
    """

    def __init__(self, paths: Sequence[Path], labels: np.ndarray, tf):
        self.paths = [Path(p) for p in paths]
        self.labels = np.asarray(labels)
        if len(self.labels) and not np.issubdtype(self.labels.dtype, np.integer):
            raise TypeError(
                f"class ids must be integers, got {self.labels.dtype}: a crop "
                "whose species is not in the class list?"
            )
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            return self.tf(img.convert("RGB")), int(self.labels[i])


def loader_kwargs(workers: int, device: str, seed: int) -> dict:
    """The DataLoader settings both loaders share, from the worker count.

    ``workers`` 0 loads on the training process and adds nothing, so a
    ``workers: 0`` run draws the same random numbers it always did. Above 0 the
    workers stay alive between epochs, and their random streams are seeded from
    ``seed``: the DataLoader draws one base seed from the generator given here
    and seeds torch, ``random`` and numpy in worker ``k`` with base + k. The
    augmentation (torch's RNG) is therefore the same on every run with the same
    worker count, and different from a ``workers: 0`` run. Pinned memory only
    helps the copy to a CUDA device.
    """
    import torch

    kwargs: dict = {"num_workers": int(workers), "pin_memory": device == "cuda"}
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["generator"] = torch.Generator().manual_seed(int(seed))
    return kwargs


def add_negatives(
    meta: pd.DataFrame,
    X: np.ndarray,
    dirs: Sequence[str],
    max_iou: Optional[float] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Append `not a fish` crops from hard_negatives.py runs as one more class.

    `dirs` are folders under OUT_DIR (`negatives_upload`, `negatives`, ...)
    holding negatives_meta.csv, negatives_embeddings.npy and crops/. A negative
    inherits its drop's split, so test-drop negatives are held out with the rest
    of that drop. `negatives_upload` needs no human pass, every box there was
    deleted by the expert who reviewed it; the `yolo` source needs the cluster
    review first.

    `max_iou` keeps only the negatives whose `best_iou` (overlap with the
    nearest expert box left on the frame) is at or below it: 0 = boxes that
    touch no expert box. A deleted box that overlaps a kept one is usually a
    duplicate or a cut-off box on a real fish, not junk (2026-11-10 review of
    the 415 SLI boxes).
    """
    split_of = (
        meta.drop_duplicates("drop").set_index("drop")["split"]
        if "split" in meta.columns
        else None
    )
    parts, embs = [meta], [X]
    for d in dirs:
        nd = OUT_DIR / d
        neg = pd.read_csv(nd / artifacts.NEGATIVES_META)
        emb = np.load(nd / artifacts.NEGATIVES_EMBEDDINGS)
        if len(emb) != len(neg):
            raise ValueError(
                f"{nd}: {len(neg)} rows in {artifacts.NEGATIVES_META} but "
                f"{len(emb)} embeddings"
            )
        if max_iou is not None:
            keep = (neg["best_iou"] <= max_iou).to_numpy()
            print(
                f"{d}: {int(keep.sum())} of {len(neg)} negatives have IoU <= {max_iou} "
                "with every expert box"
            )
            neg, emb = neg[keep].reset_index(drop=True), emb[keep]
        rows = pd.DataFrame(
            {
                "drop": neg["drop"],
                "filename": neg["filename"],
                "reserve": neg["reserve"],
                "species": NOT_FISH,
                "x_min": neg["x_min"],
                "y_min": neg["y_min"],
                "x_max": neg["x_max"],
                "y_max": neg["y_max"],
                "crop_id": neg["neg_id"],
                "crop_file": [f"{d}/crops/{n}.jpg" for n in neg["neg_id"]],
            }
        )
        if "split" in neg.columns:
            # The folder decided (static_negatives.py --test-reserves): its
            # drops are no frozen test drops, so the drop map would put every
            # crop in train and leave the class unmeasured.
            rows["split"] = neg["split"].to_numpy()
        elif split_of is not None:
            rows["split"] = neg["drop"].map(split_of).fillna("train")
        parts.append(rows)
        embs.append(emb)
        print(f"{d}: {len(neg)} `{NOT_FISH}` crops from {neg['drop'].nunique()} drops")
    return pd.concat(parts, ignore_index=True), np.vstack(embs)


def per_class(y_true, y_pred, classes) -> pd.DataFrame:
    """``per_class_table`` for the fine-tune's integer class ids."""
    return per_class_table(y_true, y_pred, classes, labels=range(len(classes)))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the tail of BioCLIP 2 on the crops."
    )
    parser.add_argument("--epochs", type=int, default=RECIPE.epochs)
    parser.add_argument("--batch", type=int, default=RECIPE.batch)
    parser.add_argument(
        "--unfreeze",
        type=int,
        default=RECIPE.unfreeze_blocks,
        help="trainable transformer blocks from the end (24 = all)",
    )
    parser.add_argument("--lr-encoder", type=float, default=RECIPE.lr_encoder)
    parser.add_argument("--lr-head", type=float, default=RECIPE.lr_head)
    parser.add_argument(
        "--sampler-power",
        type=float,
        default=1.0,
        help="class weight = count^-power: 1 = fully balanced (every class equally often), "
        "0.5 = square-root balancing, 0 = natural mix",
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="no holdout: train on every crop (test-tagged crops excluded) for the "
        "classifier that runs on unseen footage; saves final.pt, reports nothing. "
        "Pick epochs/unfreeze from a held-out run first.",
    )
    parser.add_argument(
        "--fold", type=int, default=RECIPE.fold, help="GroupKFold fold to hold out"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=RECIPE.workers,
        help="DataLoader worker processes loading and augmenting crops (0 = none)",
    )
    parser.add_argument(
        "--out",
        default="finetune",
        help=f"output folder under {OUT_DIR.name} (default finetune)",
    )
    parser.add_argument(
        "--negatives",
        nargs="*",
        default=[],
        metavar="DIR",
        help=f"hard_negatives.py output folders under {OUT_DIR.name} (negatives_upload, negatives) "
        f"whose crops train and score as `{NOT_FISH}`",
    )
    parser.add_argument(
        "--negatives-max-iou",
        type=float,
        default=None,
        metavar="IOU",
        help="keep only negatives whose overlap with the nearest expert box is at or "
        "below this (0 = touches none); default: every negative in the folders",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import open_clip
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from spyfish.config.species import species_registry

    args = parse_args(argv)
    device = pick_device()
    epochs, batch, unfreeze = args.epochs, args.batch, args.unfreeze
    lr_encoder, lr_head = args.lr_encoder, args.lr_head
    sampler_power = args.sampler_power
    workers = args.workers
    if workers < 0:
        raise ValueError(f"--workers must be >= 0, got {workers}")
    ft_dir = OUT_DIR / args.out
    print(
        f"epochs {epochs}, batch {batch}, unfreeze last {unfreeze} blocks, "
        f"lr encoder {lr_encoder}, head {lr_head}, {workers} loader workers"
    )
    torch.manual_seed(SEED)
    ft_dir.mkdir(parents=True, exist_ok=True)
    meta = read_crops_meta()
    X = np.load(OUT_DIR / artifacts.EMBEDDINGS)
    common = species_registry().common_name_map()

    target, classes, stats = classes_for(meta)
    if args.negatives:
        meta, X = add_negatives(meta, X, args.negatives, args.negatives_max_iou)
        classes = classes + [NOT_FISH]
    y_name = meta["species"].where(meta["species"].isin(target + [NOT_FISH]), OTHER)
    y = y_name.map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = meta["drop"].to_numpy()
    is_test = (
        (meta["split"] == "test").to_numpy()
        if "split" in meta.columns
        else np.zeros(len(meta), dtype=bool)
    )
    if args.all_train:
        # Final classifier for footage nobody has boxed (UoA 20 Jan): every
        # non-test crop trains, nothing is scored. Held-out runs choose the recipe.
        tr = np.flatnonzero(~is_test)
        te = np.array([], dtype=int)
        holdout = "none (all non-test crops train)"
    elif is_test.any():
        # the frozen test drops: never trained on here, nor (modulo contamination) by YOLO
        te = np.flatnonzero(is_test)
        tr = np.flatnonzero(~is_test)
        holdout = "frozen test split"
    else:
        # UoA has no boxes on its held-out day, so the only honest holdout is a
        # deployment-grouped fold of the train days, like the probe's.
        # The snapper stills (not UoA footage) train in every fold, never test.
        tr, te = grouped_folds(groups)[args.fold]
        holdout = f"GroupKFold fold {args.fold}"
    print(
        f"{holdout}: train {len(tr)} crops / {len(set(groups[tr]))} drops, "
        f"test {len(te)} crops / {sorted(set(groups[te]))}"
    )
    recipe_used = {
        **asdict(RECIPE),
        "epochs": epochs,
        "batch": batch,
        "unfreeze_blocks": unfreeze,
        "lr_encoder": lr_encoder,
        "lr_head": lr_head,
        "sampler_power": sampler_power,
        "workers": workers,
        "holdout": holdout,
        "negatives": list(args.negatives),
        "negatives_max_iou": args.negatives_max_iou,
    }
    write_class_table(
        ft_dir / CLASS_TABLE_FILENAME,
        target,
        classes,
        stats,
        extra={"recipe": recipe_used, "n_train_crops": int(len(tr))},
    )

    def crops(idx: np.ndarray, tf) -> CropDataset:
        return CropDataset(
            [OUT_DIR / f for f in meta["crop_file"].iloc[idx]], y[idx], tf
        )

    def evaluate(model, head, loader):
        model.eval()
        head.eval()
        preds, probs = [], []
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
            for x, _ in loader:
                logits = head(model.encode_image(x.to(device)).float())
                p = logits.softmax(-1)
                probs.append(p.cpu())
                preds.append(p.argmax(-1).cpu())
        return torch.cat(preds).numpy(), torch.cat(probs).numpy()

    model, preprocess, _ = load_bioclip(device)
    for p in model.parameters():
        p.requires_grad = False
    visual = model.visual
    trainable = tail_parameters(visual, unfreeze)
    for p in trainable:
        p.requires_grad = True
    head = torch.nn.Linear(X.shape[1], len(classes)).to(device)
    print(
        f"trainable encoder params: {sum(p.numel() for p in trainable) / 1e6:.1f}M, "
        f"head {head.weight.numel() / 1e3:.0f}k"
    )

    train_tf = build_train_transform(
        RECIPE.augmentation,
        open_clip.constants.OPENAI_DATASET_MEAN,
        open_clip.constants.OPENAI_DATASET_STD,
    )
    counts = np.bincount(y[tr], minlength=len(classes))
    weights = counts[y[tr]].astype(np.float64) ** -sampler_power
    sampler = WeightedRandomSampler(weights, num_samples=len(tr), replacement=True)
    train_loader = DataLoader(
        crops(tr, train_tf),
        batch_size=batch,
        sampler=sampler,
        **loader_kwargs(workers, device, SEED),
    )
    test_loader = DataLoader(
        crops(te, preprocess),
        batch_size=64,
        shuffle=False,
        **loader_kwargs(workers, device, SEED),
    )

    opt = torch.optim.AdamW(
        [
            {"params": trainable, "lr": lr_encoder},
            {"params": head.parameters(), "lr": lr_head},
        ],
        weight_decay=RECIPE.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * len(train_loader)
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=RECIPE.label_smoothing)

    def save(path: Path) -> None:
        torch.save(
            {
                "head": head.state_dict(),
                "encoder_tail": [p.detach().cpu() for p in trainable],
                "classes": classes,
                "unfreeze": unfreeze,
                "source": SOURCE,
                "recipe": recipe_used,
            },
            path,
        )

    def train_epoch(epoch: int, verbose: bool) -> float:
        model.train()
        head.train()
        t0, running = time.time(), 0.0
        for i, (x, yb) in enumerate(train_loader):
            x, yb = x.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.float16):
                logits = head(model.encode_image(x).float())
            loss = loss_fn(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            if verbose and i % 40 == 0:
                print(
                    f"  epoch {epoch} batch {i}/{len(train_loader)} loss {loss.item():.3f} {time.time() - t0:.0f}s",
                    flush=True,
                )
        return running / len(train_loader)

    if args.all_train:
        for epoch in range(epochs):
            t0 = time.time()
            loss = train_epoch(epoch, verbose=False)
            print(
                f"epoch {epoch}: loss {loss:.3f}, {time.time() - t0:.0f}s", flush=True
            )
        save(ft_dir / artifacts.FINAL_CHECKPOINT)
        (ft_dir / "final.txt").write_text(
            f"{holdout}; {len(tr)} crops / {len(set(groups[tr]))} drops; classes {classes}; "
            f"{epochs} epochs, unfreeze {unfreeze}, lr encoder {lr_encoder}, head {lr_head}, "
            f"sampler power {sampler_power}\n"
        )
        print(
            f"wrote {ft_dir / artifacts.FINAL_CHECKPOINT} (no held-out score: all-train mode)"
        )
        return

    # like-for-like probe on this fold
    probe_pred = fit_probe(X[tr], y[tr]).predict(X[te])
    probe_tab = per_class(y[te], probe_pred, classes)
    print(
        f"probe on {holdout}: acc {(probe_pred == y[te]).mean():.3f}, macro F1 {probe_tab['f1'].mean():.3f}"
    )

    best, best_tab, best_pred, best_probs, history = -1.0, None, None, None, []
    for epoch in range(epochs):
        t0 = time.time()
        loss = train_epoch(epoch, verbose=True)
        pred, probs = evaluate(model, head, test_loader)
        tab = per_class(y[te], pred, classes)
        acc, macro = (pred == y[te]).mean(), tab["f1"].mean()
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "test_acc": acc,
                "test_macro_f1": macro,
                "seconds": time.time() - t0,
            }
        )
        print(
            f"epoch {epoch}: loss {loss:.3f}, test acc {acc:.3f}, macro F1 {macro:.3f}, {time.time() - t0:.0f}s",
            flush=True,
        )
        if macro > best:
            best, best_tab, best_pred, best_probs = macro, tab, pred, probs
            save(ft_dir / artifacts.BEST_CHECKPOINT)

    def fmt(t):
        return (
            t.assign(common=[common.get(c, "") for c in t.index]).round(3).to_markdown()
        )

    # YOLO and the prior on the same held-out crops, so every column is comparable
    held = meta.iloc[te].reset_index(drop=True)
    is_neg = (held["species"] == NOT_FISH).to_numpy()
    neg_rows: list[str] = []
    if is_neg.any():
        neg_i = classes.index(NOT_FISH)
        neg_rows = [
            f"## `{NOT_FISH}` on the held-out drops",
            "",
            f"{int(is_neg.sum())} held-out negatives. Expert fish called `{NOT_FISH}`: "
            f"{(best_pred[~is_neg] == neg_i).mean():.1%}. Negatives called a fish class: "
            f"{(best_pred[is_neg] != neg_i).mean():.1%}. The YOLO rows below use the expert fish only.",
            "",
        ]
    yolo_csv = OUT_DIR / artifacts.YOLO_PREDICTIONS
    yolo_rows: list[str] = []
    if yolo_csv.exists():
        m = match_yolo(held, read_predictions(yolo_csv))
        has = m["yolo_label"].notna().to_numpy() & ~is_neg
        ylab = m["yolo_label"].where(m["yolo_label"].isin(target), OTHER)
        ylab = (
            ylab.where(m["yolo_label"] != "fish", OTHER)
            .map({c: i for i, c in enumerate(classes)})
            .to_numpy()
        )
        yolo_tab = per_class(y[te][has], ylab[has].astype(int), classes)
        ft_same = per_class(y[te][has], best_pred[has], classes)
        yolo_rows = [
            f"## YOLO on the {int(has.sum())} of these crops it detected: acc {(ylab[has] == y[te][has]).mean():.3f}, macro F1 {yolo_tab['f1'].mean():.3f}",
            "",
            fmt(yolo_tab),
            "",
            f"## Fine-tuned on the same detected crops: acc {(best_pred[has] == y[te][has]).mean():.3f}, macro F1 {ft_same['f1'].mean():.3f}",
            "",
            fmt(ft_same),
            "",
        ]
    prior_rows: list[str] = []
    if PRIOR_CSV.exists() and SOURCE != "uoa":  # UoA has no reserve, so no prior
        # strength sweep: a fine-tuned softmax is already confident, so a full-strength
        # prior (alpha 1) can suppress rare species; report what each alpha does
        prior_df = load_site_prior()
        sweep = []
        best_alpha, best_rows = None, None
        for alpha in (0.0, 0.25, 0.5, 1.0):
            pp = apply_prior(
                best_probs, classes, held["reserve"], prior_df, alpha
            ).argmax(axis=1)
            pt = per_class(y[te], pp, classes)
            sweep.append(
                {
                    "alpha": alpha,
                    "acc": round((pp == y[te]).mean(), 3),
                    "macro_f1": round(pt["f1"].mean(), 3),
                }
            )
            if best_alpha is None or pt["f1"].mean() > best_rows[0]:
                best_alpha, best_rows = alpha, (pt["f1"].mean(), pp, pt)
        prior_rows = [
            "## Fine-tuned + site prior, by prior strength alpha (0 = no prior)",
            "",
            pd.DataFrame(sweep).to_markdown(index=False),
            "",
            f"### Best alpha {best_alpha}: acc {(best_rows[1] == y[te]).mean():.3f}, macro F1 {best_rows[0]:.3f}",
            "",
            fmt(best_rows[2]),
            "",
        ]
    aug = RECIPE.augmentation
    lines = [
        f"# BioCLIP 2 fine-tune, {holdout} held out",
        "",
        f"Train {len(tr)} crops / {len(set(groups[tr]))} drops; test {len(te)} crops / {sorted(set(groups[te]))}. "
        f"Last {unfreeze} blocks + head trainable, {epochs} epochs, batch {batch}, sampler power {sampler_power} (1 = class-balanced), "
        f"augmentation: rotation +/-{aug.rotation_degrees:g}, {'flip, ' if aug.horizontal_flip else ''}"
        f"colour jitter {aug.color_jitter}, blur p={aug.blur_probability}."
        + (
            f" `{NOT_FISH}` class from {', '.join(args.negatives)}: "
            f"{int((meta['species'] == NOT_FISH).sum())} crops."
            if args.negatives
            else ""
        ),
        "",
        "## Training history",
        "",
        pd.DataFrame(history).round(3).to_markdown(index=False),
        "",
        f"## Linear probe on the same fold: acc {(probe_pred == y[te]).mean():.3f}, macro F1 {probe_tab['f1'].mean():.3f}",
        "",
        fmt(probe_tab),
        "",
        f"## Fine-tuned (best epoch): acc {(best_pred == y[te]).mean():.3f}, macro F1 {best:.3f}",
        "",
        fmt(best_tab),
        "",
    ]
    lines += neg_rows + prior_rows + yolo_rows
    (ft_dir / artifacts.FINETUNE_REPORT).write_text("\n".join(lines))
    out = meta.iloc[te][["crop_id", "drop", "reserve", "species"]].copy()
    out["expert_class"] = [classes[i] for i in y[te]]
    out["probe"] = [classes[i] for i in probe_pred]
    out["finetuned"] = [classes[i] for i in best_pred]
    out["finetuned_p"] = best_probs.max(axis=1).round(3)
    out.to_csv(ft_dir / artifacts.FINETUNE_PREDICTIONS, index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
