"""Phase 1 lite: does a BioCLIP 2 probe identify species better than the current YOLO?

Reads the outputs of build_embeddings.py and run_yolo_on_frames.py plus the site
prior CSV, and writes to the two-stage output folder (paths.sub_dirs.two_stage):
    report.md              the tables below, in one place
    oof_predictions.csv    per crop: expert species, zero-shot, probe, probe+prior, YOLO label on the matched box
    class_table.csv        crops / drops per class and which tier it landed in

Evaluations (plan §4.3–4.4):
  * zero-shot BioCLIP 2 (species name as the text prompt) – the floor
  * linear probe, drop-grouped CV, out-of-fold probabilities
  * the same probe with the site prior applied per fold
  * the current YOLO model's label on expert boxes it detected (IoU >= 0.5),
    on the same crops, and its recall per species (did it find the fish at all)

Classes: species with at least training.classifier.min_drops drops and
min_crops crops (``spyfish.ml.classifier.classes``); the rest train and score as
`other fish`. The go/no-go measurement needs the full dataset on Mahuika.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spyfish.config.species import species_registry  # noqa: E402
from spyfish.config.wrapper import config  # noqa: E402
from spyfish.ml.classifier import artifacts  # noqa: E402
from spyfish.ml.classifier.classes import (  # noqa: E402
    OTHER,
    classes_for,
    fit_probe,
    grouped_folds,
    probe_probabilities,
    scored_rows,
)
from spyfish.ml.classifier.crops import (  # noqa: E402
    OUT_DIR,
    SOURCE,
    read_crops_meta,
    read_predictions,
)
from spyfish.ml.classifier.metrics import (  # noqa: E402
    PRIOR_ALPHA,
    apply_prior,
    load_site_prior,
    match_yolo,
    per_class_table,
)
from spyfish.ml.training.test_eval import find_run_snapshot, seen_images  # noqa: E402

MODEL = config.pipeline_model_path  # the same file --ml runs
MIN_DROPS, MIN_CROPS = (
    config.classifier_training_min_drops,
    config.classifier_training_min_crops,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioCLIP probe vs YOLO on expert crops"
    )
    parser.add_argument(
        "--yolo-csv",
        default=artifacts.YOLO_PREDICTIONS,
        help="predictions file under OUT_DIR for the YOLO comparison; a missing file "
        "skips the YOLO columns (UoA: the 5-class baseline lives on NeSI)",
    )
    parser.add_argument(
        "--yolo-name", default=MODEL.stem, help="how the report names that model"
    )
    args = parser.parse_args()
    meta = read_crops_meta()
    X = np.load(OUT_DIR / artifacts.EMBEDDINGS)
    text = np.load(OUT_DIR / artifacts.TEXT_EMBEDDINGS)
    species = (OUT_DIR / artifacts.SPECIES_LIST).read_text().splitlines()
    yolo_path = OUT_DIR / args.yolo_csv
    if yolo_path.exists():
        yolo = read_predictions(yolo_path)
    else:
        logging.warning(f"{yolo_path} not found: YOLO comparison columns skipped")
        yolo = pd.DataFrame(
            columns=[
                "filename",
                "yolo_class",
                "conf",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
            ]
        )
    # No site prior for UoA: there is no reserve, and a prior built from the
    # paper's species list would be tuned on the held-out day's own truth.
    use_prior = SOURCE != "uoa"
    prior = load_site_prior() if use_prior else None
    common = species_registry().common_name_map()

    # ── classes ──
    target, classes, stats = classes_for(meta)
    stats["common_name"] = [common.get(s, "") for s in stats.index]
    stats.to_csv(OUT_DIR / "class_table.csv")
    y = meta["species"].where(meta["species"].isin(target), OTHER).to_numpy()
    groups = meta["drop"].to_numpy()

    # ── zero-shot ──
    sims = X @ text.T
    zs = np.array(species)[sims.argmax(axis=1)]
    zs = np.where(np.isin(zs, target), zs, OTHER)

    # ── probe, drop-grouped CV ──
    oof = np.zeros((len(meta), len(classes)))
    folds = grouped_folds(groups)
    for k, (tr, te) in enumerate(folds):
        oof[te] = probe_probabilities(fit_probe(X[tr], y[tr]), X[te], classes)
        print(
            f"fold {k}: train drops {len(set(groups[tr]))}, test drops {sorted(set(groups[te]))}"
        )
    probe = np.array(classes)[oof.argmax(axis=1)]
    with_prior = (
        apply_prior(oof, classes, meta["reserve"], prior, PRIOR_ALPHA)
        if use_prior
        else oof
    )
    probe_prior = np.array(classes)[with_prior.argmax(axis=1)]
    # Rows that only ever trained (the UoA snapper stills) have no out-of-fold
    # prediction: everything below scores the held-out rows only.
    keep = scored_rows(groups)
    n_train_only = int((~keep).sum())
    if n_train_only:
        meta = meta[keep].reset_index(drop=True)
        y, zs, probe, probe_prior = y[keep], zs[keep], probe[keep], probe_prior[keep]
        oof = oof[keep]
        groups = groups[keep]

    # ── YOLO on the same boxes ──
    matched = match_yolo(meta, yolo)
    meta = meta.join(matched)
    yl = meta["yolo_label"].where(meta["yolo_label"].isin(target), OTHER)
    yl = yl.where(meta["yolo_label"] != "fish", OTHER)  # unidentified fish = other
    has_yolo = meta["yolo_label"].notna().to_numpy()

    # ── tables ──
    tabs = {
        "zero-shot BioCLIP 2": per_class_table(y, zs, classes),
        "linear probe (OOF)": per_class_table(y, probe, classes),
    }
    if use_prior:
        tabs["probe + site prior"] = per_class_table(y, probe_prior, classes)
    if has_yolo.any():  # no YOLO predictions file, or none matched: nothing to score
        tabs["YOLO on detected boxes"] = per_class_table(
            y[has_yolo], yl[has_yolo].to_numpy(), classes
        )
        tabs["probe on the same detected boxes"] = per_class_table(
            y[has_yolo], probe[has_yolo], classes
        )
    recall_det = (
        meta.assign(found=has_yolo)
        .groupby("species")["found"]
        .mean()
        .rename("yolo_detect_recall")
    )
    if "split" in meta.columns and (meta["split"] == "test").any():
        test_rec = (
            meta[meta["split"] == "test"]
            .assign(found=has_yolo[(meta["split"] == "test").to_numpy()])
            .groupby("species")["found"]
            .agg(expert_boxes="size", found="sum", yolo_detect_recall="mean")
            .sort_values("expert_boxes", ascending=False)
        )

    yolo_name = args.yolo_name
    unit = "deployments" if SOURCE == "uoa" else "drops"
    class_rule = (
        f"Classes are the paper's four species, fixed; the rest are `{OTHER}`."
        if SOURCE == "uoa"
        else f"Classes need >= {MIN_DROPS} drops and >= {MIN_CROPS} crops; the rest are `{OTHER}`."
    )
    lines = [
        f"# Two-stage classifier demo: BioCLIP 2 probe vs current YOLO ({yolo_name})",
        "",
    ]
    lines += [
        f"Crops {len(meta)}, frames {meta['filename'].nunique()}, {unit} {meta['drop'].nunique()}, "
        f"reserves {sorted(meta['reserve'].unique())}. {unit.capitalize()}-grouped {len(folds)}-fold CV. "
        + class_rule
        + (
            f" Plus {n_train_only} crops that train in every fold and are never scored "
            f"(the snapper stills, not UoA footage)."
            if n_train_only
            else ""
        ),
        "",
    ]
    lines += ["## Classes", "", stats.to_markdown(), ""]
    for name, t in tabs.items():
        acc = (t["recall"] * t["n"]).sum() / t["n"].sum()
        macro = t["f1"].mean()
        lines += [
            f"## {name}",
            "",
            f"accuracy {acc:.3f}, macro F1 {macro:.3f}",
            "",
            t.assign(common=[common.get(c, "") for c in t.index])
            .round(3)
            .to_markdown(),
            "",
        ]
    if has_yolo.any():
        lines += [
            "## Detector recall of the current YOLO (expert box matched at IoU >= 0.5, any label)",
            "",
            recall_det.round(3)
            .to_frame()
            .join(stats[["crops", "drops"]])
            .to_markdown(),
            "",
        ]
    if "split" in meta.columns and (meta["split"] == "test").any():
        lines += [
            "## Detector recall on the frozen test drops",
            "",
            "Expert boxes per species on the test drops, how many the current YOLO found at IoU >= 0.5 with any "
            "label, and the share. This is the number the three-class detector has to beat.",
            "",
            test_rec.round(3)
            .assign(common=[common.get(c, "") for c in test_rec.index])
            .to_markdown(),
            "",
        ]
    # per drop: YOLO vs probe on the same detected boxes. With the assembled YOLO
    # dataset as source, `split` says which drops are the frozen test set: the
    # only drops the current YOLO never trained on, so the only fair contest.
    split = (
        meta["split"]
        if "split" in meta.columns
        else pd.Series("unknown", index=meta.index)
    )
    frame = pd.DataFrame(
        {
            "drop": meta["drop"],
            "split": split,
            "has": has_yolo,
            "yolo_ok": (yl.to_numpy() == y),
            "probe_ok": (probe == y),
            "prior_ok": (probe_prior == y),
        }
    )
    per_drop = (
        frame.query("has")
        .groupby(["split", "drop"])
        .agg(
            boxes=("has", "size"),
            yolo_acc=("yolo_ok", "mean"),
            probe_acc=("probe_ok", "mean"),
            probe_prior_acc=("prior_ok", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values(["split", "drop"])
    )
    by_split = (
        frame.query("has")
        .groupby("split")
        .agg(
            drops=("drop", "nunique"),
            boxes=("has", "size"),
            yolo=("yolo_ok", "mean"),
            probe=("probe_ok", "mean"),
            probe_prior=("prior_ok", "mean"),
        )
        .round(3)
    )
    is_test = (split == "test").to_numpy() & has_yolo
    # Contamination, as test_eval.py measures it: the frozen list postdates the
    # production model (list 2026-10-15, model 2026-08-25), so a test drop may
    # have been in that run's train or val. The run's dataset snapshot says.
    snapshot = find_run_snapshot(yolo_name) if SOURCE != "uoa" else None
    seen = seen_images(snapshot, sorted(meta["filename"].unique()))
    if seen is None:
        contaminated = pd.Series(False, index=meta.index)
        contamination_note = (
            f"No dataset snapshot found for {yolo_name} (runs/{yolo_name}/dataset), so test images that were "
            "in its train or val cannot be excluded; treat the YOLO test number as an upper bound."
        )
    else:
        seen_set = seen["train"] | seen["val"]
        contaminated = meta["filename"].isin(seen_set)
        n_test_frames = meta.loc[split == "test", "filename"].nunique()
        n_bad = meta.loc[(split == "test") & contaminated, "filename"].nunique()
        contamination_note = (
            f"Snapshot of the {yolo_name} run found: {n_bad} of {n_test_frames} test frames were in its "
            f"train ({len(seen['train'])}) or val ({len(seen['val'])}) at the time. "
            "The 'test, unseen by YOLO' row drops those frames."
        )
    unseen_test = is_test & ~contaminated.to_numpy()
    if is_test.any():
        if unseen_test.sum() < is_test.sum():
            sub = frame[unseen_test]
            by_split.loc["test, unseen by YOLO"] = [
                sub["drop"].nunique(),
                len(sub),
                round(sub["yolo_ok"].mean(), 3),
                round(sub["probe_ok"].mean(), 3),
                round(sub["prior_ok"].mean(), 3),
            ]
        lines += [
            "## Frozen test drops: the fair contest",
            "",
            f"{int(frame.loc[is_test, 'drop'].nunique())} drops, {int(is_test.sum())} expert boxes the current YOLO detected. "
            "These drops were never in YOLO's training; the probe never saw them either (out-of-fold). "
            "Train/val rows below are YOLO's own training data and only show how far it overfits. "
            + contamination_note,
            "",
            by_split.to_markdown(),
            "",
            "### Per class on the test drops",
            "",
        ]
        side = None
        for name, key, pred in (
            ("YOLO", "yolo", yl.to_numpy()),
            ("probe", "probe", probe),
            ("probe + prior", "prior", probe_prior),
        ):
            t = per_class_table(y[is_test], pred[is_test], classes)
            t = t[t["n"] > 0]
            lines += [
                f"#### {name}: accuracy {(pred[is_test] == y[is_test]).mean():.3f}, macro F1 {t['f1'].mean():.3f}",
                "",
                t.assign(common=[common.get(c, "") for c in t.index])
                .round(3)
                .to_markdown(),
                "",
            ]
            cols = t[["precision", "recall"]].rename(
                columns={"precision": f"{key}_P", "recall": f"{key}_R"}
            )
            side = cols.join(t[["n"]]) if side is None else side.join(cols)
        side = side[
            ["n", "yolo_P", "yolo_R", "probe_P", "probe_R", "prior_P", "prior_R"]
        ]
        side.insert(0, "common", [common.get(c, "") for c in side.index])
        lines += [
            "### Side by side on the test drops: precision (P) and recall (R) per species",
            "",
            "Same expert boxes, same drops, for all three. `n` = expert boxes of that species the detector found.",
            "",
            side.round(3).to_markdown(),
            "",
        ]
    if has_yolo.any():
        lines += [
            "## Per drop, on boxes the current YOLO detected",
            "",
            "`split` = test are the frozen test drops. Everything else is in the current model's "
            "training lists, so `yolo_acc` there is training-set accuracy; the probe is out-of-drop everywhere.",
            "",
            per_drop.to_markdown(index=False),
            "",
        ]
    conf_mat = pd.crosstab(pd.Series(y, name="expert"), pd.Series(probe, name="probe"))
    lines += [
        "## Probe confusion (rows expert, columns probe)",
        "",
        conf_mat.to_markdown(),
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines))

    meta.assign(
        expert_class=y,
        zero_shot=zs,
        probe=probe,
        probe_prior=probe_prior,
        probe_p=oof.max(axis=1).round(3),
        yolo_class_mapped=yl,
    ).to_csv(OUT_DIR / artifacts.OOF_PREDICTIONS, index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
