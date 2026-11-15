# Two-stage classifier: training side

**In the pipeline since 2026-11-10:** `spyfish/ml/classifier/species_classifier.py`
loads a `finetune_bioclip.py` checkpoint from `process_files/models/classifier/`
(newest stamp) and names every YOLO box during `--ml` / `--ml-infer`
(`ml_inference.classifier` in config.yaml).

**Moved out of `scripts/wip/` 2026-11-10.** The library is
`spyfish/ml/classifier/`: `crops.py` (expert boxes from the BIIGLE CSVs, the
assembled YOLO dataset or the UoA volumes; names through `canonicalize_species`;
bait, the catch-all and `training.excluded_species` dropped), `classes.py` (the
class rule and `classes.json` beside every checkpoint), `metrics.py` (box
matching, detector recall, the site prior) and `finetune.py`. Every knob is in
`config.yaml` under `training.classifier` (thresholds, recipe, augmentation,
hard negatives) or shared with inference under `ml_inference.classifier`
(crop margin, min box size, backbone). The output folder is
`paths.sub_dirs.two_stage` (`process_files/wip_two_stage/`, gitignored; the
name is historical). The scripts here are the command-line tools; the jobs are
`scripts/slurm/two_stage.sl` and `scripts/slurm/train_detector.sl`.

First working pass at `two_stage_classifier_plan.md` Phase 1, on data already on this
machine: 15 drops whose BIIGLE expert boxes and frames are under
`process_files/deployment_data/`. Run in order:

```bash
python scripts/two_stage/run_yolo_on_frames.py   # current species YOLO on every annotated frame
python scripts/two_stage/build_embeddings.py     # crop expert boxes, embed with BioCLIP 2 (MPS, ~5 min)
python scripts/two_stage/evaluate.py             # zero-shot, drop-grouped probe, + site prior, YOLO comparison
python scripts/two_stage/demo_two_stage.py       # 8 frames: YOLO boxes -> BioCLIP probe -> prior, drawn
python scripts/two_stage/make_page.py            # one self-contained HTML page of the above
```

Outputs land in `process_files/wip_two_stage/` (gitignored). Needs `open_clip_torch`,
`scikit-learn` and `tabulate` on top of requirements.txt; BioCLIP 2 weights are
downloaded once into `process_files/models/bioclip/` by `common.load_bioclip`.

Caveats that matter when reading the numbers:

* Every drop here is in the current model's training lists, so YOLO's label accuracy on
  these frames is training-set accuracy. The probe is scored out-of-drop. The go/no-go
  comparison (plan §4.4) needs the frozen test drops on Mahuika.
* Classes need `training.classifier.min_drops` drops and `min_crops` crops (2 / 30);
  the rest train as `other fish`, which is a grab-bag and scores like one. Each
  run's `classes.json` says which species made it and why.

## Running it on NeSI

Copy (code goes straight over, per the sync rule):

* `spyfish/ml/classifier/`, `scripts/two_stage/` and `scripts/slurm/two_stage.sl` (code goes straight over)
* `process_files/site_prior/species_presence_by_reserve.csv`, or run
  `scripts/build_site_prior.py` there if both DBs are current
* the model: `process_files/models/pipeline_model/species_20260825_053442.pt`
  (114 MB; check whether it is already there)
* data: with `TWO_STAGE_SOURCE=yolo_dataset` (the job's default) the scripts read
  the assembled YOLO dataset under `training.local_training_dir`/`species/`, which
  `--retrain --data-prep` builds from BIIGLE and which already holds every drop
  including the frozen test set. Nothing to copy. To reproduce the Mac run
  instead, the 15 drops' `frames/` + `annotations/` are 3.4 GB and are on S3.

One-time on a login node: `python -m pip install open_clip_torch scikit-learn tabulate`
into the venv, then `python -c "from spyfish.ml.classifier.crops import load_bioclip; load_bioclip('cpu')"`
to put the BioCLIP 2 weights in `process_files/models/bioclip/` (compute nodes may
have no internet, so the job sets `HF_HUB_OFFLINE=1` and only reads from there).

Then `sbatch scripts/slurm/two_stage.sl` runs yolo → embed → evaluate →
negatives on an l4. The fair comparison the plan asks for falls out of
`evaluate.py`'s per-drop table once the test drops are in the data: those drops
are the only ones the current YOLO has never seen.

`hard_negatives.py --source upload` takes its candidates from the BIIGLE round
trip instead of a detector run: the boxes the pipeline sent for review
(`{drop}_coco_annotations_for_biigle.json`) that are missing from the expert CSV,
so a person looked at each one and deleted it. On the two local drops with both
files that is 26 of 128 uploaded boxes: the bait bag called `fish`, starfish
called rock lobster. `--source both` adds the unreviewed current-model boxes
that are not already among them. The COCO is rewritten if ML reruns on a drop
after its review, so a drop reprocessed since then compares a newer model's
boxes with the expert's edits of an older one; the log says how many drops
contributed. On NeSI: `NEGATIVES_ARGS="--source upload" sbatch scripts/slurm/two_stage.sl negatives`,
output under `wip_two_stage/negatives_upload/`.

## Result so far (NeSI, model species_20261016, frozen test drops, 2026-11-04)

Naming on the 1,235 expert boxes the species YOLO detected, same crops for every
row:

| naming model | acc | macro F1 |
|---|---|---|
| species YOLO | 0.930 | 0.599 |
| BioCLIP probe (+ prior) | 0.866 (0.878) | |
| BioCLIP fine-tune, 6 blocks, 15 epochs, class-balanced | 0.910 | 0.628 |
| BioCLIP fine-tune, 12 blocks, 30 epochs, sampler power 0.5 | 0.951 | 0.768 |

The heavier fine-tune names better than YOLO on every species except silver
sweep (recall 0.66 vs 0.78) and ties on blue cod (0.987 vs 0.988). Snapper
0.92 vs 0.79, spotty 0.93 vs 0.69, tarakihi 5/5 vs 1/5. The site prior adds
nothing on top of it (best alpha 0; alpha 1 costs 0.05 macro F1). Caveat: the
reported epoch is picked by test macro F1, so it is optimistic by about 0.02
macro F1 (last epoch 0.711 vs best 0.733, same accuracy); a val split carved
from the training drops should pick the epoch next time.

Detection still caps everything: the species YOLO found 1,235 of 1,859 expert
boxes on those drops (spotty 14%, sweep 48%, snapper 55%, rock lobster 1 of 8),
so end to end (found x named) the two differ by 1.4 points overall, 61.8% vs
63.2%. The three-class detector below is what moves that.

## Detector: one class, or three (plan §5.2)

```bash
# one class, the preferred form (2026-11-04): every expert box, bait and lobster included,
# is `fish` to the finder, so best.pt is picked on that one class's mAP
# (bait folds into fish too: it is a dead fish, visible and labelled only in volume 30505;
#  current surveys keep it inside the box. The old --drop option was removed 2026-11-10.)
python scripts/two_stage/make_detector_dataset.py --classes fish --out detector_fish1
DATA=process_files/training/detector_fish1/data.yaml sbatch scripts/slurm/train_detector.sl
# three classes (the first run, detector_fish3_20261104_150247_img640): fish, Jasus edwardsii, bait
python scripts/two_stage/make_detector_dataset.py        # species/ -> detector/
sbatch scripts/slurm/train_detector.sl               # CFD base, h100/a100, config hyperparameters
IMGSZ=1024 BATCH=8 sbatch scripts/slurm/train_detector.sl   # second run if small fish stay missed
# score any checkpoint (best.pt is chosen by the mean mAP over classes, so with three
# classes score last.pt and the save_period checkpoints too):
YOLO_ARGS="--model <run>/weights/best.pt --out detector_predictions.csv" RECALL_FILE=detector_predictions.csv sbatch scripts/slurm/two_stage.sl yolo recall
# when done:
TWO_STAGE_SOURCE=yolo_dataset python scripts/two_stage/run_yolo_on_frames.py --model process_files/training/runs/<run>/weights/best.pt --out detector_predictions.csv
TWO_STAGE_SOURCE=yolo_dataset python scripts/two_stage/detector_recall.py detector_predictions.csv
TWO_STAGE_SOURCE=yolo_dataset python scripts/two_stage/detector_recall.py yolo_predictions.csv   # the species model, same scoring
```

`detector_recall.py` reports, per species and per split, the share of expert
boxes any detector box covers at IoU >= 0.5. Nothing here promotes a model.

## Heavier BioCLIP fine-tune

```bash
FINETUNE_ARGS="--epochs 15 --unfreeze 6 --lr-encoder 2e-5" sbatch scripts/slurm/two_stage.sl finetune
# the run that beat YOLO on naming (about 2.6 h on an l4):
FINETUNE_ARGS="--epochs 30 --unfreeze 12 --sampler-power 0.5" sbatch scripts/slurm/two_stage.sl finetune
# with a `not a fish` class from the BIIGLE-deleted boxes (run the negatives step first):
FINETUNE_ARGS="--epochs 12 --unfreeze 12 --sampler-power 0.5 --negatives negatives_upload --out finetune_neg" sbatch scripts/slurm/two_stage.sl finetune
```

`--negatives` appends the crops of one or more `hard_negatives.py` output folders
as the class `not a fish`; a negative inherits its drop's split. The report gains
a section with the two rates that matter: expert fish called `not a fish`, and
negatives called a fish class. The YOLO rows keep using the expert fish only.
`negatives_upload` can train as is; `negatives` (current-model boxes) needs the
cluster review first, since some of its boxes are fish the expert skipped.
`--out` keeps a run from overwriting `finetune/`.

## UoA mussel-farm footage (`TWO_STAGE_SOURCE=uoa`, `uoa/`)

The same scripts on the Underwood & Jeffs dataset, per
`uoa/docs/uoa_two_stage_brief.md`: a CFD-based `fish` detector fine-tuned on
the train days, BioCLIP 2 naming the paper's four species (everything else
`fish`), MaxN per 360-s interval compared with the hand counts on 20 January,
which is held out of everything. Setting `TWO_STAGE_SOURCE=uoa` makes
`common.load_boxes()` read only the three UoA volumes (`UOA_VOLUMES`), group by
**deployment** parsed from the chapter filename (both naming schemes), tag
20 Jan `test`, and send every output to `process_files/wip_two_stage/uoa/`.
`evaluate.py` then fixes the classes to the paper's four and applies no site
prior. Steps, in the brief's order:

```bash
export TWO_STAGE_SOURCE=uoa
python -m uoa.scripts.two_stage.inventory                 # 1  what is on disk, crop sizes, hand-count mix per day
python scripts/two_stage/run_yolo_on_frames.py --model process_files/models/base_model/cfd-yolov12x-1.00.pt \
    --imgsz 1280 --conf 0.15 --out cfd_base_predictions.csv     # 2  same for binary_uoa_20260609 / exp_uoa_species_v1 on NeSI
python -m uoa.scripts.two_stage.detector_dataset          # 3a one-class dataset, split by deployment, 20 Jan excluded
sbatch uoa/scripts/two_stage/uoa_train_detector.sl        # 3b CFD -> fish detector, imgsz 1280 (NeSI, h100)
python -m uoa.scripts.two_stage.detector_table            # 2/3 recall per species, every detector, + conf sweep on val
python scripts/two_stage/build_embeddings.py              # 4  every expert box, no cap
python scripts/two_stage/evaluate.py --yolo-csv exp_uoa_species_v1_predictions.csv --yolo-name exp_uoa_species_v1
python scripts/two_stage/finetune_bioclip.py --epochs 15 --unfreeze 6   # held-out deployment fold
python -m uoa.scripts.two_stage.fit_classifier            # probe on all train days + p_min sweep
python scripts/two_stage/finetune_bioclip.py --all-train --epochs 15 --unfreeze 6   # finetune/final.pt, optional
python -m uoa.scripts.two_stage.freeze_rules --detector process_files/training/runs/<run>/weights/best.pt   # 5
DAY=20220120 sbatch --array=0-7 uoa/scripts/two_stage/uoa_run_day.sl                      # 6  once
DAY=20220120 BASELINE=process_files/training/runs/exp_uoa_species_v1/weights/best.pt sbatch --array=0-7 uoa/scripts/two_stage/uoa_run_day.sl
python -m uoa.scripts.two_stage.maxn_compare --day 20220120 --run two_stage_<run>_probe --run exp_uoa_species_v1
python -m uoa.scripts.two_stage.report && python scripts/two_stage/make_page.py --report uoa_report.md --out uoa_two_stage.html
```

`uoa_two_stage.sl` chains the train-day steps on an l4 (`inventory yolo recall
embed evaluate fit finetune`, or a subset on the sbatch line). Facts the
design rests on, found 2026-11-04:

* **20 Jan has no expert boxes**, only hand counts (the three volumes are
  18 and 19 Jan), so every box-level number is a train-day number and the
  classifier is chosen by deployment-grouped CV; 20 Jan is scored at MaxN
  level only, in `maxn/`.
* `volume_30505` under `extra_no_survey_id/` is a DOC training volume (UUID
  filenames, blue cod), not UoA; the UoA scope is the `UOA_VOLUMES` whitelist.
* Locally only the 18 Jan volumes have frames; 19 Jan's (`volume_33439`) are
  on NeSI. Every loader reports what it skipped.
### Annotation round 2: one pass per day, frames for BIIGLE (2026-11-11)

The UoA detector (epoch 40 of the 100-epoch frozen run) and the UoA BioCLIP run
once per day in `run_day.py`; everything after reads its raw CSVs (detector
class in `det_class`, species in `class`). Video is decoded by ffmpeg, video
stream only (cv2 stalls on the raw GoPro chapters), at 1 fps.

```bash
export TWO_STAGE_SOURCE=uoa
# 0. models and thresholds for every day run (and the 20 Jan holdout): --dry-run first
python -m uoa.scripts.two_stage.freeze_rules --detector <run>/weights/epoch40.pt --classifier probe --ml-fps 1
# 1. one GPU pass per day -> process_files/wip_two_stage/uoa/raw/<day>/
DAY=20220118 sbatch --array=0-7 uoa/scripts/two_stage/uoa_run_day.sl     # also 20220119, 20220121
# 2. per-species MaxN per 6-min interval vs the hand counts (18 and 19 Jan)
python -m uoa.scripts.two_stage.maxn_compare --day 20220118 --run <run name>
# 3. ML-picked + blind frames from those CSVs (no second inference)
DAY=20220118 RAW_RUN=<run name> sbatch --time=02:00:00 uoa/scripts/run_uoa_videos.sl
# 4. hand-count frames: every interval with a fish, 4 each, ML peak + gaps
python -m uoa.scripts.two_stage.hand_count_frames --day 20220118 --videos-dir $VIDEOS_DIR --extract \
    --raw-dir process_files/wip_two_stage/uoa/raw/20220118 \
    --exclude-volume <step 3 run folder> uoa/kutai/process_files/deployment_data/*_20220118_*
# 5. boxes + species names on both frame sets (GPU), each into its own --out-dir
python -m uoa.scripts.two_stage.preannotate_frames --frames-dir <frames> --coco-in|--selections ... \
    --detector <run>/weights/epoch40.pt --classifier probe --out-dir <set>/preannotated
# 6. on the Mac (UoA S3 key): frames into <dep>/frames/, one volume, Pick labels
python -m uoa.scripts.upload_review_volume --volume-name "UoA 20220118 round 2" \
    --set <ml set>/preannotated/coco_annotations.json <ml set>/frames \
    --set <hand-count set>/preannotated/coco_annotations.json <hand-count set>/frames   # then --go
```

Frame rules, per deployment (`run_ml_videos_and_upload_to_biigle.py`):
2 MaxN peaks + 2 uncertain + 1 start + 3 size bands + 3 random detections
(all on the detector's single `fish` class), 1 `MaxN (<species>)` peak per
species BioCLIP names (`--species-peaks`), 8 blind; then picks under 10 s apart
in a chapter become one (a detection pick keeps its moment, blind picks merge
to the middle) and at most 3 frames the model saw no fish in stay. Hand-count
frames (`hand_count_frames.py`, Pick label `Manual count`): one at the
interval's busiest ML moment, the rest in the middle of the widest gaps around
frames already there (BIIGLE and step 3), never within 10 s of one. 21 Jan has
no hand counts. Measured on 18/19 Jan: 748 + 1,296 hand-count frames.

Unreviewed frames sit in `<dep>/frames/` next to reviewed ones: training reads
only frames of volumes marked `reviewed: true` in
`uoa/kutai/metadata/biigle_volumes/<id>.json`. Set it when the experts finish.

* `run_day.py` writes the pipeline's raw CSV schema with the classifier's
  species in `class`, so `in_progress/uoa/` reads the result unchanged.
  `frozen_rules.yaml` must exist before it runs on 20 Jan; the run is logged
  to `holdout_runs.log` and refused a second time without `--again`.
