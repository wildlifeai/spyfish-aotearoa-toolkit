# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/unit/test_config.py

# Run with verbose output
pytest -v tests/

# Linting (via pre-commit)
pre-commit run --all-files

# Individual linting tools
black spyfish/
isort spyfish/
flake8 spyfish/
mypy spyfish/

# Run the pipeline
python run_pipeline.py --ingest
python run_pipeline.py --ml
python run_pipeline.py --biigle-upload
python run_pipeline.py  # runs all steps

# Pipeline flags
# --no-upload       skip S3 uploads (local testing)
# --test-run        uses is_test_run: true in config
# --ping            print config summary and exit (connectivity check)

# Streamlit dashboard
streamlit run "app/🐟_Spyfish_Data_Tools.py"
```

---

## Architecture

Spyfish Aotearoa is a marine biodiversity data pipeline for processing underwater Baited Underwater Video (BUV) footage from New Zealand's marine reserves. It automates video ingestion → ML inference → community science export → expert annotation review → model retraining.

**Stack**: Python 3.12+, AWS S3 (`marine-buv` bucket, DOC-managed; `marine-buv-kalindi` is Kalindi's dev bucket), SQLite, Ultralytics YOLO (v8-v12), FFmpeg, Streamlit, NIWA Mahuika HPC (SLURM). Environment managed via venv/pyenv on NeSI — not Docker. Scheduling via NeSI crontab.

### Pipeline Flow — multi-section status model

The pipeline uses **independent section columns** (five core, plus two opt-in Zooniverse validation sections) on the `deployments` table, each with its own state machine. There is **no single linear status** — sections progress separately and are linked only by cross-section prerequisites.

| Column | Values |
|---|---|
| `ingest_status` | `ok`, `excluded`, `metadata_error`, `validation_error`, `removed` — set once at ingest, never advanced. Only `ok` deployments advance through processing stages. |
| `ml_status` | `ml_pending → ml_ready → ml_running → ml_complete`, `ml_error`, `ml_skipped` |
| `citsci_status` | `citsci_pending → citsci_clips_uploaded → citsci_complete`, `citsci_error`, `citsci_skipped` |
| `expert_status` | `expert_pending → expert_uploaded → expert_complete`, `expert_pending → expert_complete` (legacy direct path), `expert_error`, `expert_skipped` |
| `reporting_status` | `reporting_pending → reporting_complete`, `reporting_error` |
| `zoo_frames_status` | `zoo_frames_pending → zoo_frames_uploaded → zoo_frames_complete`, `zoo_frames_error` (no skipped: an empty drop is still a recall check) |
| `zoo_crops_status` | `zoo_crops_pending → zoo_crops_uploaded → zoo_crops_complete`, `zoo_crops_error`, `zoo_crops_skipped` |

Additional columns: `video_presence` (`present`/`archived`/`absent`/`no_video_bad_dep`), `priority` (int, higher = picked first).

**A deployment is considered "complete" when `expert_status = expert_complete` OR `reporting_status = reporting_complete`.**

**Stage eligibility:** each stage queries `DatabaseManager.get_deployments_eligible(section, statuses, prerequisites)` which always filters `ingest_status = 'ok'`, adds the stage's extra `prerequisites` dict (e.g. `{"ml_status": "ml_complete"}`), and orders by `priority DESC`. Advancement uses `db.advance_status(drop_id, section, to_status)` — a single generic method backed by the `SECTIONS` registry in `base.py` — which validates against `VALID_TRANSITIONS` and raises `InvalidTransitionError` for disallowed moves.

**`ON_HOLD` is gone.** To skip a stage, omit its CLI flag. To pause at runtime, don't run the pipeline.

Steps are invoked via CLI flags in `run_pipeline.py`:

**Data pipeline flags (run by default):**

| Flag | Section | Transition |
|---|---|---|
| `--ml` | `ml_status` | `ml_ready → ml_running → ml_complete` (or `ml_error`) |
| `--zooniverse-clips` | `citsci_status` | `citsci_pending → citsci_clips_uploaded` (requires `ml_status=ml_complete`) |
| `--zooniverse-sync` | `citsci_status` | `citsci_clips_uploaded → citsci_complete` (per-subject-set API fetch; `--force` to bypass raw CSV cache) |
| `--biigle-upload` | `expert_status` | `expert_pending → expert_uploaded` (requires `citsci_status IN (citsci_complete, citsci_skipped)` on the full-pipeline path, or `ml_status=ml_complete` on the biigle-direct path, via callable `prerequisites`) |
| `--biigle-sync` | `expert_status` | `expert_uploaded → expert_complete` |
| `--retrain` | — | Balance annotations, train YOLO, evaluate, promote model |

**Admin flags (off happy path — call explicitly):**

| Flag | Purpose |
|---|---|
| `--ingest` | Refresh metadata from SharePoint CSVs on S3, upsert deployments |
| `--check-arrivals` | Cheap S3 poll for newly arrived videos — advances `ml_pending → ml_ready` |
| `--set-targets` | Bulk-set pipeline stages from a CSV |
| `--legacy-experts` | Historical backfill: legacy expert annotations CSV from S3 |
| `--legacy-zooniverse` | Historical backfill: legacy Zooniverse classification CSV exports |
| `--db-refresh` | Reconcile DB status with on-disk artifacts and live Zooniverse/Biigle API state |

**Zooniverse validation flags (opt-in until the pipeline runs routinely; public uploads):**

| Flag | Section | What |
|---|---|---|
| `--zooniverse-frames` | `zoo_frames_status` | Frames picked by the BIIGLE selector (`frame_strategy` quotas × `zooniverse.validation.effort_multiplier`), every ML box drawn → `vframes_{drop}`. Volunteers draw missed fish (recall). |
| `--zooniverse-crops` | `zoo_crops_status` | One crop per ML box in those same frames → `crops_{drop}`. Volunteers judge right species / right box (precision). |

Both need `ml_status = ml_complete` and the video on local disk (never streamed from S3), and build everything from the existing raw CSV. They share one selection per drop (`get_validation_selection_path`). `--dry-run` renders and writes the manifests without uploading (the same flag `--retrain` uses for its fast part). Drops are chosen like every drop stage: eligible ones by `priority` (highest first), top `limit_processing` per run, so raise a drop's priority with `set_status --priority` to send it first. `--survey` bypasses the limit and does the whole survey. Sets are **attached to workflows by hand**, like clips; the upload logs which workflow each set belongs on. Sync-back of the answers is not built yet (`scripts/wip/analyse_validation_workflows.py` covers it).

### Configuration Architecture

**Single source of truth**: `config.yaml`. Missing keys raise `KeyError` — no silent defaults.

```
BaseConfig                    ← loads config.yaml, status classes, SECTIONS registry
├── PathsConfig               ← filesystem paths, S3 keys, per-drop helpers
├── ColumnsConfig             ← CSV column name accessors (from csv_mapping)
├── ValidationConfig          ← validation patterns, rules, validate_drop_id
├── MLConfig                  ← inference settings + training hyperparams
├── ExtractionConfig          ← clip/frame strategy params + ffmpeg encoding
├── BiigleConfig              ← project IDs, label mappings, polling config
├── ZooniverseConfig          ← project IDs, min_agreement_pct, clip caps
└── AWSConfig                 ← credentials from environment variables

ConfigWrapper(all of the above)  ←  singleton: `from spyfish.config.wrapper import config`
```

Secrets live in `.env` (not in git): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BIIGLE_API_EMAIL`, `BIIGLE_API_TOKEN`, `ZOONIVERSE_USER`, `ZOONIVERSE_PASSWORD`.

### Key Abstractions

- **`ConfigWrapper`** (`spyfish/config/wrapper.py`): The singleton `config` — typed properties for every config value, all path construction, validation helpers.
- **`S3Handler`** (`spyfish/storage/s3_handler.py`): Singleton for all S3 ops. Uses `aws s3 sync` without `--delete` (S3 is backup of record).
- **`DatabaseManager`** (`spyfish/database/manager.py`): SQLite CRUD for `deployments`, `sites`, and `validation_errors` tables. Single `advance_status(drop_id, section, to_status)` method backed by the `SECTIONS` registry, `get_deployments_eligible(section, statuses, prerequisites)` for stage queries, `add_or_update_deployment()` which preserves section statuses on conflict (re-ingest never resets pipeline progress). Column names validated against the live schema via `_validate_column()` to prevent SQL injection.
- **`AnnotationDatabaseManager`** (`spyfish/database/annotation_manager.py`): Separate `spyfish_annotations.db` — one row per species observation from any source. `sync_annotation_counts(drop_ids)` is the only function that writes `ml_annotations` / `citsci_annotations` / `expert_annotations` counts back to the deployments table.
- **`DataValidator`** (`spyfish/validation/data_validator.py`): Applies validation rules from `config.yaml` using the strategy pattern.
- **`MLRunner`** (`spyfish/orchestrator/ml_runner.py`): Full ML lifecycle — target selection → video download → YOLO inference → status update.

---

## Naming Conventions

- **DropID**: `{MarineReserveCode}_{YYYYMMDD}_BUV_{SiteCode}_{Replicate}` → e.g., `KSF_20240124_BUV_KSF_085_01`
- **SurveyID**: `{MarineReserveCode}_{YYYYMMDD}_BUV` → e.g., `KSF_20240124_BUV`
- Always use realistic, validation-passing IDs in tests (e.g., `KSF_20240124_BUV_KSF_085_01`), not placeholder strings.
- **MaxN CSV files**: `{drop_id}_ml_{model_name}_maxn.csv` (ML) and `{drop_id}_zooniverse_maxn.csv` (citsci). Use `get_maxn_csv_path(drop_id, model_name)` for ML and `get_zooniverse_maxn_csv_path(drop_id)` for citsci. Do not construct these paths manually.
- **Raw CSV files**: `{drop_id}_{model_name}_raw.csv` (no `_ml_` prefix — raw files are not shared with other annotation sources).

---

## Hard Rules

- **Never use raw boto3**: Always use `S3Handler()`.
- **Never hardcode config values**: No regex patterns, path strings, thresholds, or column names in business logic — all must be in `config.yaml` and accessed via `config.*`.
- **Never use `.get(key, default)` for config access**: Use `config.property_name` or `_require()` so missing config fails loudly. (`.get()` is fine for API responses, CSV lookups, etc.)
- **Never construct paths manually**: Use `config.get_drop_annotations_dir(drop_id)`, `config.get_video_path(drop_id)`, etc.
- **Never use a DropID in path construction without validation**: Always call `config.validate_drop_id(drop_id)` first (prevents path traversal).
- **Never use hardcoded CSV column names**: Use `config.drop_id_column`, `config.survey_id_column`, etc. These are an external contract with PowerApps/SharePoint — they can change without a code release.
- **DB column names and status values live in code, not config**: `MlStatus.COLUMN`, `MlStatus.READY`, etc. are internal schema — the state-machine logic depends on the exact strings, and changing them requires a code + migration pair. Access them via the status classes in `spyfish/config/base.py`, not via `config.yaml`.
- **Never add `--delete` to S3 sync**: S3 is the backup of record.
- **Never use `rm`**: Use `/usr/bin/trash` to send files to macOS Trash. `rm` is unrecoverable.
- **Ration `ssh mahuika`**: NeSI login access is rate-limited and lockout-prone. At most **one** ssh/scp/rsync to `mahuika` per user prompt, batched as a single `ssh mahuika 'cmd1; cmd2'`. Ask before connecting if the user did not explicitly ask for something on NeSI. Never retry a failed connection in a loop, report and stop.
- **Never use `/tmp` for video downloads**: Use `config.media_dir`. Videos are ~10 GB and NeSI has a `media_base_dir` override.
- **Use `logging.info(...)` directly**: Do not create named loggers (`log = logging.getLogger(__name__)`). Project convention is root-logger calls throughout.
- **DO NOT RM any files without asking for confirmation.**

---

## Project Context & Key Decisions

### Broader ecosystem (outside this repo)
The pipeline is one component. Upstream: **video-uploader app** (Gitlab/BytesNZ desktop app, concatenates GoPro segments, uploads `{DropID}.mp4` to S3) and **PowerApps** (BUV Survey + Deployment apps for ranger data entry → SharePoint). Downstream: **PowerBI** (DOC/Rangers reporting dashboard, connects to SharePoint for deployment metadata — annotation data transfer to PowerBI is currently manual, no automatic pipeline). **Streamlit** is the ops dashboard for the team and external contributors — does not require DOC access.

### Pipeline status model (see "Pipeline Flow" above for full table)
Replaced the old linear single-`status` column with **independent section columns**. `ingest_status` is the data-quality gate — all eligibility queries filter `ingest_status = 'ok'`, so `excluded` / `metadata_error` / `validation_error` / `removed` deployments are never picked up by processing stages. `video_presence` (`present`/`absent`/`no_video_bad_dep`) is its own column — it tracks S3 file state, not pipeline progress. Section errors live in `validation_errors`, whose `ErrorType` is the section's own `ERROR` status string (e.g. `MlStatus.ERROR` = `ml_error`); there is no separate mapping table. No `error_message` column. Adding a section = a status class in `base.py` added to `SECTION_STATUSES`, the column in `CREATE TABLE`, and an `_add_column_if_missing` migration.

### Annotation sources — expert wins, no formal merge
Three sources: `annotated_by = 'ml'`, `'citsci'` (Zooniverse volunteers), `'expert'` (BIIGLE). No merge step — when expert annotations exist they win. ML and citsci stay in the DB for auditing. `external_id` names the SOURCE a row came from, never the row: the model name for ML, `legacy` for the historical expert CSV backfill, `biigle_volume_{id}` for every row synced from that BIIGLE volume (constants and `biigle_external_id()` in `annotation_manager.py`). Each writer deletes only its own key space on re-ingest (`legacy` / `biigle_*` / its model name), so legacy and BIIGLE expert accounts of one deployment coexist; the dashboard ranks BIIGLE above legacy (`source_rank`) and the annotations export carries a `Source` column. Citsci rows are the exception: their `external_id` is a per-row Zooniverse subject id, so never group citsci by it.

**`annotated_by` is `'ml'` for ML rows, in the DB and the MaxN CSV alike; the model name is its own field** (`external_id` in the DB, `ModelName` in the ML CSV). Filter or group by model on those, never on `annotated_by`. Pre-2026-10-13 rows that had the model name in `annotated_by` are migrated by `AnnotationDatabaseManager.init_db`. **Newest model wins:** where several models ran on one drop, reports keep only the newest model's rows (`newest_model_only`, ranked by the training date in the model name, not by run time). `get_maxn_summary(all_models=True)` keeps them all, for provenance views.

**Absence convention:** a review that finds nothing writes one row with `scientific_name = NULL_DEPLOYMENT` (sentinel in `spyfish/config/base.py`) and `max_interval = 0` — "reviewed, nothing seen" must be distinguishable from "never looked" (no rows). Only write it when the source asserts absence: a blank field in an export means "no data", not "zero" (`parse_legacy_rows` skips those). Dashboards map the sentinel to NaN at load (`ecology_data.load_maxn`, `doc_report/data.load_annotations`); queries on nullable `scientific_name` must be null-safe (`IS`, not `=` — a `NULL = NULL` subquery bug once dropped every absence record from `get_maxn_summary`).

### ML model strategy
Currently binary (fish vs no-fish). Species-specific classes are added incrementally as training data for each species grows enough to support reliable detection. Do not force multi-class before the data supports it.

### Zooniverse classification
Volunteers classify into **top species + "other"** — intentionally simple. `--zooniverse-sync` (GlobalStage in `run_pipeline.py`) is the live path: it calls `subject_completion_from_api` once per run to check which subject sets are fully retired, fetches classifications per retired set via `fetch_classifications_for_set`, aggregates with `min_agreement_pct` threshold, and writes a raw CSV + MaxN CSV per drop before ingesting to the annotations DB. Historical backfill uses `--legacy-zooniverse` (reads classification CSV exports from `process_files/zooniverse/legacy_classifications/`). Caesar is not used. Multi-project fetch: `source_project_ids` in config. Drop completion tracked by retirement % — 180 retired subjects = one full 30-min BUV deployment.

**EY and Zooniverse are two parallel workflows of one project, both currently active.** Source IDs 14054 (main) and 30327 (EY) are the steady state — not a migration. Aggregation merges them by design.

**Subject metadata convention:** `upload.py` writes `#SubjectType: "clip"` or `#SubjectType: "frame"` on every subject. `parse_classifications.py` reads `#SubjectType` as the canonical key with legacy fallbacks. Historical frame subjects uploaded before this convention silently default to `"clip"` — they can be distinguished by subject set `display_name` (`clips_{drop_id}` vs `frames_{drop_id}`) or location mimetype (`video/mp4` vs `image/jpeg`).

**Frame subjects (Zooniverse image uploads) were removed from the pipeline.** The flow is now `citsci_pending → citsci_clips_uploaded → citsci_complete`. Frame extraction still exists for BIIGLE expert review, but Zooniverse only sees clip subjects.

**MaxN CSV naming and schema** — both ML and Zooniverse MaxN CSVs share the same 8-column schema:
- ML: `{drop_id}_ml_{model_name}_maxn.csv` — `AnnotatedBy = "ml"`, plus a `ModelName` column (and the persistence-filter columns)
- Zooniverse: `{drop_id}_zooniverse_maxn.csv` — `AnnotatedBy = "citsci"`
- Columns: `DropID, ScientificName, TimeOfMax, MaxInterval, AnnotatedBy, IntervalAnnotation, ConfidenceAgreement, TimeOfMaxSeconds`
- Path generation: use `config.get_maxn_csv_path(drop_id, model_name)` for ML only. Zooniverse path is constructed directly (not via that method).

**Aggregator outputs three count statistics per (subject, species) row:**
- `mode_count` → training labels / MaxN CSV (`MaxInterval`)
- `max_count` → ecology / dashboard reporting (true peak abundance)
- `mean_count` → BIIGLE frame ranking (continuous, no ties)
- `count_disagreement` flag → `max_count >= mode_count + 2 AND vote_count >= 3`

### BIIGLE scope
Primary use: species identification with full expert species list. Planned future uses: substrate analysis, fish size review — same upload/sync pipeline steps, different parsing when downloading annotations.

**Annotation `confidence` is write-only.** We send the YOLO score per box and BIIGLE stores it, but renders it nowhere (index endpoint strips it, no UI control, no report exports it). It is a deprecated human-uncertainty field upstream wants to delete; the ML equivalent (`ai_confidence`) is unbuilt. Never build anything that reads confidence back out of BIIGLE.

**"Pick" image labels** say why a frame was selected — `Pick > MaxN peak`, `Uncertain ID`, `ML peak`, `Fish variety`, `Spot check`, `Video start`, in workflow tree 3375, plus the species a MaxN/Uncertain frame is evidence for (not an inventory of the frame). Whole-image labels ARE displayed by BIIGLE, which is why they exist. The `SelectionReason` → label mapping is in `spyfish/biigle/selection_reason.py`; it matches on the head of the string and deliberately keeps legacy spellings (`Confusing`, `Absolute MaxN`) because `upsert_selections` keeps old reasons on disk forever. Adding a new selection bucket means adding a rule there, or its frames upload unlabelled.

### Expert review model
Target ~10 frames per deployment for expert review (can be triggered at survey level). Closer per-frame review available if ML + citsci signal looks wrong. Keeps expert effort tractable at ~1,000 deployments/year.

### Sensitive data
Do not expose in public-facing tools: GPS lat/lon of sites (illegal fishing risk), rare species presence (poaching risk), PII (names, contracts).

### Documentation
- `design_doc.md` — single source of truth: architecture, decisions, diagrams, roadmap, known gaps, CLI reference, config reference, module reference, legacy data retrieval.
- `claude_docs/todo.md` — running task list: code quality, test gaps, Zooniverse review findings, ML pipeline, BIIGLE, reliability hardening.
- `claude_docs/zooniverse_analysis.md` — **index of all Zooniverse analytical artefacts** (audits, engagement stats, parser-output reviews, notebooks, charts, production code). Start here for any Zooniverse data-quality or volunteer-behaviour question.
- `claude_docs/zooniverse_suspicious_users_report.md` — send-to-Zooniverse report on bot accounts (blank-API submissions), NH-overuse accounts, and (2026-05-25 addendum) Cat 3 speed-clickers.
- `claude_docs/zooniverse_annotations_todo.md` — agreed decisions + PR plan for annotations DB cleanup (schema, `AnnotationRecord` dataclass, bootstrap orchestrator).
- `claude_docs/folder_rename_plan.md` — confirmed folder structure rename decisions.
- `claude_docs/uoa_design.md` — UoA legacy BUV integration (Underwood & Jeffs, Jan 2022): spreadsheet→deployment cleaning, symlinks, **and the ML→BIIGLE expert-review workflow** (`scripts/wip/run_ml_videos_and_upload_to_biigle.py`, binary `fish` model, per-deployment chapter pooling, the cross-chapter `build_coco` fix).
- `claude_docs/anywhere_plan.md` — "Spyfish Anywhere": plan to generalize the pipeline to multiple camera-trap projects from one codebase (deployment↔media split, layered profiles, DwC-A export). §13 captures what the UoA one-off revealed and the convergence path back into core.
- `claude_docs/uoa_paper_plan.md` — UoA ML paper (with Underwood & Jeffs): methodology chapter outline, decisions (20 Jan holdout, 5-class model, MaxN-vs-hand-annotations headline eval), open questions, and status-at-handoff. Start here for any paper-writing session.
