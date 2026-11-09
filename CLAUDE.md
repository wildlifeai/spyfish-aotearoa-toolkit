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

**Stack**: Python 3.10–3.13, AWS S3 (`marine-buv` bucket, DOC-managed; `marine-buv-kalindi` is Kalindi's dev bucket), SQLite, Ultralytics YOLO (v8-v12), FFmpeg, Streamlit, NIWA Mahuika HPC (SLURM). Environment managed via venv/pyenv on NeSI — not Docker. Scheduling via NeSI crontab.

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

**A deployment is considered "complete" when `expert_status = expert_complete` (or equivalent terminal state for the chosen path).**
