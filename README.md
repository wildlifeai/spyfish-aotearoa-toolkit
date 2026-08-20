# Spyfish Aotearoa Pipeline

Automated pipeline for processing marine Baited Underwater Video (BUV) footage
from New Zealand's marine reserves: metadata ingestion, ML fish detection,
Zooniverse citizen science, BIIGLE expert annotation, and dashboard
visualisation.

> Work in progress: we are actively working on the pipeline, so details change
> often. Reach out if you want to use the pipeline or contribute, or if you
> have questions, ideas or feedback.

## How it works

Rangers deploy baited cameras and upload ~30-minute videos to S3. From there
the pipeline takes over:

1. **Ingest**: deployment metadata from SharePoint exports is validated and
   loaded into a SQLite database. Only valid deployments move on.
2. **ML inference**: the fine tuned Community Fish Detector, a YOLO model, scans each video for fish and writes
   per-species abundance (MaxN) estimates.
3. **Citizen science**: short clips are uploaded to
   [Zooniverse](https://www.zooniverse.org/projects/victorav/spyfish-aotearoa),
   where volunteers classify species. Their consensus is aggregated once
   subjects retire. Currently reviewing this step to make citsci annotaiton more efficient, meaning fewer volunteers per annotation.
4. **Expert review**: selected frames go to [BIIGLE](https://biigle.de), where
   marine experts confirm species identities. Expert annotations are the final truth.
5. **Reporting & retraining**: results feed the dashboard and DOC reporting,
   and confirmed annotations grow the training set for the next model.

Each stage tracks its own status per deployment, so stages progress
independently rather than in one fragile chain.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

# Secrets live in .env (never in git): AWS, BIIGLE and Zooniverse credentials
#: see design_doc.md for the full list.

python run_pipeline.py --ping     # config + connectivity check
python run_pipeline.py            # run all default pipeline steps
python run_pipeline.py --ml       # or run a single step

# Ops dashboard
streamlit run "app/🐟_Spyfish_Data_Tools.py"
```

Tests: `pytest tests/` (unit tests need no external services; integration tests
need FFmpeg).

## Running in production (HPC / SLURM)

The pipeline runs in production on New Zealand's NeSI HPC platform, not on a
server of ours. Scheduled runs are submitted as [SLURM](https://slurm.schedmd.com/)
batch jobs:

- `run_pipeline.sl`: the batch script for regular pipeline runs (GPU node for
  ML inference), triggered on a schedule via crontab on the login node.
- `spyfish/ml/training/train_job.sl`: the batch script for model retraining.

The environment on NeSI is a plain Python venv, and large video
downloads go to a scratch filesystem via the `media_base_dir` override in
`config.yaml`. Everything in Quick start above also works on a laptop: the
SLURM scripts just wrap the same `run_pipeline.py` CLI.

## Repository layout

| Path | What it holds |
|---|---|
| `spyfish/` | Pipeline library: config, database, extraction, ML, Zooniverse, BIIGLE, orchestrators |
| `app/` | Streamlit dashboard (ops views + DOC reporting) |
| `run_pipeline.py` | CLI entrypoint: one flag per pipeline stage |
| `config.yaml` | Single source of truth for all configuration |
| `tests/` | Unit + integration tests |
| `scripts/` | One-off and maintenance scripts |

## Documentation

- **[design_doc.md](design_doc.md)**: the full picture: architecture,
  decisions, CLI reference, configuration, module reference.
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: system overview and diagrams.
- **[Project wiki](https://spyfish.notion.site/overview)**: tutorials and
  contributor information.

## Links

- [Zooniverse project](https://www.zooniverse.org/projects/victorav/spyfish-aotearoa): help classify fish!
- [Project website](https://wildlife.ai/projects/spyfish-aotearoa/): news and updates.
