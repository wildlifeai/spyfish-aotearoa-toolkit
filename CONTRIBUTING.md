# Contributing to Spyfish Aotearoa

This guide is for contributors who want to run the project locally without
access to the production AWS, BIIGLE, or Zooniverse accounts. Do not put real
credentials in the repository or upload sample data to a production bucket.

## Prerequisites

Use Python 3.12 or 3.13. Install FFmpeg and the AWS CLI before creating the
virtual environment.

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install ffmpeg awscli
```

On macOS with Homebrew:

```bash
brew install ffmpeg awscli
```

Create and activate a virtual environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
```

The CPU-only PyTorch install should happen before the project requirements so
the environment does not select a CUDA build on a laptop.

## Generate sample data

The generator creates fictional metadata and videos under
`process_files/sample_bucket/`. The metadata-only mode is faster when testing
ingestion or documentation changes.

```bash
python scripts/generate_sample_metadata.py --metadata-only
# Use this instead when testing video extraction or ML input:
python scripts/generate_sample_metadata.py
```

The generated records are synthetic. They are suitable for local testing but
must not be mixed with production data.

## Run a local S3-compatible service

Moto provides an in-memory S3-compatible endpoint. Install its server extra in
the active virtual environment:

```bash
python -m pip install 'moto[server]'
moto_server -p 5000
```

Keep that process running, then add these values to a local `.env` file. The
`AWS_ENDPOINT_URL` setting is supported by `S3Handler` and is intentionally not
included in production configuration.

```dotenv
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
AWS_REGION=eu-central-1
AWS_ENDPOINT_URL=http://127.0.0.1:5000
S3_BUCKET=spyfish-local
```

Create the bucket and upload the generated sample tree:

```bash
set -a
source .env
set +a
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 mb "s3://$S3_BUCKET"
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 sync \
  process_files/sample_bucket/ "s3://$S3_BUCKET/"
```

## Run the pipeline locally

Start with the read-only check. It reports missing credentials without printing
their values and checks services only when the corresponding credentials exist:

```bash
python run_pipeline.py --ping
```

The local metadata path can then be exercised without production uploads:

```bash
python run_pipeline.py --ingest --no-upload
```

The normal stage commands are:

```bash
python run_pipeline.py --ml --no-upload
python run_pipeline.py --zooniverse-clips
python run_pipeline.py --zooniverse-sync
python run_pipeline.py --biigle-upload
python run_pipeline.py --biigle-sync
python run_pipeline.py --retrain --data-prep --no-upload
```

ML and retraining also require the configured model and enough local disk. The
Zooniverse and BIIGLE commands perform real remote operations and require
credentials and contact with those services; do not run public uploads against
an account you do not control. The `--no-upload` flag only skips the final S3
sync. Use `--dry-run` with the validation upload stages when you only need to
render manifests locally.

## Tests and style

Run the unit suite before opening a pull request:

```bash
pytest tests/unit/
git diff --check
```

Keep changes focused, explain the user-visible reason in the pull request, and
include a regression test when changing pipeline behavior.
