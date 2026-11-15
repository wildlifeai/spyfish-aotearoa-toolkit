#!/bin/bash -e
# Bulk-fetch every video of the given surveys from S3 into config.media_dir on
# NeSI, skipping files already there. Started as a hand-written NeSI script
# (copied in 2026-10-14) for the two SLI surveys.
#
# Why it exists: these surveys were restored from DEEP_ARCHIVE, and a plain
# `aws s3 cp` of a restored object fails without --force-glacier-transfer. The
# pipeline's S3Handler has no Glacier-aware download, so this is currently the
# only way to stage a restored survey before `--ml` or the review scripts.
#
#   sbatch scripts/fetch_sli.sh SLI_20240124_BUV [SLI_20220228_BUV ...]
#
# TODO (clean-up): fold the Glacier flag into S3Handler so the ML runner can
# fetch restored videos itself, and retire this script.
#SBATCH --job-name=fetch_sli
#SBATCH --account=wildlife03546
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/fetch_sli_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/fetch_sli_%j.err

if [ $# -eq 0 ]; then
    echo "fetch_sli.sh: pass survey IDs on the sbatch line, e.g. SLI_20240124_BUV" >&2
    exit 2
fi

source scripts/slurm/_env.sh

# Same folder the pipeline and the review scripts read (media_base_dir + media).
# tail: only the last line, in case importing config logs to stdout.
DEST=$(python -c "from spyfish.config.wrapper import config; print(config.media_dir)" | tail -n 1)
mkdir -p "$DEST" && cd "$DEST"
echo "fetching into $DEST"
for s in "$@"; do
  aws s3 ls "s3://marine-buv/media/$s/" --recursive | awk '{print $4}' | grep '\.mp4$' | while read -r key; do
    f=$(basename "$key")
    if [ -s "$f" ]; then echo "have $f"; continue; fi
    echo "fetch $key"
    aws s3 cp --no-progress --force-glacier-transfer "s3://marine-buv/$key" "$f"
  done
done
echo DONE
