#!/bin/bash -e
# Bulk-fetch every video of the two SLI surveys from S3 into a holding folder
# on NeSI, skipping files already there. Copied verbatim from the NeSI checkout
# on 2026-10-14, where it was written by hand and never committed.
#
# Why it exists: these surveys were restored from DEEP_ARCHIVE, and a plain
# `aws s3 cp` of a restored object fails without --force-glacier-transfer. The
# pipeline's S3Handler has no Glacier-aware download, so this is currently the
# only way to stage a restored survey before `--ml`.
#
# TODO (clean-up): take the survey ids as arguments, read the destination from
# config (media_base_dir), and fold the Glacier flag into S3Handler so the ML
# runner can fetch restored videos itself. Until then: edit the `for s in`
# line and `sbatch scripts/fetch_sli.sh`.
#SBATCH --job-name=fetch_sli
#SBATCH --account=wildlife03546
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/fetch_sli_%j.out
#SBATCH --error=/nesi/project/wildlife03546/spyfish-aotearoa-toolkit/slurm_logs/fetch_sli_%j.err
export PATH=/opt/nesi/bin:$PATH
DEST=/home/kalindi/00_nesi_projects/wildlife03546_nobackup/media_hold
mkdir -p "$DEST" && cd "$DEST"
for s in SLI_20220228_BUV SLI_20240124_BUV; do
  aws s3 ls "s3://marine-buv/media/$s/" --recursive | awk '{print $4}' | grep '\.mp4$' | while read -r key; do
    f=$(basename "$key")
    if [ -s "$f" ]; then echo "have $f"; continue; fi
    echo "fetch $key"
    aws s3 cp --force-glacier-transfer "s3://marine-buv/$key" "$f"
  done
done
echo DONE
