# Video Management Documentation

This directory contains Jupyter notebooks and scripts for managing video files in the Spyfish Aotearoa project. These tools help you work with videos stored in AWS S3, including concatenating multi-part GoPro files, previewing videos, and renaming files in bulk.

## Available Tools

1. **`concat_existing_go_pro.ipynb`** - Concatenate multi-part GoPro video files
2. **`gopro_concat.py`** - Command-line version for HPC/batch processing
3. **`local_preview_movies_aws.ipynb`** - Preview videos from S3 in your browser
4. **`preview_movies_aws.ipynb`** - Alternative preview tool (legacy)
5. **`rename_files_in_aws_from_csv.ipynb`** - Bulk rename files using a CSV

## Quick Start Guide

### Prerequisites

All notebooks require:
```bash
pip install boto3 pandas python-dotenv jupyter
```

AWS credentials must be configured in `~/.env` file:
```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=ap-southeast-2
```

### Which Tool Should I Use?

| Task | Recommended Tool |
|------|------------------|
| Combine split GoPro videos | `concat_existing_go_pro.ipynb` or `gopro_concat.py` |
| Watch videos before downloading | `local_preview_movies_aws.ipynb` |
| Rename multiple files at once | `rename_files_in_aws_from_csv.ipynb` |
| Batch processing on HPC | `gopro_concat.py` |

---

## Tool 1: Concatenating GoPro Videos

### Using the Notebook: `concat_existing_go_pro.ipynb`

**Purpose:** Automatically find and combine multi-part GoPro video files into single videos.

#### Step-by-Step Instructions

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook concat_existing_go_pro.ipynb
   ```

2. **Import Required Libraries** (First cell)
   ```python
   from sftk.s3_handler import S3Handler
   from sftk.video_handler import VideoProcessor
   ```

3. **Configure Your Settings** (Second cell)
   ```python
   # --- Configuration ---
   S3_PREFIX = "media/HOR_20240408_BUV"  # Your survey folder
   GOPRO_PREFIX = "G"                     # Usually "G" for GoPro files
   DELETE_ORIGINALS = False               # Set True to delete parts after merging
   TEST_MODE = False                      # Set True to test on one drop only
   PARALLEL_DROPS = 2                     # Process 2 drops at once
   DOWNLOAD_THREADS = 4                   # 4 parallel downloads per drop
   SEQUENTIAL_DOWNLOAD = False            # Set True for slower, more stable downloads
   ```

4. **Initialize and Preview** (Still second cell)
   ```python
   # Initialize processor
   s3_handler = S3Handler()
   processor = VideoProcessor(
       s3_handler,
       prefix=S3_PREFIX,
       gopro_prefix=GOPRO_PREFIX,
       delete_originals=DELETE_ORIGINALS,
       test_mode=TEST_MODE,
       download_threads=DOWNLOAD_THREADS,
       parallel_drops=PARALLEL_DROPS,
       sequential_download=SEQUENTIAL_DOWNLOAD
   )
   
   # Preview files that will be processed
   display(processor.filtered_df)
   ```

5. **Run the Processing** (Third cell)
   ```python
   processor.process_gopro_videos()
   ```

6. **Optional: Clean Up Redundant Files** (Fourth cell onwards)
   - The notebook includes additional cells to find and remove individual files where a concatenated version already exists
   - Review the list before uncommenting the delete command

#### What Happens During Processing?

- **Discovery:** Finds all GoPro files in your specified S3 prefix
- **Grouping:** Groups files by drop and sequence number (e.g., GOPR0298, GP010298, GP020298)
- **Download:** Downloads video parts in parallel
- **Validation:** Checks each video file is valid
- **Concatenation:** Combines parts into single video files
- **Upload:** Uploads concatenated videos back to S3
- **Cleanup:** Removes local temporary files (and optionally S3 originals)

#### GoPro File Naming Explained

GoPro cameras split long recordings into multiple files:
- `GOPR0298.MP4` - First file in recording #0298
- `GP010298.MP4` - Second file in recording #0298  
- `GP020298.MP4` - Third file in recording #0298
- `GOPR0392.MP4` - New recording #0392 starts

The tool automatically detects these sequences and combines them.

#### Multiple Sequences in One Drop

If a drop folder contains multiple GoPro sequences, they are saved with suffix letters:
- Sequence 0298 → `DropID_A.mp4`
- Sequence 0392 → `DropID_B.mp4`
- Sequence 0425 → `DropID_C.mp4`

---

## Tool 2: Previewing Videos

### Using the Notebook: `local_preview_movies_aws.ipynb`

**Purpose:** Watch videos stored in S3 without downloading them to your computer.

#### Step-by-Step Instructions

1. **Launch the Notebook**
   ```bash
   jupyter notebook local_preview_movies_aws.ipynb
   ```

2. **Import Libraries** (First cell)
   ```python
   from sftk.s3_handler import S3Handler
   from sftk.video_handler import VideoProcessor
   ```

3. **Initialize Handlers** (Second cell)
   ```python
   s3_handler = S3Handler()
   processor = VideoProcessor(s3_handler)
   ```

4. **Preview a Specific Video** (Third cell)
   ```python
   movie_key = "media/HOR_20240408_BUV/HOR_20240408_BUV_HOR_096_01/HOR_20240408_BUV_HOR_096_01.mp4"
   display(processor.preview_movie(movie_key))
   ```

5. **List All Videos in a Folder** (Fourth cell)
   ```python
   S3_PREFIX = "media/HOR_20240408_BUV/HOR_20240408_BUV_HOR_096_01/"
   
   import pandas as pd
   pd.set_option('display.max_colwidth', None)  # Show full paths
   
   movies_df = processor.get_movies_df(prefix=S3_PREFIX)
   display(movies_df)
   ```

#### Tips for Previewing

- Videos stream directly from S3 - no download needed
- The preview URL expires after a few hours
- Use this to verify videos before processing
- Great for checking if concatenation worked correctly

---

## Tool 3: Bulk Renaming Files

### Using the Notebook: `rename_files_in_aws_from_csv.ipynb`

**Purpose:** Rename multiple video files in S3 using a CSV file with old and new names.

#### Step-by-Step Instructions

1. **Prepare Your CSV File**
   
   Create a CSV with two columns containing old and new S3 keys:
   ```csv
   OLD,NEW
   media/survey/drop1/video_old.mp4,media/survey/drop1/video_new.mp4
   media/survey/drop2/GOPR0123.mp4,media/survey/drop2/drop2_concatenated.mp4
   ```
   
   Save this as `rename_movies.csv` in your data folder.

2. **Launch the Notebook**
   ```bash
   jupyter notebook rename_files_in_aws_from_csv.ipynb
   ```

3. **Import Libraries** (First cell)
   ```python
   import pandas as pd
   import os
   from sftk.common import LOCAL_DATA_FOLDER_PATH, MOVIE_EXTENSIONS
   from sftk.s3_handler import S3Handler
   ```

4. **Connect to S3** (Second cell)
   ```python
   s3_handler = S3Handler()
   ```

5. **Configure File Paths** (Third cell)
   ```python
   file_path = os.path.join(LOCAL_DATA_FOLDER_PATH, "rename_movies.csv")
   new_name_column = "NEW"
   old_name_column = "OLD"
   ```

6. **Load and Preview Changes** (Fourth cell)
   ```python
   rename_csv_df = pd.read_csv(file_path)
   rename_pairs = dict(zip(rename_csv_df[old_name_column], rename_csv_df[new_name_column]))
   print(rename_pairs)  # Review what will be renamed
   ```

7. **Execute Renaming** (Fifth cell)
   ```python
   s3_handler.rename_s3_objects_from_dict(
       rename_pairs,
       suffixes=MOVIE_EXTENSIONS,
       try_run=True  # Set to False when ready to actually rename
   )
   ```

#### Important Notes

- **Always test first:** Use `try_run=True` to preview changes without making them
- **Backup important data:** Renaming cannot be easily undone
- **Check paths carefully:** A typo can rename the wrong file
- **Verify after renaming:** Use the preview notebook to confirm changes

#### Optional: Export S3 File List

The notebook includes a section to export all S3 paths to CSV:
```python
import csv

foo = s3_handler.get_file_paths_set_from_s3(prefix="media/AHE", suffixes=".mp4")

with open('file_paths_per_row.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["File Path"])
    for item in foo:
        writer.writerow([item])
```

This creates a list you can use to build your rename CSV.

---

## Command-Line Tool: `gopro_concat.py`

**Purpose:** Run GoPro concatenation from the command line or on HPC systems like NeSI.

### Local Usage

#### Prerequisites

```bash
# Install dependencies
pip install boto3 pandas python-dotenv

# Ensure FFmpeg is installed
ffmpeg -version
```

#### Basic Usage

```bash
# Process videos with auto-detected paths
python gopro_concat.py --prefix "media/BNP_20210127"
```

#### Full Options

```bash
python gopro_concat.py \
  --prefix "media/SURVEY_ID" \
  --gopro-prefix "G" \
  --parallel-drops 2 \
  --download-threads 4 \
  --delete-originals
```

#### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--prefix` | S3 prefix for video files (required) | - |
| `--gopro-prefix` | Prefix for GoPro files (G, GX, GH, etc.) | `G` |
| `--delete-originals` | Delete original parts from S3 after concatenation | `False` |
| `--test-mode` | Process only last drop, skip upload | `False` |
| `--parallel-drops` | Number of drops to process simultaneously | `1` |
| `--download-threads` | Parallel downloads per drop | `4` |
| `--sequential-download` | Download files one at a time | `False` |
| `--use-nesi-ffmpeg` | Load FFmpeg from NeSI modules | `False` |
| `--toolkit-path` | Path to toolkit (auto-detected if omitted) | Auto |

#### Get Help

```bash
python gopro_concat.py --help
```

---

## Running on NeSI HPC

### Setup Virtual Environment (One-Time)

```bash
# Load Python module
module load Python/3.11.3-gimkl-2022a

# Create virtual environment
python -m venv ~/venvs/spyfish-env

# Activate environment
source ~/venvs/spyfish-env/bin/activate

# Install dependencies
pip install boto3 pandas python-dotenv
```

### Method 1: Interactive Session

```bash
# Request interactive resources
salloc --time=2:00:00 --mem=8G --cpus-per-task=4

# Load modules
module load Python/3.11.3-gimkl-2022a
module load FFmpeg/4.2.2-GCCcore-9.2.0

# Activate environment
source ~/venvs/spyfish-env/bin/activate

# Run script
cd /nesi/project/wildlife03546/Spyfish-Aotearoa-toolkit/video_management
python gopro_concat.py \
  --prefix "media/BNP_20210127" \
  --parallel-drops 2
```

### Method 2: Batch Job (Recommended)

Create a Slurm script `run_gopro_concat.sl`:

```bash
#!/bin/bash -e
#SBATCH --job-name=gopro_concat
#SBATCH --account=wildlife03546        # Your NeSI project
#SBATCH --time=24:00:00                # Max runtime
#SBATCH --mem=16G                      # Memory allocation
#SBATCH --cpus-per-task=4              # CPU cores
#SBATCH --output=logs/gopro_%j.log    # Output log file
#SBATCH --error=logs/gopro_%j.err     # Error log file

# Load required modules
module purge
module load Python/3.11.3-gimkl-2022a
module load FFmpeg/4.2.2-GCCcore-9.2.0

# Activate virtual environment
source $HOME/venvs/spyfish-env/bin/activate

# Navigate to script directory
cd /nesi/project/wildlife03546/Spyfish-Aotearoa-toolkit/video_management

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the concatenation script
python gopro_concat.py \
  --prefix "media/BNP_20210127" \
  --parallel-drops 2 \
  --download-threads 4 \
  --delete-originals

echo "Processing complete at $(date)"
```

Submit the job:

```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch run_gopro_concat.sl

# Check job status
squeue -u $USER

# View output
tail -f logs/gopro_<jobid>.log
```

### Method 3: Processing Multiple Surveys

Create `run_multiple_surveys.sl`:

```bash
#!/bin/bash -e
#SBATCH --job-name=gopro_multi
#SBATCH --account=wildlife03546
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gopro_multi_%j.log

module purge
module load Python/3.11.3-gimkl-2022a
module load FFmpeg/4.2.2-GCCcore-9.2.0

source $HOME/venvs/spyfish-env/bin/activate
cd /nesi/project/wildlife03546/Spyfish-Aotearoa-toolkit/video_management

# List of surveys to process
SURVEYS=(
  "media/BNP_20210127"
  "media/HOR_20240408_BUV"
  "media/SURVEY_20250101"
)

# Process each survey
for SURVEY in "${SURVEYS[@]}"; do
  echo "=========================================="
  echo "Processing survey: $SURVEY"
  echo "=========================================="
  
  python gopro_concat.py \
    --prefix "$SURVEY" \
    --parallel-drops 2 \
    --download-threads 4
  
  if [ $? -eq 0 ]; then
    echo "✓ Successfully processed $SURVEY"
  else
    echo "✗ Failed to process $SURVEY"
  fi
  echo ""
done

echo "All surveys processed at $(date)"
```

---

## Common Workflows

### Workflow 1: Processing a New Survey

1. **First, preview what videos exist:**
   ```python
   # In local_preview_movies_aws.ipynb
   movies_df = processor.get_movies_df(prefix="media/YOUR_SURVEY/")
   display(movies_df)
   ```

2. **Then concatenate GoPro files:**
   ```python
   # In concat_existing_go_pro.ipynb
   S3_PREFIX = "media/YOUR_SURVEY"
   TEST_MODE = True  # Test with one drop first
   # ... run processing
   ```

3. **Verify results:**
   ```python
   # In local_preview_movies_aws.ipynb
   movie_key = "media/YOUR_SURVEY/DROP_ID/DROP_ID_A.mp4"
   display(processor.preview_movie(movie_key))
   ```

4. **Clean up if successful:**
   ```python
   # In concat_existing_go_pro.ipynb
   DELETE_ORIGINALS = True
   TEST_MODE = False
   # ... run processing again
   ```

### Workflow 2: Fixing Incorrectly Named Files

1. **Export current file list:**
   ```python
   # In rename_files_in_aws_from_csv.ipynb
   foo = s3_handler.get_file_paths_set_from_s3(prefix="media/YOUR_SURVEY/")
   # ... export to CSV
   ```

2. **Edit CSV in Excel/Sheets:**
   - Add OLD and NEW columns
   - Fill in correct names

3. **Preview and execute renames:**
   ```python
   # Still in rename_files_in_aws_from_csv.ipynb
   s3_handler.rename_s3_objects_from_dict(rename_pairs, try_run=True)
   # Review output, then set try_run=False
   ```

### Workflow 3: Batch Processing on NeSI

1. **Test locally first:**
   ```python
   # In concat_existing_go_pro.ipynb with TEST_MODE = True
   ```

2. **Create Slurm job:**
   ```bash
   # Use gopro_concat.py with --use-nesi-ffmpeg
   ```

3. **Monitor and verify:**
   ```bash
   tail -f logs/gopro_<jobid>.log
   ```

---

## Troubleshooting

### "Cannot connect to S3"
- Check your `~/.env` file exists and contains valid AWS credentials
- Verify credentials: `cat ~/.env`
- Ensure you have network access to AWS

### "Module sftk not found"
- Make sure you're running notebooks from the correct directory
- The `sftk` package should be in the parent directory
- Check with: `import sys; print(sys.path)`

### "FFmpeg not found"
- On local machine: Install FFmpeg (`brew install ffmpeg` on Mac, or download from ffmpeg.org)
- On NeSI: Use `--use-nesi-ffmpeg` flag or load module manually

### Videos won't play in preview
- Preview URLs expire after a few hours - regenerate them
- Check browser console for errors
- Verify the file exists in S3 with correct permissions

### Concatenation fails partway through
- Check available disk space (needs 2x video size)
- Verify all source files are valid videos
- Try with `SEQUENTIAL_DOWNLOAD = True` for more stability
- Review error messages for specific file issues

### Renaming affects wrong files
- **Always use `try_run=True` first!**
- Double-check your CSV column names match configuration
- Verify paths are complete S3 keys, not just filenames

---

## Best Practices

1. **Always test first:**
   - Use `TEST_MODE = True` for concatenation
   - Use `try_run=True` for renaming
   - Preview one video before processing hundreds

2. **Keep backups:**
   - Don't delete originals immediately
   - Verify concatenated videos play correctly
   - Export file lists before bulk operations

3. **Work in stages:**
   - Process one survey at a time
   - Start with small `PARALLEL_DROPS` values
   - Increase parallelization once stable

4. **Monitor resources:**
   - Watch disk space during processing
   - Check memory usage on HPC
   - Keep log files for debugging

5. **Document changes:**
   - Save CSV files used for renaming
   - Keep notes on which surveys were processed
   - Record any issues encountered

---

## Performance Tips

### Local Processing
- **Fast internet:** Increase `DOWNLOAD_THREADS` to 8
- **Limited bandwidth:** Use `SEQUENTIAL_DOWNLOAD = True`
- **Many drops:** Increase `PARALLEL_DROPS` to 3-4
- **Low memory:** Process one drop at a time (`PARALLEL_DROPS = 1`)

### HPC Processing
```bash
# For large surveys (100+ drops):
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
--parallel-drops 4 --download-threads 8

# For small surveys (< 20 drops):
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
--parallel-drops 1 --download-threads 4
```

---

## Monitoring Jobs on NeSI

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# View job details
scontrol show job <jobid>

# Cancel a job
scancel <jobid>
```

### View Logs

```bash
# Watch live output
tail -f logs/gopro_<jobid>.log

# Search for errors
grep -i error logs/gopro_<jobid>.log

# Check completed jobs
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed
```

---

## Contact

For issues or questions, contact the Spyfish Aotearoa team or submit an issue on the repository.