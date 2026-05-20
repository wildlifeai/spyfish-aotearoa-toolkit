import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager, Queue
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from IPython.display import HTML

from sftk.s3_handler import ProgressTracker, S3Handler

logger = logging.getLogger(__name__)


def _extract_gopro_sequence_id(
    filename: str, gopro_prefix: str = "GX"
) -> Optional[str]:
    """
    Extract the sequence ID from a filename. Files with _1, _2, etc. are parts
    of the SAME sequence and should be grouped together.

    Supports multiple naming patterns:
        Underscore format (primary):
        - S2-2 14012022_1.MP4 -> "S2-2 14012022" (parts 1,2,etc. belong together)
        - S2-1 14012022_2.MP4 -> "S2-1 14012022" (parts 1,2,etc. belong together)
        - TUK_1.MP4 -> "TUK" (parts 1,2,etc. belong together)

        Standard GoPro (fallback):
        - GOPR0298.MP4 -> "0298"
        - GP010298.MP4 -> "0298"
        - GX010425.MP4 -> "0425"

    Args:
        filename: The filename to parse
        gopro_prefix: The prefix to look for (e.g., "TUK", "Orau_", "GX")

    Returns:
        The sequence ID (base name without part number), or None if not a valid format
    """
    # Primary pattern: Extract everything BEFORE _<number>.MP4
    # This groups files with _1, _2, _3 etc. as the same sequence
    pattern = r"^(.+)_(\d+)\.MP4$"
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        # Return the base name without the part number
        # e.g., "S2-2 14012022_1.MP4" -> "S2-2 14012022"
        return match.group(1)
    
    # Fallback to standard GoPro format if underscore pattern not found
    # Match GoPro pattern: GOPR or G[A-Z][0-9][0-9] followed by 4 digits
    gopro_pattern = rf"^(GOPR|G[A-Z]\d{{2}})(\d{{4}})\.MP4$"
    gopro_match = re.match(gopro_pattern, filename, re.IGNORECASE)
    if gopro_match:
        return gopro_match.group(2)  # Return the 4-digit sequence ID
    
    return None


def _group_videos_by_sequence(
    keys_with_sizes: List[tuple], gopro_prefix: str = "GX"
) -> Dict[str, List[tuple]]:
    """
    Group video files by their sequence ID.

    Args:
        keys_with_sizes: List of (key, size) tuples
        gopro_prefix: The prefix to use for grouping (e.g., "GX", "Orau_")

    Returns:
        Dictionary mapping sequence_id -> list of (key, size) tuples
    """
    sequences = {}

    for key, size in keys_with_sizes:
        filename = Path(key).name
        seq_id = _extract_gopro_sequence_id(filename, gopro_prefix)

        if seq_id:
            if seq_id not in sequences:
                sequences[seq_id] = []
            sequences[seq_id].append((key, size))
        else:
            # If we can't extract sequence ID, treat as its own sequence
            logger.warning(
                f"Could not extract sequence ID from {filename}, treating as separate sequence"
            )
            sequences[filename] = [(key, size)]

    return sequences


def _check_disk_space(path: Path, required_gb: float = 10) -> bool:
    """Check if sufficient disk space is available."""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < required_gb:
        logger.error(
            f"❌ Insufficient disk space: {free_gb:.1f} GB free, {required_gb:.1f} GB required"
        )
        return False
    logger.info(f"✓ Disk space check passed: {free_gb:.1f} GB free")
    return True


def _find_ffmpeg() -> Optional[str]:
    """Find ffmpeg executable."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    possible_paths = [
        Path.cwd() / "ffmpeg.exe",
        Path.cwd() / "bin" / "ffmpeg.exe",
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)
    return None


def _worker_log_config(log_queue: Queue) -> None:
    """Configure logging for a worker process to send logs to a queue."""
    # Clear existing handlers first
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Now add the queue handler
    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)

    # Suppress noisy S3Handler initialization logs in workers
    # This must be done AFTER creating the handler
    logging.getLogger("sftk.s3_handler").setLevel(logging.WARNING)
    logging.getLogger("s3_handler").setLevel(logging.WARNING)  # Try both module names

    # Suppress noisy S3Handler initialization logs in workers
    s3_logger = logging.getLogger("sftk.s3_handler")
    s3_logger.setLevel(logging.WARNING)  # Only show warnings and errors from S3Handler


def _download_with_timeout(
    s3_handler, key: str, local_path: str, s3_size: int, timeout: int = 3600
) -> bool:
    """Wrapper to download with progress tracking and timeout protection."""
    try:
        filename = Path(key).name
        logger.info(f"[DOWNLOAD-START] {filename} ({s3_size / (1024*1024):.1f} MB)")

        # Create progress tracker
        progress = ProgressTracker(filename, s3_size, log_interval=10.0)

        # Download with progress callback
        result = s3_handler.download_object_from_s3(key, local_path, callback=progress)

        if result:
            progress.complete()
        else:
            logger.error(f"[DOWNLOAD-FAILED] {filename}")

        return result

    except Exception as e:
        logger.error(f"[DOWNLOAD-ERROR] {Path(key).name}: {e}")
        return False


def _should_download_file(
    local_path: Path, s3_size: int, tolerance: float = 0.01
) -> bool:
    """
    Check if a file needs to be downloaded by comparing local and S3 file sizes.

    Args:
        local_path: Path to the local file
        s3_size: Size of the file in S3 (in bytes)
        tolerance: Acceptable size difference as a fraction (default 1%)

    Returns:
        True if file should be downloaded, False if local file is valid
    """
    if not local_path.exists():
        return True

    local_size = local_path.stat().st_size

    # Check if file is empty
    if local_size == 0:
        logger.warning(f"Local file {local_path.name} is empty, will re-download")
        return True

    # Check if sizes match within tolerance
    if s3_size > 0:
        size_diff = abs(local_size - s3_size) / s3_size
        if size_diff <= tolerance:
            logger.info(
                f"✓ File {local_path.name} already exists with correct size, skipping download"
            )
            return False
        else:
            logger.warning(
                f"⚠ Local file {local_path.name} size mismatch "
                f"(local: {local_size:,} bytes, S3: {s3_size:,} bytes), will re-download"
            )

    return True


def _verify_video_file_deep(file_path: Path) -> bool:
    """Deep verification using ffprobe."""
    # Check if ffprobe exists
    if not shutil.which("ffprobe"):
        logger.warning(
            f"ffprobe not found, skipping deep verification for {file_path.name}"
        )
        # Fall back to basic file size check
        return file_path.exists() and file_path.stat().st_size > 0

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=True
        )
        data = json.loads(result.stdout)
        if "format" in data and "streams" in data:
            logger.debug(f"✓ Video file verified: {file_path.name}")
            return True
        logger.error(f"✗ Invalid video format for {file_path.name}")
        return False
    except subprocess.CalledProcessError as e:
        # Exit code 127 specifically means command not found
        if e.returncode == 127:
            logger.error(f"ffprobe command not found - cannot verify {file_path.name}")
            # Optionally: return True here if you trust your downloads
            return file_path.exists() and file_path.stat().st_size > 0
        logger.error(f"✗ Error verifying video file {file_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error verifying video file {file_path.name}: {e}")
        return False


def _try_repair_with_untrunc(
    corrupted_path: Path, temp_dir: Path, all_video_paths: List[Path]
) -> Optional[Path]:
    """Repair using untrunc - excellent for missing moov atoms."""
    repaired_path = temp_dir / f"untrunc_{corrupted_path.name}"

    # Find a valid reference video from the same drop
    reference_video = None
    for path in all_video_paths:
        if path != corrupted_path and _verify_video_file_deep(path):
            reference_video = path
            break

    if not reference_video:
        logger.error("✗ No reference video found for untrunc repair")
        return None

    try:
        logger.info("🔧 Attempting untrunc repair...")
        cmd = ["untrunc", str(reference_video), str(corrupted_path)]
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)

        # untrunc creates a file with _fixed suffix
        fixed_file = (
            corrupted_path.parent
            / f"{corrupted_path.stem}_fixed{corrupted_path.suffix}"
        )
        if fixed_file.exists():
            shutil.move(str(fixed_file), str(repaired_path))
            if _verify_video_file_deep(repaired_path):
                logger.info("✅ untrunc repair successful!")
                return repaired_path

        logger.warning("⚠ untrunc ran but did not produce a valid output")
        return None
    except Exception as e:
        logger.warning(f"⚠ untrunc repair failed: {e}")
        return None


def _try_repair_video(
    corrupted_path: Path, temp_dir: Path, all_video_paths: List[Path]
) -> Optional[Path]:
    """Attempts to repair a corrupted video file using multiple strategies."""
    logger.warning(f"⚠ Video file corrupted: {corrupted_path.name}")

    # Strategy 1: untrunc (best for missing moov atom)
    if shutil.which("untrunc"):
        repaired = _try_repair_with_untrunc(corrupted_path, temp_dir, all_video_paths)
        if repaired:
            return repaired

    # Other strategies could be added here

    logger.error(f"❌ All repair strategies failed for {corrupted_path.name}")
    return None


def _concatenate_videos(
    video_paths: List[Path], output_path: Path, ffmpeg_path: str
) -> tuple[bool, str]:
    """Concatenate multiple videos, attempting to repair any corrupted files first."""
    if not video_paths:
        error_msg = "No video paths provided for concatenation."
        logger.error(f"❌ {error_msg}")
        return False, error_msg

    list_file = None
    temp_dir = Path(tempfile.mkdtemp(prefix="video_repair_"))

    try:
        videos_to_concat = []
        needs_repair = []

        # First pass: identify files needing repair
        for path in video_paths:
            if not path.exists():
                error_msg = f"Input file '{path.name}' does not exist."
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            if path.stat().st_size == 0:
                error_msg = f"Input file '{path.name}' is empty (0 bytes)."
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            if _verify_video_file_deep(path):
                videos_to_concat.append(path)
            else:
                needs_repair.append(path)

        # Second pass: attempt repairs
        if needs_repair:
            logger.warning(f"⚠ {len(needs_repair)} file(s) need repair")
            for corrupted_path in needs_repair:
                repaired_path = _try_repair_video(corrupted_path, temp_dir, video_paths)
                if repaired_path:
                    videos_to_concat.append(repaired_path)
                else:
                    error_msg = f"Could not repair '{corrupted_path.name}'. Aborting concatenation."
                    logger.critical(f"❌ {error_msg}")
                    return False, error_msg

        if not videos_to_concat:
            error_msg = (
                "No valid videos available to concatenate after verification/repair."
            )
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        # Store file list in temp directory
        list_file = temp_dir / "file_list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for path in videos_to_concat:
                line = f"file '{path.resolve().as_posix()}'\n"
                f.write(line)

        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output_path),
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            timeout=3600,
        )

        elapsed = time.time() - start_time
        logger.info(f"✅ Concatenation successful! Took {elapsed:.2f} seconds")

        if not output_path.exists() or output_path.stat().st_size == 0:
            error_msg = f"Output file '{output_path.name}' is missing or empty after concatenation."
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        return True, "Concatenation successful."

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = getattr(e, "stderr", "N/A")
        error_msg = (
            f"CONCATENATION FAILED for {output_path.name}: {e}\n"
            f"--- FFMPEG STDERR ---\n{stderr}\n--- END FFMPEG STDERR ---"
        )
        logger.error(f"❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during concatenation: {e}"
        logger.error(f"❌ {error_msg}")
        return False, error_msg
    finally:
        # Clean up temporary files and directory
        if list_file and list_file.exists():
            list_file.unlink()
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _upload_and_cleanup_s3(
    s3_handler: S3Handler, output_path: Path, upload_data: dict, delete_originals: bool
) -> None:
    """Upload concatenated video and delete original parts from S3 if configured."""
    drop_id = upload_data["DropID"]
    survey_id = upload_data["SurveyID"]
    sequence_suffix = upload_data.get("sequence_suffix", "")
    new_key = f"media/{survey_id}/{drop_id}/{drop_id}{sequence_suffix}.mp4"

    logger.info(f"⬆️ Uploading to S3: {new_key}")
    file_size = output_path.stat().st_size

    upload_progress = ProgressTracker(
        filename=output_path.name, total_size=file_size, log_interval=10.0
    )

    s3_handler.upload_file_to_s3(
        str(output_path),
        new_key,
        callback=upload_progress,
    )

    upload_progress.complete()

    if delete_originals:
        logger.info(f"🗑️ Deleting {len(upload_data['Key'])} original file(s) from S3...")
        deleted_count = 0
        failed_count = 0
        for key in upload_data["Key"]:
            try:
                s3_handler.s3.delete_object(Bucket=s3_handler.bucket, Key=key)
                deleted_count += 1
                logger.info(f"   ✓ Deleted from S3: {Path(key).name}")
            except Exception as e:
                failed_count += 1
                logger.error(f"   ✗ Failed to delete {Path(key).name}: {e}")

        if failed_count > 0:
            logger.warning(
                f"⚠ Deleted {deleted_count}/{len(upload_data['Key'])} files ({failed_count} failed)"
            )
        else:
            logger.info(
                f"✅ Successfully deleted all {deleted_count} original files from S3"
            )
    else:
        logger.info("ℹ️ Keeping original files in S3 (delete_originals=False)")


def _cleanup_local_files(
    downloaded_files: List[Path],
    output_paths: List[Path],
    drop_specific_download_dir: Optional[Path],
    drop_id: str,
    test_mode: bool,
) -> None:
    """Clean up local files after processing."""
    if not test_mode:
        logger.info(f"🗑️ Cleaning up local files for {drop_id}...")
        deleted_count = 0
        for file_path in downloaded_files:
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"   ✗ Could not delete {file_path.name}: {e}")

        logger.info(
            f"   Deleted {deleted_count}/{len(downloaded_files)} local video file(s)"
        )

        if drop_specific_download_dir and drop_specific_download_dir.exists():
            try:
                drop_specific_download_dir.rmdir()
                logger.info(
                    f"   ✓ Removed drop directory: {drop_specific_download_dir.name}"
                )
            except OSError:
                logger.debug(
                    f"   Could not remove {drop_specific_download_dir.name} (not empty)"
                )
    else:
        logger.info(
            f"🧪 Test mode: Keeping {len(downloaded_files)} downloaded file(s) for {drop_id}"
        )

    if not test_mode:
        for output_path in output_paths:
            if output_path and output_path.exists():
                try:
                    output_path.unlink()
                    logger.info(f"   ✓ Deleted local output file: {output_path.name}")
                except Exception as e:
                    logger.warning(
                        f"   ✗ Could not delete output file {output_path.name}: {e}"
                    )


def _process_single_drop(
    drop_data_dict: dict,
    download_dir: str,
    output_dir: str,
    delete_originals: bool,
    test_mode: bool,
    max_workers: int,
    sequential_download: bool,
    ffmpeg_path: str,
    gopro_prefix: str,
    log_queue: Optional[Queue] = None,
) -> None:
    """
    Process a single drop's worth of videos, handling multiple sequences within the drop.
    """
    if log_queue:
        _worker_log_config(log_queue)

    # Get logger AFTER configuring it
    process_logger = logging.getLogger(__name__)

    drop_id = drop_data_dict.get("drop_id", "UNKNOWN")
    keys = drop_data_dict["keys"]
    sizes = drop_data_dict["sizes"]
    survey_id = drop_data_dict["survey_id"]

    process_logger.info(f"{'='*60}")
    process_logger.info(f"🎬 Processing DropID: {drop_id}")
    process_logger.info(f"   Files to process: {len(keys)}")
    process_logger.info(f"{'='*60}")

    # Suppress S3Handler logging before creating it
    logging.getLogger("sftk.s3_handler").setLevel(logging.WARNING)
    logging.getLogger("s3_handler").setLevel(logging.WARNING)

    # Create S3Handler in worker process (unavoidable for multiprocessing)
    from sftk.s3_handler import S3Handler

    s3_handler = S3Handler()
    process_logger.debug(f"Initialized S3Handler for {drop_id}")

    downloaded_files = []
    output_paths = []
    drop_download_dir: Optional[Path] = None

    # Check disk space before downloading
    total_size_gb = sum(sizes) / (1024**3)
    if not _check_disk_space(Path(download_dir), required_gb=total_size_gb * 1.5):
        raise RuntimeError(f"Insufficient disk space for {drop_id}")

    def _download_videos_parallel(keys_with_sizes: list) -> List[Path]:
        """Download all videos for a drop in parallel."""
        downloaded_files_local = []
        futures = {}
        files_to_download = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if drop_download_dir is None:
                raise ValueError(
                    "drop_download_dir must be set before parallel download."
                )

            for key, s3_size in keys_with_sizes:
                local_path = drop_download_dir / Path(key).name

                # Check if download is needed
                if not _should_download_file(local_path, s3_size):
                    downloaded_files_local.append(local_path)
                    continue

                files_to_download += 1
                process_logger.debug(
                    f"[DOWNLOAD] Submitting: {Path(key).name} ({s3_size / (1024*1024):.1f} MB)"
                )
                future = executor.submit(
                    _download_with_timeout,
                    s3_handler,
                    key,
                    str(local_path),
                    s3_size,
                    3600,
                )

                futures[future] = (key, local_path, time.time())

            if files_to_download > 0:
                process_logger.info(
                    f"⬇ Downloading {files_to_download} file(s) in parallel..."
                )

            completed = 0
            for future in as_completed(futures, timeout=3600):
                key, local_path, start_time = futures[future]
                completed += 1
                try:
                    result = future.result()
                    if result:
                        downloaded_files_local.append(local_path)
                        elapsed = time.time() - start_time
                        file_size_mb = (
                            local_path.stat().st_size / (1024 * 1024)
                            if local_path.exists()
                            else 0
                        )
                        process_logger.info(
                            f"✓ Downloaded: {local_path.name} ({file_size_mb:.1f} MB in {elapsed:.1f}s) "
                            f"[{completed}/{files_to_download}]"
                        )
                    else:
                        process_logger.error(
                            f"✗ Download failed for {Path(key).name} (returned False)"
                        )
                except TimeoutError as e:
                    process_logger.error(
                        f"✗ Download timeout for {Path(key).name}: {e}"
                    )
                except Exception as e:
                    process_logger.error(
                        f"✗ Download exception for {Path(key).name}: {e}"
                    )

        return sorted(downloaded_files_local)

    try:
        # Create a drop-specific download directory
        base_download_dir = Path(download_dir)
        drop_download_dir = base_download_dir / drop_id
        drop_download_dir.mkdir(parents=True, exist_ok=True)

        # Create list of (key, size) tuples
        keys_with_sizes = list(zip(keys, sizes))

        # Group videos by sequence
        sequences = _group_videos_by_sequence(keys_with_sizes, gopro_prefix)

        process_logger.info(
            f"📊 Found {len(sequences)} video sequence(s) in drop {drop_id}"
        )

        # Sort sequences by their ID to ensure consistent ordering
        sorted_sequences = sorted(sequences.items())

        # Download all files (once for all sequences)
        if not sequential_download:
            downloaded_files = _download_videos_parallel(keys_with_sizes)
        else:
            downloaded_files = []
            files_to_download = sum(
                1
                for key, s3_size in keys_with_sizes
                if _should_download_file(drop_download_dir / Path(key).name, s3_size)
            )

            if files_to_download > 0:
                process_logger.info(
                    f"⬇ Downloading {files_to_download} file(s) sequentially..."
                )

            completed = 0
            for key, s3_size in keys_with_sizes:
                local_path = drop_download_dir / Path(key).name

                # Check if download is needed
                if not _should_download_file(local_path, s3_size):
                    downloaded_files.append(local_path)
                    continue

                completed += 1
                start_time = time.time()
                if s3_handler.download_object_from_s3(key, str(local_path)):
                    downloaded_files.append(local_path)
                    elapsed = time.time() - start_time
                    file_size_mb = (
                        local_path.stat().st_size / (1024 * 1024)
                        if local_path.exists()
                        else 0
                    )
                    process_logger.info(
                        f"✓ Downloaded: {local_path.name} ({file_size_mb:.1f} MB in {elapsed:.1f}s) "
                        f"[{completed}/{files_to_download}]"
                    )
                else:
                    process_logger.error(
                        f"✗ Sequential download failed for {Path(key).name}"
                    )

            downloaded_files = sorted(downloaded_files)

        if not downloaded_files:
            raise RuntimeError("No files were successfully downloaded or found locally")

        # Validate all downloaded files
        process_logger.info(f"🔍 Validating {len(downloaded_files)} file(s)...")
        for f in downloaded_files:
            if not f.exists():
                raise RuntimeError(f"File {f.name} does not exist after download")
            if f.stat().st_size == 0:
                raise RuntimeError(f"File {f.name} is empty (0 bytes)")

        # Process each sequence separately
        for seq_index, (seq_id, seq_files) in enumerate(sorted_sequences):
            # Determine suffix (_A, _B, _C, etc.)
            if len(sorted_sequences) > 1:
                suffix = f"_{chr(65 + seq_index)}"  # 65 is ASCII for 'A'
            else:
                suffix = ""

            output_path = Path(output_dir) / f"{drop_id}{suffix}.mp4"
            output_paths.append(output_path)

            # Check if output already exists
            if output_path.exists():
                process_logger.warning(
                    f"⚠ Output file {output_path.name} already exists, skipping"
                )
                continue

            # Get local paths for this sequence
            seq_local_paths = []
            for key, size in seq_files:
                local_path = drop_download_dir / Path(key).name
                if local_path in downloaded_files:
                    seq_local_paths.append(local_path)

            seq_local_paths = sorted(seq_local_paths)

            process_logger.info(
                f"🎞️ Concatenating sequence {seq_id} ({len(seq_local_paths)} files) -> {output_path.name}"
            )
            concatenation_success, concatenation_message = _concatenate_videos(
                seq_local_paths, output_path, ffmpeg_path
            )

            if not concatenation_success:
                raise RuntimeError(
                    f"Video concatenation failed for sequence {seq_id}: {concatenation_message}"
                )

            if not test_mode:
                # Get keys for this sequence
                seq_keys = [key for key, size in seq_files]

                upload_data = {
                    "DropID": drop_id,
                    "SurveyID": survey_id,
                    "Key": seq_keys,
                    "sequence_suffix": suffix,
                }
                _upload_and_cleanup_s3(
                    s3_handler, output_path, upload_data, delete_originals
                )
            else:
                process_logger.info(
                    f"🧪 Test mode: Skipping S3 upload for {output_path.name}"
                )

    except Exception as e:
        process_logger.error(f"❌ Failed to process drop {drop_id}: {e}")
        raise
    finally:
        if drop_download_dir:
            _cleanup_local_files(
                downloaded_files, output_paths, drop_download_dir, drop_id, test_mode
            )


class VideoProcessor:
    """
    A class to handle video processing tasks like verification, repair, and concatenation.
    """

    MOVIE_EXTENSIONS = {".wmv", ".mpg", ".mov", ".avi", ".mp4", ".MOV", ".MP4"}

    def __init__(
        self,
        s3_handler: S3Handler,
        prefix: str = "",
        gopro_prefix: str = "GX",
        delete_originals: bool = False,
        test_mode: bool = True,
        download_threads: int = 4,  # Threads for downloading files within a drop
        parallel_drops: int = 3,  # Number of drops to process simultaneously
        sequential_download: bool = False,
        _skip_init_logs: bool = False,
    ):
        self.s3_handler = s3_handler
        self.prefix = prefix.strip()
        self.gopro_prefix = gopro_prefix
        self.delete_originals = delete_originals
        self.test_mode = test_mode
        self.download_threads = download_threads
        self.parallel_drops = parallel_drops
        self.sequential_download = sequential_download
        self._skip_init_logs = _skip_init_logs  # Store the flag

        # Suppress S3Handler logs in worker processes
        if _skip_init_logs:
            logging.getLogger("sftk.s3_handler").setLevel(logging.WARNING)

        if not _skip_init_logs:
            logger.info(f"{'='*80}")
            logger.info("🎥 VideoProcessor Initialization")
            logger.info(f"{'='*80}")
            logger.info(f"   Prefix: '{prefix}'")
            logger.info(f"   GoPro prefix: '{gopro_prefix}'")
            logger.info(f"   Test mode: {test_mode}")
            logger.info(f"   Parallel drops: {parallel_drops}")
            logger.info(f"   Download threads: {download_threads}")
            logger.info(f"   Sequential download: {sequential_download}")
            logger.info(f"   Delete originals: {delete_originals}")
            logger.info(f"{'='*80}")

        self.download_dir = Path("downloaded_movies")
        self.output_dir = Path("concatenated_videos")

        self.download_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.ffmpeg_path = _find_ffmpeg()
        if not self.ffmpeg_path:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH."
            )
        else:
            logger.info(f"✓ Found ffmpeg at: {self.ffmpeg_path}")

        # Pre-fetch and filter movies to provide upfront information
        logger.info(f"🔍 Fetching movie list from S3 with prefix: '{self.prefix}'")
        self.movies_df = self.get_movies_df(prefix=self.prefix)
        logger.info(
            f"📁 Found {len(self.movies_df):,} movie files with the specified prefix"
        )

        movies_df_with_parts = _add_path_parts_to_df(self.movies_df)
        self.filtered_df = get_filtered_movies_df(
            movies_df=movies_df_with_parts.copy(), gopro_prefix=self.gopro_prefix
        )

        num_drops = (
            self.filtered_df["DropID"].nunique() if not self.filtered_df.empty else 0
        )
        logger.info(f"🎬 Found {num_drops} drop(s) that require concatenation")
        logger.info(f"{'='*80}\n")

    def process_gopro_videos(self) -> None:
        """Process GoPro videos by DropID."""
        if self.filtered_df.empty:
            logger.info("ℹ️  No GoPro videos found that require concatenation")
            return

        # CRITICAL: Use drop_duplicates on DropID to ensure each drop appears only once
        unique_drops_df = self.filtered_df.drop_duplicates(subset=["DropID"])
        drop_ids = unique_drops_df["DropID"].unique()

        logger.info(f"🔍 Found {len(drop_ids)} unique drop IDs to process")

        valid_drops = []
        valid_drop_ids_set = set()  # Extra safety: track which drops we've added

        for drop_id in drop_ids:
            # Double-check we haven't added this drop already
            if drop_id in valid_drop_ids_set:
                logger.warning(f"⚠ Skipping duplicate DropID in loop: {drop_id}")
                continue

            # Get all rows for this drop_id from the ORIGINAL filtered_df
            drop_data = self.filtered_df[self.filtered_df["DropID"] == drop_id]

            # Verify all files start with gopro_prefix
            if all(str(name).startswith(self.gopro_prefix) for name in drop_data["fileName"]):  # type: ignore
                # Convert DataFrame to dict for pickling, include sizes
                drop_dict = {
                    "keys": drop_data["Key"].tolist(),
                    "sizes": drop_data["Size"].tolist(),
                    "drop_id": drop_data["DropID"].iloc[0],
                    "survey_id": drop_data["SurveyID"].iloc[0],
                }
                valid_drops.append((drop_id, drop_dict))
                valid_drop_ids_set.add(drop_id)  # Mark as added
            else:
                logger.warning(
                    f"⚠ Skipping DropID {drop_id}: Not all videos start with {self.gopro_prefix}"
                )

        if self.test_mode and valid_drops:
            valid_drops = valid_drops[-1:]
            logger.info("🧪 Test mode enabled: Processing only the last drop")

        # FINAL SAFETY CHECK: Verify no duplicates in valid_drops
        drop_id_list = [drop_id for drop_id, _ in valid_drops]
        unique_drop_ids = set(drop_id_list)

        if len(drop_id_list) != len(unique_drop_ids):
            from collections import Counter

            counts = Counter(drop_id_list)
            duplicates = {item: count for item, count in counts.items() if count > 1}

            logger.error(
                "❌ CRITICAL ERROR: Duplicate drop IDs detected in valid_drops!"
            )
            logger.error(f"   Total drops in list: {len(drop_id_list)}")
            logger.error(f"   Unique drops: {len(unique_drop_ids)}")
            logger.error(f"   Duplicates: {duplicates}")

            # Remove duplicates by keeping only first occurrence
            seen = set()
            valid_drops_deduped = []
            for drop_id, drop_dict in valid_drops:
                if drop_id not in seen:
                    valid_drops_deduped.append((drop_id, drop_dict))
                    seen.add(drop_id)

            valid_drops = valid_drops_deduped
            logger.warning(f"   ⚠ After forced deduplication: {len(valid_drops)} drops")

        logger.info(f"📊 Processing {len(valid_drops)} valid drop(s)\n")

        # Setup logging for parallel execution
        log_queue = None
        log_listener = None
        manager = None

        if self.parallel_drops > 1:
            manager = Manager()
            log_queue = manager.Queue(-1)
            log_listener = QueueListener(log_queue, logging.StreamHandler())
            log_listener.start()

        processing_start = time.time()
        successful = 0
        failed = 0

        logger.info(
            f"🚀 Starting parallel processing with {self.parallel_drops} drops...\n"
        )

        try:
            if self.parallel_drops > 1:
                with ProcessPoolExecutor(max_workers=self.parallel_drops) as executor:
                    futures_to_drop_id = {}
                    for drop_id, drop_dict in valid_drops:
                        future = executor.submit(
                            _process_single_drop,
                            drop_dict,
                            str(self.download_dir),
                            str(self.output_dir),
                            self.delete_originals,
                            self.test_mode,
                            self.download_threads,  # Updated variable name
                            self.sequential_download,
                            self.ffmpeg_path,
                            self.gopro_prefix,
                            log_queue,
                        )
                        futures_to_drop_id[future] = drop_id

                    logger.info(
                        f"✅ Submitted {len(futures_to_drop_id)} task(s) to executor\n"
                    )

                    # Wait for all futures to complete
                    for i, future in enumerate(as_completed(futures_to_drop_id), 1):
                        drop_id = futures_to_drop_id[future]
                        logger.debug(
                            f"Processing result {i}/{len(futures_to_drop_id)} for {drop_id}"
                        )
                        try:
                            future.result(timeout=7200)  # 2 hour timeout per drop
                            successful += 1
                            logger.info(
                                f"✅ Successfully processed DropID {drop_id} ({successful}/{len(futures_to_drop_id)})\n"
                            )
                        except TimeoutError:
                            failed += 1
                            logger.error(
                                f"❌ Timeout processing DropID {drop_id} (exceeded 2 hours)\n"
                            )
                        except Exception as e:
                            failed += 1
                            logger.error(f"❌ Error processing DropID {drop_id}: {e}\n")
                            logger.exception("Full traceback:")
            else:
                for drop_id, drop_dict in valid_drops:
                    try:
                        _process_single_drop(
                            drop_dict,
                            str(self.download_dir),
                            str(self.output_dir),
                            self.delete_originals,
                            self.test_mode,
                            self.download_threads,
                            self.sequential_download,
                            self.ffmpeg_path,
                            self.gopro_prefix,
                            None,
                        )
                        successful += 1
                        logger.info(f"✅ Successfully processed DropID {drop_id}\n")
                    except Exception as e:
                        failed += 1
                        logger.error(f"❌ Error processing DropID {drop_id}: {e}\n")
                        continue
        finally:
            if log_listener:
                log_listener.stop()

            # Summary
            total_time = time.time() - processing_start
            logger.info(f"\n{'='*80}")
            logger.info("📊 Processing Summary")
            logger.info(f"{'='*80}")
            logger.info(f"   Total drops: {len(valid_drops)}")
            logger.info(f"   ✅ Successful: {successful}")
            logger.info(f"   ❌ Failed: {failed}")
            logger.info(f"   ⏱️  Total time: {total_time:.2f} seconds")
            logger.info(f"{'='*80}\n")

    def get_movies_df(self, prefix: str = "") -> pd.DataFrame:
        """Get DataFrame of movie files in S3 bucket with their sizes."""
        objects = self.s3_handler.get_objects_from_s3(
            prefix=prefix, suffixes=tuple(self.MOVIE_EXTENSIONS), keys_only=False
        )
        assert isinstance(objects, list), "Expected list of objects"
        movie_data = [{"Key": obj["Key"], "Size": obj["Size"]} for obj in objects]
        return pd.DataFrame(movie_data)

    def find_already_concatenated_movies_df(
        self, size_tolerance: float = 0.01
    ) -> pd.DataFrame:
        """Find individual movie files that can be removed because concatenated version exists."""
        movies_df = self.get_movies_df(prefix=self.prefix)
        movies_df_with_parts = _add_path_parts_to_df(movies_df)

        concatenated_movies = movies_df_with_parts[
            movies_df_with_parts["fileNameNoExt"] == movies_df_with_parts["DropID"]
        ]
        gopro_movies = movies_df_with_parts[
            movies_df_with_parts.fileName.str.startswith(self.gopro_prefix, na=False)
        ]

        files_to_remove_list = []

        for _, concatenated_movie in concatenated_movies.iterrows():
            drop_id = concatenated_movie["DropID"]
            gopro_parts = gopro_movies[gopro_movies["DropID"] == drop_id]

            if not gopro_parts.empty:
                total_parts_size = gopro_parts["Size"].sum()
                concatenated_size = concatenated_movie["Size"]

                # Check if sizes match within tolerance
                if (
                    abs(total_parts_size - concatenated_size) / concatenated_size
                    < size_tolerance
                ):
                    files_to_remove_list.append(gopro_parts)

        if not files_to_remove_list:
            return pd.DataFrame()

        return pd.concat(files_to_remove_list)

    def preview_movie(self, key: str, expiration: int = 26400) -> Optional[HTML]:
        """
        Generates an HTML video player for a movie in S3.

        Args:
            key: The S3 key of the movie file
            expiration: The URL's expiration time in seconds

        Returns:
            IPython HTML object for display, or None on failure
        """
        movie_url = self.s3_handler.generate_presigned_url(key, expiration=expiration)

        if not movie_url:
            logger.error(f"Could not generate preview URL for {key}")
            return None

        html_code = f"""
        <div style="display: flex; align-items: center; width: 100%;">
            <div style="width: 60%; padding-right: 10px;">
                <video width="100%" controls>
                    <source src="{movie_url}">
                </video>
            </div>
        </div>
        """
        return HTML(html_code)


def _add_path_parts_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SurveyID, DropID, fileName, and fileNameNoExt from the 'Key' column."""
    if df.empty or "Key" not in df.columns:
        return df

    parts = df["Key"].str.split("/", expand=True)
    if parts.shape[1] < 4:
        logger.warning(
            "⚠ Key structure does not match expected format (media/survey/drop/file)"
        )
        return df

    df = df.assign(SurveyID=parts[1], DropID=parts[2], fileName=parts[3])
    df["fileNameNoExt"] = df["fileName"].str.replace(
        ".mp4", "", case=False, regex=False
    )
    return df


def get_filtered_movies_df(
    movies_df: pd.DataFrame, gopro_prefix: str = "GX"
) -> pd.DataFrame:
    """Filter movies DataFrame to find GoPro groups that need concatenation."""
    if movies_df.empty or "fileName" not in movies_df.columns:
        return pd.DataFrame()

    go_pro_movies_df = movies_df[
        movies_df.fileName.str.startswith(gopro_prefix, na=False)
    ].copy()

    # Find DropIDs that already have a concatenated file
    movies_df["fileNameNoExt"] = movies_df["fileName"].str.replace(
        ".mp4", "", case=False, regex=False
    )
    matching_dropids = movies_df[movies_df["fileNameNoExt"] == movies_df["DropID"]][
        "DropID"
    ].unique()

    # Exclude groups that already have a concatenated file
    df_no_matching = go_pro_movies_df[
        ~go_pro_movies_df["DropID"].isin(matching_dropids)
    ]

    # Only consider groups with more than one video to concatenate
    grouped_counts = df_no_matching.groupby("DropID")["fileName"].nunique()
    filtered_dropids = grouped_counts[grouped_counts > 1].index

    result_df = df_no_matching[df_no_matching["DropID"].isin(filtered_dropids)]

    # CRITICAL FIX: Remove duplicate rows based on Key (same file shouldn't appear twice)
    result_df = result_df.drop_duplicates(subset=["Key"])

    return result_df
