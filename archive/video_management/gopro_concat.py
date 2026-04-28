#!/usr/bin/env python3
"""
GoPro Video Concatenation Script

This script finds and concatenates multi-part GoPro video files stored in an S3 bucket.
It identifies groups of videos belonging to the same DropID and combines them into single MP4 files.

Usage:
    python gopro_concat.py --prefix "media/SURVEY_ID" [options]
    
    Or make it executable:
    chmod +x gopro_concat.py
    ./gopro_concat.py --prefix "media/SURVEY_ID" [options]
"""

import sys
import os
import subprocess
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_ffmpeg_nesi():
    """Setup FFmpeg on NeSI by loading the module in the Python environment."""
    
    logger.info("Setting up FFmpeg on NeSI...")
    
    # Get the module initialization script
    module_init_script = """
source /etc/profile.d/modules.sh
module purge
module load FFmpeg/4.2.2-GCCcore-9.2.0
env
"""
    
    # Run the script and capture the environment
    try:
        result = subprocess.run(
            ["bash", "-c", module_init_script],
            capture_output=True,
            text=True,
            timeout=30
        )
    except subprocess.TimeoutExpired:
        logger.error("Timeout while loading FFmpeg module")
        return False
    
    if result.returncode != 0:
        logger.error(f"Error loading module: {result.stderr}")
        return False
    
    # Parse the environment variables from the module
    for line in result.stdout.split('\n'):
        if '=' in line:
            key, _, value = line.partition('=')
            # Only update PATH and LD_LIBRARY_PATH related vars
            if key in ['PATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH']:
                os.environ[key] = value
    
    # Verify ffmpeg works
    try:
        test_result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if test_result.returncode == 0:
            logger.info("✓ FFmpeg loaded successfully via module system")
            logger.info(test_result.stdout.split('\n')[0])
            return True
        else:
            logger.error(f"FFmpeg test failed: {test_result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing ffmpeg: {e}")
        return False


def main():
    """Main function to process GoPro videos."""
    
    parser = argparse.ArgumentParser(
        description='Concatenate multi-part GoPro videos from S3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--prefix',
        type=str,
        required=True,
        help='S3 prefix for video files (e.g., "media/SURVEY_ID")'
    )
    
    # Optional arguments
    parser.add_argument(
        '--gopro-prefix',
        type=str,
        default='G',
        help='Prefix for GoPro video files (e.g., "G", "GX", "GH")'
    )
    
    parser.add_argument(
        '--delete-originals',
        action='store_true',
        help='Delete original video parts from S3 after concatenation'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: skip upload and keep all files locally'
    )
    
    parser.add_argument(
        '--parallel-drops',
        type=int,
        default=1,
        help='Number of drops to process simultaneously'
    )
    
    parser.add_argument(
        '--download-threads',
        type=int,
        default=4,
        help='Max number of parallel downloads per drop'
    )
    
    parser.add_argument(
        '--sequential-download',
        action='store_true',
        help='Download files one at a time within each drop'
    )
    
    parser.add_argument(
        '--use-nesi-ffmpeg',
        action='store_true',
        help='Load FFmpeg from NeSI module system (for NeSI HPC)'
    )
    
    parser.add_argument(
        '--toolkit-path',
        type=str,
        default=None,
        help='Path to Spyfish-Aotearoa-toolkit directory'
    )
    
    args = parser.parse_args()
    
    # Add toolkit to path if specified, otherwise try to find it
    if args.toolkit_path:
        sys.path.insert(0, args.toolkit_path)
        logger.info(f"Added toolkit path: {args.toolkit_path}")
    else:
        # Try to automatically find the toolkit directory
        # Look in parent directory or common locations
        script_dir = Path(__file__).resolve().parent
        possible_paths = [
            script_dir.parent,  # One level up from script
            script_dir.parent.parent,  # Two levels up
            Path('/nesi/project/wildlife03546/Spyfish-Aotearoa-toolkit'),  # NeSI default
            Path.home() / 'Spyfish-Aotearoa-toolkit',  # Home directory
        ]
        
        for path in possible_paths:
            sftk_path = path / 'sftk'
            if sftk_path.exists() and sftk_path.is_dir():
                sys.path.insert(0, str(path))
                logger.info(f"Auto-detected toolkit path: {path}")
                break
        else:
            logger.warning("Could not auto-detect toolkit path. You may need to specify --toolkit-path")
    
    # Import after adding to path
    try:
        from sftk.s3_handler import S3Handler
        from sftk.video_handler import VideoProcessor
    except ImportError as e:
        logger.error(f"Failed to import sftk modules: {e}")
        logger.error("Please specify the toolkit path using --toolkit-path")
        logger.error("Example: --toolkit-path /path/to/Spyfish-Aotearoa-toolkit")
        return 1
    
    # Setup FFmpeg on NeSI if requested
    if args.use_nesi_ffmpeg:
        if not setup_ffmpeg_nesi():
            logger.error("Failed to setup FFmpeg on NeSI. Exiting.")
            sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("GoPro Video Concatenation Script")
    logger.info("=" * 80)
    logger.info(f"S3 Prefix: {args.prefix}")
    logger.info(f"GoPro Prefix: {args.gopro_prefix}")
    logger.info(f"Delete Originals: {args.delete_originals}")
    logger.info(f"Test Mode: {args.test_mode}")
    logger.info(f"Parallel Drops: {args.parallel_drops}")
    logger.info(f"Download Threads: {args.download_threads}")
    logger.info(f"Sequential Download: {args.sequential_download}")
    logger.info("=" * 80)
    
    try:
        # Initialize S3Handler and VideoProcessor
        logger.info("Initializing S3 handler...")
        s3_handler = S3Handler()
        
        logger.info("Initializing video processor...")
        processor = VideoProcessor(
            s3_handler, 
            prefix=args.prefix, 
            gopro_prefix=args.gopro_prefix, 
            delete_originals=args.delete_originals, 
            test_mode=args.test_mode, 
            download_threads=args.download_threads,
            parallel_drops=args.parallel_drops,  
            sequential_download=args.sequential_download
        )
        
        # Display what will be processed
        num_drops = processor.filtered_df['DropID'].nunique() if not processor.filtered_df.empty else 0
        num_files = len(processor.filtered_df)
        logger.info(f"Found {num_drops} drop(s) with {num_files} file(s) to process")
        
        if num_drops == 0:
            logger.info("No videos to process. Exiting.")
            return 0
        
        # Process GoPro videos
        logger.info("\nStarting video processing...\n")
        processor.process_gopro_videos()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Processing complete!")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\nProcess interrupted by user (Ctrl+C)")
        return 130
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())