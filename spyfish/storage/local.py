import logging
import os
from pathlib import Path


class LocalStorageHandler:
    def __init__(self, video_folder: str):
        self.video_folder = Path(video_folder)
        if not self.video_folder.exists():
            logging.warning(
                f"Local storage directory does not exist: {self.video_folder}"
            )

    def get_all_videos(self) -> set:
        """Scan the local media directory once and return a set of all relative file paths."""
        logging.info(f"Scanning local file system uniformly: {self.video_folder}")
        if not self.video_folder.exists():
            return set()

        known_files = set()
        for root, _, files in os.walk(self.video_folder):
            for file in files:
                # Store relative to video_folder to match DB entries
                full_path = Path(root) / file
                rel_path = full_path.relative_to(self.video_folder)
                known_files.add(str(rel_path))
                # Also add just the filename in case the DB stores it that way
                known_files.add(str(file))

        return known_files
