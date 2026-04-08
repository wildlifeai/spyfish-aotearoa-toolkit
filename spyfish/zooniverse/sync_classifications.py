"""
Zooniverse volunteer annotation sync — pipeline integration point.

This module is the entry point called by run_pipeline.py for step 5b.
The heavy parsing work (API fetch, vote aggregation, completion gate,
MaxN CSV export) lives in parse_zooniverse_classifications.py and should
be run separately (manually or via cron) before this step is invoked.

Once parse_zooniverse_classifications.py has written a MaxN CSV for a drop,
sync_zooniverse_drop() ingests it into spyfish_annotations.db and signals
the pipeline to advance to CITSCI_COMPLETE.

TODO: Integrate Caesar completion check directly so this step can
auto-detect subject retirement without requiring the operator to run
parse_zooniverse_classifications.py separately.
"""

import logging

from spyfish.config.base import PipelineStatus
from spyfish.zooniverse.parse_classifications import ingest_zooniverse_annotations


def sync_zooniverse_drop(drop_id: str) -> str | None:
    """
    Ingest Zooniverse volunteer annotations for a single drop if ready.

    Checks whether parse_zooniverse_classifications.py has already written
    a MaxN CSV for this drop. If found, ingests annotations into
    spyfish_annotations.db with annotated_by='citsci' and returns
    CITSCI_COMPLETE. If not found, returns None to retry on the next run.

    Args:
        drop_id: The deployment ID to sync.

    Returns:
        PipelineStatus.CITSCI_COMPLETE if annotations were ingested.
        None if the MaxN CSV is not yet present.
    """
    count = ingest_zooniverse_annotations(drop_id)
    if count == 0:
        logging.info(
            f"zooniverse-sync: No MaxN CSV found for {drop_id}. "
            "Run parse_zooniverse_classifications.py once volunteers are done. "
            "Leaving at AWAITING_CITSCI_FRAMES."
        )
        return None

    logging.info(
        f"zooniverse-sync: Ingested {count} citsci annotations for {drop_id} → CITSCI_COMPLETE"
    )
    return PipelineStatus.CITSCI_COMPLETE
