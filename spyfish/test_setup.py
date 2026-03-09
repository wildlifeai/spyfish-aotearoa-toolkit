import logging
import pandas as pd
from typing import Set


# Hardcoded metadata for local test execution

def inject_test_drops(
    deployments_df: pd.DataFrame = None,
    db = None,
    use_pipeline_status: bool = False
) -> pd.DataFrame:
    """
    Single function to inject test configurations either for full ingestion or staged testing,
    reading directly from the test CSV.

    If use_pipeline_status is True:
        Acts like a staged test, seeding the SQLite DB directly with specific stages and volume IDs.
    If use_pipeline_status is False:
        Acts for Step 1 (ingestion), appending test drops to the deployments_df at READY_FOR_ML.
    """
    from spyfish.config import PipelineStatus, config
    from pathlib import Path

    csv_path = config.project_root / config.test_deployment_metadata_csv
    if not csv_path.exists():
        logging.warning(f"Test metadata CSV not found at {csv_path}")
        return deployments_df

    test_drops_df = pd.read_csv(csv_path)

    if db is None:
        raise ValueError("db must be provided when use_pipeline_status=True")

    logging.info(f"Injecting {len(test_drops_df)} staged test drop(s) into the database from CSV...")
    for _, row in test_drops_df.iterrows():

        drop_id = str(row['drop_id'])
        video_path = str(row['video_path'])
        sampling_start = int(row['sampling_start'])
        sampling_end = int(row['sampling_end'])
        if use_pipeline_status:
            status = str(row['status']).strip()
        else:
            status = PipelineStatus.READY_FOR_ML

        db.add_or_update_deployment(
            drop_id=drop_id,
            status=status,
            video_path=video_path,
            is_bad_deployment=False,
            sampling_start=sampling_start,
            sampling_end=sampling_end,
        )
        logging.info(f"  ✅ Seeded {drop_id} → {status}")
