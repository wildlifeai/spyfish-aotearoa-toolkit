import logging
import pandas as pd
from typing import Set

# Hardcoded metadata for local test execution
TEST_DROPS = [
    ("KSF_20240124_BUV_KSF_085_01", "media/KSF_20240124_BUV/KSF_20240124_BUV_KSF_085_01/KSF_20240124_BUV_KSF_085_01.mp4", 1, 29),
    ("KSF_20240124_BUV_KSF_085_02", "media/KSF_20240124_BUV/KSF_20240124_BUV_KSF_085_02/KSF_20240124_BUV_KSF_085_02.mp4", 1, 29)
]

# Preconfigured test drops for testing Biigle annotation sync (Step 6).
# These entries are inserted directly into the DB with READY_FOR_EXPERT status
# and a known biigle_volume_id, bypassing the full ML+upload flow.
# Update the volume_id values to match real Biigle volumes in your project.
BIIGLE_TEST_DROPS = [
    # (drop_id, biigle_volume_id, sampling_start, sampling_end)
    ("KSF_20240124_BUV_KSF_085_01", 30544, 1, 29),
    ("KSF_20240124_BUV_KSF_085_02", 30547, 1, 29),
]

# Staggered pipeline stage drops
# Used to test the pipeline by picking up drops that are waiting at different pipeline steps
PIPELINE_STAGES_TEST_DROPS = [
    ("KSF_20240124_BUV_KSF_085_01", "media/KSF_20240124_BUV/KSF_20240124_BUV_KSF_085_01/KSF_20240124_BUV_KSF_085_01.mp4", 1, 29, "READY_FOR_ML"),
    ("KSF_20240124_BUV_KSF_085_02", "media/KSF_20240124_BUV/KSF_20240124_BUV_KSF_085_02/KSF_20240124_BUV_KSF_085_02.mp4", 1, 29, "ML_COMPLETE"),
    ("KSF_20240124_BUV_KSF_085_03", "media/KSF_20240124_BUV/KSF_20240124_BUV_KSF_085_03/KSF_20240124_BUV_KSF_085_03.mp4", 1, 29, "READY_FOR_CITSCI"),
    ("KSF_20240124_BUV_KSF_085_04", "media/KSF_20240124_BUV/KSF_20240124_BUV_KSF_085_04/KSF_20240124_BUV_KSF_085_04.mp4", 1, 29, "CITSCI_COMPLETE"),
    ("KSF_20240124_BUV_KSF_085_05", "media/KSF_20240124_BUV/KSF_20240124_BUV_KSF_085_05/KSF_20240124_BUV_KSF_085_05.mp4", 1, 29, "PROCESSING_EXPERT"),
]

# mapping to old videos
# KSF_20240124_BUV_KSF_085_01 -> TUH_20210309_BUV_TUH_034_01.mp4_clip_1869_30.mp4
# KSF_20240124_BUV_KSF_085_02 -> KOK_20240219_BUV_KOK_060_01.mp4_clip_1572_30.mp4
# KSF_20240124_BUV_KSF_085_03 -> AHE_20220425_BUV_AHE_119_01.mp4_clip_900_30.mp4
# KSF_20240124_BUV_KSF_085_04 -> TON_20221205_BUV_TON_044_01.mp4_clip_835_30.mp4
# KSF_20240124_BUV_KSF_085_05 -> KOK_20240219_BUV_KOK_060_01.mp4_clip_1542_30.mp4


def inject_test_data(deployments_df: pd.DataFrame, known_files: Set[str]) -> pd.DataFrame:
    """
    Injects test configurations into the deployed manifest and known files list.
    Used for Step 1 (ingestion) to make test drops visible to the pipeline.
    """
    for test_id, test_key, sampling_start, sampling_end in TEST_DROPS:
        if test_key:
            known_files.add(test_key)

        fake_row = pd.DataFrame([{
            "DropID": test_id,
            "SurveyID": test_id[:16],
            "SiteID": test_id[17:24],
            "IsBadDeployment": "False",
            "LinkToVideoFile": test_key,
            "SamplingStart": sampling_start,
            "SamplingEnd": sampling_end
        }])
        deployments_df = pd.concat([deployments_df, fake_row], ignore_index=True)

    return deployments_df


def inject_biigle_test_drops(db) -> None:
    """
    Directly seeds the database with test drops that already have a biigle_volume_id,
    placing them at PROCESSING EXPERT status so Step 6 (Biigle sync) can be tested
    without running the full ingestion → ML → upload flow.

    Call this from the pipeline or CLI when you want to test Biigle annotation syncing.
    """
    from spyfish.config import PipelineStatus

    logging.info(f"Injecting {len(BIIGLE_TEST_DROPS)} Biigle test drop(s) into the database...")
    for drop_id, volume_id, sampling_start, sampling_end in BIIGLE_TEST_DROPS:
        survey_id = drop_id[:16]

        # Always overwrite the deployment for the test drops so we can test the sync step
        # repeatedly, regardless of where they were in the pipeline before.

        db.add_or_update_deployment(
            drop_id=drop_id,
            status=PipelineStatus.PROCESSING_EXPERT,
            video_path=f"media/{survey_id}/{drop_id}/{drop_id}.mp4",
            is_bad_deployment=False,
            sampling_start=sampling_start,
            sampling_end=sampling_end,
            biigle_volume_id=str(volume_id),
        )
        logging.info(f"  ✅ Seeded {drop_id} → PROCESSING_EXPERT (biigle_volume_id={volume_id})")

def inject_staged_test_drops(db) -> None:
    """
    Directly seeds the database with drops at assorted pipeline stages.
    """
    from spyfish.config import PipelineStatus

    logging.info(f"Injecting {len(PIPELINE_STAGES_TEST_DROPS)} staged test drop(s) into the database...")
    for drop_id, video_path, sampling_start, sampling_end, status in PIPELINE_STAGES_TEST_DROPS:
        db.add_or_update_deployment(
            drop_id=drop_id,
            status=getattr(PipelineStatus, status),
            video_path=video_path,
            is_bad_deployment=False,
            sampling_start=sampling_start,
            sampling_end=sampling_end,
        )
        logging.info(f"  ✅ Seeded {drop_id} → {status}")
