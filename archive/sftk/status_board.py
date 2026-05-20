"""
Status Board Module for Spyfish Aotearoa Toolkit.

Generates status reports at two levels:
1. Deployment (DropID) level - individual video deployment status
2. Survey level - aggregated survey statistics

Status includes:
- Video file presence in S3 (Present/Missing/NoLink/Excluded)
- Annotation counts (expert, ML, Biigle)
- Bad deployment flags
"""

import logging
from pathlib import Path

import pandas as pd

from sftk.common import (
    DEPLOYMENT_STATUS_FILENAME,
    EXPORT_LOCAL,
    LOCAL_DATA_FOLDER_PATH,
    S3_DEPLOYMENT_STATUS_CSV,
    S3_KSO_ANNOTATIONS_CSV,
    S3_SHAREPOINT_DEPLOYMENT_CSV,
    S3_SHAREPOINT_SURVEY_CSV,
    S3_SURVEY_STATUS_CSV,
    SURVEY_STATUS_FILENAME,
)
from sftk.data_validator import DataValidator


def classify_video_status(link: str, missing_files: set, paired_files: set) -> str:
    """
    Classify video status based on LinkToVideoFile entry.

    Returns one of:
    - 'Present': Video file exists in S3
    - 'Missing': Referenced in deployment list but not found in S3
    - 'NoLink': No LinkToVideoFile value in deployment record
    - 'Excluded': Bad deployment, no video expected (marked "NO VIDEO BAD DEPLOYMENT")
    - 'Unknown': Status couldn't be determined (should be rare)
    """
    if pd.isna(link):
        return "NoLink"
    if link == "NO VIDEO BAD DEPLOYMENT":
        return "Excluded"
    if link in missing_files:
        return "Missing"
    if link in paired_files:
        return "Present"
    return "Unknown"


class StatusBoard:
    """Generate status reports for surveys and deployments."""

    def __init__(
        self,
        data_validator: DataValidator | None = None,
    ):
        """
        Initialize StatusBoard.

        Args:
            data_validator: Optional DataValidator instance to reuse (avoids duplicate S3 calls).
                            If provided, uses its S3Handler for consistency.
        """
        self.data_validator = data_validator or DataValidator()
        self.s3_handler = self.data_validator.s3_handler
        self.local_folder_path = LOCAL_DATA_FOLDER_PATH

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load surveys, deployments, and annotations from S3."""
        surveys_df = self.s3_handler.read_df_from_s3_csv(S3_SHAREPOINT_SURVEY_CSV)
        deployments_df = self.s3_handler.read_df_from_s3_csv(
            S3_SHAREPOINT_DEPLOYMENT_CSV
        )
        annotations_df = self.s3_handler.read_df_from_s3_csv(S3_KSO_ANNOTATIONS_CSV)
        return surveys_df, deployments_df, annotations_df

    def get_file_status(self) -> tuple[set, set]:
        """
        Get file presence status.

        Uses DataValidator's cached file differences to avoid duplicate S3 calls.
        """
        all_files, missing_files, extra_files = (
            self.data_validator.get_file_differences()
        )
        paired_files = all_files - extra_files
        return missing_files, paired_files

    def build_deployment_status(
        self,
        deployments_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
        missing_files: set,
        paired_files: set,
    ) -> pd.DataFrame:
        """
        Build deployment-level status DataFrame.

        Columns:
        - SurveyID: Survey identifier (e.g., SLI_20240124_BUV)
        - DropID: Deployment identifier
        - VideoStatus: Present, Missing, NoLink, Excluded, Unknown
        - IsBadDeployment: True if marked as bad deployment
        - HasAnnotations: True if any annotations exist
        - TotalAnnotations: Sum of all annotation types
        - ExpertAnnotations: Count of expert annotations
        - MlAnnotations: Placeholder for ML annotations (future)
        - BiigleAnnotations: Placeholder for Biigle annotations (future)
        - NeedsAction: True if NOT ((has video AND annotations) OR bad deployment)
        """
        # Only include deployments (no extra files, no surveys without deployments)
        dep_df = deployments_df[
            ["SurveyID", "DropID", "LinkToVideoFile", "IsBadDeployment"]
        ].copy()

        # Classify video status
        dep_df["VideoStatus"] = dep_df["LinkToVideoFile"].apply(
            lambda x: classify_video_status(x, missing_files, paired_files)
        )

        # Clean up IsBadDeployment (fix FutureWarning)
        dep_df["IsBadDeployment"] = dep_df["IsBadDeployment"].fillna(False)
        dep_df["IsBadDeployment"] = dep_df["IsBadDeployment"].astype(bool)

        # Annotation columns
        annotation_counts = annotations_df["DropID"].value_counts()
        dep_df["ExpertAnnotations"] = (
            dep_df["DropID"].map(annotation_counts).fillna(0).astype(int)
        )
        dep_df["MlAnnotations"] = 0  # Placeholder for future
        dep_df["BiigleAnnotations"] = 0  # Placeholder for future
        dep_df["TotalAnnotations"] = dep_df[
            ["ExpertAnnotations", "MlAnnotations", "BiigleAnnotations"]
        ].sum(axis=1)
        dep_df["HasAnnotations"] = dep_df["TotalAnnotations"] > 0

        # NeedsAction: NOT ((has video AND has annotations) OR is bad deployment)
        has_video = dep_df["VideoStatus"] == "Present"
        is_complete = (has_video & dep_df["HasAnnotations"]) | dep_df["IsBadDeployment"]
        dep_df["NeedsAction"] = ~is_complete

        # Select and order columns (identifiers left, action status right)
        output_columns = [
            # Identifiers
            "SurveyID",
            "DropID",
            # Video status
            "VideoStatus",
            "IsBadDeployment",
            # Annotations
            "HasAnnotations",
            "TotalAnnotations",
            "ExpertAnnotations",
            "MlAnnotations",
            "BiigleAnnotations",
            # Action status (right side for quick filtering)
            "NeedsAction",
        ]

        return dep_df[output_columns]

    def build_survey_status(
        self,
        surveys_df: pd.DataFrame,
        deployment_status_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build survey-level status summary from deployment status.

        Includes all surveys from surveys_df, even those with no deployments.

        Columns:
        - SurveyID: Survey identifier
        - TotalDeployments: Count of deployments in survey
        - CompleteDeployments: Count fully complete (video+annotations OR excluded)
        - VideosNoActionNeeded: Count with VideoStatus Present or Excluded
        - VideosMissing: Count referenced but not in S3
        - VideosNoLink: Count with no link value
        - BadDeployments: Count of bad deployments
        - AnnotatedDeployments: Count with at least one annotation
        - TotalAnnotations: Sum of all annotations
        - CompletionPct: % of deployments complete (based on IsComplete)
        - AnnotationPct: % of deployments with annotations
        - BadDeploymentPct: % of deployments that are bad
        - NeedsAction: True if any deployment needs action (or 0 deployments)
        """
        # Group by SurveyID and aggregate using deployment status columns
        survey_summary_agg = (
            deployment_status_df.groupby("SurveyID")
            .agg(
                TotalDeployments=("DropID", "nunique"),
                IncompleteDeployments=("NeedsAction", "sum"),
                VideosPresent=("VideoStatus", lambda x: (x == "Present").sum()),
                VideosExcluded=("VideoStatus", lambda x: (x == "Excluded").sum()),
                VideosMissing=("VideoStatus", lambda x: (x == "Missing").sum()),
                VideosNoLink=("VideoStatus", lambda x: (x == "NoLink").sum()),
                BadDeployments=("IsBadDeployment", "sum"),
                AnnotatedDeployments=("HasAnnotations", "sum"),
                TotalAnnotations=("TotalAnnotations", "sum"),
            )
            .reset_index()
        )

        # Include surveys with no deployments by merging with the full survey list
        all_surveys_df = surveys_df[["SurveyID"]].drop_duplicates()
        survey_summary = pd.merge(
            all_surveys_df, survey_summary_agg, on="SurveyID", how="left"
        ).fillna(0)

        # CompleteDeployments = Total - Incomplete
        survey_summary["CompleteDeployments"] = (
            survey_summary["TotalDeployments"] - survey_summary["IncompleteDeployments"]
        )

        # VideosNoActionNeeded: Present + Excluded
        survey_summary["VideosNoActionNeeded"] = (
            survey_summary["VideosPresent"] + survey_summary["VideosExcluded"]
        )

        # Calculate percentages (handle division by zero for surveys with no deployments)
        total_deployments = survey_summary["TotalDeployments"]
        for numerator_col, pct_col in [
            ("CompleteDeployments", "CompletionPct"),
            ("AnnotatedDeployments", "AnnotationPct"),
            ("BadDeployments", "BadDeploymentPct"),
        ]:
            # Vectorized calculation is more performant than apply.
            # Division by zero results in inf/nan, which we can clean up.
            pct = survey_summary[numerator_col] / total_deployments * 100
            # Replace inf/-inf with nan, then fill all nan with 0, then round.
            survey_summary[pct_col] = (
                pct.replace([float("inf"), -float("inf")], float("nan"))
                .fillna(0)
                .round(1)
            )

        # NeedsAction: True if not 100% complete (includes surveys with 0 deployments)
        survey_summary["NeedsAction"] = survey_summary["CompletionPct"] < 100

        # Select and order columns (identifiers left, action status right)
        output_columns = [
            # Identifier
            "SurveyID",
            # Totals
            "TotalDeployments",
            "CompleteDeployments",
            "AnnotatedDeployments",
            "TotalAnnotations",
            # Video status breakdown
            "VideosNoActionNeeded",
            "VideosMissing",
            "VideosNoLink",
            "BadDeployments",
            # Progress & action (right side for quick filtering)
            "CompletionPct",
            "AnnotationPct",
            "BadDeploymentPct",
            "NeedsAction",
        ]

        return survey_summary[output_columns]

    def generate_status_report(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete status report.

        Returns:
            tuple: (deployment_status_df, survey_status_df)
        """
        # Load data
        surveys_df, deployments_df, annotations_df = self.load_data()

        # Get file status (uses DataValidator's cache)
        missing_files, paired_files = self.get_file_status()

        # Build deployment status
        deployment_status = self.build_deployment_status(
            deployments_df,
            annotations_df,
            missing_files,
            paired_files,
        )

        # Build survey status (includes surveys with no deployments)
        survey_status = self.build_survey_status(surveys_df, deployment_status)

        return deployment_status, survey_status

    def export_to_csv(
        self,
        deployment_status: pd.DataFrame,
        survey_status: pd.DataFrame,
    ) -> None:
        """
        Export status reports to CSV files (local or S3).

        Args:
            deployment_status: Deployment-level status DataFrame
            survey_status: Survey-level status DataFrame
        """
        if EXPORT_LOCAL:
            # Export locally
            path = Path(self.local_folder_path)
            path.mkdir(parents=True, exist_ok=True)

            deployment_path = path / DEPLOYMENT_STATUS_FILENAME
            survey_path = path / SURVEY_STATUS_FILENAME

            deployment_status.to_csv(deployment_path, index=False)
            survey_status.to_csv(survey_path, index=False)

            logging.info(f"Deployment status exported to {deployment_path}")
            logging.info(f"Survey status exported to {survey_path}")
        else:
            # Export to S3
            self.s3_handler.upload_updated_df_to_s3(
                df=deployment_status,
                key=S3_DEPLOYMENT_STATUS_CSV,
                filename=DEPLOYMENT_STATUS_FILENAME,
                keep_df_index=False,
            )
            self.s3_handler.upload_updated_df_to_s3(
                df=survey_status,
                key=S3_SURVEY_STATUS_CSV,
                filename=SURVEY_STATUS_FILENAME,
                keep_df_index=False,
            )

            logging.info(
                f"Deployment status exported to S3: {S3_DEPLOYMENT_STATUS_CSV}"
            )
            logging.info(f"Survey status exported to S3: {S3_SURVEY_STATUS_CSV}")

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate and export status reports.

        Returns:
            tuple: (deployment_status_df, survey_status_df)
        """
        deployment_status, survey_status = self.generate_status_report()
        self.export_to_csv(deployment_status, survey_status)
        logging.info(
            f"Status board completed: {len(deployment_status)} deployments, "
            f"{len(survey_status)} surveys"
        )
        return deployment_status, survey_status


if __name__ == "__main__":
    # Example usage
    board = StatusBoard()
    deployment_df, survey_df = board.run()
