import os

from spyfish.config.base import BaseConfig, get_required


class ZooniverseConfig(BaseConfig):

    @property
    def user(self):
        return os.getenv("ZOONIVERSE_USER")

    @property
    def password(self):
        return os.getenv("ZOONIVERSE_PASSWORD")

    @property
    def _zooniverse(self) -> dict:
        return get_required(self._yaml_config, "zooniverse", "")

    @property
    def zooniverse_project_id(self) -> int:
        """Upload target — single project."""
        return int(get_required(self._zooniverse, "project_id", "zooniverse"))

    @property
    def zooniverse_source_project_ids(self) -> list[int]:
        """All projects to fetch classifications from (can be multiple)."""
        ids = get_required(self._zooniverse, "source_project_ids", "zooniverse")
        return [int(i) for i in ids]

    @property
    def size_limit_mb(self) -> float:
        return float(get_required(self._zooniverse, "size_limit_mb", "zooniverse"))

    @property
    def min_clips_per_video(self) -> int:
        return int(get_required(self._zooniverse, "min_clips_per_video", "zooniverse"))

    @property
    def zooniverse_min_agreement_pct(self) -> float:
        """Aggregator filter threshold.

        A (subject, species) row passes the filter when
        ``vote_count / total_classifiers * 100 >= min_agreement_pct``.
        Handles the full workflow-retirement range (1 → 30+) without
        per-workflow tuning: an expert workflow with retirement=1 passes
        at 100% agreement; a broad workflow with retirement=30 needs at
        least ``min_agreement_pct%`` of voters to agree.
        """
        return float(get_required(self._zooniverse, "min_agreement_pct", "zooniverse"))

    @property
    def zooniverse_consensus_something_here_pct(self) -> float:
        """Consensus-fish rule threshold.

        When no single named species at a subject clears `min_agreement_pct`,
        but at least this fraction of voters saw SOMETHING (i.e.,
        `(total_classifiers − nothing_here_votes) / total_classifiers * 100 >=
        consensus_something_here_pct`), the aggregator emits a single row
        with `species="fish"` representing the collective "fish here,
        species ambiguous" signal. Replaces what would otherwise be a set
        of weak per-species rows that fail the agreement gate.
        """
        return float(
            get_required(self._zooniverse, "consensus_something_here_pct", "zooniverse")
        )

    @property
    def zooniverse_max_frames_per_run(self) -> int:
        return int(get_required(self._zooniverse, "max_frames_per_run", "zooniverse"))


zooniverse_config = ZooniverseConfig()
