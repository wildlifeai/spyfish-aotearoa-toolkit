"""
Declarative pipeline stage infrastructure.

Two stage types:
  GlobalStage — runs once, manages its own drop iteration internally
                (ingest, ML inference, Biigle sync, retrain)
  DropStage   — StageRunner iterates over drops eligible for this section,
                calls fn(drop_id) -> target_status, then advances each drop.

Adding a new pipeline stage
---------------------------
1. Write the step function in run_pipeline.py.
   - GlobalStage fn:  () -> None
   - DropStage fn:    (drop_id: str) -> str | None   (None = not ready, leave unchanged)
2. Add one entry to STAGES in run_pipeline.py.
That's it — argparse, eligibility, status transitions, and logging are automatic.
"""

from __future__ import annotations

import argparse
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from spyfish.config.base import SECTIONS
from spyfish.database.manager import DatabaseManager
from spyfish.log_config import log_header


@dataclass
class GlobalStage:
    """A stage that runs once and manages its own drop iteration internally."""

    flag: str  # CLI flag name, e.g. "ingest" → --ingest
    description: str  # argparse help text and log header
    fn: Callable[[], Any]
    run_in_all: bool = True  # included when no step flags are given


@dataclass
class DropStage:
    """A stage that processes drops one-by-one.

    section: which DB status column this stage owns (e.g. "citsci_status").
             The runner queries drops where section is in input_statuses, calls
             fn(drop_id), and advances section to the returned status.

    input_statuses: values to look for in section. May be a static list or a
                    callable (args, run_all) -> list[str] for dynamic cases.

    prerequisites: optional extra column=value conditions AND'd into the query
                   (e.g. {"ml_status": "ml_complete"} for zooniverse-clips).
                   May be a static dict or a callable (args, run_all) -> dict.
    """

    flag: str
    description: str
    fn: Callable[[str], Optional[str]]
    section: str
    input_statuses: list[str] | Callable[[argparse.Namespace, bool], list[str]]
    run_in_all: bool = True
    prerequisites: (
        dict[str, str | list[str]]
        | Callable[[argparse.Namespace, bool], dict[str, str | list[str]]]
        | None
    ) = None


StageSpec = GlobalStage | DropStage


class StageRunner:
    """Builds argparse and orchestrates stage execution from a declarative stage list.

    After `run()` completes, `failed_stages` lists any stages whose top-level
    exception was caught — drop-stage per-drop failures still just mark the
    failed drop as errored and don't appear here. Callers can check
    `runner.failed_stages` to decide whether to exit non-zero.
    """

    def __init__(self, stages: list[StageSpec], db: DatabaseManager):
        self.stages = stages
        self.db = db
        self.failed_stages: list[str] = []

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Run the Spyfish pipeline. Runs all steps by default."
        )
        for stage in self.stages:
            parser.add_argument(
                f"--{stage.flag}",
                action="store_true",
                help=stage.description,
                dest=stage.flag.replace("-", "_"),
            )
        return parser

    def _is_run_all(self, args: argparse.Namespace) -> bool:
        """True when no stage flag was explicitly set.

        Any explicit flag — including off-happy-path ones like --legacy or
        --check-arrivals — scopes the run to just the chosen stages.
        """
        return not any(
            getattr(args, s.flag.replace("-", "_"), False) for s in self.stages
        )

    def run(self, args: argparse.Namespace) -> None:
        run_all = self._is_run_all(args)
        self.failed_stages = []

        active = [
            s.flag
            for s in self.stages
            if (run_all and s.run_in_all)
            or getattr(args, s.flag.replace("-", "_"), False)
        ]
        logging.info(f"Active stages: {', '.join(active) or '(none)'}")

        for stage in self.stages:
            flag_attr = stage.flag.replace("-", "_")
            is_active = (run_all and stage.run_in_all) or bool(
                getattr(args, flag_attr, False)
            )

            if not is_active:
                logging.info(f"─── {stage.flag.upper()}: SKIPPED ───")
                continue

            log_header(f"{stage.description}")

            if isinstance(stage, GlobalStage):
                self._run_global(stage)
            else:
                self._run_drop_stage(stage, args, run_all)

        if self.failed_stages:
            logging.error(
                f"Pipeline completed with {len(self.failed_stages)} failed stage(s): "
                f"{', '.join(self.failed_stages)}"
            )

    def _run_global(self, stage: GlobalStage) -> None:
        """Runs a global stage. On exception, logs and records the failure
        but does NOT re-raise — a transient error in one global stage (e.g.
        a network blip during ingest) should not abort unrelated downstream
        stages. The runner surfaces failures via `self.failed_stages` so
        `main()` can still exit non-zero at the end if anything broke.
        """
        try:
            stage.fn()
        except Exception as e:
            logging.error(f"{stage.flag} FAILED: {e}")
            logging.error(traceback.format_exc())
            self.failed_stages.append(stage.flag)

    def _run_drop_stage(
        self, stage: DropStage, args: argparse.Namespace, run_all: bool
    ) -> None:
        statuses = (
            stage.input_statuses(args, run_all)
            if callable(stage.input_statuses)
            else stage.input_statuses
        )
        prereqs = (
            stage.prerequisites(args, run_all)
            if callable(stage.prerequisites)
            else stage.prerequisites
        )

        records = self.db.get_deployments_eligible(stage.section, statuses, prereqs)
        drop_ids = [r["drop_id"] for r in records]

        if not drop_ids:
            logging.info(f"No eligible deployments for {stage.flag}. Skipping.")
            return

        logging.info(f"Processing {len(drop_ids)} drops for {stage.flag}...")

        for drop_id in drop_ids:
            try:
                next_status = stage.fn(drop_id)
                if next_status is None:
                    logging.info(f"  → {drop_id}: not ready, leaving status unchanged")
                else:
                    self.db.advance_status(drop_id, stage.section, next_status)
                    logging.info(f"  → {drop_id}: {stage.section} → {next_status}")
            except Exception as e:
                logging.error(f"{stage.flag} failed for {drop_id}: {e}")
                logging.error(traceback.format_exc())
                self.db.update_section_status(
                    drop_id, stage.section, SECTIONS[stage.section].ERROR
                )
