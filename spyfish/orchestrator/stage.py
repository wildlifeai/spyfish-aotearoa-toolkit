"""
Declarative pipeline stage infrastructure.

Two stage types:
  GlobalStage — runs once, manages its own drop iteration internally
                (ingest, ML inference, Biigle sync, retrain)
  DropStage   — StageRunner iterates over drops in input_statuses,
                calls fn(drop_id) -> target_status, then advances each drop.

Adding a new pipeline stage
---------------------------
1. Write the step function in run_pipeline.py.
   - GlobalStage fn:  () -> None
   - DropStage fn:    (drop_id: str) -> str   (returns the target PipelineStatus)
2. Add one entry to STAGES in run_pipeline.py.
That's it — argparse, eligibility, status transitions, and logging are automatic.
"""

from __future__ import annotations

import argparse
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

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

    The runner queries drops in input_statuses, calls fn(drop_id) for each,
    and advances the drop to the returned status via db.advance_status().

    input_statuses may be a static list[str] or a callable
    (args: Namespace, run_all: bool) -> list[str] for dynamic cases
    (e.g. Biigle-direct path that also picks up ML_COMPLETE).

    queue_status: if set, drops not already at this status are pre-advanced
    to it before fn runs. Use when a stage accepts an "earlier" trigger status
    but the state machine requires an intermediate queue state en route to the
    final status (e.g. ML_COMPLETE → AWAITING_CITSCI_CLIPS → CITSCI_CLIPS_COMPLETE).
    """

    flag: str
    description: str
    fn: Callable[[str], Optional[str]]
    input_statuses: list[str] | Callable[[argparse.Namespace, bool], list[str]]
    run_in_all: bool = True
    queue_status: str | None = None


StageSpec = GlobalStage | DropStage


class StageRunner:
    """Builds argparse and orchestrates stage execution from a declarative stage list."""

    def __init__(self, stages: list[StageSpec], db: DatabaseManager):
        self.stages = stages
        self.db = db

    def build_parser(self) -> argparse.ArgumentParser:
        """Build argparse from the stage registry. Extra non-stage flags are added by main()."""
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
        """True when no run_in_all stage flag was explicitly set."""
        return not any(
            getattr(args, s.flag.replace("-", "_"), False)
            for s in self.stages
            if s.run_in_all
        )

    def run(self, args: argparse.Namespace) -> None:
        run_all = self._is_run_all(args)

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

    def _run_global(self, stage: GlobalStage) -> None:
        try:
            stage.fn()
        except Exception as e:
            logging.error(f"{stage.flag} FAILED: {e}")
            logging.error(traceback.format_exc())
            raise

    def _run_drop_stage(
        self, stage: DropStage, args: argparse.Namespace, run_all: bool
    ) -> None:
        statuses = (
            stage.input_statuses(args, run_all)
            if callable(stage.input_statuses)
            else stage.input_statuses
        )
        records = self.db.get_deployments_by_statuses(statuses)
        drop_ids = [r["drop_id"] for r in records]

        if not drop_ids:
            logging.info(f"No deployments in {statuses} for {stage.flag}. Skipping.")
            return

        logging.info(f"Processing {len(drop_ids)} drops for {stage.flag}...")

        status_by_drop = {r["drop_id"]: r["status"] for r in records}

        for drop_id in drop_ids:
            try:
                if (
                    stage.queue_status
                    and status_by_drop.get(drop_id) != stage.queue_status
                ):
                    self.db.advance_status(drop_id, stage.queue_status)
                    logging.info(f"  → {drop_id}: queued as {stage.queue_status}")
                next_status = stage.fn(drop_id)
                if next_status is None:
                    logging.info(f"  → {drop_id}: not ready, leaving status unchanged")
                else:
                    self.db.advance_status(drop_id, next_status)
                    logging.info(f"  → {drop_id}: advanced to {next_status}")
            except Exception as e:
                logging.error(f"{stage.flag} failed for {drop_id}: {e}")
                logging.error(traceback.format_exc())
