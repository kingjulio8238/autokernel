"""Schedule optimization runs for later execution.

Usage:
    /schedule softmax.py --at "2026-04-19 02:00" --config configs/minimax_default.yaml
    /schedule list
    /schedule run  (execute all due runs)

    # Or from CLI for overnight/cron execution:
    kernel-code run-scheduled
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ScheduledRun:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    reference_path: str = ""
    backend: str = "triton"
    config_path: str | None = None
    max_iterations: int = 50
    scheduled_time: str = ""  # ISO format
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"  # pending | running | done | failed
    result_speedup: float = 0.0
    result_cost: float = 0.0
    error: str = ""


class RunScheduler:
    """Manages scheduled optimization runs.

    Stores each run as a JSON file under ``schedule_dir``.  No daemon --
    call ``get_due_runs()`` from a cron job or the CLI to execute pending work.
    """

    def __init__(self, schedule_dir: str | Path = ".kernel-code/scheduled"):
        self._dir = Path(schedule_dir)
        self._runs: list[ScheduledRun] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        reference: str,
        backend: str = "triton",
        config_path: str | None = None,
        max_iterations: int = 50,
        scheduled_time: str | None = None,
    ) -> ScheduledRun:
        """Schedule a new optimization run.

        If *scheduled_time* is ``None``, the run is scheduled for immediate
        execution (i.e. the next ``get_due_runs()`` call will pick it up).
        """
        now = datetime.now(timezone.utc).isoformat()
        run = ScheduledRun(
            reference_path=reference,
            backend=backend,
            config_path=config_path,
            max_iterations=max_iterations,
            scheduled_time=scheduled_time or now,
        )
        self._runs.append(run)
        self._save()
        return run

    def get_due_runs(self) -> list[ScheduledRun]:
        """Return all pending runs whose scheduled time has passed."""
        now = datetime.now(timezone.utc)
        due: list[ScheduledRun] = []
        for run in self._runs:
            if run.status != "pending":
                continue
            try:
                scheduled = datetime.fromisoformat(run.scheduled_time)
            except (ValueError, TypeError):
                continue
            # Treat naive datetimes as UTC
            if scheduled.tzinfo is None:
                scheduled = scheduled.replace(tzinfo=timezone.utc)
            if scheduled <= now:
                due.append(run)
        return due

    def list_runs(self) -> list[ScheduledRun]:
        """List all scheduled runs (pending + completed)."""
        return list(self._runs)

    def mark_running(self, run_id: str) -> None:
        run = self._find(run_id)
        run.status = "running"
        self._save()

    def mark_done(self, run_id: str, speedup: float, cost: float) -> None:
        run = self._find(run_id)
        run.status = "done"
        run.result_speedup = speedup
        run.result_cost = cost
        self._save()

    def mark_failed(self, run_id: str, error: str) -> None:
        run = self._find(run_id)
        run.status = "failed"
        run.error = error
        self._save()

    def cancel(self, run_id: str) -> bool:
        """Cancel a pending run.  Returns ``True`` if it was pending."""
        run = self._find(run_id)
        if run.status != "pending":
            return False
        run.status = "cancelled"
        self._save()
        return True

    def format_status(self) -> str:
        """Format all runs as a human-readable table string.

        Compatible with Rich console output when printed directly.
        """
        if not self._runs:
            return "No scheduled runs."

        header = f"{'ID':<10} {'Status':<10} {'Reference':<30} {'Scheduled':<26} {'Speedup':>8}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in self._runs:
            speedup_str = f"{r.result_speedup:.2f}x" if r.result_speedup else ""
            # Truncate reference for display
            ref = r.reference_path
            if len(ref) > 28:
                ref = "..." + ref[-25:]
            lines.append(
                f"{r.id:<10} {r.status:<10} {ref:<30} {r.scheduled_time:<26} {speedup_str:>8}"
            )
        return "\n".join(lines)

    def generate_cron_script(self) -> str:
        """Generate a shell script for cron/nohup execution.

        Output: a bash script that runs ``kernel-code run-scheduled``.
        """
        return (
            "#!/usr/bin/env bash\n"
            "# Auto-generated by kernel-code scheduler.\n"
            "# Add to crontab:  */15 * * * * /path/to/run_scheduled.sh\n"
            "\n"
            "set -euo pipefail\n"
            'LOGFILE="${HOME}/.kernel-code/scheduler.log"\n'
            'echo "$(date -u +\'%Y-%m-%dT%H:%M:%SZ\') Running scheduled optimizations" >> "$LOGFILE"\n'
            'kernel-code run-scheduled >> "$LOGFILE" 2>&1\n'
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _find(self, run_id: str) -> ScheduledRun:
        for run in self._runs:
            if run.id == run_id:
                return run
        raise KeyError(f"No scheduled run with id '{run_id}'")

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        for run in self._runs:
            path = self._dir / f"{run.id}.json"
            path.write_text(json.dumps(asdict(run), indent=2) + "\n")

    def _load(self) -> None:
        self._runs = []
        if not self._dir.is_dir():
            return
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                self._runs.append(ScheduledRun(**data))
            except (json.JSONDecodeError, TypeError):
                # Skip corrupted files
                continue
