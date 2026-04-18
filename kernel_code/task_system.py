"""Task system for multi-kernel optimization campaigns.

Manage multiple kernel optimization tasks as a campaign:
  /tasks create my-inference softmax.py layernorm.py attention.py
  /tasks run
  /tasks status
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CAMPAIGNS_DIR = _PROJECT_ROOT / ".kernel-code" / "campaigns"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class OptimizationTask:
    """A single kernel optimization task within a campaign."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    reference_path: str = ""
    backend: str = "triton"
    target_speedup: float | None = None
    status: str = "pending"  # pending | running | done | failed | skipped
    result_speedup: float = 0.0
    result_cost: float = 0.0
    result_time_seconds: float = 0.0
    error: str = ""


@dataclass
class Campaign:
    """A named group of optimization tasks."""

    name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tasks: list[OptimizationTask] = field(default_factory=list)

    @property
    def pending_tasks(self) -> list[OptimizationTask]:
        return [t for t in self.tasks if t.status == "pending"]

    @property
    def done_tasks(self) -> list[OptimizationTask]:
        return [t for t in self.tasks if t.status == "done"]

    @property
    def total_cost(self) -> float:
        return sum(t.result_cost for t in self.tasks)

    @property
    def best_overall_speedup(self) -> float:
        done = [t for t in self.tasks if t.status == "done" and t.result_speedup > 0]
        return max((t.result_speedup for t in done), default=0.0)


# ---------------------------------------------------------------------------
# Campaign manager
# ---------------------------------------------------------------------------


class CampaignManager:
    """Manages optimization campaigns with JSON persistence."""

    def __init__(self, campaigns_dir: str | Path = _CAMPAIGNS_DIR):
        self._dir = Path(campaigns_dir)
        self._campaigns: dict[str, Campaign] = {}
        self._load_all()

    # -- public API ----------------------------------------------------------

    def create_campaign(
        self,
        name: str,
        reference_files: list[str],
        backend: str = "triton",
    ) -> Campaign:
        """Create a new campaign with one task per reference file."""
        if name in self._campaigns:
            raise ValueError(f"Campaign '{name}' already exists")

        tasks = [
            OptimizationTask(reference_path=ref, backend=backend)
            for ref in reference_files
        ]
        campaign = Campaign(name=name, tasks=tasks)
        self._campaigns[name] = campaign
        self._save(campaign)
        logger.info(
            "Created campaign '%s' with %d task(s)", name, len(tasks)
        )
        return campaign

    def get_campaign(self, name: str) -> Campaign | None:
        """Return a campaign by name, or ``None`` if it does not exist."""
        return self._campaigns.get(name)

    def list_campaigns(self) -> list[Campaign]:
        """Return all campaigns sorted by creation time (newest first)."""
        campaigns = list(self._campaigns.values())
        campaigns.sort(key=lambda c: c.created_at, reverse=True)
        return campaigns

    def get_next_task(self, campaign_name: str) -> OptimizationTask | None:
        """Get the next pending task in a campaign."""
        campaign = self._campaigns.get(campaign_name)
        if campaign is None:
            return None
        pending = campaign.pending_tasks
        return pending[0] if pending else None

    def mark_task_running(self, campaign_name: str, task_id: str) -> None:
        """Mark a task as running."""
        task = self._find_task(campaign_name, task_id)
        if task is None:
            return
        task.status = "running"
        self._save(self._campaigns[campaign_name])

    def mark_task_done(
        self,
        campaign_name: str,
        task_id: str,
        speedup: float,
        cost: float,
        time_s: float,
    ) -> None:
        """Record a successful optimization result for a task."""
        task = self._find_task(campaign_name, task_id)
        if task is None:
            return
        task.status = "done"
        task.result_speedup = speedup
        task.result_cost = cost
        task.result_time_seconds = time_s
        self._save(self._campaigns[campaign_name])

    def mark_task_failed(
        self, campaign_name: str, task_id: str, error: str
    ) -> None:
        """Record a failed optimization attempt for a task."""
        task = self._find_task(campaign_name, task_id)
        if task is None:
            return
        task.status = "failed"
        task.error = error
        self._save(self._campaigns[campaign_name])

    def get_campaign_status(self, campaign_name: str) -> str:
        """Format campaign status as a Rich-compatible string."""
        campaign = self._campaigns.get(campaign_name)
        if campaign is None:
            return f"[red]Campaign '{campaign_name}' not found.[/red]"

        total = len(campaign.tasks)
        done = len(campaign.done_tasks)
        pending = len(campaign.pending_tasks)
        running = len([t for t in campaign.tasks if t.status == "running"])
        failed = len([t for t in campaign.tasks if t.status == "failed"])
        skipped = len([t for t in campaign.tasks if t.status == "skipped"])

        lines: list[str] = []
        lines.append(
            f"[bold]Campaign:[/bold] {campaign.name}  "
            f"[dim]({campaign.id})[/dim]"
        )
        lines.append(
            f"[bold]Progress:[/bold] {done}/{total} done  "
            f"| {pending} pending | {running} running "
            f"| {failed} failed | {skipped} skipped"
        )
        if campaign.total_cost > 0:
            lines.append(
                f"[bold]Cost:[/bold] ${campaign.total_cost:.4f}  "
                f"[bold]Best speedup:[/bold] {campaign.best_overall_speedup:.2f}x"
            )
        lines.append("")

        _STATUS_STYLE = {
            "pending": "dim",
            "running": "cyan",
            "done": "green",
            "failed": "red",
            "skipped": "yellow",
        }

        for task in campaign.tasks:
            style = _STATUS_STYLE.get(task.status, "")
            ref = Path(task.reference_path).name
            detail = ""
            if task.status == "done":
                detail = (
                    f"  {task.result_speedup:.2f}x  "
                    f"${task.result_cost:.4f}  "
                    f"{task.result_time_seconds:.0f}s"
                )
            elif task.status == "failed":
                detail = f"  {task.error}" if task.error else ""
            lines.append(
                f"  [{style}]{task.id}  {task.status:<8}  {ref}{detail}[/{style}]"
            )

        return "\n".join(lines)

    # -- persistence ---------------------------------------------------------

    def _save(self, campaign: Campaign) -> None:
        """Save campaign to .kernel-code/campaigns/{name}.json."""
        self._dir.mkdir(parents=True, exist_ok=True)
        dest = self._dir / f"{campaign.name}.json"
        dest.write_text(json.dumps(self._campaign_to_dict(campaign), indent=2))
        logger.debug("Campaign saved: %s", dest)

    def _load_all(self) -> None:
        """Load all campaigns from disk."""
        if not self._dir.exists():
            return
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                campaign = self._dict_to_campaign(data)
                self._campaigns[campaign.name] = campaign
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Skipping malformed campaign file: %s", path)

    def _campaign_to_dict(self, campaign: Campaign) -> dict:
        """Serialize a Campaign to a plain dict."""
        return {
            "name": campaign.name,
            "id": campaign.id,
            "created_at": campaign.created_at,
            "tasks": [asdict(t) for t in campaign.tasks],
        }

    def _dict_to_campaign(self, data: dict) -> Campaign:
        """Deserialize a Campaign from a plain dict."""
        tasks = [OptimizationTask(**t) for t in data.get("tasks", [])]
        return Campaign(
            name=data["name"],
            id=data["id"],
            created_at=data["created_at"],
            tasks=tasks,
        )

    # -- helpers -------------------------------------------------------------

    def _find_task(
        self, campaign_name: str, task_id: str
    ) -> OptimizationTask | None:
        """Look up a task by campaign name and task id."""
        campaign = self._campaigns.get(campaign_name)
        if campaign is None:
            logger.warning("Campaign '%s' not found", campaign_name)
            return None
        for task in campaign.tasks:
            if task.id == task_id:
                return task
        logger.warning(
            "Task '%s' not found in campaign '%s'", task_id, campaign_name
        )
        return None
