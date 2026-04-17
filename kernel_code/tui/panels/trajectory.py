"""Trajectory panel -- optimization speedup chart using SparklineWidget.

Shows speedup progression over iterations with color-coded status markers
and running-best tracking.
"""

from __future__ import annotations

from rich.text import Text

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from kernel_code.tui.widgets.sparkline import SparklineWidget


class TrajectoryPanel(Widget):
    """Optimization trajectory chart panel."""

    DEFAULT_CSS = """
    TrajectoryPanel {
        height: auto;
        min-height: 8;
        background: #24231f;
        border: solid #3d3a36;
        padding: 0;
    }
    TrajectoryPanel > Static {
        height: 1;
        background: #2e2c28;
        color: #a09890;
        text-style: bold;
        padding: 0 1;
    }
    """

    def __init__(self, iterations: list[dict] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._iterations: list[dict] = iterations or []

    def compose(self) -> ComposeResult:
        title = Text("OPTIMIZATION TRAJECTORY", style="bold")
        yield Static(title)
        values, statuses, running_best = self._extract_data()
        yield SparklineWidget(
            values=values,
            statuses=statuses,
            running_best=running_best,
            id="trajectory-sparkline",
        )

    def update_iterations(self, iterations: list[dict]) -> None:
        """Update trajectory with new iteration data."""
        self._iterations = iterations
        values, statuses, running_best = self._extract_data()
        try:
            sparkline = self.query_one("#trajectory-sparkline", SparklineWidget)
            sparkline.update_data(values, statuses, running_best)
        except Exception:
            pass

    def _extract_data(self) -> tuple[list[float], list[str], list[float]]:
        values: list[float] = []
        statuses: list[str] = []
        running_best: list[float] = []
        best_so_far = 0.0

        for it in self._iterations:
            speedup = it.get("speedup", 0.0)
            status = it.get("decision", it.get("status", ""))
            values.append(speedup)
            statuses.append(status)
            best_so_far = max(best_so_far, speedup)
            running_best.append(best_so_far)

        return values, statuses, running_best
