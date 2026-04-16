"""Trajectory panel — optimization speedup chart using SparklineWidget.

Shows speedup progression over iterations with color-coded status markers.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from kernel_code.tui.widgets.sparkline import SparklineWidget


class TrajectoryPanel(Widget):
    """Optimization trajectory chart panel."""

    DEFAULT_CSS = """
    TrajectoryPanel {
        height: auto;
        min-height: 5;
        border: solid $accent;
        padding: 0;
    }
    TrajectoryPanel > Static {
        height: 1;
        background: $accent;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    """

    def __init__(self, iterations: list[dict] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._iterations: list[dict] = iterations or []

    def compose(self) -> ComposeResult:
        yield Static("Optimization Trajectory")
        values, statuses = self._extract_data()
        yield SparklineWidget(values=values, statuses=statuses, id="trajectory-sparkline")

    def update_iterations(self, iterations: list[dict]) -> None:
        """Update trajectory with new iteration data."""
        self._iterations = iterations
        values, statuses = self._extract_data()
        try:
            sparkline = self.query_one("#trajectory-sparkline", SparklineWidget)
            sparkline.update_data(values, statuses)
        except Exception:
            pass

    def _extract_data(self) -> tuple[list[float], list[str]]:
        values = []
        statuses = []
        for it in self._iterations:
            speedup = it.get("speedup", 0.0)
            status = it.get("decision", it.get("status", ""))
            values.append(speedup)
            statuses.append(status)
        return values, statuses
