"""Experiment log panel — scrollable colored table of all iterations.

Uses ColoredTable widget to render iteration history with status colors.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from kernel_code.tui.widgets.colored_table import ColoredTable


class ExperimentLogPanel(Widget):
    """Scrollable experiment log showing all iterations."""

    DEFAULT_CSS = """
    ExperimentLogPanel {
        height: 1fr;
        border: solid $accent;
        padding: 0;
    }
    ExperimentLogPanel > Static {
        height: 1;
        background: $accent;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    ExperimentLogPanel > VerticalScroll {
        height: 1fr;
    }
    """

    def __init__(self, iterations: list[dict] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._iterations: list[dict] = iterations or []

    def compose(self) -> ComposeResult:
        yield Static("Experiment Log")
        with VerticalScroll():
            yield ColoredTable(rows=self._iterations, id="experiment-table")

    def update_iterations(self, iterations: list[dict]) -> None:
        """Update the experiment log with new iteration data."""
        self._iterations = iterations
        try:
            table = self.query_one("#experiment-table", ColoredTable)
            table.update_rows(iterations)
        except Exception:
            pass
