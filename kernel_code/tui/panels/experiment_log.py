"""Experiment log panel -- scrollable colored table of all iterations.

Uses ColoredTable widget to render iteration history with status colors.
Title shows count: "EXPERIMENT LOG (N iterations)".
Auto-scrolls to bottom when new iterations arrive.
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
        background: #252220;
        border: solid #3d3835;
        padding: 0;
    }
    ExperimentLogPanel > #log-title {
        height: 1;
        background: #302c28;
        color: #a09890;
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
        yield Static(self._title_text(), id="log-title")
        with VerticalScroll(id="log-scroll"):
            yield ColoredTable(rows=self._iterations, id="experiment-table")

    def _title_text(self) -> str:
        count = len(self._iterations)
        if count > 0:
            return f" EXPERIMENT LOG ({count} iterations)"
        return " EXPERIMENT LOG"

    def update_iterations(self, iterations: list[dict]) -> None:
        """Update the experiment log with new iteration data."""
        self._iterations = iterations

        # Update title with count
        try:
            title = self.query_one("#log-title", Static)
            title.update(self._title_text())
        except Exception:
            pass

        # Update table
        try:
            table = self.query_one("#experiment-table", ColoredTable)
            table.update_rows(iterations)
        except Exception:
            pass

        # Auto-scroll to bottom
        try:
            scroll = self.query_one("#log-scroll", VerticalScroll)
            scroll.scroll_end(animate=False)
        except Exception:
            pass
