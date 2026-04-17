"""Colored table widget for the experiment log.

Rows are colored by iteration status:
  - green for keep
  - red for discard / compile_error
  - yellow for incorrect
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

_STATUS_COLORS = {
    "keep": "green",
    "discard": "red",
    "compile_error": "bright_red",
    "incorrect": "yellow",
    "error": "bright_red",
    "running": "cyan",
}

_STATUS_ICONS = {
    "keep": "✓",
    "discard": "×",
    "compile_error": "!",
    "incorrect": "△",
    "error": "!",
    "running": "···",
}


class ColoredTable(Widget):
    """A table with rows colored by status.

    Columns: #, Speedup, Status, Intent
    """

    DEFAULT_CSS = """
    ColoredTable {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, rows: list[dict] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rows: list[dict] = rows or []

    def compose(self) -> ComposeResult:
        yield Static(self._render_table(), id="table-content")

    def update_rows(self, rows: list[dict]) -> None:
        """Update table data and re-render."""
        self._rows = rows
        try:
            content = self.query_one("#table-content", Static)
            content.update(self._render_table())
        except Exception:
            pass

    def _render_table(self) -> str:
        if not self._rows:
            return "No iterations yet."

        # Header
        header = f"{'#':>3}  {'Speedup':>8}  {'Status':<8}  {'Intent'}"
        separator = "─" * 50
        lines = [header, separator]

        for row in self._rows:
            iteration = row.get("iteration", 0)
            speedup = row.get("speedup", 0.0)
            status = row.get("status", "")
            intent = row.get("intent", "")
            decision = row.get("decision", status)

            color = _STATUS_COLORS.get(decision, "white")
            icon = _STATUS_ICONS.get(decision, " ")

            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "  -- "
            # Truncate intent to fit
            intent_short = intent[:25] if len(intent) > 25 else intent

            line = f"[{color}]{iteration:>3}  {speedup_str:>8}  {icon} {decision:<6}  {intent_short}[/]"
            lines.append(line)

        return "\n".join(lines)
