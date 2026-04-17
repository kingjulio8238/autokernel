"""Colored table widget for the experiment log.

Rows are colored by iteration status:
  - bold green for keep (with dim highlight for best row)
  - dim red for discard / compile_error
  - dim yellow for incorrect

Columns: # | Speedup | Best | Status | Intent
"""

from __future__ import annotations

from rich.text import Text

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

_STATUS_COLORS = {
    "keep": "#4ade80",
    "discard": "#f87171",
    "compile_error": "#f87171",
    "incorrect": "#fbbf24",
    "error": "#f87171",
    "running": "cyan",
}

_STATUS_ICONS = {
    "keep": "\u2713",
    "discard": "\u00d7",
    "compile_error": "!",
    "incorrect": "\u25b3",
    "error": "!",
    "running": "\u00b7\u00b7\u00b7",
}


class ColoredTable(Widget):
    """A table with rows colored by status.

    Columns: #, Speedup, Best, Status, Intent
    """

    DEFAULT_CSS = """
    ColoredTable {
        height: auto;
        padding: 0 1;
        color: #e8e0d8;
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

    def _render_table(self) -> Text:
        if not self._rows:
            return Text("No iterations yet.", style="dim")

        # --- Precompute running best and identify the overall best row ---
        running_best = 0.0
        best_at: list[float] = []
        overall_best_speedup = 0.0
        overall_best_idx = -1

        for i, row in enumerate(self._rows):
            speedup = row.get("speedup", 0.0)
            decision = row.get("decision", row.get("status", ""))
            if decision == "keep" and speedup > running_best:
                running_best = speedup
            best_at.append(running_best)
            if speedup > overall_best_speedup and decision == "keep":
                overall_best_speedup = speedup
                overall_best_idx = i

        # --- Build header ---
        output = Text()

        header_text = f"  {'#':>3}  {'SPEEDUP':>8}  {'BEST':>8}  {'STATUS':<10}  {'INTENT'}"
        output.append(header_text, style="bold underline dim")
        output.append("\n")

        # --- Build rows ---
        for i, row in enumerate(self._rows):
            iteration = row.get("iteration", 0)
            speedup = row.get("speedup", 0.0)
            status = row.get("status", "")
            intent = row.get("intent", "")
            decision = row.get("decision", status)

            color = _STATUS_COLORS.get(decision, "white")
            icon = _STATUS_ICONS.get(decision, " ")

            is_best_row = i == overall_best_idx
            is_keep = decision == "keep"
            is_discard = decision in ("discard", "compile_error", "incorrect", "error")

            # Format values
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "  --  "
            best_str = f"{best_at[i]:.2f}x" if best_at[i] > 0 else "  --  "

            # Show at least 50 chars of intent
            intent_display = intent[:50] if len(intent) > 50 else intent

            # Row prefix
            prefix = "\u2605 " if is_best_row else "  "

            # Build the line
            line = f"{prefix}{iteration:>3}  {speedup_str:>8}  {best_str:>8}  {icon} {decision:<8}  {intent_display}"

            # Determine style
            if is_best_row:
                style = "bold #4ade80 on #1a3a2a"
            elif is_keep:
                style = f"bold {color}"
            elif is_discard:
                style = f"dim {color}"
            else:
                style = color

            output.append(line, style=style)
            output.append("\n")

        return output
