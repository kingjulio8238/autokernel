"""Sparkline widget for terminal-based charts.

Renders a horizontal bar chart using Unicode block characters,
with per-bar coloring based on status (keep/discard/error).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

# Unicode block characters ordered by height (1/8 to 8/8)
_BLOCKS = " ▁▂▃▄▅▆▇█"


def _bar_char(value: float, min_val: float, max_val: float) -> str:
    """Map a value to a block character."""
    if max_val <= min_val or value <= min_val:
        return _BLOCKS[1]
    ratio = (value - min_val) / (max_val - min_val)
    idx = max(1, min(len(_BLOCKS) - 1, int(ratio * (len(_BLOCKS) - 1))))
    return _BLOCKS[idx]


_STATUS_COLORS = {
    "keep": "green",
    "discard": "red",
    "compile_error": "bright_red",
    "incorrect": "yellow",
    "error": "bright_red",
    "running": "cyan",
}


class SparklineWidget(Widget):
    """A sparkline chart rendered with Unicode block characters.

    Each bar is colored according to its status.
    """

    DEFAULT_CSS = """
    SparklineWidget {
        height: 3;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        values: list[float] | None = None,
        statuses: list[str] | None = None,
        label: str = "Speedup",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._values: list[float] = values or []
        self._statuses: list[str] = statuses or []
        self._label = label

    def compose(self) -> ComposeResult:
        yield Static(self._render_sparkline(), id="sparkline-content")

    def update_data(self, values: list[float], statuses: list[str]) -> None:
        """Update sparkline with new data and re-render."""
        self._values = values
        self._statuses = statuses
        try:
            content = self.query_one("#sparkline-content", Static)
            content.update(self._render_sparkline())
        except Exception:
            pass

    def _render_sparkline(self) -> str:
        if not self._values:
            return f"{self._label}: (no data)"

        min_val = 0.0
        max_val = max(self._values) if self._values else 1.0
        if max_val < 1.0:
            max_val = 1.0

        parts: list[str] = []
        for i, val in enumerate(self._values):
            char = _bar_char(val, min_val, max_val)
            status = self._statuses[i] if i < len(self._statuses) else "keep"
            color = _STATUS_COLORS.get(status, "white")
            parts.append(f"[{color}]{char}[/]")

        bars = "".join(parts)
        best = max(self._values)
        current = self._values[-1] if self._values else 0.0

        header = f"{self._label}  best: {best:.2f}x  current: {current:.2f}x"
        legend = "[green]●[/]keep  [red]×[/]discard  [yellow]△[/]error"
        return f"{header}\n{bars}\n{legend}"
