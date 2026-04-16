"""Horizontal gauge widget for profiling metrics.

Renders a labeled bar like: "BW   ▓▓▓▓▓▓▓░░░ 72%"
Color: green >0.8, yellow 0.5-0.8, red <0.5.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

_FILLED = "▓"
_EMPTY = "░"


def _gauge_color(ratio: float) -> str:
    if ratio >= 0.8:
        return "green"
    elif ratio >= 0.5:
        return "yellow"
    else:
        return "red"


class GaugeWidget(Widget):
    """A horizontal bar gauge for displaying utilization metrics."""

    DEFAULT_CSS = """
    GaugeWidget {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        label: str = "",
        value: float = 0.0,
        max_value: float = 1.0,
        bar_width: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._max_value = max_value
        self._bar_width = bar_width

    def compose(self) -> ComposeResult:
        yield Static(self._render_gauge(), id="gauge-content")

    def update_value(self, value: float) -> None:
        """Update the gauge value and re-render."""
        self._value = value
        try:
            content = self.query_one("#gauge-content", Static)
            content.update(self._render_gauge())
        except Exception:
            pass

    def _render_gauge(self) -> str:
        ratio = self._value / self._max_value if self._max_value > 0 else 0.0
        ratio = max(0.0, min(1.0, ratio))

        filled_count = int(ratio * self._bar_width)
        empty_count = self._bar_width - filled_count

        color = _gauge_color(ratio)
        bar = f"[{color}]{_FILLED * filled_count}[/]{_EMPTY * empty_count}"

        pct = int(ratio * 100)
        # Pad label to 5 chars for alignment
        padded_label = self._label.ljust(5)
        return f"{padded_label} {bar} {pct}%"
