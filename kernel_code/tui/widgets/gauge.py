"""Horizontal gauge widget for profiling metrics.

Renders a labeled bar with target threshold, delta indicator, and color coding.
Example: "BW    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░ 73% (+5%^) -> 85%"
Color: green >0.8, yellow 0.5-0.8, red <0.5.
"""

from __future__ import annotations

from rich.text import Text

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

_FILLED = "\u2593"
_EMPTY = "\u2591"


def _gauge_color(ratio: float) -> str:
    if ratio >= 0.8:
        return "#4ade80"
    elif ratio >= 0.5:
        return "#fbbf24"
    else:
        return "#f87171"


class GaugeWidget(Widget):
    """A horizontal bar gauge for displaying utilization metrics."""

    DEFAULT_CSS = """
    GaugeWidget {
        height: 1;
        padding: 0 1;
        color: #e8e0d8;
    }
    """

    def __init__(
        self,
        label: str = "",
        value: float = 0.0,
        max_value: float = 1.0,
        bar_width: int = 20,
        target: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._prev_value: float | None = None
        self._max_value = max_value
        self._bar_width = bar_width
        self._target = target

    def compose(self) -> ComposeResult:
        yield Static(self._render_gauge())

    def update_value(self, value: float) -> None:
        """Update the gauge value and re-render."""
        self._prev_value = self._value
        self._value = value
        try:
            content = self.query_one(Static)
            content.update(self._render_gauge())
        except Exception:
            pass

    def set_target(self, target: float | None) -> None:
        """Set or clear the target threshold."""
        self._target = target

    def _render_gauge(self) -> Text:
        ratio = self._value / self._max_value if self._max_value > 0 else 0.0
        ratio = max(0.0, min(1.0, ratio))

        filled_count = int(ratio * self._bar_width)
        empty_count = self._bar_width - filled_count

        color = _gauge_color(ratio)
        pct = int(ratio * 100)

        # Pad label to 5 chars for alignment
        padded_label = self._label.ljust(5)

        result = Text()
        result.append(f"{padded_label} ")
        result.append(_FILLED * filled_count, style=color)
        result.append(_EMPTY * empty_count)
        result.append(f" {pct:>3}%")

        # Delta from previous iteration
        if self._prev_value is not None:
            prev_ratio = self._prev_value / self._max_value if self._max_value > 0 else 0.0
            prev_pct = int(max(0.0, min(1.0, prev_ratio)) * 100)
            delta = pct - prev_pct
            if delta > 0:
                result.append(f" (+{delta}%\u2191)", style="#4ade80")
            elif delta < 0:
                result.append(f" ({delta}%\u2193)", style="#f87171")

        # Target threshold
        if self._target is not None:
            target_pct = int(max(0.0, min(1.0, self._target)) * 100)
            result.append(f" \u2192 {target_pct}%", style="dim")

        return result
