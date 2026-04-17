"""Sparkline widget for terminal-based charts.

Renders a multi-line horizontal bar chart using Unicode block characters,
with per-bar coloring based on status (keep/discard/error) and running-best markers.
"""

from __future__ import annotations

from rich.text import Text

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

# Unicode block characters ordered by height (1/8 to 8/8)
_BLOCKS = " ▁▂▃▄▅▆▇█"

# For 3-row rendering: each cell maps to 8 sub-levels, so 3 rows = 24 levels
_NUM_ROWS = 3
_LEVELS_PER_ROW = len(_BLOCKS) - 1  # 8 levels per row (indices 1-8)
_TOTAL_LEVELS = _NUM_ROWS * _LEVELS_PER_ROW  # 24 total levels


def _bar_levels(value: float, min_val: float, max_val: float) -> int:
    """Map a value to an integer level (0 .. _TOTAL_LEVELS)."""
    if max_val <= min_val or value <= min_val:
        return 0
    ratio = (value - min_val) / (max_val - min_val)
    return max(0, min(_TOTAL_LEVELS, int(ratio * _TOTAL_LEVELS)))


_STATUS_COLORS = {
    "keep": "#4ade80",
    "discard": "#f87171",
    "compile_error": "#f87171",
    "incorrect": "#fbbf24",
    "error": "#f87171",
    "running": "cyan",
}


class SparklineWidget(Widget):
    """A multi-row sparkline chart rendered with Unicode block characters.

    Each bar is colored according to its status, with the running best
    marked using a distinct character.
    """

    DEFAULT_CSS = """
    SparklineWidget {
        height: auto;
        min-height: 7;
        padding: 0 1;
        color: #e8e0d8;
    }
    """

    def __init__(
        self,
        values: list[float] | None = None,
        statuses: list[str] | None = None,
        running_best: list[float] | None = None,
        label: str = "Speedup",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._values: list[float] = values or []
        self._statuses: list[str] = statuses or []
        self._running_best: list[float] = running_best or []
        self._label = label

    def compose(self) -> ComposeResult:
        yield Static(self._render_sparkline(), id="sparkline-content")

    def update_data(
        self,
        values: list[float],
        statuses: list[str],
        running_best: list[float] | None = None,
    ) -> None:
        """Update sparkline with new data and re-render."""
        self._values = values
        self._statuses = statuses
        self._running_best = running_best or []
        try:
            content = self.query_one("#sparkline-content", Static)
            content.update(self._render_sparkline())
        except Exception:
            pass

    def _render_sparkline(self) -> Text:
        if not self._values:
            return Text(f"{self._label}: (no data)", style="dim")

        min_val = 0.0
        max_val = max(self._values) if self._values else 1.0
        if max_val < 1.0:
            max_val = 1.0

        # Compute running best if not provided
        running_best = self._running_best
        if not running_best and self._values:
            best_so_far = 0.0
            running_best = []
            for v in self._values:
                best_so_far = max(best_so_far, v)
                running_best.append(best_so_far)

        # Find the global best index
        best_val = max(self._values)
        best_idx = -1
        for i, v in enumerate(self._values):
            if v == best_val:
                best_idx = i

        # Pre-compute levels for each data point
        levels = [_bar_levels(v, min_val, max_val) for v in self._values]

        # Determine if each bar is the running best at that point
        is_running_best = []
        for i, v in enumerate(self._values):
            if i < len(running_best):
                is_running_best.append(abs(v - running_best[i]) < 1e-9 and v > 0)
            else:
                is_running_best.append(False)

        # Y-axis labels
        max_label = f"{max_val:.1f}x"
        mid_val = (min_val + max_val) / 2
        mid_label = f"{mid_val:.1f}x"
        min_label = f"{min_val:.1f}x"
        label_width = max(len(max_label), len(mid_label), len(min_label)) + 1

        # Build 3 rows of the chart (top row = highest values)
        result = Text()

        for row in range(_NUM_ROWS):
            # Row 0 = top (highest), Row 2 = bottom (lowest)
            row_from_bottom = _NUM_ROWS - 1 - row
            row_base = row_from_bottom * _LEVELS_PER_ROW

            # Y-axis label for this row
            if row == 0:
                y_label = max_label.rjust(label_width)
            elif row == _NUM_ROWS - 1:
                y_label = min_label.rjust(label_width)
            else:
                y_label = mid_label.rjust(label_width)

            result.append(y_label, style="dim")
            result.append(" ")

            for i, level in enumerate(levels):
                # How many levels fall into this row
                row_level = max(0, min(_LEVELS_PER_ROW, level - row_base))

                if row_level <= 0:
                    char = " "
                else:
                    char = _BLOCKS[row_level]

                status = self._statuses[i] if i < len(self._statuses) else "keep"
                color = _STATUS_COLORS.get(status, "white")

                # Use full block for running best, regular block for others
                if is_running_best[i] and row_level > 0:
                    # Running best bars use bold style
                    result.append(char, style=f"bold {color}")
                else:
                    result.append(char, style=color)

            result.append("\n")

        # Iteration number axis
        axis_padding = " " * (label_width + 1)
        result.append(axis_padding, style="dim")

        # Build axis markers: show first, best, and latest
        n = len(self._values)
        markers: dict[int, str] = {}
        if n > 0:
            markers[0] = "1"
        if best_idx >= 0 and best_idx != 0 and best_idx != n - 1:
            markers[best_idx] = f"*{best_idx + 1}"
        if n > 1:
            markers[n - 1] = str(n)

        # Render axis line
        axis_chars = [" "] * n
        for pos, label_str in sorted(markers.items()):
            for ci, ch in enumerate(label_str):
                write_pos = pos + ci
                if 0 <= write_pos < n:
                    axis_chars[write_pos] = ch

        result.append("".join(axis_chars), style="dim")
        result.append("\n")

        # Summary line
        current = self._values[-1] if self._values else 0.0
        summary = Text()
        summary.append("Best: ", style="dim")
        summary.append(f"{best_val:.2f}x", style="bold #4ade80")
        summary.append(f" at #{best_idx + 1}", style="dim")
        summary.append(" \u2502 ", style="dim")
        summary.append("Current: ", style="dim")
        current_color = "#4ade80" if current >= best_val * 0.9 else "#fbbf24"
        summary.append(f"{current:.2f}x", style=f"bold {current_color}")
        summary.append(" \u2502 ", style="dim")
        summary.append("Baseline: ", style="dim")
        summary.append("1.0x", style="dim bold")

        result.append(summary)

        return result
