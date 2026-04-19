"""Inline Rich.Live optimization display — no full-screen TUI.

Renders a live-updating display in the terminal output stream:
- Header with problem/hardware/backend info
- Unicode sparkline trajectory (▁▂▃▅▆▇█)
- Compact iteration results table
- Animated spinner with current phase

Uses Rich.Live to redraw in-place without clearing the screen.
Everything stays in the shell — zero context switching.

Usage::

    from kernel_code.live_display import LiveOptimizationDisplay

    display = LiveOptimizationDisplay(console, problem="matmul 4096x4096")
    display.start()
    display.update_iteration(1, 0.85, "discard", "tiled matmul")
    display.update_phase("Evaluating on L40S (correctness + benchmark)")
    display.update_iteration(2, 1.20, "keep", "vectorized loads")
    display.finish(stop_reason="Converged")
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.panel import Panel

if TYPE_CHECKING:
    pass

# Unicode block characters for sparklines (8 levels + space)
_BLOCKS = " ▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------


def _sparkline(values: list[float], width: int = 40) -> Text:
    """Render a unicode sparkline from speedup values."""
    if not values:
        return Text("  No data yet", style="white")

    mx = max(values) if max(values) > 0 else 1.0

    # Pad or truncate to width
    display_vals = values[-width:]

    chars = ""
    for v in display_vals:
        idx = min(8, int(v / mx * 8)) if mx > 0 else 0
        chars += _BLOCKS[idx]

    best = max(values)
    best_style = "#4ade80" if best > 1.0 else "#fbbf24" if best > 0 else "#ef4444"

    result = Text()
    result.append("  ")
    result.append(chars, style=best_style)
    result.append(f"  {best:.2f}x best", style=f"bold {best_style}")
    return result


# ---------------------------------------------------------------------------
# Iteration table
# ---------------------------------------------------------------------------


def _iteration_table(iterations: list[dict], width: int = 80) -> Table:
    """Render the iteration results as a compact table."""
    table = Table(
        box=None,
        padding=(0, 2),
        show_header=True,
        header_style="bold white",
        expand=False,
        width=min(width, 85),
    )
    table.add_column("#", width=4, justify="right", style="white")
    table.add_column("Speedup", width=8, justify="right")
    table.add_column("Status", width=10)
    table.add_column("Intent", min_width=30, max_width=50)

    for it in iterations:
        num = str(it.get("num", "?"))
        speedup = it.get("speedup", 0.0)
        status = it.get("status", "?")
        intent = it.get("intent", "")[:50]

        # Color coding
        if status == "keep":
            speed_style = "bold #4ade80"
            status_text = Text("✓ keep", style="#4ade80")
        elif status == "discard":
            speed_style = "white"
            status_text = Text("✗ disc", style="#888888")
        elif status in ("compile_error", "error", "incorrect"):
            speed_style = "#ef4444"
            status_text = Text("! error", style="#ef4444")
        else:
            speed_style = "white"
            status_text = Text(status[:8], style="white")

        is_best = it.get("is_best", False)
        if is_best:
            status_text = Text("★ best", style="bold #4ade80")

        table.add_row(
            num,
            Text(f"{speedup:.2f}x", style=speed_style),
            status_text,
            Text(intent, style="white"),
        )

    return table


# ---------------------------------------------------------------------------
# Live Display
# ---------------------------------------------------------------------------


class LiveOptimizationDisplay:
    """Inline optimization display using Rich.Live.

    Renders sparkline + table + spinner in-place in the terminal.
    No full-screen app, no context switching.
    """

    def __init__(
        self,
        console: Console | None = None,
        problem: str = "",
        hardware: str = "L40S",
        backend: str = "triton",
        max_iterations: int = 10,
    ) -> None:
        self._console = console or Console()
        self._problem = problem
        self._hardware = hardware
        self._backend = backend
        self._max_iterations = max_iterations

        self._iterations: list[dict] = []
        self._speedups: list[float] = []
        self._best_speedup: float = 0.0
        self._kept_count: int = 0
        self._current_phase: str = ""
        self._start_time: float = 0.0
        self._live: Live | None = None

    def start(self) -> None:
        """Start the live display."""
        self._start_time = time.time()
        self._live = Live(
            self._build(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def update_iteration(
        self,
        num: int,
        speedup: float,
        status: str,
        intent: str,
    ) -> None:
        """Record an iteration result and refresh the display."""
        is_best = status == "keep" and speedup > self._best_speedup
        if status == "keep" and speedup > self._best_speedup:
            self._best_speedup = speedup
        if status == "keep":
            self._kept_count += 1

        self._iterations.append({
            "num": num,
            "speedup": speedup,
            "status": status,
            "intent": intent,
            "is_best": is_best,
        })
        self._speedups.append(speedup)
        self._refresh()

    def update_phase(self, message: str) -> None:
        """Update the current phase message (spinner text)."""
        self._current_phase = message
        self._refresh()

    def finish(self, stop_reason: str = "") -> None:
        """Stop the live display, leaving final state visible."""
        self._current_phase = ""
        self._refresh()
        if self._live:
            self._live.stop()
            self._live = None

        # Print stop reason below the display
        if stop_reason:
            self._console.print(
                f"\n  [#fbbf24]Stopped:[/#fbbf24] [white]{stop_reason}[/white]"
            )

    def print_permanent(self, message: str) -> None:
        """Print a permanent line above the live region."""
        if self._live:
            self._live.console.print(message)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Rebuild and update the live display."""
        if self._live:
            self._live.update(self._build())

    def _build(self) -> Group:
        """Build the full display as a Group of renderables."""
        width = min(self._console.width, 90)
        elapsed = time.time() - self._start_time
        parts: list = []

        # Header
        header = Text()
        header.append(f"\n  Optimizing", style="bold white")
        if self._problem:
            header.append(f"  {self._problem[:50]}", style="white")
        header.append(f"  ({self._hardware}, {self._backend})", style="#888888")
        parts.append(header)
        parts.append(Text(""))

        # Sparkline
        if self._speedups:
            parts.append(_sparkline(self._speedups, width=width - 20))
            parts.append(Text(""))

        # Iteration table
        if self._iterations:
            parts.append(_iteration_table(self._iterations, width=width))
            parts.append(Text(""))

        # Status line: spinner + phase + elapsed
        if self._current_phase:
            elapsed_str = f"{elapsed:.0f}s"
            spinner_text = Text()
            spinner_text.append(f"  {self._current_phase}", style="white")
            spinner_text.append(f"  ({elapsed_str})", style="#888888")
            parts.append(Spinner("dots", text=spinner_text))
        else:
            # Show summary when not actively working
            summary = Text()
            summary.append("  ")
            kept = self._kept_count
            total = len(self._iterations)
            summary.append(f"{kept}/{total} kept", style="white")
            summary.append("  |  ", style="#888888")
            summary.append(f"best: {self._best_speedup:.2f}x", style="bold #4ade80")
            summary.append("  |  ", style="#888888")
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
            summary.append(time_str, style="white")
            parts.append(summary)

        return Group(*parts)
