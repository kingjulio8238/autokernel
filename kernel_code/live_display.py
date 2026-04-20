"""Inline Rich.Live optimization display — Claude Code-aligned.

Uses Claude Code's visual language:
- ⎿ connectors for sub-info
- ── rules for sections
- Bold headers, dim metadata
- Brand clay (#d77757), success (#4eba65), error (#ff6b80)
- Minimal chrome, no heavy panels
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from rich.live import Live

# Claude Code palette
_CLAY = "#d77757"
_SUCCESS = "#4eba65"
_WARNING = "#ffc107"
_ERROR = "#ff6b80"
_DIM = "#999999"
_MUTED = "#777777"

_BLOCKS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def _sparkline(values: list[float], width: int = 40) -> Text:
    """Render a unicode sparkline."""
    if not values:
        return Text()

    best = max(values)
    if best <= 0:
        chars = "\u2581" * len(values[-width:])
        t = Text()
        t.append(f"  {chars}", style=_ERROR)
        t.append(f"  0.00x", style=f"bold {_ERROR}")
        return t

    display_vals = values[-width:]
    chars = ""
    for v in display_vals:
        idx = max(1, min(8, int(v / best * 8))) if v > 0 else 1
        chars += _BLOCKS[idx]

    color = _SUCCESS if best > 1.0 else _WARNING
    t = Text()
    t.append(f"  {chars}", style=color)
    t.append(f"  {best:.2f}x best", style=f"bold {color}")
    return t


def _iteration_table(iterations: list[dict], width: int = 80) -> Table:
    """Compact iteration table."""
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
    table.add_column("Intent", min_width=30, max_width=55)

    for it in iterations:
        num = str(it.get("num", "?"))
        speedup = it.get("speedup", 0.0)
        status = it.get("status", "?")
        raw_intent = it.get("intent", "")
        intent = raw_intent[:55].rsplit(" ", 1)[0] + "..." if len(raw_intent) > 55 else raw_intent

        if status == "keep":
            speed_style = f"bold {_SUCCESS}"
            is_best = it.get("is_best", False)
            status_text = Text("\u2605 best", style=f"bold {_SUCCESS}") if is_best else Text("\u2713 keep", style=_SUCCESS)
        elif status == "discard":
            speed_style = "white"
            status_text = Text("\u2717 disc", style=_DIM)
        elif status in ("compile_error", "error", "incorrect"):
            speed_style = _ERROR
            status_text = Text("! error", style=_ERROR)
        else:
            speed_style = "white"
            status_text = Text(status[:8], style="white")

        table.add_row(
            num,
            Text(f"{speedup:.2f}x", style=speed_style),
            status_text,
            Text(intent, style="white"),
        )

    return table


class LiveOptimizationDisplay:
    """Inline display using Claude Code visual patterns."""

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
        self._current_round: int = 0
        self._current_strategy: str = ""
        self._target_speedup: float | None = None
        self._round_markers: list[int] = []
        self._worker_states: list[dict] = []

    def start(self) -> None:
        self._start_time = time.time()
        self._live = Live(
            self._build(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def set_target(self, target: float) -> None:
        self._target_speedup = target

    def start_round(self, round_num: int, strategy: str) -> None:
        self._current_round = round_num
        self._current_strategy = strategy
        self._round_markers.append(len(self._iterations))
        self._refresh()

    def update_workers(self, workers: list[dict]) -> None:
        self._worker_states = workers
        self._refresh()

    def update_iteration(self, num: int, speedup: float, status: str, intent: str) -> None:
        is_best = status == "keep" and speedup > self._best_speedup
        if status == "keep" and speedup > self._best_speedup:
            self._best_speedup = speedup
        if status == "keep":
            self._kept_count += 1
        global_num = len(self._iterations) + 1
        self._iterations.append({
            "num": global_num, "speedup": speedup, "status": status,
            "intent": intent, "is_best": is_best,
        })
        self._speedups.append(speedup)
        self._refresh()

    def update_phase(self, message: str) -> None:
        self._current_phase = message
        self._refresh()

    def finish(self, stop_reason: str = "") -> None:
        self._current_phase = ""
        self._refresh()
        if self._live:
            self._live.stop()
            self._live = None
        if stop_reason:
            self._console.print(f"  \u23bf  [{_DIM}]{stop_reason}[/{_DIM}]")

    def print_permanent(self, message: str) -> None:
        if self._live:
            self._live.console.print(message)

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._build())

    def _build(self) -> Group:
        width = min(self._console.width, 90)
        elapsed = time.time() - self._start_time
        parts: list = []

        # Header — Claude Code style: bold title + dim metadata
        header = Text()
        header.append("\n  \u2500\u2500 ", style=_DIM)
        if self._problem:
            header.append(self._problem[:55], style="bold white")
        else:
            header.append("Optimizing", style="bold white")
        header.append(" \u2500\u2500", style=_DIM)
        parts.append(header)

        # Sub-info via ⎿
        sub = Text()
        sub_parts = [self._hardware, self._backend]
        if self._current_round > 0:
            sub_parts.append(f"round {self._current_round}")
        sub.append(f"  \u23bf  ", style=_DIM)
        sub.append(" \u00b7 ".join(sub_parts), style=_DIM)
        parts.append(sub)

        # Strategy
        if self._current_strategy:
            strat = self._current_strategy
            if len(strat) > 60:
                strat = strat[:60].rsplit(" ", 1)[0] + "\u2026"
            parts.append(Text(f"  \u23bf  {strat}", style=_MUTED))

        # Target progress bar
        if self._target_speedup is not None:
            pct = min(100, int(self._best_speedup / self._target_speedup * 100))
            bar_w = 20
            filled = int(pct / 100 * bar_w)
            color = _SUCCESS if pct >= 100 else _WARNING if pct >= 50 else _ERROR
            tgt = Text()
            tgt.append("  \u23bf  target ", style=_DIM)
            tgt.append("\u2588" * filled, style=color)
            tgt.append("\u2591" * (bar_w - filled), style="#333333")
            tgt.append(f" {self._best_speedup:.2f}x/{self._target_speedup:.1f}x", style=f"bold {color}")
            parts.append(tgt)

        parts.append(Text(""))

        # Workers
        if self._worker_states:
            for w in self._worker_states:
                wid = w.get("id", 0)
                rnd = w.get("round", 0)
                max_rnd = w.get("max_rounds", 10)
                status = w.get("status", "working")
                action = w.get("action", "")

                wl = Text()
                wl.append(f"  Worker {wid + 1}  ", style="bold white")
                bar_w = 14
                filled = int(rnd / max(max_rnd, 1) * bar_w)

                if status == "passed":
                    wl.append("\u2588" * bar_w, style=_SUCCESS)
                    wl.append("  \u2713 correct kernel", style=f"bold {_SUCCESS}")
                elif status == "stopped":
                    wl.append("\u2588" * filled + "\u2591" * (bar_w - filled), style=_DIM)
                    wl.append("  stopped", style=_DIM)
                elif status == "waiting":
                    wl.append("\u2591" * bar_w, style="#333333")
                    wl.append("  waiting\u2026", style=_DIM)
                else:
                    wl.append("\u2588" * filled, style=_CLAY)
                    wl.append("\u2591" * (bar_w - filled), style="#333333")
                    wl.append(f"  {rnd}/{max_rnd}", style="white")
                    if action:
                        wl.append(f"  {action}", style=_DIM)
                parts.append(wl)
            parts.append(Text(""))

        # Sparkline
        if self._speedups:
            parts.append(_sparkline(self._speedups, width=width - 20))
            parts.append(Text(""))

        # Iteration table
        if self._iterations:
            parts.append(_iteration_table(self._iterations, width=width))
            parts.append(Text(""))

        # Timer — Claude Code style: ✻ thinking with elapsed
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        elapsed_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

        if self._current_phase:
            phase = self._current_phase
            if len(phase) > 55:
                phase = phase[:55].rsplit(" ", 1)[0] + "\u2026"
            spinner_chars = "\u2818\u2838\u2830\u2834\u2826\u2827\u2807\u280f"
            sc = spinner_chars[int(elapsed * 4) % len(spinner_chars)]
            status_line = Text()
            status_line.append(f"  {sc} ", style=_CLAY)
            status_line.append(phase, style="white")
            status_line.append(f"  ({elapsed_str})", style=_DIM)
            parts.append(status_line)
        else:
            summary = Text()
            summary.append(f"  \u23bf  ", style=_DIM)
            kept = self._kept_count
            total = len(self._iterations)
            summary.append(f"{kept}/{total} kept", style="white")
            summary.append(f" \u00b7 best: {self._best_speedup:.2f}x", style=f"bold {_SUCCESS}" if self._best_speedup > 0 else "white")
            summary.append(f" \u00b7 {elapsed_str}", style=_DIM)
            parts.append(summary)

        return Group(*parts)
