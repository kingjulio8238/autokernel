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


def _iteration_log_compact(iterations: list[dict], width: int = 80) -> list:
    """BAND 3 - compact one-line-per-round log.

    Format: "  R1  * SOL 0.42 \u00b7 0.91x  shared memory tiling..." when SOL is
    present on that iteration, else "  R1  * 0.91x  shared memory tiling..."
    (speedup-only, no SOL column \u2014 BAND 1 already shows the best-SOL rollup).
    """
    lines: list = []
    if not iterations:
        return lines

    header = Text()
    header.append("  History", style=f"bold {_DIM}")
    lines.append(header)

    for it in iterations[-6:]:  # last 6; older via /show results
        num = it.get("num", "?")
        speedup = it.get("speedup", 0.0)
        sol_score = float(it.get("sol_score", 0.0) or 0.0)
        status = it.get("status", "?")
        is_best = it.get("is_best", False)
        intent = it.get("intent", "")
        # Dual-display eats ~9 extra chars; tighten intent when SOL is shown.
        intent_budget = 36 if sol_score > 0 else 45
        if len(intent) > intent_budget:
            intent = intent[:intent_budget - 2].rstrip() + "\u2026"

        if is_best:
            glyph, glyph_style = "\u2605", f"bold {_SUCCESS}"
            sp_style = f"bold {_SUCCESS}"
        elif status == "keep":
            glyph, glyph_style = "\u2713", _SUCCESS
            sp_style = _SUCCESS
        elif status == "discard":
            glyph, glyph_style = "\u2717", _DIM
            sp_style = "white"
        elif status in ("compile_error", "error", "incorrect"):
            glyph, glyph_style = "!", _ERROR
            sp_style = _ERROR
        else:
            glyph, glyph_style = "\u00b7", _DIM
            sp_style = "white"

        line = Text()
        line.append(f"    R{num} ", style=_DIM)
        line.append(f"{glyph} ", style=glyph_style)
        if sol_score > 0:
            line.append(f"SOL {sol_score:.2f}", style=sp_style)
            line.append(f" \u00b7 {speedup:.2f}x", style=_DIM)
        else:
            line.append(f"{speedup:.2f}x", style=sp_style)
        line.append("  ")
        line.append(intent, style="white" if is_best else _DIM)
        lines.append(line)

    return lines


class LiveOptimizationDisplay:
    """Inline display using Claude Code visual patterns."""

    def __init__(
        self,
        console: Console | None = None,
        problem: str = "",
        hardware: str = "L40S",
        backend: str = "triton",
        max_iterations: int = 10,
        problem_tier: str = "",
        problem_type: str = "",
        is_memory_bound: bool | None = None,
    ) -> None:
        self._console = console or Console()
        self._problem = problem
        self._hardware = hardware
        self._backend = backend
        self._max_iterations = max_iterations
        self._problem_tier = problem_tier
        self._problem_type = problem_type
        self._is_memory_bound = is_memory_bound

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
        self._worker_speedups: dict[int, list[tuple[int, float, float]]] = {}
        self._show_plots: bool = True
        self._current_bw_pct: float = 0.0
        self._current_compute_pct: float = 0.0
        self._run_finalized: bool = False

    def start(self) -> None:
        self._start_time = time.time()
        self._run_finalized = False
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
        # Capture per-worker speedup history for plot A using GLOBAL id
        now = time.time() - self._start_time
        for w in workers:
            gid = w.get("global_id", w.get("id", 0))  # global ID for plot tracking
            rnd = w.get("round", 0)
            speedup = w.get("speedup", 0.0)
            history = self._worker_speedups.setdefault(gid, [])
            if rnd > 0 and speedup > 0 and (not history or history[-1][0] < rnd):
                history.append((rnd, speedup, now))
        self._refresh()

    def update_iteration(
        self, num: int, speedup: float, status: str, intent: str,
        sol_score: float = 0.0,
    ) -> None:
        is_best = status == "keep" and speedup > self._best_speedup
        if status == "keep" and speedup > self._best_speedup:
            self._best_speedup = speedup
        if status == "keep":
            self._kept_count += 1
        global_num = len(self._iterations) + 1
        self._iterations.append({
            "num": global_num, "speedup": speedup, "status": status,
            "intent": intent, "is_best": is_best, "sol_score": sol_score,
        })
        self._speedups.append(speedup)
        self._refresh()

    def update_phase(self, message: str) -> None:
        self._current_phase = message
        self._refresh()

    def update_profile(self, bw_pct: float = 0.0, compute_pct: float = 0.0) -> None:
        """Update current profile metrics for display."""
        self._current_bw_pct = min(100.0, max(0.0, bw_pct))
        self._current_compute_pct = min(100.0, max(0.0, compute_pct))
        self._refresh()

    def finish(self, stop_reason: str = "") -> None:
        self._current_phase = ""
        self._run_finalized = True
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
        """Option A layout: 3 bands — status, live work, history."""
        width = min(self._console.width, 90)
        elapsed = time.time() - self._start_time
        parts: list = []

        # Compute best SOL score from iterations
        best_sol = 0.0
        for it in self._iterations:
            s = it.get("sol_score", 0.0)
            if s > best_sol:
                best_sol = s

        # ============ BAND 1: STATUS ============
        # Line 1: title + context
        header = Text()
        header.append("\n  \u2500\u2500 ", style=_DIM)
        problem = self._problem or "Optimizing"
        header.append(problem[:40], style="bold white")
        header.append(" \u2500\u2500  ", style=_DIM)
        ctx_parts = [self._hardware, self._backend]
        if self._problem_tier:
            ctx_parts.append(self._problem_tier)
        if self._problem_type:
            ctx_parts.append(self._problem_type)
        header.append(" \u00b7 ".join(ctx_parts), style=_DIM)
        if self._is_memory_bound is not None:
            bound = "mem-bound" if self._is_memory_bound else "compute-bound"
            header.append(f"  [{bound}]", style="#22d3ee")
        parts.append(header)

        # Line 2: round + best metric + time.
        # Dual-display: SOL primary when present, speedup as dim secondary.
        ctx2 = Text()
        ctx2.append("  \u23bf  ", style=_DIM)
        if self._current_round > 0:
            ctx2.append(f"Round {self._current_round}", style="white")
            ctx2.append(f" \u00b7 ", style=_DIM)
        ctx2.append(f"best ", style=_DIM)
        sp_color = _SUCCESS if self._best_speedup >= (self._target_speedup or 1.0) else _WARNING if self._best_speedup >= 1.0 else _ERROR
        if best_sol > 0:
            sol_color = _SUCCESS if best_sol >= 0.7 else _WARNING if best_sol >= 0.4 else _ERROR
            ctx2.append(f"SOL {best_sol:.2f}", style=f"bold {sol_color}")
            ctx2.append(f" \u00b7 {self._best_speedup:.2f}x", style=_DIM)
        else:
            ctx2.append(f"{self._best_speedup:.2f}x", style=f"bold {sp_color}")
        if self._target_speedup:
            ctx2.append(f" / {self._target_speedup:.1f}x target", style=_DIM)
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        elapsed_str = f"{mins}m{secs:02d}s" if mins > 0 else f"{secs}s"
        ctx2.append(f" \u00b7 {elapsed_str}", style=_DIM)
        parts.append(ctx2)

        # Line 3: combined progress bars. SOL-primary: when SOL is available
        # it renders first (wider, labeled), and the target/speedup bar moves
        # to the secondary/right position. Target bar stays for transitional
        # dual display \u2014 Phase 4 removes it when target_speedup is retired.
        if self._target_speedup is not None or best_sol > 0:
            prog = Text()
            prog.append("  \u23bf  ", style=_DIM)

            if best_sol > 0:
                sol_bar_w = 20
                sol_filled = min(sol_bar_w, max(0, int(best_sol * sol_bar_w)))
                sol_color = _SUCCESS if best_sol >= 0.7 else _WARNING if best_sol >= 0.4 else _ERROR
                prog.append("SOL ", style="bold white")
                prog.append("\u2588" * sol_filled, style=sol_color)
                prog.append("\u2591" * (sol_bar_w - sol_filled), style="#333333")
                prog.append(f" {best_sol:.2f}", style=f"bold {sol_color}")

            if self._target_speedup is not None:
                pct = min(100, int(self._best_speedup / self._target_speedup * 100))
                bar_w = 12 if best_sol > 0 else 20
                filled = int(pct / 100 * bar_w)
                color = _SUCCESS if pct >= 100 else _WARNING if pct >= 50 else _ERROR
                if best_sol > 0:
                    # Secondary position: dim label, narrower bar.
                    prog.append("   target ", style=_DIM)
                    prog.append("\u2588" * filled, style=color)
                    prog.append("\u2591" * (bar_w - filled), style="#333333")
                    prog.append(f" {pct}%", style=_DIM)
                else:
                    # No SOL \u2014 target bar in primary position.
                    prog.append("\u2588" * filled, style=color)
                    prog.append("\u2591" * (bar_w - filled), style="#333333")
                    prog.append(f" {pct}%", style=f"bold {color}")

            parts.append(prog)

        parts.append(Text(""))

        # ============ BAND 2: LIVE WORK ============
        # Strategy line
        if self._current_strategy:
            strat = self._current_strategy
            if len(strat) > 70:
                strat = strat[:68].rsplit(" ", 1)[0] + "\u2026"
            s_line = Text()
            s_line.append("  \u25b6 ", style=_CLAY)
            s_line.append(strat, style="white")
            parts.append(s_line)

        # Plot A: compact dot scatter (keep)
        if self._worker_speedups and self._show_plots:
            from kernel_code.worker_plots import render_live_lines
            plot = render_live_lines(self._worker_speedups, elapsed, width=min(width, 55))
            parts.append(plot)

        # Current round workers — ONE compact line
        if self._worker_states:
            _WCOLORS = ["#d77757", "#f5a850", "#4eba65", "#5a9ec9",
                         "#8b6fa8", "#c97b9b", "#6ec0a0", "#b8a050"]
            wline = Text("  ")
            for w in self._worker_states:
                wid = w.get("id", 0)
                gid = w.get("global_id", wid)
                rnd = w.get("round", 0)
                max_rnd = w.get("max_rounds", 10)
                status = w.get("status", "working")
                wcolor = _WCOLORS[gid % len(_WCOLORS)]

                wline.append(f"W{wid + 1}", style=f"bold {wcolor}")
                if status == "passed":
                    speedup = w.get("speedup", 0.0)
                    wline.append(f" \u2713{speedup:.2f}x", style=_SUCCESS) if speedup > 0 else wline.append(" \u2713", style=_SUCCESS)
                elif self._run_finalized:
                    # Run ended (target reached / budget / error) before this worker finished.
                    # Show cancelled instead of the stale in-progress attempt counter.
                    wline.append(" \u2014", style=_DIM)
                elif status == "waiting":
                    wline.append(" \u2026", style=_DIM)
                elif status == "stopped":
                    wline.append(" stop", style=_DIM)
                else:
                    wline.append(f" {rnd}/{max_rnd}", style=_DIM)
                wline.append("  ", style=_DIM)
            parts.append(wline)

            # Utilization gauges (if available) - compact
            if self._current_bw_pct > 0 or self._current_compute_pct > 0:
                util = Text("  ")
                if self._current_bw_pct > 0:
                    util.append(f"BW {self._current_bw_pct:.0f}%", style="#5a9ec9")
                    util.append("  ", style=_DIM)
                if self._current_compute_pct > 0:
                    util.append(f"Compute {self._current_compute_pct:.0f}%", style="#d77757")
                parts.append(util)

        parts.append(Text(""))

        # ============ BAND 3: HISTORY ============
        if self._iterations:
            for line in _iteration_log_compact(self._iterations, width=width):
                parts.append(line)

        # Phase/spinner line (only if active phase)
        if self._current_phase:
            phase = self._current_phase
            if len(phase) > 60:
                phase = phase[:58].rsplit(" ", 1)[0] + "\u2026"
            spinner_chars = "\u2818\u2838\u2830\u2834\u2826\u2827\u2807\u280f"
            sc = spinner_chars[int(elapsed * 4) % len(spinner_chars)]
            status_line = Text()
            status_line.append(f"  {sc} ", style=_CLAY)
            status_line.append(phase, style=_DIM)
            parts.append(status_line)

        return Group(*parts)
