"""Inline Rich.Live optimization display — Claude Code-aligned.

Uses Claude Code's visual language:
- ⎿ connectors for sub-info
- ── rules for sections
- Bold headers, dim metadata
- Brand clay (#d77757), success (#4eba65), error (#ff6b80)
- Minimal chrome, no heavy panels
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from rich import box
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
        self._target_sol: float | None = None
        self._round_markers: list[int] = []
        self._worker_states: list[dict] = []
        self._worker_speedups: dict[int, list[tuple[int, float, float]]] = {}
        self._show_plots: bool = True
        self._current_bw_pct: float = 0.0
        self._current_compute_pct: float = 0.0
        self._run_finalized: bool = False

        # --- Table layout (Option B) --------------------------------------
        # Default presentation: a single growing table, one row per completed
        # worker. The prior multi-band chart layout is preserved verbatim in
        # ``_build_legacy_view`` and opted into with OPENKERNEL_LEGACY_DISPLAY=1.
        self._use_table_layout: bool = (
            os.environ.get("OPENKERNEL_LEGACY_DISPLAY", "").strip() != "1"
        )
        # Live rows, in append order. Each row:
        #   {id, global_id, round_num, speedup, runtime_us, sol_score,
        #    status: "working" | "passed" | "failed" | "cancelled" | "stopped"}
        # Rows are created the first time a worker reports any progress and
        # updated in place on subsequent polls so the table animates as
        # rounds complete instead of waiting until finish().
        self._worker_rows: list[dict] = []
        # Index: global_id -> row dict (reference into _worker_rows) for O(1)
        # in-place updates from the 2-Hz poller.
        self._rows_by_gid: dict[int, dict] = {}
        # Global IDs that have reached a terminal state ("passed" / "failed"
        # / "cancelled" / "stopped"). Terminal rows stop updating so a late
        # "working" tick can't overwrite a final result.
        self._terminal_gids: set[int] = set()
        # Baseline runtime (µs) captured from the pre-run reference profile.
        # ``None`` means the caller didn't supply one — runtime column renders
        # as "—" in that case.
        self._baseline_us: float | None = None

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

    def set_target_sol(self, target_sol: float) -> None:
        """SOL target (0.0-1.0). Rendered alongside the best-SOL line when set."""
        self._target_sol = target_sol

    def set_baseline(self, ref_runtime_us: float) -> None:
        """Record the reference runtime (µs) used to derive per-worker
        runtime in the table view. No-op when <= 0."""
        if ref_runtime_us and ref_runtime_us > 0:
            self._baseline_us = float(ref_runtime_us)

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

        # Table layout: create a row the first time a worker reports
        # progress, and update it in place on every subsequent poll so the
        # table animates as rounds complete. Terminal states
        # ("passed"/"failed"/"cancelled"/"stopped") are locked once reached.
        if self._use_table_layout:
            for w in workers:
                self._upsert_worker_row(w)
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
        # Table layout: iteration data arrives only for the round's winning
        # kernel (after Modal re-eval). Latch its SOL onto the most recent
        # row that's still missing one so the % SOL column lights up.
        if self._use_table_layout and sol_score and sol_score > 0.0:
            for row in reversed(self._worker_rows):
                if row.get("sol_score") is None and row.get("status") == "passed":
                    row["sol_score"] = float(sol_score)
                    break
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
        # Table layout: lock any still-live row to a terminal state so the
        # final frame shows the full 1:1 worker→row mapping.
        if self._use_table_layout:
            for w in self._worker_states:
                gid = w.get("global_id", w.get("id", 0))
                if gid in self._terminal_gids:
                    continue
                status = w.get("status", "")
                terminal = "stopped" if status == "stopped" else "cancelled"
                if gid not in self._rows_by_gid:
                    self._upsert_worker_row(dict(w, status=terminal))
                else:
                    self._rows_by_gid[gid]["status"] = terminal
                self._terminal_gids.add(gid)
        self._refresh()
        if self._live:
            self._live.stop()
            self._live = None
        if stop_reason:
            self._console.print(f"  \u23bf  [{_DIM}]{stop_reason}[/{_DIM}]")

    def print_permanent(self, message: str) -> None:
        if self._live:
            self._live.console.print(message)

    _TERMINAL_STATUSES = ("passed", "failed", "cancelled", "stopped")

    def _upsert_worker_row(self, worker: dict) -> None:
        """Create or update the table row for ``worker``. Rows stay live
        (status/speedup/runtime update each poll) until the worker reports
        a terminal status — once terminal, the row is locked so a late
        "working" tick can't overwrite a final result.
        """
        gid = worker.get("global_id", worker.get("id", 0))
        if gid in self._terminal_gids:
            return

        status = worker.get("status", "") or "working"
        speedup = float(worker.get("speedup", 0.0) or 0.0)
        runtime_us: float | None = None
        if speedup > 0.0 and self._baseline_us and self._baseline_us > 0.0:
            runtime_us = self._baseline_us / speedup

        action = (worker.get("action") or "").strip()
        row = self._rows_by_gid.get(gid)
        if row is None:
            row = {
                "id": worker.get("id", gid),
                "global_id": gid,
                "round_num": self._current_round,
                "speedup": speedup,
                "runtime_us": runtime_us,
                "sol_score": None,
                "status": status,
                "action": action,
            }
            self._rows_by_gid[gid] = row
            self._worker_rows.append(row)
        else:
            # Monotonic on speedup — never regress the displayed best.
            if speedup > float(row.get("speedup") or 0.0):
                row["speedup"] = speedup
                row["runtime_us"] = runtime_us
            row["status"] = status
            # Only overwrite action on a non-empty update — terminal sweeps
            # carry diagnostics that the 2-Hz poller doesn't, and we want
            # them to stick.
            if action:
                row["action"] = action
            if self._current_round and not row.get("round_num"):
                row["round_num"] = self._current_round

        if status in self._TERMINAL_STATUSES:
            self._terminal_gids.add(gid)

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._build())

    def _build(self) -> Group:
        """Dispatch between the default table view (Option B) and the
        legacy 3-band chart layout. Toggle with OPENKERNEL_LEGACY_DISPLAY=1."""
        if self._use_table_layout:
            return self._build_table_view()
        return self._build_legacy_view()

    # ------------------------------------------------------------------
    # Option B — single growing table
    # ------------------------------------------------------------------

    def _build_table_view(self) -> Group:
        """One table, one row per completed worker. Header line above the
        table carries the run context + baseline so each row only shows
        per-worker information."""
        parts: list = []
        elapsed = time.time() - self._start_time

        # ---- Header: problem · hw · backend · baseline · elapsed ------
        header = Text()
        header.append("\n  \u2500\u2500 ", style="white")
        header.append((self._problem or "Optimizing")[:40], style="bold white")
        ctx = [self._hardware, self._backend]
        if self._problem_tier:
            ctx.append(self._problem_tier)
        if self._problem_type:
            ctx.append(self._problem_type)
        header.append(" \u00b7 " + " \u00b7 ".join(ctx), style="white")
        if self._baseline_us:
            header.append(f" \u00b7 baseline {self._baseline_us:.0f} \u00b5s", style="white")
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        header.append(f" \u00b7 {elapsed_str}", style="white")
        header.append(" \u2500\u2500", style="white")
        parts.append(header)

        # ---- Strategy (one line) ------------------------------------
        if self._current_strategy:
            strat = self._current_strategy
            if len(strat) > 72:
                strat = strat[:70].rsplit(" ", 1)[0] + "\u2026"
            s_line = Text()
            s_line.append("  \u25b6 ", style="white")
            s_line.append(strat, style="white")
            parts.append(s_line)

        parts.append(Text(""))

        # ---- Table -----------------------------------------------------
        table = Table(
            box=box.ROUNDED,  # full borders (top/bottom/sides) around the table
            show_header=True,
            show_edge=True,
            show_lines=True,  # horizontal divider between every row
            header_style="bold white",
            border_style="white",
            # pad_edge=True so the leftmost/rightmost columns get symmetric
            # padding. Combined with justify="center" on every column this
            # makes headers and cells visually centered under/over each other.
            pad_edge=True,
            padding=(0, 2),
            expand=False,
        )
        # All columns center-aligned (headers + cells) for a consistent
        # visual grid. Rich applies column ``justify`` to both header and
        # body cells, so centering here fixes the user-visible misalignment
        # on column headings that was most noticeable on the numeric columns.
        table.add_column("worker", style="white", justify="center", header_style="bold white", no_wrap=True)
        table.add_column("runtime", style="white", justify="center", header_style="bold white", no_wrap=True)
        table.add_column("% SOL", style="white", justify="center", header_style="bold white", no_wrap=True)
        table.add_column("speedup", style="white", justify="center", header_style="bold white", no_wrap=True)
        table.add_column("status", style="white", justify="center", header_style="bold white", no_wrap=True)

        # Status label mapping \u2014 no color, no bold. Minimalist. The status
        # column reads as plain white text for every outcome; "best" is
        # communicated via a bold speedup cell on the winning row instead.
        _STATUS_LABEL = {
            "passed":    "success",
            "failed":    "failed",
            "cancelled": "cancelled",
            "stopped":   "stopped",
            "working":   "running",
            "waiting":   "queued",
        }

        # Find the winning row: passed worker with the maximum speedup. Ties
        # break on first occurrence so the marker is deterministic.
        best_gid: object = None
        best_speedup = 0.0
        for row in self._worker_rows:
            if row.get("status") != "passed":
                continue
            sp = float(row.get("speedup") or 0.0)
            if sp > best_speedup:
                best_speedup = sp
                best_gid = row.get("global_id", row.get("id", 0))

        for row in self._worker_rows:
            gid = row.get("global_id", row.get("id", 0))
            worker_label = f"W{gid + 1}"
            if row.get("round_num"):
                worker_label = f"R{row['round_num']}\u00b7W{gid + 1}"

            speedup = float(row.get("speedup") or 0.0)
            runtime_us = row.get("runtime_us")
            sol_score = row.get("sol_score")
            status = row.get("status", "") or "waiting"

            runtime_cell = (
                f"{runtime_us:.0f} \u00b5s" if runtime_us is not None else "\u2014"
            )
            sol_cell = (
                f"{int(round(sol_score * 100))}%" if sol_score else "\u2014"
            )
            speedup_cell = f"{speedup:.2f}\u00d7" if speedup > 0.0 else "\u2014"
            status_label = _STATUS_LABEL.get(status, status or "\u2014")
            # Append the diagnostic action on non-success terminals so the
            # user can see *why* a row ended. Example: "failed (syntax
            # error)", "cancelled (LLM timeout)".
            action = (row.get("action") or "").strip()
            if action and status in ("failed", "cancelled", "stopped"):
                status_label = f"{status_label} ({action})"

            # Only the single winning row gets a bold speedup cell \u2014 that's
            # the sole visual signal of "this is the best". No color coding
            # on status, no "(best)" appendage: the bold number is the
            # signal.
            is_best = best_gid is not None and gid == best_gid
            speedup_style = "bold white" if is_best else "white"

            table.add_row(
                Text(worker_label, style="white"),
                Text(runtime_cell, style="white"),
                Text(sol_cell, style="white"),
                Text(speedup_cell, style=speedup_style),
                Text(status_label, style="white"),
            )

        if not self._worker_rows:
            # Empty-state placeholder so the table renders something immediately.
            table.add_row(
                Text("\u2014", style="white"),
                Text("\u2014", style="white"),
                Text("\u2014", style="white"),
                Text("\u2014", style="white"),
                Text("waiting\u2026", style="#999999"),
            )

        parts.append(table)

        # ---- Phase line (spinner) — only while a phase is active ------
        if self._current_phase and not self._run_finalized:
            phase = self._current_phase
            if len(phase) > 60:
                phase = phase[:58].rsplit(" ", 1)[0] + "\u2026"
            spinner_chars = "\u2818\u2838\u2830\u2834\u2826\u2827\u2807\u280f"
            sc = spinner_chars[int(elapsed * 4) % len(spinner_chars)]
            status_line = Text()
            status_line.append(f"  {sc} ", style="white")
            status_line.append(phase, style="white")
            parts.append(status_line)

        return Group(*parts)

    # ------------------------------------------------------------------
    # Legacy 3-band chart layout (opt-in via OPENKERNEL_LEGACY_DISPLAY=1)
    # ------------------------------------------------------------------

    def _build_legacy_view(self) -> Group:
        """Legacy layout: 3 bands — status, live work, history."""
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
            if self._target_sol:
                ctx2.append(f" / {self._target_sol:.2f} target", style=_DIM)
            ctx2.append(f" \u00b7 {self._best_speedup:.2f}x", style=_DIM)
        else:
            ctx2.append(f"{self._best_speedup:.2f}x", style=f"bold {sp_color}")
        # Show speedup target ONLY when user didn't pick a SOL target.
        # (When they picked SOL, the SOL target is the authoritative display above.)
        if self._target_speedup and not self._target_sol:
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
