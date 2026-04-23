"""Worker-progress plots for /optimize (plot A) and /autopilot (plot C).

Two pure Rich renderers. Both return a Group ready to drop into Rich.Live
or console.print. Colors match the live_display palette.
"""

from __future__ import annotations

from typing import Sequence

from rich.console import Group
from rich.text import Text

# Palette
_CLAY = "#d77757"
_SUCCESS = "#4eba65"
_DIM = "#7a7a7a"
_GRID = "#3a3a3a"

_WCOLORS = [
    "#d77757", "#f5a850", "#4eba65", "#5a9ec9",
    "#8b6fa8", "#c97b9b", "#6ec0a0", "#b8a050",
]

_FULL = "\u2588"
_H_LINE = "\u2500"
_V_LINE = "\u2502"


def _worker_color(wid: int) -> str:
    return _WCOLORS[wid % len(_WCOLORS)]


def _best_so_far(points: Sequence[tuple[int, float, float]]) -> list[tuple[float, float]]:
    """(round, speedup, t) -> [(t, best_so_far)]"""
    out: list[tuple[float, float]] = []
    best = 0.0
    for _rnd, sp, t in sorted(points, key=lambda p: p[2]):
        if sp > best:
            best = sp
        out.append((t, best))
    return out


# ═══════════════════════════════════════════════════════════
# PLOT A — live multi-line (time x speedup)
# ═══════════════════════════════════════════════════════════
def render_live_lines(
    worker_speedups: dict[int, list[tuple[int, float, float]]],
    elapsed: float,
    width: int = 55,
    height: int = 9,
) -> Group:
    """Plot A — one step-line per worker plus global frontier."""
    if not worker_speedups:
        return Group(Text(""))

    per_worker = {wid: _best_so_far(pts) for wid, pts in worker_speedups.items()}
    all_points = [(t, b, wid) for wid, pts in per_worker.items() for (t, b) in pts]
    all_points.sort()

    frontier: list[tuple[float, float]] = []
    best = 0.0
    for t, b, _wid in all_points:
        if b > best:
            best = b
            frontier.append((t, best))

    max_speedup = best * 1.3 if best > 0 else 1.2  # 30% headroom above best
    t_max = max(elapsed, 1.0)

    def _x(t: float) -> int:
        return min(width - 1, max(0, int(t / t_max * (width - 1))))

    def _y(s: float) -> int:
        return min(height - 1, max(0, int(s / max_speedup * (height - 1))))

    # grid[y][x] = (char, style) — y=0 is TOP
    grid: list[list[tuple[str, str]]] = [
        [(" ", "") for _ in range(width)] for _ in range(height)
    ]

    # 1.0x baseline
    by = height - 1 - _y(1.0)
    if 0 <= by < height:
        for x in range(width):
            if x % 3 != 0:
                grid[by][x] = (_H_LINE, _GRID)

    # Plot ALL data as discrete dots only — no lines, no frontier line
    _DOT_CHAR = "\u25cf"  # ●
    for wid in sorted(per_worker):
        style = _worker_color(wid)
        for t, s in per_worker[wid]:
            x = _x(t)
            y = height - 1 - _y(s)
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = (_DOT_CHAR, style)

    # Derive round count and per-round concurrency from the raw data.
    # Each worker_speedups entry is (round, speedup, t); global wids may span rounds.
    wids_by_round: dict[int, list[int]] = {}
    for wid, pts in worker_speedups.items():
        for rnd, _sp, _t in pts:
            if rnd > 0 and wid not in wids_by_round.setdefault(rnd, []):
                wids_by_round[rnd].append(wid)
    rounds_seen = sorted(wids_by_round.keys())
    round_count = len(rounds_seen)
    concurrent = max((len(v) for v in wids_by_round.values()), default=len(per_worker))

    parts: list = []
    header = Text()
    header.append("  WORKERS  ", style=f"bold {_CLAY}")
    if round_count > 1:
        header.append(f"{concurrent} workers \u00d7 {round_count} rounds ", style=_DIM)
    else:
        header.append(f"{concurrent} workers ", style=_DIM)
    header.append("(self-reported, confirmed in history)", style="#555555 italic")
    parts.append(header)

    for yi, row in enumerate(grid):
        s_at = (height - 1 - yi) / (height - 1) * max_speedup
        line = Text()
        if yi % 2 == 0:
            line.append(f" {s_at:4.1f}\u00d7 ", style=_DIM)
        else:
            line.append("       ")
        line.append(_V_LINE, style=_DIM)
        for ch, st in row:
            line.append(ch, style=st or _DIM)
        parts.append(line)

    axis = Text()
    axis.append("       \u2514" + _H_LINE * width, style=_DIM)
    parts.append(axis)

    # X-axis labels with intermediate time markers
    xl = Text()
    xl.append("        ", style=_DIM)
    if elapsed <= 60:
        # Short: just 0s and end
        xl.append("0s", style=_DIM)
        xl.append(" " * max(0, width - 8))
        xl.append(f"{int(elapsed)}s", style=_DIM)
    else:
        # Longer: show markers at regular intervals, then always cap at end
        interval = 30 if elapsed <= 180 else 60 if elapsed <= 600 else 120
        last_drawn = 0
        t = 0
        while t <= elapsed:
            label = f"{int(t)}s"
            xl.append(label, style=_DIM)
            next_t = t + interval
            if next_t <= elapsed:
                # Pad the gap proportionally to x-axis scaling
                pad = max(1, int((next_t - t) / t_max * (width - 1)) - len(label))
                xl.append(" " * pad)
            last_drawn = t
            t = next_t
        # Append a final label at elapsed, but only if it's far enough from
        # the last regular tick to be visually distinct (≥ half an interval).
        # Otherwise the end-cap collides with the previous tick (e.g. "420s 421s").
        if int(elapsed) - last_drawn >= max(1, interval // 2):
            end_label = f"{int(elapsed)}s"
            pad = max(1, int((elapsed - last_drawn) / t_max * (width - 1)) - len(f"{int(last_drawn)}s"))
            xl.append(" " * pad)
            xl.append(end_label, style=_DIM)
    parts.append(xl)

    # Legend — compact horizontal, top workers only, skip 0.00x
    # NOTE: these are worker-reported speedups (from round JSON stdout),
    # NOT Modal-confirmed. The confirmed best is in the iteration table.
    ranked = [(wid, pts) for wid, pts in sorted(
        per_worker.items(),
        key=lambda kv: kv[1][-1][1] if kv[1] else 0,
        reverse=True,
    ) if pts and pts[-1][1] > 0]

    if ranked:
        # Per-round local IDs: within each round, sort global wids ascending
        # and assign 1-based positions. Keeps legend labels aligned with the
        # bar row's per-round W{id+1} scheme so the two don't diverge.
        local_id: dict[tuple[int, int], int] = {}
        for _rnd, _wids in wids_by_round.items():
            for _i, _w in enumerate(sorted(_wids)):
                local_id[(_rnd, _w)] = _i + 1

        latest_round: dict[int, int] = {}
        for _wid, _pts in worker_speedups.items():
            _rnds = [r for r, _s, _t in _pts if r > 0]
            if _rnds:
                latest_round[_wid] = max(_rnds)

        row = Text("  ")
        for i, (wid, pts) in enumerate(ranked[:6]):  # show top 6
            cur = pts[-1][1]
            rnd = latest_round.get(wid, 0)
            lid = local_id.get((rnd, wid), wid + 1)
            label = f"R{rnd}W{lid}" if round_count > 1 and rnd > 0 else f"W{lid}"
            row.append(f"\u25cf ", style=_worker_color(wid))
            row.append(f"{label} {cur:.2f}\u00d7", style=_worker_color(wid))
            row.append("  ")
        if len(ranked) > 6:
            row.append(f"+{len(ranked) - 6} more", style=_DIM)
        parts.append(row)

    return Group(*parts)


# ═══════════════════════════════════════════════════════════
# PLOT C — stacked columns by round
# ═══════════════════════════════════════════════════════════
def render_round_columns(
    round_history: list[dict],
    col_width: int = 8,
    height: int = 9,
) -> Group:
    """Plot C - per-round confirmed speedup bars.

    Each column = one round, bar height = Modal-confirmed speedup
    for that round (the "keep" decision metric). Per-worker self-reported
    speedups are NOT shown here — they appear in the live worker plot only.
    """
    if not round_history:
        return Group(Text(""))

    # Confirmed speedup for each round (Modal eval result)
    def _round_speedup(r: dict) -> float:
        # OptimizationLog format uses "speedup"; legacy uses "best_speedup"
        return r.get("speedup") or r.get("best_speedup") or 0.0

    speedups = [_round_speedup(r) for r in round_history]
    max_sp = max(speedups + [1.2])

    def _bar_h(s: float) -> int:
        if s <= 0:
            return 0
        return max(1, int(s / max_sp * (height - 1)))

    parts: list = []
    hdr = Text()
    hdr.append("  BY ROUND  ", style=f"bold {_CLAY}")
    best_confirmed = max(speedups) if speedups else 0.0
    hdr.append(f"{len(round_history)} rounds ", style=_DIM)
    hdr.append(f"\u00b7 best ", style=_DIM)
    best_color = _SUCCESS if best_confirmed >= 1.0 else _DIM
    hdr.append(f"{best_confirmed:.2f}\u00d7", style=f"bold {best_color}")
    hdr.append(" (confirmed)", style=_DIM)
    parts.append(hdr)

    # Baseline dashed line at 1.0x
    by = height - 1 - _bar_h(1.0)

    for yi in range(height - 1, -1, -1):
        s_at = (yi / max(height - 1, 1)) * max_sp
        line = Text()
        if (height - 1 - yi) % 2 == 0:
            line.append(f" {s_at:4.1f}\u00d7 ", style=_DIM)
        else:
            line.append("       ")
        line.append(_V_LINE, style=_DIM)

        for i, rnd in enumerate(round_history):
            s = _round_speedup(rnd)
            bh = _bar_h(s)
            ch = _FULL if yi < bh else " "
            # Baseline dashed line (only where there is no bar)
            if yi == by and ch == " ":
                ch = _H_LINE
                line.append(" " + ch * (col_width - 2) + " ", style=_GRID)
                continue
            # Color: green if beat 1.0x, clay otherwise
            is_best = (s == max(speedups)) and s > 0
            color = _SUCCESS if s >= 1.0 else (_CLAY if is_best else "#8b6fa8")
            line.append(" " + ch * (col_width - 2) + " ", style=color)
        parts.append(line)

    axis = Text()
    axis.append("       \u2514", style=_DIM)
    for _ in round_history:
        axis.append(_H_LINE * col_width, style=_DIM)
    parts.append(axis)

    xl = Text("        ")
    for rnd in round_history:
        label = f"r{rnd.get('round', '?')}"
        xl.append(label.center(col_width), style=_DIM)
    parts.append(xl)

    # Per-round speedup legend (confirmed values)
    for i, rnd in enumerate(round_history):
        s = _round_speedup(rnd)
        strat = rnd.get("strategy", "")
        if len(strat) > 40:
            strat = strat[:38] + "\u2026"
        is_best = s == max(speedups) and s > 0
        row = Text()
        row.append(f"  r{rnd.get('round', i+1)} ", style=_DIM)
        color = _SUCCESS if s >= 1.0 else "white"
        row.append(f"{s:.2f}\u00d7", style=f"bold {color}")
        if is_best:
            row.append("  \u2605", style=f"bold {_SUCCESS}")
        if strat:
            row.append(f"  {strat}", style=_DIM)
        parts.append(row)

    return Group(*parts)
