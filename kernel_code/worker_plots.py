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

    max_speedup = max(best, 1.2)
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

    def _plot(points: list[tuple[float, float]], style: str) -> None:
        if not points:
            return
        prev_x, prev_y = 0, height - 1
        for t, s in points:
            x = _x(t)
            y = height - 1 - _y(s)
            for xi in range(prev_x, x + 1):
                if grid[prev_y][xi][0] in (" ", _H_LINE):
                    grid[prev_y][xi] = (_H_LINE, style)
            lo, hi = sorted([prev_y, y])
            for yi in range(lo, hi + 1):
                if yi == y:
                    grid[yi][x] = (_FULL, style)
                elif grid[yi][x][0] in (" ", _H_LINE):
                    grid[yi][x] = (_V_LINE, style)
            prev_x, prev_y = x, y
        for xi in range(prev_x + 1, width):
            if grid[prev_y][xi][0] in (" ", _H_LINE):
                grid[prev_y][xi] = (_H_LINE, style)

    for wid in sorted(per_worker):
        _plot(per_worker[wid], _worker_color(wid))
    _plot(frontier, _CLAY)

    parts: list = []
    header = Text()
    header.append("  WORKERS  ", style=f"bold {_CLAY}")
    header.append(f"best {best:.2f}\u00d7", style="bold white")
    header.append(f"  frontier across {len(per_worker)} workers", style=_DIM)
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

    xl = Text()
    xl.append("        0s", style=_DIM)
    xl.append(" " * max(0, width - 10))
    xl.append(f"{int(elapsed)}s", style=_DIM)
    parts.append(xl)

    # Legend
    ranked = sorted(per_worker.items(),
                    key=lambda kv: kv[1][-1][1] if kv[1] else 0,
                    reverse=True)
    for i, (wid, pts) in enumerate(ranked):
        cur = pts[-1][1] if pts else 0.0
        row = Text()
        row.append("  \u25cf ", style=_worker_color(wid))
        row.append(f"w{wid} {cur:.2f}\u00d7", style=f"bold {_worker_color(wid)}")
        if i == 0 and cur > 0:
            row.append(" \u2605", style=f"bold {_SUCCESS}")
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
    """Plot C — one column group per round, one bar per worker."""
    if not round_history:
        return Group(Text(""))

    wids: set[int] = set()
    for r in round_history:
        for pw in r.get("per_worker", []) or []:
            wids.add(int(pw.get("id", 0)))
    if not wids:
        wids = {0}
    worker_ids = sorted(wids)
    n_w = len(worker_ids)

    max_speedup = max(
        [r.get("best_speedup", 0.0) for r in round_history] + [1.2]
    )

    def _bar_h(s: float) -> int:
        if s <= 0:
            return 0
        return max(1, int(s / max_speedup * (height - 1)))

    parts: list = []
    hdr = Text()
    hdr.append("  BY ROUND  ", style=f"bold {_CLAY}")
    hdr.append(f"{len(round_history)} rounds \u00b7 {n_w} workers", style=_DIM)
    parts.append(hdr)

    bar_w = max(1, (col_width - 2) // max(n_w, 1))

    for yi in range(height - 1, -1, -1):
        s_at = (yi / max(height - 1, 1)) * max_speedup
        line = Text()
        if (height - 1 - yi) % 2 == 0:
            line.append(f" {s_at:4.1f}\u00d7 ", style=_DIM)
        else:
            line.append("       ")
        line.append(_V_LINE, style=_DIM)

        for rnd in round_history:
            pw_by_id = {int(p.get("id", 0)): p.get("speedup", 0.0)
                        for p in rnd.get("per_worker", []) or []}
            line.append(" ")
            if pw_by_id:
                for wid in worker_ids:
                    s = pw_by_id.get(wid, 0.0)
                    bh = _bar_h(s)
                    ch = _FULL if yi < bh else " "
                    line.append(ch * bar_w, style=_worker_color(wid))
            else:
                # Fallback: single bar from best_speedup
                s = rnd.get("best_speedup", 0.0)
                bh = _bar_h(s)
                ch = _FULL if yi < bh else " "
                line.append(ch * (col_width - 2), style=_CLAY)
            line.append(" ")
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

    # Legend
    best_by_wid: dict[int, float] = {wid: 0.0 for wid in worker_ids}
    for r in round_history:
        for pw in r.get("per_worker", []) or []:
            wid = int(pw.get("id", 0))
            s = float(pw.get("speedup", 0.0))
            if s > best_by_wid.get(wid, 0.0):
                best_by_wid[wid] = s
    ranked = sorted(best_by_wid.items(), key=lambda kv: kv[1], reverse=True)
    for i, (wid, bst) in enumerate(ranked):
        row = Text()
        row.append("  \u25a0 ", style=_worker_color(wid))
        row.append(f"worker {wid}  {bst:.2f}\u00d7",
                   style=f"bold {_worker_color(wid)}")
        if i == 0 and bst > 0:
            row.append("  \u2605", style=f"bold {_SUCCESS}")
        parts.append(row)

    return Group(*parts)
