"""Worker progress plots for /optimize and /autopilot.

Plot A — live multi-line chart during /optimize:
  Each worker gets a step-line showing speedup over time.
  A "frontier" line tracks the best-so-far across all workers.

Plot C — stacked columns by round for /autopilot:
  After each round, shows a column chart of per-worker speedups.

Usage::

    from kernel_code.worker_plots import render_live_lines, render_round_columns
"""

from __future__ import annotations

from rich.text import Text

_ACCENT = "#d77757"
_SUCCESS = "#4eba65"
_DIM = "#999999"
_BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
_WORKER_COLORS = ["#22d3ee", "#4ade80", "#fbbf24", "#c084fc"]


def render_live_lines(
    worker_speedups: dict[int, list[tuple[int, float, float]]],
    elapsed: float,
    width: int = 50,
    height: int = 8,
) -> Text:
    """Render live multi-line speedup chart (Plot A).

    Args:
        worker_speedups: {worker_id: [(round, speedup, time_s), ...]}
        elapsed: total elapsed seconds
        width: chart width in chars
        height: chart height in rows
    """
    if not worker_speedups:
        return Text()

    # Find ranges
    all_speedups = [s for wdata in worker_speedups.values() for _, s, _ in wdata if s > 0]
    if not all_speedups:
        return Text()

    max_speedup = max(all_speedups) * 1.1
    max_speedup = max(max_speedup, 1.1)  # at least show 1.0x line
    frontier = max(all_speedups)

    result = Text()
    result.append(f"  WORKERS", style="bold white")
    result.append(f"  best {frontier:.2f}\u00d7\n", style=f"bold {_SUCCESS}")

    # Build grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Y axis
    for y in range(height):
        grid[y][6] = "\u2502"
    # X axis
    for x in range(6, width):
        grid[height - 1][x] = "\u2500"
    grid[height - 1][6] = "\u2514"

    # Y labels
    for y_idx in [0, height // 2, height - 2]:
        val = max_speedup * (1.0 - y_idx / (height - 1))
        label = f"{val:4.1f}\u00d7"
        for i, ch in enumerate(label[:5]):
            if i < 6:
                grid[y_idx][i] = ch

    # 1.0x reference line
    ref_y = int((1.0 - 1.0 / max_speedup) * (height - 2))
    ref_y = max(0, min(height - 2, ref_y))
    for x in range(7, width):
        if grid[ref_y][x] == " ":
            grid[ref_y][x] = "\u2504"  # dotted line at 1.0x

    # Plot each worker's speedup trajectory
    for wid, data in sorted(worker_speedups.items()):
        if not data:
            continue
        color_idx = wid % len(_WORKER_COLORS)

        for i, (rnd, speedup, t) in enumerate(data):
            if speedup <= 0:
                continue
            # Map time to x position
            x = 7 + int(t / max(elapsed, 1) * (width - 9))
            x = max(7, min(width - 2, x))
            # Map speedup to y position
            y = int((1.0 - speedup / max_speedup) * (height - 2))
            y = max(0, min(height - 2, y))
            grid[y][x] = "\u25cf"  # ●

    # Render grid
    for row in grid:
        line = "  " + "".join(row)
        result.append(line + "\n")

    # X axis labels
    mid_t = elapsed / 2
    result.append(f"         0s")
    result.append(f"{'':>{width // 3}}{mid_t:.0f}s")
    result.append(f"{'':>{width // 3}}{elapsed:.0f}s\n")

    # Legend
    for wid, data in sorted(worker_speedups.items()):
        if not data:
            continue
        color = _WORKER_COLORS[wid % len(_WORKER_COLORS)]
        best = max(s for _, s, _ in data) if data else 0
        result.append(f"  [{color}]\u25cf[/{color}] w{wid} {best:.2f}\u00d7")
        if wid == max(worker_speedups.keys()):
            result.append("\n")
        else:
            result.append("  ")

    return result


def render_round_columns(
    round_history: list[dict],
    width: int = 50,
    height: int = 10,
) -> Text:
    """Render stacked columns by round (Plot C).

    Args:
        round_history: list of round dicts with best_speedup
    """
    if not round_history:
        return Text()

    result = Text()
    result.append("\n  BY ROUND\n", style="bold white")

    max_speedup = max(r.get("best_speedup", 0) for r in round_history)
    max_speedup = max(max_speedup, 1.1)

    # Column chart using block characters
    for y_level in range(height, 0, -1):
        threshold = y_level / height * max_speedup
        line = f"  {threshold:5.1f}\u00d7 \u2502"

        for i, rnd in enumerate(round_history):
            speedup = rnd.get("best_speedup", 0)
            if speedup >= threshold:
                line += " \u2588\u2588"
            else:
                line += "   "

        result.append(line + "\n")

    # X axis
    result.append(f"        \u2514")
    for i in range(len(round_history)):
        result.append("\u2500\u2500\u2500")
    result.append("\n")

    # Round labels
    result.append("         ")
    for i, rnd in enumerate(round_history):
        result.append(f"R{rnd.get('round', i + 1):<2}")
    result.append("\n")

    # Best per round
    result.append("\n")
    for rnd in round_history:
        num = rnd.get("round", "?")
        speedup = rnd.get("best_speedup", 0)
        strategy = rnd.get("strategy", "")[:25]
        color = _SUCCESS if speedup > 1.0 else _DIM
        result.append(f"  [{color}]R{num}: {speedup:.2f}\u00d7[/{color}]")
        if strategy:
            result.append(f"  [{_DIM}]{strategy}[/{_DIM}]")
        result.append("\n")

    return result
