"""SOL-based visualization plots for kernel optimization.

Follows Cursor/NVIDIA's SOL-ExecBench visualization patterns:
- SOL Trajectory: Y=SOL score (0-1), X=iteration, with baseline/ceiling lines
- Strategy Timeline: horizontal dot progression with SOL at each stage

Both render as Rich Group objects for terminal display.
"""

from __future__ import annotations

from typing import Sequence

from rich.console import Group
from rich.text import Text

# Palette (matches worker_plots.py)
_CLAY = "#d77757"
_SUCCESS = "#4eba65"
_DIM = "#7a7a7a"
_GRID = "#3a3a3a"
_CYAN = "#22d3ee"
_GOLD = "#f5a850"
_WHITE = "white"

_FULL = "\u2588"
_H_LINE = "\u2500"
_V_LINE = "\u2502"
_DOT = "\u25cf"
_DOT_SM = "\u2022"
_ARROW = "\u2192"


# ═══════════════════════════════════════════════════════════
# SOL TRAJECTORY — Y: SOL score (0-1), X: iteration
# ═══════════════════════════════════════════════════════════

def render_sol_trajectory(
    rounds: list[dict],
    width: int = 55,
    height: int = 11,
) -> Group:
    """Render SOL score trajectory with baseline/ceiling lines.

    Follows Cursor's annotated trajectory visualization:
    - Y-axis: SOL score 0.0 to 1.0
    - X-axis: Round/iteration number
    - Dashed line at 0.5 (baseline)
    - Dashed line at 1.0 (theoretical limit)
    - Strategy annotations at inflection points

    Args:
        rounds: List of round dicts from optimization_log.to_dict_list().
            Each must have 'round', 'speedup', 'strategy', and optionally
            'profile' dict with 'sol_score'.
        width: Chart width in characters.
        height: Chart height in characters.
    """
    if not rounds:
        return Group(Text("  No SOL data yet.", style=_DIM))

    # Extract SOL scores and strategies
    data: list[tuple[int, float, str]] = []
    for r in rounds:
        round_num = r.get("round", 0)
        sol = 0.0
        profile = r.get("profile", {})
        if isinstance(profile, dict):
            sol = profile.get("sol_score", 0.0)
        # Fallback: estimate SOL from speedup (0.5 * speedup, capped at 1.0)
        if sol <= 0 and r.get("speedup", 0) > 0:
            sol = min(1.0, 0.5 * r["speedup"])
        strategy = r.get("strategy", "")
        data.append((round_num, sol, strategy))

    if not data:
        return Group(Text("  No SOL data yet.", style=_DIM))

    max_round = max(d[0] for d in data)
    best_sol = max(d[1] for d in data)

    # Grid: y=0 is TOP, y=height-1 is BOTTOM
    # SOL range: 0.0 (bottom) to 1.0 (top)
    grid: list[list[tuple[str, str]]] = [
        [(" ", "") for _ in range(width)] for _ in range(height)
    ]

    def _x(round_num: int) -> int:
        if max_round <= 1:
            return 0
        return min(width - 1, max(0, int((round_num - 1) / max(max_round - 1, 1) * (width - 1))))

    def _y(sol: float) -> int:
        """Map SOL score (0-1) to grid y (0=top, height-1=bottom)."""
        row = int((1.0 - sol) * (height - 1))
        return min(height - 1, max(0, row))

    baseline_y = _y(0.5)
    ceiling_y = _y(1.0)  # should be row 0

    # Use distinct dot glyph so data doesn't blend with baseline line
    _DOT = "\u25cf"  # bullet

    # Plot SOL data points as DOTS only (no connecting lines -
    # lines blend with baseline when points sit near 0.5).
    for round_num, sol, _strategy in data:
        x = _x(round_num)
        y = _y(sol)
        grid[y][x] = (_DOT, f"bold {_CLAY}")

    # Shift baseline row if any data dot falls exactly on it, so
    # both the dot and the baseline line remain visible.
    dots_on_baseline = any(_y(sol) == baseline_y for _, sol, _ in data)
    effective_baseline_y = baseline_y
    if dots_on_baseline and baseline_y + 1 < height:
        effective_baseline_y = baseline_y + 1

    # Draw horizontal reference lines AFTER data (empty cells only)
    for x in range(width):
        if x % 3 != 0:
            if grid[effective_baseline_y][x][0] == " ":
                grid[effective_baseline_y][x] = ("\u2508", _DIM)
            if grid[ceiling_y][x][0] == " ":
                grid[ceiling_y][x] = ("\u2508", _GRID)

    # Detect strategy inflection points (where strategy changed)
    annotations: list[tuple[int, float, str]] = []
    prev_strategy = ""
    for round_num, sol, strategy in data:
        if strategy and strategy != prev_strategy and prev_strategy:
            short = strategy[:20] + "\u2026" if len(strategy) > 20 else strategy
            annotations.append((round_num, sol, short))
        prev_strategy = strategy

    # Render
    parts: list = []

    # Header
    header = Text()
    header.append("  SOL TRAJECTORY  ", style=f"bold {_CYAN}")
    header.append(f"best {best_sol:.2f}/1.00 ceiling", style="bold white")
    header.append(f"  ({len(data)} round{'s' if len(data) != 1 else ''})", style=_DIM)
    parts.append(header)

    # Subtitle: define SOL inline so KEs don't have to remember the acronym
    subtitle = Text(
        "  SOL = Speed of Light (fraction of hardware peak). 1.0 = theoretical maximum.",
        style=_DIM,
    )
    parts.append(subtitle)

    # Grid rows with Y-axis labels
    for yi in range(height):
        sol_at = 1.0 - yi / (height - 1)
        line = Text()
        # Y-axis label at key positions
        if abs(sol_at - 1.0) < 0.05:
            line.append(" 1.0  ", style=_CYAN)
        elif abs(sol_at - 0.75) < 0.05:
            line.append(" 0.75 ", style=_DIM)
        elif abs(sol_at - 0.5) < 0.05:
            line.append(" 0.5  ", style=_DIM)
        elif abs(sol_at - 0.25) < 0.05:
            line.append(" 0.25 ", style=_DIM)
        elif abs(sol_at) < 0.05:
            line.append(" 0.0  ", style=_DIM)
        else:
            line.append("      ")
        line.append(_V_LINE, style=_DIM)
        for ch, st in grid[yi]:
            line.append(ch, style=st or _DIM)

        # Add reference labels at right edge
        if yi == ceiling_y:
            line.append("  Peak (1.0×)", style=_GRID)
        elif yi == baseline_y:
            line.append("  Reference (1.0× speedup)", style=_DIM)

        parts.append(line)

    # X-axis
    axis = Text()
    axis.append("      \u2514" + _H_LINE * width, style=_DIM)
    parts.append(axis)

    # X-axis labels
    xl = Text()
    xl.append("       ")
    if max_round <= 5:
        for i in range(1, max_round + 1):
            pos = _x(i)
            xl.append(f"R{i}", style=_DIM)
            spacing = (_x(i + 1) if i < max_round else width) - pos - 2
            if spacing > 0:
                xl.append(" " * spacing)
    else:
        xl.append("R1", style=_DIM)
        xl.append(" " * max(0, width - 8))
        xl.append(f"R{max_round}", style=_DIM)
    parts.append(xl)

    # Annotations (strategy inflection points)
    if annotations:
        ann_line = Text("  ")
        ann_line.append("Strategy changes: ", style=f"bold {_DIM}")
        for i, (rnd, sol, label) in enumerate(annotations[:3]):
            if i > 0:
                ann_line.append(f" {_ARROW} ", style=_DIM)
            ann_line.append(f"R{rnd}", style=_GOLD)
            ann_line.append(f" {label}", style=_WHITE)
            ann_line.append(f" ({sol:.2f})", style=_DIM)
        parts.append(ann_line)

    return Group(*parts)


# ═══════════════════════════════════════════════════════════
# STRATEGY PROGRESSION TIMELINE — horizontal dot chart
# ═══════════════════════════════════════════════════════════

def render_strategy_timeline(
    rounds: list[dict],
    width: int = 60,
) -> Group:
    """Render horizontal strategy progression timeline.

    Shows optimization strategy stages as labeled dots on a
    horizontal bar, like Cursor's GEMM progression chart:
    General 4.5% → Blackwell instructions 15.8% → ...

    Args:
        rounds: List of round dicts with 'strategy', 'speedup',
            and optionally 'profile.sol_score'.
    """
    if not rounds:
        return Group(Text("  No strategy data yet.", style=_DIM))

    # Group consecutive rounds by strategy
    stages: list[dict] = []
    current_strategy = ""
    for r in rounds:
        strategy = r.get("strategy", "general optimization")
        speedup = r.get("speedup", 0.0)
        sol = 0.0
        profile = r.get("profile", {})
        if isinstance(profile, dict):
            sol = profile.get("sol_score", 0.0)

        if strategy != current_strategy:
            stages.append({
                "strategy": strategy,
                "best_speedup": speedup,
                "best_sol": sol,
                "rounds": 1,
                "start_round": r.get("round", 0),
            })
            current_strategy = strategy
        else:
            if stages:
                stages[-1]["best_speedup"] = max(stages[-1]["best_speedup"], speedup)
                stages[-1]["best_sol"] = max(stages[-1]["best_sol"], sol)
                stages[-1]["rounds"] += 1

    if not stages:
        return Group(Text("  No strategy data yet.", style=_DIM))

    parts: list = []

    # Header
    header = Text()
    header.append("  STRATEGY PROGRESSION  ", style=f"bold {_GOLD}")
    header.append(f"{len(stages)} stage{'s' if len(stages) != 1 else ''}", style=_DIM)
    parts.append(header)

    # Timeline bar
    num_stages = len(stages)
    dot_spacing = max(1, (width - 4) // max(num_stages, 1))

    # Dot line
    dot_line = Text("  ")
    for i, stage in enumerate(stages):
        # Color: gray for early, gold for middle, green for best
        sol = stage["best_sol"]
        speedup = stage["best_speedup"]
        if i == num_stages - 1:
            color = _SUCCESS
        elif sol > 0.7 or speedup > 2.0:
            color = _GOLD
        else:
            color = _DIM

        dot_line.append(_DOT, style=f"bold {color}")

        # Connection line to next dot
        if i < num_stages - 1:
            dash_count = dot_spacing - 1
            dot_line.append(_H_LINE * dash_count, style=_GRID)

    parts.append(dot_line)

    # Labels line (strategy name + metric)
    label_line = Text("  ")
    for i, stage in enumerate(stages):
        short_name = stage["strategy"]
        if len(short_name) > 15:
            short_name = short_name[:14] + "\u2026"

        # Show SOL if available, otherwise speedup
        if stage["best_sol"] > 0:
            metric = f"{stage['best_sol']:.2f}"
        elif stage["best_speedup"] > 0:
            metric = f"{stage['best_speedup']:.1f}x"
        else:
            metric = "?"

        label = f"{short_name} {metric}"
        max_label_width = dot_spacing
        if len(label) > max_label_width:
            label = label[:max_label_width - 1] + "\u2026"

        color = _SUCCESS if i == num_stages - 1 else _DIM
        label_line.append(label, style=color)

        padding = max(0, dot_spacing - len(label))
        if i < num_stages - 1 and padding > 0:
            label_line.append(" " * padding)

    parts.append(label_line)

    # Summary line — show vs-baseline context, not just inter-round delta
    if stages:
        first_metric = stages[0]["best_speedup"]
        last_metric = stages[-1]["best_speedup"]
        summary = Text("  ")
        # Color end based on baseline (1.0x), not vs start
        end_color = _SUCCESS if last_metric >= 1.0 else _GOLD if last_metric >= 0.9 else "#ff6b80"

        # Only show Start → End when there are multiple stages; otherwise
        # first_metric == last_metric and the arrow is tautological.
        if len(stages) > 1:
            summary.append(f"Start: {first_metric:.2f}x", style=_DIM)
            summary.append(f"  {_ARROW}  ", style=_GRID)
            summary.append(f"End: {last_metric:.2f}x", style=f"bold {end_color}")

            # Inter-round delta (informational, parenthesized)
            if last_metric != first_metric and first_metric > 0:
                delta = (last_metric - first_metric) / first_metric * 100
                sign = "+" if delta >= 0 else ""
                summary.append(f"  ({sign}{delta:.0f}% round-over-round)", style=_DIM)
        else:
            # Single-stage run: report best directly rather than Start→End
            summary.append(f"Best: {last_metric:.2f}x", style=f"bold {end_color}")

        # vs-baseline: the number the KE actually cares about
        if last_metric <= 0.0:
            summary.append("  — no correct kernel produced", style="#ff6b80")
        elif last_metric < 1.0:
            below = (1.0 - last_metric) * 100
            summary.append(f"  — still {below:.0f}% below baseline", style="#ff6b80")
        else:
            above = (last_metric - 1.0) * 100
            summary.append(f"  — {above:.0f}% above baseline", style=_SUCCESS)
        parts.append(summary)

    return Group(*parts)
