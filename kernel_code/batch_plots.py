"""Batch evaluation visualizations for multi-problem kernel optimization.

Renders Cursor-style aggregate views:
- Speedup distribution histogram (log-scale, gray/orange bars)
- Geomean comparison bar chart
- Problem taxonomy table with per-tier stats

All render as Rich Group objects for terminal display.
"""

from __future__ import annotations

import math
from typing import Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

# Palette (matches worker_plots.py)
_CLAY = "#d77757"
_SUCCESS = "#4eba65"
_DIM = "#7a7a7a"
_GRID = "#3a3a3a"
_CYAN = "#22d3ee"
_GOLD = "#f5a850"
_RED = "#e05252"
_WHITE = "white"

_FULL = "\u2588"
_H_LINE = "\u2500"
_V_LINE = "\u2502"


# ═══════════════════════════════════════════════════════════
# SPEEDUP DISTRIBUTION HISTOGRAM (log-scale X-axis)
# ═══════════════════════════════════════════════════════════

def render_speedup_histogram(
    speedups: Sequence[float],
    width: int = 50,
    height: int = 10,
) -> Group:
    """Render log-scale speedup distribution histogram.

    X-axis: log-scale speedup bins (0.25x to 256x)
    Y-axis: problem count
    Gray bars below 1.0x baseline, orange/green above.
    Includes geomean line annotation.

    Args:
        speedups: List of speedup values from batch results.
        width: Chart width.
        height: Chart height.
    """
    if not speedups:
        return Group(Text("  No speedup data.", style=_DIM))

    # Filter out zeros (failed problems)
    valid = [s for s in speedups if s > 0]
    if not valid:
        return Group(Text("  No valid speedup data.", style=_DIM))

    # Log-scale bins: 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256
    bin_edges = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    bin_labels = ["0.25x", "0.5x", "1x", "2x", "4x", "8x", "16x", "32x", "64x", "128x", "256x"]

    # Count problems per bin
    counts = [0] * len(bin_edges)
    for s in valid:
        placed = False
        for i, edge in enumerate(bin_edges):
            if i < len(bin_edges) - 1:
                if s < bin_edges[i + 1]:
                    counts[i] += 1
                    placed = True
                    break
        if not placed:
            counts[-1] += 1

    max_count = max(counts) if counts else 1

    # Compute geomean
    log_sum = sum(math.log(max(s, 0.01)) for s in valid)
    geomean = math.exp(log_sum / len(valid))

    parts: list = []

    # Header
    header = Text()
    header.append("  SPEEDUP DISTRIBUTION  ", style=f"bold {_CYAN}")
    header.append(f"{len(valid)} problems", style=_DIM)
    header.append(f"  geomean={geomean:.2f}x", style=f"bold {_GOLD}")
    parts.append(header)

    # Bars (vertical, drawn row by row from top)
    num_bins = len(bin_edges)
    bar_width = max(1, (width - 8) // num_bins)

    for row in range(height, 0, -1):
        threshold = row / height * max_count
        line = Text()
        # Y-axis label
        if row == height:
            line.append(f" {max_count:>3} ", style=_DIM)
        elif row == height // 2:
            line.append(f" {max_count // 2:>3} ", style=_DIM)
        else:
            line.append("     ")
        line.append(_V_LINE, style=_DIM)

        for i, count in enumerate(counts):
            if count >= threshold:
                # Color: gray below baseline (bins 0,1), orange/green above
                if i < 2:  # below 1.0x
                    color = _DIM
                elif i >= 3:  # >= 2.0x
                    color = _SUCCESS
                else:  # 1.0x - 2.0x
                    color = _CLAY
                line.append(_FULL * bar_width, style=color)
            else:
                line.append(" " * bar_width)

        parts.append(line)

    # X-axis
    axis = Text()
    axis.append("     \u2514" + _H_LINE * (bar_width * num_bins), style=_DIM)
    parts.append(axis)

    # X-axis labels
    xl = Text("      ")
    for i, label in enumerate(bin_labels):
        padded = label.center(bar_width)
        xl.append(padded, style=_DIM)
    parts.append(xl)

    # Baseline marker
    baseline_text = Text("  ")
    baseline_bin = 2  # 1.0x bin
    baseline_pos = baseline_bin * bar_width + 5
    baseline_text.append(" " * baseline_pos)
    baseline_text.append("\u25b2 baseline", style=_DIM)
    parts.append(baseline_text)

    return Group(*parts)


# ═══════════════════════════════════════════════════════════
# GEOMEAN COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════

def render_geomean_bars(
    systems: dict[str, float],
    width: int = 40,
) -> Group:
    """Render horizontal bar chart comparing system geomean speedups.

    Like Cursor's "Naive PyTorch 0.17x | Optimized PyTorch 1x | Agent 1.38x"

    Args:
        systems: Dict of system_name -> geomean_speedup.
            Example: {"Naive PyTorch": 0.17, "Optimized PyTorch": 1.0, "autokernel": 1.38}
    """
    if not systems:
        return Group(Text("  No comparison data.", style=_DIM))

    max_val = max(systems.values())
    if max_val <= 0:
        return Group(Text("  No positive speedups.", style=_DIM))

    parts: list = []

    header = Text()
    header.append("  PERFORMANCE COMPARISON  ", style=f"bold {_GOLD}")
    parts.append(header)

    for name, value in systems.items():
        line = Text()
        # Name (right-aligned in 18 chars)
        padded_name = name.rjust(18)
        line.append(f"  {padded_name} ", style=_WHITE)
        line.append(_V_LINE, style=_DIM)

        # Bar
        bar_len = int(value / max_val * width)
        bar_len = max(1, bar_len)

        # Color based on performance
        if value >= max_val * 0.95:
            color = _SUCCESS
        elif value >= 1.0:
            color = _CLAY
        else:
            color = _DIM

        line.append(_FULL * bar_len, style=color)
        line.append(f" {value:.2f}x", style=f"bold {color}")

        parts.append(line)

    # Baseline marker
    baseline_pos = int(1.0 / max_val * width) + 21  # offset for name column
    if baseline_pos < 80:
        bl = Text(" " * baseline_pos)
        bl.append("\u2502 1.0x", style=_DIM)
        parts.append(bl)

    return Group(*parts)


# ═══════════════════════════════════════════════════════════
# PROBLEM TAXONOMY TABLE
# ═══════════════════════════════════════════════════════════

def render_taxonomy_table(
    per_tier_stats: dict[str, dict[str, float]],
    total_problems: int = 0,
    geomean_speedup: float = 0.0,
) -> Group:
    """Render problem taxonomy table with per-tier stats.

    Like SOL-ExecBench's problem segmentation by type.

    Args:
        per_tier_stats: Dict from BatchResult.per_tier_stats.
        total_problems: Total problem count.
        geomean_speedup: Overall geomean speedup.
    """
    if not per_tier_stats:
        return Group(Text("  No tier data.", style=_DIM))

    table = Table(
        title="Problem Taxonomy",
        show_header=True,
        header_style=f"bold {_CYAN}",
        border_style=_GRID,
        title_style=f"bold {_GOLD}",
    )
    table.add_column("Tier", style=_WHITE, width=10)
    table.add_column("Count", style=_DIM, justify="right", width=7)
    table.add_column("Geomean", style=_CLAY, justify="right", width=9)
    table.add_column(">1x", style=_SUCCESS, justify="right", width=7)
    table.add_column(">=2x", style=_GOLD, justify="right", width=7)

    for tier in sorted(per_tier_stats.keys()):
        stats = per_tier_stats[tier]
        count = int(stats.get("count", 0))
        geomean = stats.get("geomean_speedup", 0.0)
        beating = stats.get("pct_beating_baseline", 0.0)
        exceeding = stats.get("pct_exceeding_2x", 0.0)

        table.add_row(
            tier,
            str(count),
            f"{geomean:.2f}x",
            f"{beating:.0f}%",
            f"{exceeding:.0f}%",
        )

    # Total row
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_problems}[/bold]",
        f"[bold]{geomean_speedup:.2f}x[/bold]",
        "",
        "",
        style="bold",
    )

    return Group(table)


# ═══════════════════════════════════════════════════════════
# FULL BATCH DASHBOARD (combines all three)
# ═══════════════════════════════════════════════════════════

def render_batch_dashboard(
    batch_result_dict: dict,
) -> Group:
    """Render complete batch results dashboard.

    Combines histogram + geomean bars + taxonomy table.

    Args:
        batch_result_dict: Output of BatchResult.to_dict().
    """
    parts: list = []

    # Title
    title = Text()
    title.append("\n  " + "=" * 56, style=_GRID)
    parts.append(title)
    title2 = Text()
    title2.append("  BATCH OPTIMIZATION RESULTS", style=f"bold {_CYAN}")
    parts.append(title2)
    title3 = Text()
    title3.append("  " + "=" * 56, style=_GRID)
    parts.append(title3)

    # Key metrics
    metrics = Text()
    total = batch_result_dict.get("total_problems", 0)
    geomean = batch_result_dict.get("geomean_speedup", 0.0)
    beating = batch_result_dict.get("pct_beating_baseline", 0.0)
    exceeding = batch_result_dict.get("pct_exceeding_2x", 0.0)
    sol = batch_result_dict.get("mean_sol_score", 0.0)
    cost = batch_result_dict.get("total_cost_usd", 0.0)

    metrics.append(f"  {total} problems", style=_WHITE)
    metrics.append(f"  |  Geomean: ", style=_DIM)
    metrics.append(f"{geomean:.2f}x", style=f"bold {_CLAY}")
    metrics.append(f"  |  SOL: ", style=_DIM)
    metrics.append(f"{sol:.2f}", style=f"bold {_CYAN}")
    metrics.append(f"  |  >1x: ", style=_DIM)
    metrics.append(f"{beating:.0f}%", style=_SUCCESS)
    metrics.append(f"  |  >=2x: ", style=_DIM)
    metrics.append(f"{exceeding:.0f}%", style=_GOLD)
    metrics.append(f"  |  ${cost:.2f}", style=_DIM)
    parts.append(metrics)
    parts.append(Text())

    # Histogram
    problem_speedups = [
        p.get("speedup", 0.0)
        for p in batch_result_dict.get("problems", [])
    ]
    parts.append(render_speedup_histogram(problem_speedups))
    parts.append(Text())

    # Taxonomy table
    parts.append(render_taxonomy_table(
        batch_result_dict.get("per_tier_stats", {}),
        total_problems=total,
        geomean_speedup=geomean,
    ))

    return Group(*parts)
