"""Iteration heatmap — visual summary of optimization runs.

Like a GitHub contribution graph but for kernel optimization iterations.
Each cell represents one iteration, colored by speedup.
"""
from rich.text import Text

# Color scale: low speedup → high speedup
_SPEEDUP_COLORS = [
    (0.0, "#6b6360"),   # gray — error/no data
    (0.5, "#ef4444"),   # red — slower than baseline
    (1.0, "#fbbf24"),   # yellow — around baseline
    (1.5, "#4ade80"),   # green — good speedup
    (2.0, "#22d3ee"),   # cyan — great speedup
    (3.0, "#c084fc"),   # purple — excellent
]

def _speedup_color(speedup: float) -> str:
    """Map speedup to a color."""
    if speedup <= 0:
        return _SPEEDUP_COLORS[0][1]
    for i in range(len(_SPEEDUP_COLORS) - 1):
        lo_s, lo_c = _SPEEDUP_COLORS[i]
        hi_s, hi_c = _SPEEDUP_COLORS[i + 1]
        if speedup <= hi_s:
            return hi_c
    return _SPEEDUP_COLORS[-1][1]

def _status_char(status: str) -> str:
    """Map status to a display character."""
    return {
        "keep": "█",
        "discard": "▓",
        "error": "░",
        "compile_error": "░",
        "incorrect": "▒",
    }.get(status, "·")

def render_iteration_heatmap(
    iterations: list[dict],
    width: int = 60,
    show_legend: bool = True,
) -> Text:
    """Render a compact heatmap of iteration results.

    Each iteration is one cell, colored by speedup.
    Best iteration is marked with ★.
    """
    if not iterations:
        return Text("[dim]No iterations to display[/dim]")

    result = Text()
    result.append("Iterations: ", style="bold")

    best_idx = 0
    best_speedup = 0
    for i, it in enumerate(iterations):
        sp = it.get("speedup", 0)
        if sp > best_speedup:
            best_speedup = sp
            best_idx = i

    # Render cells
    for i, it in enumerate(iterations):
        speedup = it.get("speedup", 0)
        status = it.get("decision", it.get("status", ""))
        color = _speedup_color(speedup)
        char = _status_char(status)

        if i == best_idx and best_speedup > 0:
            result.append("★", style=f"bold {color}")
        else:
            result.append(char, style=color)

    result.append(f"  ({len(iterations)} total)\n")

    # Markers
    if iterations:
        result.append(f"  1", style="dim")
        spacer = len(iterations) - 4
        if spacer > 0:
            result.append(" " * spacer, style="dim")
        result.append(f"{len(iterations)}\n", style="dim")

    # Legend
    if show_legend:
        result.append("  ", style="dim")
        result.append("█", style="#4ade80")
        result.append(" keep ", style="dim")
        result.append("▓", style="#fbbf24")
        result.append(" discard ", style="dim")
        result.append("░", style="#ef4444")
        result.append(" error ", style="dim")
        result.append("★", style="bold #22d3ee")
        result.append(" best\n", style="dim")

    return result
