"""Compact one-line-per-iteration formatting for optimization output.

Replaces the verbose multi-line iteration output with aligned, scannable lines:
  #1  0.63x  ✗ naive tiling BLOCK_SIZE=128
  #2  ──     ! compile error
  #3  1.21x  ✓ vectorized float4 loads          ← new best
"""
from rich.text import Text


def format_iteration_line(
    iteration: int,
    speedup: float,
    status: str,  # "keep", "discard", "error", "compile_error", "incorrect"
    intent: str,
    is_new_best: bool = False,
    max_intent_width: int = 45,
) -> Text:
    """Format a single iteration as a compact one-liner.

    Returns a Rich Text object with colored markers and aligned columns.
    """
    line = Text()

    # Iteration number (right-aligned, 3 chars)
    line.append(f" #{iteration:<3}", style="bold")

    # Speedup (right-aligned, 6 chars)
    if status in ("error", "compile_error"):
        line.append("  ──   ", style="dim")
    else:
        color = "#4ade80" if status == "keep" else "#ef4444" if speedup < 1.0 else "#e0ddd8"
        line.append(f" {speedup:5.2f}x ", style=color)

    # Status marker
    if status == "keep":
        line.append("✓ ", style="#4ade80 bold")
    elif status in ("error", "compile_error"):
        line.append("! ", style="#fbbf24 bold")
    elif status == "incorrect":
        line.append("? ", style="#fbbf24")
    else:
        line.append("✗ ", style="#ef4444")

    # Intent (truncated)
    intent_display = intent[:max_intent_width]
    if len(intent) > max_intent_width:
        intent_display = intent_display[:-1] + "…"

    if status == "keep":
        line.append(intent_display, style="#4ade80")
    elif status in ("error", "compile_error"):
        line.append(intent_display, style="#fbbf24 dim")
    else:
        line.append(intent_display, style="dim")

    # New best marker
    if is_new_best:
        line.append("  ← new best", style="#4ade80 bold")

    return line


def format_optimization_header(
    iterations: int,
    backend: str = "triton",
    hardware: str = "L40S",
    estimated_cost: float = 0.0,
) -> Text:
    """Format the optimization start header."""
    header = Text()
    header.append("\nStarting optimization: ", style="bold")
    header.append(f"{iterations} iterations", style="bold #4ade80")
    header.append(" | ", style="dim")
    header.append(backend, style="bold")
    header.append(" | ", style="dim")
    header.append(hardware, style="bold")
    header.append(" | ", style="dim")
    header.append(f"${estimated_cost:.2f} est.", style="bold #fbbf24")
    header.append("\n")
    return header
