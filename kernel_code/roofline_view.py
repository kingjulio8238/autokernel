"""ASCII roofline plot for terminal display.

Three views:
- B3.0: Compute view with landmark kernels
- B3.1: Your kernel pinned (after /optimize)
- B3.2: Memory/bandwidth view

Usage::

    from kernel_code.roofline_view import render_roofline
    render_roofline(console, view="compute", user_speedup=2.1, user_label="my_kernel")
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

_ACCENT = "magenta"
_DIM = "#999999"
_SUCCESS = "#4eba65"

# Landmark kernels: (name, operational_intensity, achieved_tflops_fraction)
_LANDMARKS = [
    ("vector add", 0.1, 0.02),
    ("softmax", 1.0, 0.08),
    ("layernorm", 2.0, 0.12),
    ("conv2d 3x3", 10.0, 0.35),
    ("GEMM 4K", 100.0, 0.70),
    ("flash attn", 50.0, 0.55),
    ("cuBLAS", 200.0, 0.85),
]

# Hardware specs: (peak_tflops_fp32, peak_bandwidth_gbs)
# All values are fp32 for consistent roofline comparison
_HW_SPECS: dict[str, tuple[float, float]] = {
    "L40S": (91.6, 864),         # fp32 / GDDR6X
    "H100": (756.0, 3350),      # fp32 dense / HBM3 (SXM5)
    "A100-80GB": (156.0, 2039),  # fp32 / HBM2e
    "A100-40GB": (156.0, 1555),  # fp32 / HBM2
}


def _get_hw_specs(hardware: str) -> tuple[float, float, float]:
    """Get (peak_tflops, peak_bw_gbs, ridge_oi) for hardware."""
    peak_tflops, peak_bw = _HW_SPECS.get(hardware, _HW_SPECS["L40S"])
    ridge_oi = peak_tflops * 1000 / peak_bw  # FLOP/byte
    return peak_tflops, peak_bw, ridge_oi


def render_roofline(
    console: Console,
    view: str = "compute",
    user_speedup: float = 0.0,
    user_label: str = "your kernel",
    hardware: str = "L40S",
    profile: dict | None = None,
) -> None:
    """Render an ASCII roofline plot.

    Args:
        profile: Optional profile dict from Modal eval containing
                 'operational_intensity', 'total_flops', 'total_bytes'.
                 When provided, uses measured OI for exact kernel placement.
    """
    peak_tflops, peak_bw, ridge_oi = _get_hw_specs(hardware)

    console.print()
    console.print(
        f"  [bold white]\u2500\u2500 Roofline ({hardware}) "
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold white]"
    )
    console.print()

    # Plot dimensions
    width = 50
    height = 12

    # Build the plot grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Draw axes
    for y in range(height):
        grid[y][4] = "\u2502"  # vertical axis
    for x in range(4, width):
        grid[height - 1][x] = "\u2500"  # horizontal axis
    grid[height - 1][4] = "\u2514"  # corner

    # Draw roofline: memory slope + compute ceiling
    ridge_x = int(4 + (_ridge_to_x(ridge_oi) * (width - 6)))
    ridge_x = min(ridge_x, width - 2)

    # Memory slope (left side)
    for x in range(5, ridge_x):
        y = height - 2 - int((x - 5) / max(ridge_x - 5, 1) * (height - 3))
        y = max(0, min(height - 2, y))
        grid[y][x] = "\u2571"  # /

    # Compute ceiling (right side)
    ceiling_y = 1
    for x in range(ridge_x, width - 1):
        grid[ceiling_y][x] = "\u2500"  # flat ceiling

    # Label the ceiling
    _place_label(grid, 0, ridge_x + 2, f"{peak_tflops:.0f} TFLOPs", width)

    # Plot landmarks
    for name, oi, frac in _LANDMARKS:
        x = int(4 + (_ridge_to_x(oi) * (width - 6)))
        y = height - 2 - int(frac * (height - 3))
        x = max(5, min(width - 2, x))
        y = max(1, min(height - 2, y))
        grid[y][x] = "\u25cf"  # ●

    # Plot user kernel
    oi_measured = False
    if user_speedup > 0 and view == "me":
        # Use measured OI from profile if available, else estimate
        user_oi = 1.0  # default heuristic
        if profile and profile.get("operational_intensity", 0) > 0:
            user_oi = profile["operational_intensity"]
            oi_measured = True
        user_frac = min(0.9, user_speedup * 0.08)
        x = int(4 + (_ridge_to_x(user_oi) * (width - 6)))
        y = height - 2 - int(user_frac * (height - 3))
        x = max(5, min(width - 2, x))
        y = max(1, min(height - 2, y))
        grid[y][x] = "\u2605"  # ★

    # Render
    for row in grid:
        line = "".join(row)
        # Color the roofline characters
        line = line.replace("\u2571", f"[{_ACCENT}]\u2571[/{_ACCENT}]")
        line = line.replace("\u25cf", f"[white]\u25cf[/white]")
        line = line.replace("\u2605", f"[bold {_SUCCESS}]\u2605[/bold {_SUCCESS}]")
        console.print(f"  {line}")

    # Axis labels
    console.print(f"       [{_DIM}]0.1        1         10        100   FLOP/byte[/{_DIM}]")
    console.print()

    # Legend
    if view == "me" and user_speedup > 0:
        oi_note = "measured OI" if oi_measured else "estimated OI"
        oi_val = f" · OI={user_oi:.1f}" if oi_measured else ""
        console.print(
            f"  [{_SUCCESS}]\u2605[/{_SUCCESS}] {user_label} "
            f"({user_speedup:.2f}x{oi_val}) [{_DIM}]{oi_note}[/{_DIM}]"
        )

    # Landmark legend
    legend = Text()
    legend.append("  Landmarks: ", style=_DIM)
    for i, (name, _, _) in enumerate(_LANDMARKS[:4]):
        if i > 0:
            legend.append(" \u00b7 ", style=_DIM)
        legend.append(f"\u25cf {name}", style="white")
    console.print(legend)

    if view == "mem":
        console.print(f"\n  [{_DIM}]Memory roof: {peak_bw} GB/s  |  Ridge point: {ridge_oi:.0f} FLOP/byte[/{_DIM}]")

    console.print()


def _ridge_to_x(oi: float) -> float:
    """Map operational intensity to x position (log scale 0.1 to 300)."""
    import math
    if oi <= 0:
        return 0
    log_min = math.log10(0.1)
    log_max = math.log10(300)
    return (math.log10(oi) - log_min) / (log_max - log_min)


def _place_label(grid: list[list[str]], y: int, x: int, label: str, max_x: int) -> None:
    """Place a label on the grid without overflow."""
    for i, ch in enumerate(label):
        px = x + i
        if 0 <= px < max_x and 0 <= y < len(grid):
            grid[y][px] = ch
