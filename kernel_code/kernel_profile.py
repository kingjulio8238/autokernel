"""Kernel profiling report — ncu-style inline visualization.

Renders a dense, kernel-engineer-focused profile report after optimization:
- Speed of Light gauges (compute SOL, memory SOL, occupancy)
- Roofline position indicator
- Before/after comparison table
- Bottleneck identification in under 5 lines

Inspired by Nsight Compute's SOL summary — the most valuable thing
a kernel engineer can see.

Usage::

    from kernel_code.kernel_profile import render_kernel_profile

    render_kernel_profile(
        speedup=2.14,
        ref_runtime_us=2900,
        kernel_runtime_us=1355,
        profile={"compute_util": 0.34, "bandwidth_util": 0.87, ...},
        console=console,
    )
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# Warm earth tone palette (Claude Code aligned)
_ACCENT = "#d97757"     # clay/terracotta
_SUCCESS = "#4ade80"     # green (kept for success)
_WARNING = "#fbbf24"     # amber
_ERROR = "#ef4444"       # red
_MUTED = "#888888"       # muted gray
_DIM = "#555555"         # dim gray


def _sol_gauge(label: str, value: float, width: int = 20) -> Text:
    """Render a Speed of Light gauge with bottleneck indicator."""
    pct = int(value * 100)
    filled = int(value * width)

    if pct >= 80:
        color = _ERROR  # at ceiling — this IS the bottleneck
        indicator = " \u2190 bottleneck" if pct >= 70 else ""
    elif pct >= 50:
        color = _WARNING
        indicator = ""
    else:
        color = _MUTED
        indicator = ""

    t = Text()
    t.append(f"  {label:<12}", style="bold white")
    t.append("\u2588" * filled, style=color)
    t.append("\u2591" * (width - filled), style=_DIM)
    t.append(f"  {pct}%", style=f"bold {color}")
    t.append(indicator, style=f"bold {_ERROR}")
    return t


def _metric_row(label: str, before: str, after: str, delta: str, delta_style: str) -> list:
    """Build a row for the before/after table."""
    return [label, before, after, delta]


def render_kernel_profile(
    speedup: float = 0.0,
    ref_runtime_us: float = 0.0,
    kernel_runtime_us: float = 0.0,
    profile: dict | None = None,
    hardware: str = "L40S",
    console: Console | None = None,
) -> None:
    """Render a kernel profiling report as an inline Rich panel."""
    c = console or Console()
    prof = profile or {}

    content = Text()
    content.append("\n")

    # === Speedup headline ===
    if speedup > 1.0:
        content.append(f"  Speedup     ", style="bold white")
        content.append(f"{speedup:.2f}x", style=f"bold {_SUCCESS}")
    elif speedup > 0:
        content.append(f"  Speedup     ", style="bold white")
        content.append(f"{speedup:.2f}x", style=f"bold {_WARNING}")
    else:
        content.append(f"  Speedup     ", style="bold white")
        content.append("no correct kernel", style=f"bold {_ERROR}")

    if ref_runtime_us > 0 and kernel_runtime_us > 0:
        content.append(
            f"  (ref: {ref_runtime_us:.0f}\u03bcs \u2192 kernel: {kernel_runtime_us:.0f}\u03bcs)",
            style=_MUTED,
        )
    content.append("\n\n")

    # === Speed of Light gauges ===
    compute_util = prof.get("compute_util", prof.get("compute_utilization", 0))
    bw_util = prof.get("bandwidth_util", prof.get("bandwidth_utilization", 0))
    occupancy = prof.get("occupancy", 0)

    if any([compute_util, bw_util, occupancy]):
        content.append("  Speed of Light\n", style="bold white")
        if compute_util > 0:
            content.append_text(_sol_gauge("Compute", compute_util))
            content.append("\n")
        if bw_util > 0:
            content.append_text(_sol_gauge("Memory", bw_util))
            content.append("\n")
        if occupancy > 0:
            content.append_text(_sol_gauge("Occupancy", occupancy))
            content.append("\n")
        content.append("\n")

    # === Bottleneck identification ===
    bottleneck = prof.get("bottleneck_type", "unknown")
    if bottleneck and bottleneck != "unknown":
        bn_color = _ERROR if "memory" in bottleneck else _WARNING if "compute" in bottleneck else _ACCENT
        content.append("  Bottleneck  ", style="bold white")
        content.append(bottleneck.upper().replace("_", " "), style=f"bold {bn_color}")
        content.append("\n")

        # Actionable hint based on bottleneck
        if "memory" in bottleneck:
            content.append("              Optimize memory access: vectorize loads, improve coalescing, use shared memory\n", style=_MUTED)
        elif "compute" in bottleneck:
            content.append("              Optimize compute: use tensor cores (tl.dot), increase arithmetic intensity\n", style=_MUTED)
        content.append("\n")

    # === Memory metrics ===
    cache_eff = prof.get("cache_efficiency", 0)
    if cache_eff > 0 or bw_util > 0:
        content.append("  Memory      ", style="bold white")
        parts = []
        if cache_eff > 0:
            parts.append(f"L2 hit: {cache_eff:.0%}")
        if bw_util > 0:
            # Estimate actual bandwidth from utilization + hardware peak
            hw_bw = {"L40S": 864, "H100": 3350, "A100-80GB": 2039}.get(hardware, 864)
            actual_bw = bw_util * hw_bw
            parts.append(f"BW: {actual_bw:.0f}/{hw_bw} GB/s")
        content.append("  ".join(parts), style="white")
        content.append("\n")

    # === Before/After comparison ===
    if ref_runtime_us > 0 and kernel_runtime_us > 0:
        content.append("\n")
        content.append("  Before \u2192 After\n", style="bold white")

        # Runtime
        runtime_delta = (kernel_runtime_us - ref_runtime_us) / ref_runtime_us * 100
        rt_style = _SUCCESS if runtime_delta < 0 else _ERROR
        content.append(f"  {'Runtime':<12}", style="white")
        content.append(f"{ref_runtime_us:>8.0f}\u03bcs", style=_MUTED)
        content.append("  \u2192  ", style=_DIM)
        content.append(f"{kernel_runtime_us:>8.0f}\u03bcs", style="white")
        content.append(f"   {runtime_delta:+.0f}%\n", style=f"bold {rt_style}")

    content.append("")

    # Render panel
    panel = Panel(
        content,
        title="[bold white] Kernel Profile [/bold white]",
        border_style=_ACCENT,
        padding=(0, 2),
        width=min(c.width, 80),
    )
    c.print(panel)
