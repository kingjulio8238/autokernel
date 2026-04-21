"""Kernel profiling report — Claude Code-aligned inline visualization.

Uses Claude Code's visual language:
- ⎿ connectors for sub-items
- ── horizontal rules for sections
- Bold headers, dim metadata
- Brand clay (#d77757), success green (#4eba65), error (#ff6b80)

Usage::

    from kernel_code.kernel_profile import render_kernel_profile
    render_kernel_profile(speedup=2.14, ...)
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

# Claude Code color palette
_CLAY = "#d77757"
_SUCCESS = "#4eba65"
_WARNING = "#ffc107"
_ERROR = "#ff6b80"
_DIM = "#999999"
_MUTED = "#777777"


def _gauge(label: str, value: float, width: int = 20) -> Text:
    """SOL gauge with bottleneck arrow."""
    pct = int(value * 100)
    filled = int(value * width)

    if pct >= 75:
        color = _ERROR
        suffix = "  \u2190 bottleneck"
    elif pct >= 50:
        color = _WARNING
        suffix = ""
    elif pct > 0:
        color = _DIM
        suffix = ""
    else:
        return Text()

    t = Text()
    t.append(f"  {label:<12}", style="bold white")
    t.append("\u2588" * filled, style=color)
    t.append("\u2591" * (width - filled), style="#333333")
    t.append(f"  {pct}%", style=f"bold {color}")
    t.append(suffix, style=f"{_ERROR}")
    return t


def render_kernel_profile(
    speedup: float = 0.0,
    ref_runtime_us: float = 0.0,
    kernel_runtime_us: float = 0.0,
    profile: dict | None = None,
    hardware: str = "L40S",
    console: Console | None = None,
    is_baseline: bool = False,
) -> None:
    """Render kernel profile inline using Claude Code visual language.

    When ``is_baseline=True`` the call is a reference-vs-reference baseline
    characterization: the Speedup headline and Runtime delta row are omitted
    and a single Reference runtime line is shown instead.
    """
    c = console or Console()
    prof = profile or {}

    c.print()
    c.print(f"  [bold white]\u2500\u2500 Kernel Profile \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold white]")
    c.print()

    if is_baseline:
        # Reference-vs-reference baseline: no speedup/delta to show.
        if ref_runtime_us > 0:
            c.print(
                f"  [bold white]Reference[/bold white]   "
                f"[white]{ref_runtime_us:.0f}μs[/white]   "
                f"[{_DIM}](5-trial mean)[/{_DIM}]"
            )
        else:
            c.print(f"  [bold white]Reference[/bold white]   [{_DIM}]runtime unavailable[/{_DIM}]")
    elif speedup > 1.0:
        c.print(f"  [bold white]Speedup[/bold white]   [{_SUCCESS}]{speedup:.2f}x[/{_SUCCESS}]", end="")
    elif speedup > 0:
        c.print(f"  [bold white]Speedup[/bold white]   [{_WARNING}]{speedup:.2f}x[/{_WARNING}]", end="")
    else:
        c.print(f"  [bold white]Speedup[/bold white]   [{_ERROR}]no correct kernel[/{_ERROR}]", end="")

    if is_baseline:
        pass  # Reference line above already ended with a newline.
    elif ref_runtime_us > 0 and kernel_runtime_us > 0:
        c.print(f"  [{_DIM}]({ref_runtime_us:.0f}\u03bcs \u2192 {kernel_runtime_us:.0f}\u03bcs)[/{_DIM}]")
    else:
        c.print()

    # SOL gauges
    compute = prof.get("compute_util", prof.get("compute_utilization", 0))
    memory = prof.get("bandwidth_util", prof.get("bandwidth_utilization", 0))
    occupancy = prof.get("occupancy", 0)

    if any([compute, memory, occupancy]):
        c.print()
        if compute > 0:
            c.print(_gauge("Compute", compute))
        if memory > 0:
            c.print(_gauge("Memory", memory))
        if occupancy > 0:
            c.print(_gauge("Occupancy", occupancy))

    # Bottleneck + memory sub-info via ⎿
    bottleneck = prof.get("bottleneck_type", "unknown")
    cache_eff = prof.get("cache_efficiency", 0)
    hw_bw = {"L40S": 864, "H100": 3350, "A100-80GB": 2039}.get(hardware, 864)

    sub_parts = []
    if bottleneck and bottleneck != "unknown":
        sub_parts.append(bottleneck.replace("_", " "))
    if cache_eff > 0:
        sub_parts.append(f"L2: {cache_eff:.0%}")
    if memory > 0:
        sub_parts.append(f"BW: {memory * hw_bw:.0f}/{hw_bw} GB/s")

    if sub_parts:
        c.print(f"  \u23bf  [{_DIM}]{' \u00b7 '.join(sub_parts)}[/{_DIM}]")

    # Before/after
    if not is_baseline and ref_runtime_us > 0 and kernel_runtime_us > 0:
        delta_pct = (kernel_runtime_us - ref_runtime_us) / ref_runtime_us * 100
        delta_color = _SUCCESS if delta_pct < 0 else _ERROR
        c.print()
        c.print(
            f"  [bold white]Runtime[/bold white]   "
            f"[{_DIM}]{ref_runtime_us:.0f}\u03bcs[/{_DIM}]"
            f"  \u2192  "
            f"[white]{kernel_runtime_us:.0f}\u03bcs[/white]"
            f"   [{delta_color}]{delta_pct:+.0f}%[/{delta_color}]"
        )

    c.print()
