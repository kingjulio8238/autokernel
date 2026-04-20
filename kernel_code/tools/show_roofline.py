"""Tool: show_roofline -- format roofline analysis for a kernel iteration."""

from __future__ import annotations

from typing import Any


def execute(session_context: dict, **kwargs: Any) -> str:
    """Format roofline analysis: arithmetic intensity, peak BW, peak compute,
    and where the kernel sits relative to the roofline ceiling.

    Optional kwargs:
        iteration (int): specific iteration. Defaults to latest with profile data.
    """
    iterations = session_context.get("iterations", [])
    if not iterations:
        return "No iterations available. Run /optimize first."

    iteration_num = kwargs.get("iteration")

    target = None
    if iteration_num is not None:
        for it in iterations:
            if it.get("iteration") == iteration_num:
                target = it
                break
        if target is None:
            return (
                f"Iteration #{iteration_num} not found. "
                f"Available iterations: 1-{len(iterations)}"
            )
    else:
        for it in reversed(iterations):
            if it.get("profile") and it.get("status") not in ("compile_error", "error"):
                target = it
                break
        if target is None:
            return "No profiling data available for roofline analysis."

    profile = target.get("profile", {})
    if not profile:
        return f"Iteration #{target.get('iteration')} has no profiling data."

    bottleneck = profile.get("bottleneck_type", "unknown")
    bw = profile.get("bandwidth_util", 0)
    cu = profile.get("compute_util", 0)
    roofline_pos = profile.get("roofline_position", 0)
    ce = profile.get("cache_efficiency", 0)
    occ = profile.get("occupancy", 0)

    # Determine which ceiling the kernel is under
    if bottleneck == "memory":
        ceiling = "memory bandwidth ceiling"
        utilisation = bw
    elif bottleneck == "compute":
        ceiling = "compute throughput ceiling"
        utilisation = cu
    else:
        ceiling = "latency/other ceiling"
        utilisation = max(bw, cu)

    # Estimate arithmetic intensity regime
    if bw > cu and bw > 0:
        ai_regime = "low arithmetic intensity (memory-bound regime)"
    elif cu > bw:
        ai_regime = "high arithmetic intensity (compute-bound regime)"
    else:
        ai_regime = "near the ridge point"

    lines = [
        f"=== Roofline Analysis -- Iteration #{target.get('iteration')} ===",
        "",
        f"Arithmetic intensity regime: {ai_regime}",
        f"Active ceiling:              {ceiling}",
        f"Roofline position:           {roofline_pos:.0%} of theoretical peak",
        "",
        "Hardware utilisation:",
        f"  Memory bandwidth:  {bw:.0%} of peak",
        f"  Compute throughput: {cu:.0%} of peak",
        f"  Cache efficiency:  {ce:.0%}",
        f"  Occupancy:         {occ:.2f}",
        "",
    ]

    # Gap analysis
    gap = 1.0 - roofline_pos
    if gap > 0.4:
        lines.append(f"Significant gap to ceiling ({gap:.0%}). Major optimisation opportunity.")
    elif gap > 0.15:
        lines.append(f"Moderate gap to ceiling ({gap:.0%}). Targeted optimisation possible.")
    else:
        lines.append(f"Close to ceiling ({gap:.0%} gap). Near-optimal for current algorithm.")

    headroom = profile.get("estimated_headroom", "unknown")
    if headroom != "unknown":
        lines.append(f"Estimated remaining headroom: {headroom}")

    return "\n".join(lines)
