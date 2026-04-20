"""Tool: profile_kernel -- analyse profiler data from the session."""

from __future__ import annotations

from typing import Any


def execute(session_context: dict, **kwargs: Any) -> str:
    """Analyse profiler data from the session, classify bottleneck, return summary.

    Optional kwargs:
        iteration (int): specific iteration to profile. Defaults to latest.
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
        # Find latest iteration with profile data
        for it in reversed(iterations):
            if it.get("profile") and it.get("status") not in ("compile_error", "error"):
                target = it
                break
        if target is None:
            return "No profiling data available in any iteration."

    profile = target.get("profile", {})
    if not profile:
        return f"Iteration #{target.get('iteration')} has no profiling data."

    bottleneck = profile.get("bottleneck_type", "unknown")
    bw = profile.get("bandwidth_util", 0)
    cu = profile.get("compute_util", 0)
    ce = profile.get("cache_efficiency", 0)
    occ = profile.get("occupancy", 0)
    headroom = profile.get("estimated_headroom", "unknown")
    stalls = profile.get("top_stalls", [])

    # Classify bottleneck severity
    if bottleneck == "memory":
        if bw > 0.8:
            severity = "near peak bandwidth -- limited headroom via memory path"
        elif bw > 0.5:
            severity = "moderate bandwidth utilisation -- room for coalescing/caching"
        else:
            severity = "low bandwidth utilisation -- likely redundant or strided accesses"
    elif bottleneck == "compute":
        if cu > 0.8:
            severity = "near peak compute -- limited headroom"
        elif cu > 0.5:
            severity = "moderate compute utilisation -- ILP or tensor core opportunity"
        else:
            severity = "low compute utilisation -- occupancy or divergence issue"
    elif bottleneck == "latency":
        severity = "latency-bound -- occupancy or prefetching may help"
    else:
        severity = "bottleneck type unknown"

    lines = [
        f"Profile analysis for iteration #{target.get('iteration')}",
        f"  Bottleneck:      {bottleneck} ({severity})",
        f"  Bandwidth util:  {bw:.0%}",
        f"  Compute util:    {cu:.0%}",
        f"  Cache efficiency: {ce:.0%}",
        f"  Occupancy:       {occ:.2f}",
        f"  Headroom:        {headroom}",
    ]
    if stalls:
        lines.append(f"  Top stalls:      {', '.join(stalls[:5])}")

    return "\n".join(lines)
