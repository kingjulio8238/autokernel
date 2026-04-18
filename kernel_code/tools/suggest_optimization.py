"""Tool: suggest_optimization -- suggest the next technique based on current state."""

from __future__ import annotations

from typing import Any


def execute(session_context: dict, **kwargs: Any) -> str:
    """Based on current bottleneck and what has been tried, suggest the next
    optimisation technique.
    """
    iterations = session_context.get("iterations", [])
    if not iterations:
        return "No optimization data yet. Run /optimize first."

    # Find the latest valid profile
    latest_profile = None
    latest_iter = None
    for it in reversed(iterations):
        profile = it.get("profile")
        if profile and it.get("status") not in ("compile_error", "error"):
            latest_profile = profile
            latest_iter = it
            break

    if latest_profile is None:
        return "No profiling data available to make a suggestion."

    bottleneck = latest_profile.get("bottleneck_type", "unknown")
    bw = latest_profile.get("bandwidth_util", 0)
    cu = latest_profile.get("compute_util", 0)
    ce = latest_profile.get("cache_efficiency", 0)
    occ = latest_profile.get("occupancy", 0)
    headroom = latest_profile.get("estimated_headroom", "unknown")

    lines = [
        f"Analysis based on iteration #{latest_iter.get('iteration')} profile:",
        f"  Bottleneck: {bottleneck}",
        f"  Bandwidth util: {bw:.0%}  |  Compute util: {cu:.0%}",
        f"  Cache efficiency: {ce:.0%}  |  Occupancy: {occ:.2f}",
        f"  Estimated headroom: {headroom}",
        "",
    ]

    # Bottleneck-specific suggestions
    if bottleneck == "memory":
        suggestions = [
            "Try memory coalescing -- ensure threads in a warp access consecutive addresses.",
            "Increase shared memory usage to reduce global memory traffic.",
            "If bandwidth utilization is low, look for redundant loads/stores.",
        ]
    elif bottleneck == "compute":
        suggestions = [
            "Try instruction-level parallelism -- unroll inner loops.",
            "Consider using tensor cores / warp-level matrix ops if applicable.",
            "Reduce register pressure to improve occupancy.",
        ]
    elif bottleneck == "latency":
        suggestions = [
            "Increase occupancy to hide latency -- reduce register/shared memory usage.",
            "Try prefetching data into shared memory.",
            "Check for warp divergence in conditional branches.",
        ]
    else:
        suggestions = [
            "Review the top stall reasons and address them directly.",
            "Try a different tiling strategy.",
            "Consider fusing adjacent operations.",
        ]

    if ce < 0.6:
        suggestions.insert(0, f"Cache efficiency is low ({ce:.0%}) -- consider tiling or blocking.")
    if occ < 0.5:
        suggestions.insert(0, f"Occupancy is low ({occ:.2f}) -- reduce register/smem usage.")

    lines.append("Suggestions:")
    for i, s in enumerate(suggestions[:3], 1):
        lines.append(f"  {i}. {s}")

    return "\n".join(lines)
