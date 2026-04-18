"""Tool: generate_kernel -- describe what optimisation the Generator would try next."""

from __future__ import annotations

from typing import Any


def execute(session_context: dict, **kwargs: Any) -> str:
    """Describe what optimisation the Generator would try next.

    Does not actually generate a kernel -- just explains the intent based on
    current profiling data and iteration history.
    """
    iterations = session_context.get("iterations", [])
    if not iterations:
        return (
            "No iterations yet. The Generator would start with a baseline "
            "optimisation (e.g. basic tiling or vectorisation) based on the "
            "problem type and hardware."
        )

    # Find latest valid profile
    latest_profile = None
    latest_iter = None
    for it in reversed(iterations):
        profile = it.get("profile")
        if profile and it.get("status") not in ("compile_error", "error"):
            latest_profile = profile
            latest_iter = it
            break

    # Gather what has been tried
    tried_intents = []
    for it in iterations:
        intent = it.get("intent", "")
        if intent:
            tried_intents.append(
                f"  #{it.get('iteration')}: {intent} "
                f"(status={it.get('status')}, speedup={it.get('speedup', 0):.2f}x)"
            )

    lines = ["=== Generator next-step analysis ===", ""]

    if latest_profile and latest_iter:
        bottleneck = latest_profile.get("bottleneck_type", "unknown")
        bw = latest_profile.get("bandwidth_util", 0)
        cu = latest_profile.get("compute_util", 0)
        ce = latest_profile.get("cache_efficiency", 0)
        occ = latest_profile.get("occupancy", 0)

        lines.append(f"Based on iteration #{latest_iter.get('iteration')} profile:")
        lines.append(f"  Bottleneck: {bottleneck}")
        lines.append(f"  BW util: {bw:.0%}  |  Compute util: {cu:.0%}")
        lines.append(f"  Cache eff: {ce:.0%}  |  Occupancy: {occ:.2f}")
        lines.append("")

        # Determine next approach
        if bottleneck == "memory":
            if ce < 0.6:
                next_approach = "tiling/blocking to improve cache locality"
            elif bw < 0.5:
                next_approach = "memory coalescing or vectorised loads (float4)"
            else:
                next_approach = "shared memory staging to reduce global memory traffic"
        elif bottleneck == "compute":
            if occ < 0.5:
                next_approach = "reduce register pressure to improve occupancy"
            else:
                next_approach = "loop unrolling or tensor core usage for ILP"
        elif bottleneck == "latency":
            if occ < 0.5:
                next_approach = "increase occupancy via smaller block size or reduced register usage"
            else:
                next_approach = "software prefetching / double-buffering"
        else:
            next_approach = "general tiling + vectorisation sweep"

        lines.append(f"The Generator would try: {next_approach}")
    else:
        lines.append("No profiling data available. The Generator would run a baseline approach.")

    if tried_intents:
        lines.append("")
        lines.append("Previously tried:")
        lines.extend(tried_intents[-5:])  # last 5

    return "\n".join(lines)
