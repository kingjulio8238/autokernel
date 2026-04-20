"""Tool: explain_iteration -- detailed explanation of a specific iteration."""

from __future__ import annotations

from typing import Any


def execute(session_context: dict, **kwargs: Any) -> str:
    """Provide a detailed explanation of a specific iteration.

    Required kwargs:
        iteration (int): the iteration number to explain.
    """
    iteration_num = kwargs.get("iteration")
    if iteration_num is None:
        return "Missing required parameter 'iteration'."

    iterations = session_context.get("iterations", [])
    target = None
    for it in iterations:
        if it.get("iteration") == iteration_num:
            target = it
            break

    if target is None:
        return (
            f"Iteration #{iteration_num} not found. "
            f"Available iterations: 1-{len(iterations)}"
        )

    lines = [
        f"Iteration #{iteration_num}",
        f"  Status:  {target.get('status', 'unknown')}",
        f"  Speedup: {target.get('speedup', 0):.2f}x",
        f"  Intent:  {target.get('intent', 'unknown')}",
    ]

    if target.get("runtime_us"):
        lines.append(f"  Runtime: {target['runtime_us']:.1f} us")
    if target.get("ref_runtime_us"):
        lines.append(f"  Ref runtime: {target['ref_runtime_us']:.1f} us")

    profile = target.get("profile", {})
    if profile:
        lines.append("  Profile:")
        lines.append(f"    Bottleneck:    {profile.get('bottleneck_type', 'unknown')}")
        lines.append(f"    Bandwidth:     {profile.get('bandwidth_util', 0):.0%}")
        lines.append(f"    Compute:       {profile.get('compute_util', 0):.0%}")
        lines.append(f"    Cache eff.:    {profile.get('cache_efficiency', 0):.0%}")
        lines.append(f"    Occupancy:     {profile.get('occupancy', 0):.2f}")
        stalls = profile.get("top_stalls")
        if stalls:
            lines.append(f"    Top stalls:    {', '.join(stalls[:3])}")

    if target.get("error"):
        err = target["error"]
        if len(err) > 200:
            err = err[:200] + "..."
        lines.append(f"  Error: {err}")

    code = target.get("kernel_code_snippet", "")
    if code:
        code_lines = code.strip().splitlines()[:10]
        lines.append("  Code (first 10 lines):")
        for cl in code_lines:
            lines.append(f"    {cl}")

    return "\n".join(lines)
