"""Tool: evaluate_kernel -- get eval results for a specific iteration or the best kernel."""

from __future__ import annotations

from typing import Any


def execute(session_context: dict, **kwargs: Any) -> str:
    """Get evaluation results for a specific iteration or the best kernel.

    Optional kwargs:
        iteration (int): specific iteration number. If omitted, returns the best.
    """
    iterations = session_context.get("iterations", [])
    if not iterations:
        return "No iterations available. Run /optimize first."

    iteration_num = kwargs.get("iteration")

    if iteration_num is not None:
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
    else:
        # Find the best kernel
        best_speedup = session_context.get("best_speedup", 0.0)
        target = None
        for it in iterations:
            if it.get("status") == "keep":
                sp = it.get("speedup", 0.0)
                if sp >= best_speedup or (target is None and sp > 0):
                    target = it
                    best_speedup = sp
        if target is None:
            return "No successful kernel found. All iterations failed or were discarded."

    lines = [
        f"Eval results for iteration #{target.get('iteration')}",
        f"  Status:       {target.get('status', 'unknown')}",
        f"  Speedup:      {target.get('speedup', 0):.2f}x",
        f"  Runtime:      {target.get('runtime_us', 0):.1f} us",
        f"  Ref runtime:  {target.get('ref_runtime_us', 0):.1f} us",
        f"  Intent:       {target.get('intent', 'unknown')}",
    ]

    if target.get("error"):
        err = target["error"]
        if len(err) > 200:
            err = err[:200] + "..."
        lines.append(f"  Error:        {err}")

    profile = target.get("profile", {})
    if profile:
        lines.append(f"  Bottleneck:   {profile.get('bottleneck_type', 'unknown')}")
        lines.append(f"  Bandwidth:    {profile.get('bandwidth_util', 0):.0%}")
        lines.append(f"  Compute:      {profile.get('compute_util', 0):.0%}")

    return "\n".join(lines)
