"""Tool: compare_kernels -- compare two iterations side by side."""

from __future__ import annotations

from typing import Any


def execute(session_context: dict, **kwargs: Any) -> str:
    """Compare two iterations side by side (speedup, profiling metrics, key differences).

    Required kwargs:
        iter_a (int): first iteration number.
        iter_b (int): second iteration number.
    """
    iterations = session_context.get("iterations", [])
    iter_a = kwargs.get("iter_a")
    iter_b = kwargs.get("iter_b")

    if iter_a is None or iter_b is None:
        return "Both iter_a and iter_b are required."

    a = b = None
    for it in iterations:
        num = it.get("iteration")
        if num == iter_a:
            a = it
        if num == iter_b:
            b = it

    missing = []
    if a is None:
        missing.append(str(iter_a))
    if b is None:
        missing.append(str(iter_b))
    if missing:
        return (
            f"Iteration(s) #{', #'.join(missing)} not found. "
            f"Available: 1-{len(iterations)}"
        )

    def _fmt(it: dict) -> list[str]:
        lines = [
            f"  Status:    {it.get('status', 'unknown')}",
            f"  Speedup:   {it.get('speedup', 0):.2f}x",
            f"  Intent:    {it.get('intent', 'unknown')}",
        ]
        if it.get("runtime_us"):
            lines.append(f"  Runtime:   {it['runtime_us']:.1f} us")
        profile = it.get("profile", {})
        if profile:
            lines.append(f"  Bottleneck: {profile.get('bottleneck_type', 'unknown')}")
            lines.append(f"  Bandwidth:  {profile.get('bandwidth_util', 0):.0%}")
            lines.append(f"  Compute:    {profile.get('compute_util', 0):.0%}")
            lines.append(f"  Cache eff.: {profile.get('cache_efficiency', 0):.0%}")
            lines.append(f"  Occupancy:  {profile.get('occupancy', 0):.2f}")
        return lines

    lines = [f"=== Iteration #{iter_a} vs #{iter_b} ===", ""]
    lines.append(f"--- #{iter_a} ---")
    lines.extend(_fmt(a))
    lines.append("")
    lines.append(f"--- #{iter_b} ---")
    lines.extend(_fmt(b))

    # Deltas
    sp_a = a.get("speedup", 0)
    sp_b = b.get("speedup", 0)
    delta = sp_b - sp_a
    lines.append("")
    lines.append(
        f"Delta: #{iter_b} is {'+' if delta >= 0 else ''}{delta:.2f}x "
        f"{'faster' if delta > 0 else 'slower' if delta < 0 else 'same'} than #{iter_a}"
    )

    # Profile deltas if both have profiles
    pa = a.get("profile", {})
    pb = b.get("profile", {})
    if pa and pb:
        lines.append("")
        lines.append("Metric deltas (B - A):")
        for key, label in [
            ("bandwidth_util", "Bandwidth"),
            ("compute_util", "Compute"),
            ("cache_efficiency", "Cache eff."),
            ("occupancy", "Occupancy"),
        ]:
            va = pa.get(key, 0)
            vb = pb.get(key, 0)
            d = vb - va
            if key != "occupancy":
                lines.append(f"  {label}: {'+' if d >= 0 else ''}{d:.0%}")
            else:
                lines.append(f"  {label}: {'+' if d >= 0 else ''}{d:.2f}")

    return "\n".join(lines)
