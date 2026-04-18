"""Proactive advisor for stuck kernel optimization.

Tracks optimization progress and suggests next steps when the optimizer
plateaus -- i.e. when consecutive iterations fail to improve on the
current best speedup.

The advisor monitors two signals:

1. **Consecutive non-improvements** -- 5+ iterations without a new best.
2. **Plateau detection** -- no improvement in the last 10 iterations.

When triggered, the advisor inspects which approach *categories* have
already been tried (tiling, vectorization, tensor cores, etc.) and
cross-references them with the detected bottleneck type to generate
targeted suggestions.

Usage::

    from kernel_code.advisor import AdvisorState, should_advise, get_advice

    advisor = AdvisorState()
    # ... inside the optimization loop:
    advisor.record_iteration(speedup, decision, intent, bottleneck)
    if should_advise(advisor):
        print(get_advice(advisor))
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdvisorState:
    """Tracks optimization progress for proactive advice."""

    total_iterations: int = 0
    consecutive_non_improvements: int = 0
    last_improvement_iteration: int = 0
    current_best_speedup: float = 0.0
    current_bottleneck: str = "unknown"
    approaches_tried: list[str] = field(default_factory=list)  # categorized approaches
    approaches_kept: list[str] = field(default_factory=list)

    def record_iteration(
        self,
        speedup: float,
        decision: str,
        intent: str,
        bottleneck: str = "",
    ) -> None:
        """Record an iteration result.

        Args:
            speedup: The speedup achieved by this iteration.
            decision: ``"keep"`` or ``"discard"`` (or error status).
            intent: Free-text description of what the iteration tried.
            bottleneck: Current bottleneck type (e.g. ``"memory_bound"``).
        """
        self.total_iterations += 1
        if bottleneck:
            self.current_bottleneck = bottleneck

        # Categorize the approach
        category = _categorize_approach(intent)
        if category not in self.approaches_tried:
            self.approaches_tried.append(category)

        if decision == "keep" and speedup > self.current_best_speedup:
            self.current_best_speedup = speedup
            self.last_improvement_iteration = self.total_iterations
            self.consecutive_non_improvements = 0
            if category not in self.approaches_kept:
                self.approaches_kept.append(category)
        else:
            self.consecutive_non_improvements += 1


# ---------------------------------------------------------------------------
# Approach categories
# ---------------------------------------------------------------------------

_APPROACH_CATEGORIES: dict[str, list[str]] = {
    "tiling": ["tile", "block", "block_m", "block_n", "block_k"],
    "vectorization": ["vectorize", "float4", "vector", "coalesce"],
    "shared_memory": ["shared", "smem", "cache", "reuse"],
    "tensor_cores": ["tensor", "wmma", "mma", "tl.dot"],
    "fusion": ["fuse", "fusion", "epilogue", "inline"],
    "pipeline": ["pipeline", "async", "prefetch", "double buffer"],
    "reduction": ["reduction", "online", "welford", "scan"],
    "parallelism": ["split-k", "stream-k", "persistent", "cooperative"],
    "register": ["register", "unroll", "occupancy"],
    "algorithm": ["algorithm", "approach", "rewrite", "restructure"],
    "library": ["cublas", "cutlass", "cudnn", "library"],
    "backend_switch": ["cuda", "triton", "switch"],
}


def _categorize_approach(intent: str) -> str:
    """Categorize an optimization approach by keywords in *intent*."""
    intent_lower = intent.lower()
    for category, keywords in _APPROACH_CATEGORIES.items():
        if any(kw in intent_lower for kw in keywords):
            return category
    return "other"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def should_advise(state: AdvisorState) -> bool:
    """Check if we should proactively suggest next steps.

    Returns ``True`` when:
    - 5+ consecutive iterations produced no improvement, **or**
    - no improvement in the last 10 iterations (plateau).
    """
    # Advise after 5+ consecutive non-improvements
    if state.consecutive_non_improvements >= 5:
        return True
    # Advise if plateau: no improvement in last 10 iterations
    if (
        state.total_iterations - state.last_improvement_iteration >= 10
        and state.total_iterations > 0
    ):
        return True
    return False


def get_advice(state: AdvisorState, session_data: dict | None = None) -> str:
    """Generate contextual advice for a stuck optimizer.

    Examines which approach categories have been tried, which have not,
    and the current bottleneck to produce up to 3 targeted suggestions.

    Args:
        state: Current :class:`AdvisorState`.
        session_data: Optional session dict for additional context
            (currently unused but reserved for future enrichment).

    Returns:
        A multi-line string ready for console output.
    """
    parts: list[str] = []

    # Header
    stuck_for = state.consecutive_non_improvements
    parts.append(
        f"Stuck at {state.current_best_speedup:.2f}x for {stuck_for} iterations."
    )

    # What's been tried
    if state.approaches_tried:
        tried = ", ".join(state.approaches_tried)
        parts.append(f"Tried: {tried}")

    # Untried approaches
    all_categories = set(_APPROACH_CATEGORIES.keys())
    untried = all_categories - set(state.approaches_tried)

    # Generate suggestions based on bottleneck + untried
    suggestions: list[str] = []
    if state.current_bottleneck == "memory_bound":
        if "vectorization" not in state.approaches_tried:
            suggestions.append("Try vectorized float4 loads for better bandwidth")
        if "shared_memory" not in state.approaches_tried:
            suggestions.append("Add shared memory tiling to reduce DRAM traffic")
        if "fusion" not in state.approaches_tried:
            suggestions.append(
                "Fuse with adjacent ops to eliminate memory round-trips"
            )
    elif state.current_bottleneck == "compute_bound":
        if "tensor_cores" not in state.approaches_tried:
            suggestions.append("Use tensor core instructions (tl.dot for Triton)")
        if "pipeline" not in state.approaches_tried:
            suggestions.append(
                "Add software pipelining to overlap compute and memory"
            )
        if "register" not in state.approaches_tried:
            suggestions.append("Optimize register usage to improve occupancy")

    # Always suggest these if not tried
    if "backend_switch" not in state.approaches_tried:
        suggestions.append("Switch backend (Triton -> CUDA or vice versa)")
    if "algorithm" not in state.approaches_tried:
        suggestions.append("Try a fundamentally different algorithm")
    if "parallelism" not in state.approaches_tried:
        suggestions.append(
            "Try split-K or persistent kernel for better parallelism"
        )

    if suggestions:
        parts.append("Suggestions:")
        for i, s in enumerate(suggestions[:3], 1):
            parts.append(f"  {i}. {s}")

    return "\n".join(parts)
