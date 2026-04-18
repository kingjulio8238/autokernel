"""Context compaction for long optimization sessions.

When an optimization session grows beyond ~20 iterations, the raw history
becomes too large (and too noisy) for the LLM context window.  This module
summarises long sessions into a concise representation that preserves
WHAT WORKED, WHAT DIDN'T, and WHAT TO TRY NEXT -- without the raw details
of every individual attempt.
"""

from __future__ import annotations

from collections import defaultdict


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4


def should_compact(session_data: dict, max_context_tokens: int = 3000) -> bool:
    """Check if session data would overflow context window.

    Returns True when the number of iterations exceeds 20 -- the threshold
    at which the full formatting in ``format_session_context`` starts to
    bloat the prompt beyond useful limits.
    """
    iterations = session_data.get("iterations", [])
    return len(iterations) > 20


def compact_session(
    session_data: dict,
    max_tokens: int = 3000,
    strategy: str = "balanced",
) -> str:
    """Summarise a long session into concise context.

    Strategies:
      - "balanced" (default): top 5 kept, failed categories, bottleneck,
        untried approaches.  Target ~3000 tokens.
      - "aggressive": only best kernel code (10 lines) + best speedup +
        problem info.  Everything else dropped.  Target ~500 tokens.
      - "minimal": single-line summary.  Target ~200 tokens.

    Keeps (in priority order for *balanced*):
      1. Current best kernel code (first 15 lines)
      2. Best speedup + problem info
      3. Top 5 kept optimisations (iter#, speedup, intent)
      4. Failed approach CATEGORIES (not individual iterations)
         e.g. "Tried tiling (5 attempts, best 1.3x), tried fusion (3 attempts, best 0.8x)"
      5. Current bottleneck from latest profiling
      6. What hasn't been tried yet

    Drops:
      - Raw profiler dumps
      - Intermediate kernel versions
      - Individual discarded iteration details
      - Duplicate / similar intents
    """
    iterations: list[dict] = session_data.get("iterations", [])
    kept = [it for it in iterations if it.get("decision") == "keep" or it.get("status") == "keep"]

    best_speedup = session_data.get("best_speedup", 0.0)
    for it in iterations:
        sp = it.get("speedup", 0.0)
        if sp and sp > best_speedup:
            best_speedup = sp

    # ------------------------------------------------------------------
    # "minimal" strategy -- single-line summary (~200 tokens)
    # ------------------------------------------------------------------
    if strategy == "minimal":
        max_chars = 200 * 4
        latest_profile = _latest_profile(iterations)
        bottleneck = "unknown"
        if latest_profile:
            bottleneck = latest_profile.get("bottleneck_type", "unknown")
        context = (
            f"Best: {best_speedup:.1f}x on {session_data.get('problem', 'unknown')}, "
            f"tried {len(iterations)} approaches, current bottleneck: {bottleneck}"
        )
        return context[:max_chars]

    # ------------------------------------------------------------------
    # "aggressive" strategy -- best kernel + key stats (~500 tokens)
    # ------------------------------------------------------------------
    if strategy == "aggressive":
        max_chars = 500 * 4
        parts: list[str] = []
        parts.append(
            f"=== Compacted (aggressive) ===\n"
            f"Problem: {session_data.get('problem', 'unknown')}\n"
            f"Total iterations: {len(iterations)}  |  Best speedup: {best_speedup:.2f}x"
        )
        best_code = _best_kernel_snippet(iterations, best_speedup)
        if best_code:
            code_lines = best_code.strip().splitlines()[:10]
            parts.append("=== Best Kernel (first 10 lines) ===\n" + "\n".join(code_lines))
        context = "\n\n".join(parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... (truncated)"
        return context

    # ------------------------------------------------------------------
    # "balanced" strategy (default) -- full summary (~3000 tokens)
    # ------------------------------------------------------------------
    max_chars = max_tokens * 4  # inverse of estimate_tokens
    parts: list[str] = []

    errors = [it for it in iterations if it.get("status") in ("compile_error", "incorrect", "error")]
    discarded = [it for it in iterations if it.get("decision") == "discard" or it.get("status") == "discard"]

    # ------------------------------------------------------------------
    # 1 & 2. Problem info + best speedup summary
    # ------------------------------------------------------------------
    summary = (
        f"=== Compacted Session Summary ===\n"
        f"Problem: {session_data.get('problem', 'unknown')}\n"
        f"Hardware: {session_data.get('hardware', 'unknown')}\n"
        f"Backend: {session_data.get('backend', 'unknown')}\n"
        f"Model: {session_data.get('model', 'unknown')}\n"
        f"Ref runtime: {session_data.get('ref_runtime_us', 0.0):.1f} us\n"
        f"Total iterations: {len(iterations)}\n"
        f"Kept: {len(kept)}  |  Discarded: {len(discarded)}  |  Errors: {len(errors)}\n"
        f"Best speedup: {best_speedup:.2f}x"
    )
    parts.append(summary)

    # ------------------------------------------------------------------
    # 1 (cont). Best kernel code snippet (first 15 lines)
    # ------------------------------------------------------------------
    best_code = _best_kernel_snippet(iterations, best_speedup)
    if best_code:
        code_lines = best_code.strip().splitlines()[:15]
        parts.append("=== Best Kernel (first 15 lines) ===\n" + "\n".join(code_lines))

    # ------------------------------------------------------------------
    # 3. Top 5 kept optimisations
    # ------------------------------------------------------------------
    if kept:
        sorted_kept = sorted(kept, key=lambda it: it.get("speedup", 0.0), reverse=True)[:5]
        lines = ["=== Top Kept Optimizations ==="]
        for it in sorted_kept:
            lines.append(
                f"  Iter #{it.get('iteration', '?')}: "
                f"{it.get('speedup', 0.0):.2f}x -- {it.get('intent', 'unknown')}"
            )
        parts.append("\n".join(lines))

    # ------------------------------------------------------------------
    # 4. Failed approach categories
    # ------------------------------------------------------------------
    approach_stats = _categorise_approaches(iterations)
    if approach_stats:
        lines = ["=== Failed Approach Categories ==="]
        for category, stats in approach_stats.items():
            lines.append(
                f"  {category}: {stats['attempts']} attempts, "
                f"best {stats['best_speedup']:.2f}x, "
                f"kept {stats['kept']}/{stats['attempts']}"
            )
        parts.append("\n".join(lines))

    # ------------------------------------------------------------------
    # 5. Current bottleneck from latest profiling
    # ------------------------------------------------------------------
    latest_profile = _latest_profile(iterations)
    if latest_profile:
        lines = ["=== Current Bottleneck ==="]
        lines.append(f"  Type: {latest_profile.get('bottleneck_type', 'unknown')}")
        bw = latest_profile.get("bandwidth_util")
        cu = latest_profile.get("compute_util")
        ce = latest_profile.get("cache_efficiency")
        occ = latest_profile.get("occupancy")
        if bw is not None:
            lines.append(f"  Bandwidth util: {bw:.0%}")
        if cu is not None:
            lines.append(f"  Compute util: {cu:.0%}")
        if ce is not None:
            lines.append(f"  Cache efficiency: {ce:.0%}")
        if occ is not None:
            lines.append(f"  Occupancy: {occ:.2f}")
        headroom = latest_profile.get("estimated_headroom")
        if headroom is not None:
            lines.append(f"  Estimated headroom: {headroom}")
        stalls = latest_profile.get("top_stalls")
        if stalls:
            lines.append(f"  Top stalls: {', '.join(stalls[:3])}")
        parts.append("\n".join(lines))

    # ------------------------------------------------------------------
    # 6. What hasn't been tried yet
    # ------------------------------------------------------------------
    untried = _untried_approaches(iterations)
    if untried:
        parts.append("=== Untried Approaches ===\n  " + ", ".join(untried))

    # ------------------------------------------------------------------
    # Assemble and enforce token budget
    # ------------------------------------------------------------------
    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n... (truncated)"
    return context


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_APPROACH_KEYWORDS: dict[str, list[str]] = {
    "tiling / blocking": ["tile", "tiling", "block", "blocking"],
    "vectorization": ["vector", "float4", "int4", "vectoriz"],
    "loop unrolling": ["unroll", "pragma unroll"],
    "shared memory": ["shared mem", "smem", "__shared__"],
    "memory coalescing": ["coalesc"],
    "fusion": ["fuse", "fusion", "fused"],
    "register blocking": ["register block", "register tile"],
    "occupancy tuning": ["occupancy", "block size", "blockdim"],
    "tensor cores": ["tensor core", "wmma", "mma.sync"],
    "prefetching": ["prefetch", "double buffer", "software pipelin"],
    "warp-level": ["warp shuffle", "warp reduce", "warp-level", "__shfl"],
    "algorithmic": ["algorithm", "restructur", "reorder", "transpose"],
}


def _classify_intent(intent: str) -> str:
    """Map an iteration intent string to a broad approach category."""
    intent_lower = intent.lower()
    for category, keywords in _APPROACH_KEYWORDS.items():
        for kw in keywords:
            if kw in intent_lower:
                return category
    return "other"


def _categorise_approaches(iterations: list[dict]) -> dict[str, dict]:
    """Group non-kept iterations by approach category.

    Returns a dict of category -> {attempts, best_speedup, kept} for
    categories that have at least one non-kept attempt.
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {"attempts": 0, "best_speedup": 0.0, "kept": 0}
    )

    for it in iterations:
        intent = it.get("intent", "")
        if not intent:
            continue
        category = _classify_intent(intent)
        entry = stats[category]
        entry["attempts"] += 1
        sp = it.get("speedup", 0.0)
        if sp > entry["best_speedup"]:
            entry["best_speedup"] = sp
        if it.get("decision") == "keep" or it.get("status") == "keep":
            entry["kept"] += 1

    # Only report categories with at least one non-kept attempt (i.e. failures)
    return {
        cat: s for cat, s in sorted(stats.items(), key=lambda x: -x[1]["attempts"])
        if s["attempts"] > s["kept"]
    }


def _untried_approaches(iterations: list[dict]) -> list[str]:
    """Return approach categories that have NOT been attempted."""
    tried: set[str] = set()
    for it in iterations:
        intent = it.get("intent", "")
        if intent:
            tried.add(_classify_intent(intent))

    all_categories = set(_APPROACH_KEYWORDS.keys())
    untried = sorted(all_categories - tried)
    return untried


def _latest_profile(iterations: list[dict]) -> dict | None:
    """Return the profile dict from the most recent correct iteration."""
    for it in reversed(iterations):
        profile = it.get("profile")
        if profile and it.get("status") not in ("compile_error", "error"):
            return profile
    if iterations:
        return iterations[-1].get("profile")
    return None


def _best_kernel_snippet(iterations: list[dict], best_speedup: float) -> str | None:
    """Return the kernel code from the iteration with the best speedup."""
    for it in iterations:
        if abs(it.get("speedup", 0.0) - best_speedup) < 0.01:
            snippet = it.get("kernel_code_snippet")
            if snippet:
                return snippet
    for it in reversed(iterations):
        if (it.get("decision") == "keep" or it.get("status") == "keep") and it.get("kernel_code_snippet"):
            return it["kernel_code_snippet"]
    return None
