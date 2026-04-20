"""Tool: search_skills -- search skill library by keyword."""

from __future__ import annotations

from typing import Any


# Keyword categories (same set as the original implementation)
_SKILL_CATEGORIES: dict[str, str] = {
    "tiling": "Block/tile the computation to improve cache locality. Common tile sizes: 32x32, 64x64, 128x128.",
    "vectorization": "Use vector loads (float4, int4) to increase memory throughput per instruction.",
    "coalescing": "Ensure threads in a warp access consecutive memory addresses for maximum bandwidth.",
    "shared memory": "Stage data in shared memory to reduce global memory traffic. Use __syncthreads().",
    "loop unrolling": "Unroll inner loops with #pragma unroll to increase ILP and reduce branch overhead.",
    "fusion": "Fuse multiple operations into a single kernel to eliminate intermediate global memory writes.",
    "register blocking": "Keep working data in registers to avoid shared memory bank conflicts.",
    "occupancy": "Tune block size and resource usage to maximize the number of active warps.",
    "tensor cores": "Use wmma or mma.sync instructions for matrix operations on Tensor Core hardware.",
    "prefetching": "Use double-buffering or software pipelining to overlap compute with memory loads.",
}


def execute(session_context: dict, **kwargs: Any) -> str:
    """Search the skill library by keyword, return matching skills with templates.

    Required kwargs:
        query (str): what to search for.
    """
    query = kwargs.get("query", "")
    if not query:
        return "Please provide a search query."

    iterations = session_context.get("iterations", [])
    query_lower = query.lower()

    matches: list[str] = []

    # Search skill categories
    for cat, desc in _SKILL_CATEGORIES.items():
        if query_lower in cat or cat in query_lower:
            matches.append(f"[{cat}] {desc}")

    # Search iteration intents
    intent_matches: list[str] = []
    for it in iterations:
        intent = it.get("intent", "").lower()
        if query_lower in intent:
            intent_matches.append(
                f"  Iter #{it.get('iteration')}: {it.get('intent')} "
                f"(status={it.get('status')}, speedup={it.get('speedup', 0):.2f}x)"
            )

    if not matches and not intent_matches:
        return (
            f"No skills found matching '{query}'. "
            "Try: tiling, vectorization, coalescing, fusion, shared memory, loop unrolling."
        )

    lines: list[str] = []
    if matches:
        lines.append("Matching skill patterns:")
        for m in matches:
            lines.append(f"  {m}")
    if intent_matches:
        lines.append("\nPast iterations using similar techniques:")
        lines.extend(intent_matches)

    return "\n".join(lines)
