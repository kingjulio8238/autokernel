"""LLM reflection between autonomous optimization rounds.

After each round, the LLM reviews the trajectory and decides:
- CONTINUE — refine the current best approach
- PIVOT — abandon current strategy, try something fundamentally different
- STOP — goal is unreachable or no headroom left

This is the "reflect" step in the evaluate-reflect-adapt loop.

Usage::

    from kernel_code.meta_reflect import reflect_on_round, MetaReflection

    reflection = await reflect_on_round(round_history, best=1.8, target=2.0, llm=llm)
    if reflection.action == "pivot":
        next_strategy = reflection.next_strategy
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from openkernel.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class MetaReflection:
    """Result of the LLM reflecting on a round."""

    action: str = "continue"  # "continue" | "pivot" | "stop"
    next_strategy: str = ""  # what to try next (for continue/pivot)
    reason: str = ""  # why this decision
    confidence: float = 0.5  # 0-1, how confident the LLM is


def _build_reflection_prompt(
    rounds: list[dict],
    best_speedup: float,
    target_speedup: float,
    budget_remaining: float,
    problem_context: str = "",
    strategy_hints: list[str] | None = None,
) -> str:
    """Build a compact reflection prompt (~450 tokens).

    Args:
        problem_context: Optional string describing problem type + bottleneck
            (e.g. "L1 elementwise, launch-bound"). Used to filter strategies.
        strategy_hints: Optional problem-specific best-practice bullets from
            the classifier (e.g. histogram → "warp-aggregated atomics on
            high-contention inputs"). Surfaced verbatim so the pivot prompt
            picks a strategy that matches the real bottleneck.
    """
    round_summaries = []
    for r in rounds:
        strategy = r.get("strategy", "general optimization")
        best = r.get("best_speedup", 0.0)
        kept = r.get("kept", 0)
        total = r.get("total", 0)
        errors = r.get("errors", 0)
        bottleneck = r.get("bottleneck", "unknown")
        round_summaries.append(
            f"  Round {r.get('round', '?')}: strategy=\"{strategy}\" "
            f"best={best:.2f}x, {kept}/{total} kept, {errors} errors, "
            f"bottleneck={bottleneck}"
        )

    trajectory = "\n".join(round_summaries)

    # Build problem-aware guidance
    problem_section = ""
    if problem_context:
        problem_section = f"\nPROBLEM: {problem_context}\n"
        if "launch-bound" in problem_context.lower() or "launch-overhead" in problem_context.lower():
            problem_section += (
                "\nIMPORTANT: This is a launch-overhead-bound problem. "
                "PyTorch's fused elementwise kernels are already near-optimal at this size.\n"
                "AVOID suggesting: block size tuning, shared memory tiling, tensor cores, "
                "TMEM, pipeline fusion — these don't help when kernel launch dominates.\n"
                "CONSIDER instead: grid-stride loops (process more elements per launch), "
                "multi-op fusion, or STOP if 2+ rounds can't beat baseline.\n"
            )

    hints_section = ""
    if strategy_hints:
        bullets = "\n".join(f"- {h}" for h in strategy_hints)
        hints_section = (
            "\n## Problem-specific strategy hints (from classifier)\n"
            f"{bullets}\n\n"
            "Strongly consider these when choosing the next strategy — they "
            "reflect known best practices for this op type.\n"
        )

    return f"""\
You are managing an autonomous GPU kernel optimization run.

TARGET: {target_speedup:.1f}x speedup
CURRENT BEST: {best_speedup:.2f}x
BUDGET REMAINING: ${budget_remaining:.2f}{problem_section}{hints_section}
ROUNDS SO FAR:
{trajectory}

Based on the trajectory, decide the next action:

CONTINUE: <strategy description> — if current approach has room to improve
PIVOT: <new strategy description> — if current approach is exhausted, try something different
STOP: <reason> — if target is unreachable or we're at the theoretical ceiling

Consider these levels of optimization, from most to least conventional:

LEVEL 1 — Parameter tuning (try different block sizes, tile shapes, num_warps):
- Is the current tiling/blocking configuration suboptimal?
- Are there autotune configurations we haven't explored?

LEVEL 2 — Algorithmic change (different reduction strategy, different memory access pattern):
- Should we switch from global memory to shared memory tiling?
- Should we use warp-level shuffle reduction instead of shared memory reduction?
- Should we fuse multiple operations into a single kernel pass?

LEVEL 3 — Parallelism axis (MOST IMPACTFUL — question the fundamental decomposition):
- Is the parallelism axis wrong? (e.g., experts→outputs for MoE decode)
- Are there pipeline stages that can be ELIMINATED entirely? (5/8 stages in traditional MoE are bookkeeping)
- Can intermediate buffers be removed by accumulating in registers?

LEVEL 4 — Hardware-native operations (bypass software implementation entirely):
- Can the hardware do this operation natively? (e.g., tcgen05.mma block_scale for dequantization)
- Is intermediate precision loss avoidable? (e.g., keep BF16 activations instead of quantizing to MXFP8)
- Are there architecture-specific instructions we're not using? (TMA, TMEM, shfl.sync.bfly)

If stuck at Level 1 for 2+ rounds, PIVOT to Level 2 or 3.
If stuck at Level 2, consider Level 3 (parallelism axis flip) or Level 4 (hardware-native ops).

Respond with EXACTLY one line in one of these formats:
CONTINUE: <what to refine next, 10-15 words>
PIVOT: <fundamentally different approach, 10-15 words>
STOP: <reason, 10 words>"""


async def reflect_on_round(
    rounds: list[dict],
    best_speedup: float,
    target_speedup: float,
    budget_remaining: float,
    llm: LLMProvider,
    problem_context: str = "",
    strategy_hints: list[str] | None = None,
) -> MetaReflection:
    """Ask the LLM to reflect on progress and decide next action."""
    prompt = _build_reflection_prompt(
        rounds, best_speedup, target_speedup, budget_remaining,
        problem_context=problem_context,
        strategy_hints=strategy_hints,
    )

    try:
        response = await llm.generate(prompt)

        # Dev mode: append reflection to per-run log file
        if os.environ.get("OPENKERNEL_DEV_LOG") == "1":
            import json
            from pathlib import Path
            run_id = os.environ.get("OPENKERNEL_RUN_ID", "unknown")
            log_file = Path(f".kernel-code/dev_logs/run_{run_id}.jsonl")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps({
                        "type": "reflection",
                        "prompt": prompt,
                        "response": response.strip(),
                        "rounds_seen": len(rounds),
                        "best_speedup": best_speedup,
                        "target_speedup": target_speedup,
                    }) + "\n")
            except Exception:
                pass

        return _parse_reflection(response.strip())
    except Exception as exc:
        logger.warning("Meta-reflection failed: %s", exc)
        return MetaReflection(action="continue", reason=f"reflection failed: {exc}")


def _parse_reflection(response: str) -> MetaReflection:
    """Parse the LLM's structured response."""
    line = ""
    for candidate in response.strip().split("\n"):
        candidate = candidate.strip()
        if candidate:
            line = candidate
            break

    if not line:
        return MetaReflection(action="continue", reason="empty response")

    upper = line.upper()

    if upper.startswith("STOP"):
        reason = line.split(":", 1)[1].strip() if ":" in line else "LLM advised stop"
        return MetaReflection(action="stop", reason=reason)

    if upper.startswith("PIVOT"):
        strategy = line.split(":", 1)[1].strip() if ":" in line else "try a different approach"
        # Clean up LLM artifacts like "10 words", trailing word counts
        import re
        strategy = re.sub(r",?\s*\d+\s*words?\s*$", "", strategy).strip()
        return MetaReflection(action="pivot", next_strategy=strategy, reason="strategy exhausted")

    if upper.startswith("CONTINUE"):
        strategy = line.split(":", 1)[1].strip() if ":" in line else "refine current approach"
        return MetaReflection(action="continue", next_strategy=strategy, reason="progress detected")

    # Default: continue
    return MetaReflection(action="continue", next_strategy=line[:60], reason="unstructured response")
