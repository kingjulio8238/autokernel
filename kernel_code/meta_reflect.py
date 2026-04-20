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
) -> str:
    """Build a compact reflection prompt (~300 tokens)."""
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

    return f"""\
You are managing an autonomous GPU kernel optimization run.

TARGET: {target_speedup:.1f}x speedup
CURRENT BEST: {best_speedup:.2f}x
BUDGET REMAINING: ${budget_remaining:.2f}
ROUNDS SO FAR:
{trajectory}

Based on the trajectory, decide the next action:

CONTINUE: <strategy description> — if current approach has room to improve
PIVOT: <new strategy description> — if current approach is exhausted, try something different
STOP: <reason> — if target is unreachable or we're at the theoretical ceiling

Consider:
- Are we making progress toward the target?
- Is the bottleneck type suggesting a specific optimization direction?
- Have we exhausted the obvious approaches?
- Is the remaining budget sufficient for another meaningful round?

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
) -> MetaReflection:
    """Ask the LLM to reflect on progress and decide next action."""
    prompt = _build_reflection_prompt(rounds, best_speedup, target_speedup, budget_remaining)

    try:
        response = await llm.generate(prompt)
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
