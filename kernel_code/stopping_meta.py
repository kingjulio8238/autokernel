"""LLM meta-advisor for adaptive stopping decisions.

Called every N iterations by the StoppingController.  Sends a compact
trajectory summary to the LLM and parses a structured CONTINUE/STOP/ADJUST
response.

The meta-advisor adds intelligence that rules can't provide:
- Recognizing strategy exhaustion ("tried tiling, vectorization, shared memory
  — remaining approaches are unlikely to help for this problem type")
- Detecting systematic errors ("3 compile errors on the same tl.dot shape
  mismatch — the model keeps making the same mistake")
- Estimating headroom from bottleneck ("85% bandwidth utilization, only ~15%
  headroom left — diminishing returns")

Cost: ~200 tokens input, ~50 tokens output.  On Groq free tier: <1s, $0.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from openkernel.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class MetaDecision:
    """Parsed decision from the meta-advisor LLM call."""

    action: str = "continue"  # "continue" | "stop" | "adjust"
    reason: str = ""
    adjustments: dict = field(default_factory=dict)  # e.g. {"convergence_patience": 5}


def _build_trajectory_summary(history: list[dict], best_speedup: float) -> str:
    """Build a compact trajectory summary for the LLM (~150 tokens)."""
    if not history:
        return "No iterations completed yet."

    lines: list[str] = []
    lines.append(f"Iterations: {len(history)}, Best speedup: {best_speedup:.2f}x")

    # Speedup trajectory
    speedups = [h.get("speedup", 0.0) for h in history]
    statuses = [h.get("status", "?") for h in history]
    lines.append(f"Trajectory: {' → '.join(f'{s:.2f}x({st[:1]})' for s, st in zip(speedups, statuses))}")

    # Approaches tried
    intents = [h.get("intent", "")[:40] for h in history if h.get("intent")]
    if intents:
        lines.append(f"Approaches: {', '.join(intents)}")

    # Error patterns
    errors = [h for h in history if h.get("status") in ("compile_error", "error", "incorrect")]
    if errors:
        error_types = [h.get("status", "") for h in errors]
        lines.append(f"Errors: {len(errors)}/{len(history)} ({', '.join(error_types)})")

    # Latest bottleneck
    for h in reversed(history):
        profile = h.get("profile", {})
        bn = profile.get("bottleneck_type", "")
        if bn and bn != "unknown":
            bw = profile.get("bandwidth_util", 0)
            comp = profile.get("compute_util", 0)
            lines.append(f"Bottleneck: {bn} (BW: {bw:.0%}, Compute: {comp:.0%})")
            break

    return "\n".join(lines)


_META_PROMPT = """\
You are monitoring a GPU kernel optimization run. Based on the trajectory below, decide whether to CONTINUE, STOP, or ADJUST the strategy.

{trajectory}

Consider:
- Is the speedup still improving, or has it plateaued?
- Are errors systematic (same type repeating) or incidental?
- Have we exhausted the obvious optimization approaches?
- Is there meaningful headroom left based on the bottleneck?

Respond with EXACTLY one line in one of these formats:
CONTINUE
STOP: <reason in 10 words or less>
ADJUST convergence_patience=<N>: <reason in 10 words or less>
"""


async def meta_evaluate(
    history: list[dict],
    best_speedup: float,
    llm: LLMProvider,
) -> MetaDecision:
    """Ask the LLM whether to continue, stop, or adjust.

    Parameters
    ----------
    history : list[dict]
        Iteration history (speedup, status, intent, profile per iteration).
    best_speedup : float
        Best speedup achieved so far.
    llm : LLMProvider
        The LLM provider (same one used for kernel generation).

    Returns
    -------
    MetaDecision
        Parsed decision with action, reason, and optional adjustments.
    """
    trajectory = _build_trajectory_summary(history, best_speedup)
    prompt = _META_PROMPT.format(trajectory=trajectory)

    try:
        response = await llm.generate(prompt)
        return _parse_response(response.strip())
    except Exception as exc:
        logger.warning("Meta-advisor LLM call failed: %s", exc)
        return MetaDecision(action="continue", reason=f"meta-call failed: {exc}")


def _parse_response(response: str) -> MetaDecision:
    """Parse the LLM's structured response into a MetaDecision."""
    # Take only the first non-empty line
    line = ""
    for candidate in response.strip().split("\n"):
        candidate = candidate.strip()
        if candidate:
            line = candidate
            break

    if not line:
        return MetaDecision(action="continue", reason="empty response")

    upper = line.upper()

    # STOP: reason
    if upper.startswith("STOP"):
        reason = line.split(":", 1)[1].strip() if ":" in line else "LLM advised stop"
        return MetaDecision(action="stop", reason=reason)

    # ADJUST param=value: reason
    if upper.startswith("ADJUST"):
        reason = ""
        adjustments: dict = {}

        # Extract reason after colon
        if ":" in line:
            before_colon, reason = line.split(":", 1)
            reason = reason.strip()
        else:
            before_colon = line

        # Parse key=value pairs
        for match in re.finditer(r"(\w+)\s*=\s*(\d+\.?\d*)", before_colon):
            key = match.group(1)
            val = match.group(2)
            adjustments[key] = float(val) if "." in val else int(val)

        if not adjustments:
            # Default adjustment: extend patience
            adjustments = {"convergence_patience": 5}

        return MetaDecision(action="adjust", reason=reason, adjustments=adjustments)

    # CONTINUE (or anything else)
    return MetaDecision(action="continue", reason="LLM advised continue")
