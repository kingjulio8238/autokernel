"""Planning agent — decomposes optimization into structured diagnostics.

Sits between the Critic and Generator in the agent pipeline.
The Critic identifies bottlenecks; the Planner produces a concrete,
single-method modification plan with numbered code actions.

This decomposition follows the arxiv paper 2509.07506 finding that
explicit agent specialization (Testing, Profiling, Planning, Coding)
outperforms monolithic approaches (1.32x vs 1.08x).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from openkernel.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class PlanDiagnostic:
    """Structured output from the planning agent.

    Contains a single-method optimization plan with
    concrete code actions, following the one-method discipline
    from KernelMem.
    """
    primary_method: str                   # Exactly one optimization method
    bottleneck_summary: str               # One-line bottleneck description
    modification_plan: list[str]          # Numbered code action checklist
    evidence: list[str]                   # Numeric metrics supporting the plan
    expected_improvements: dict[str, str] # metric_name -> expected_direction
    headroom: str                         # "high", "medium", or "low"
    confidence: float = 0.0              # 0.0 - 1.0
    warnings: list[str] = field(default_factory=list)

    def to_prompt_string(self) -> str:
        """Format for injection into the generator prompt."""
        plan_str = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(self.modification_plan))
        evidence_str = "\n".join(f"  - {e}" for e in self.evidence)
        improvements_str = "\n".join(f"  - {k}: {v}" for k, v in self.expected_improvements.items())

        parts = [
            f"## Optimization Plan",
            f"**Method**: {self.primary_method}",
            f"**Bottleneck**: {self.bottleneck_summary}",
            f"**Headroom**: {self.headroom} (confidence: {self.confidence:.0%})",
            f"\n**Modification Plan**:\n{plan_str}",
            f"\n**Evidence**:\n{evidence_str}",
            f"\n**Expected Improvements**:\n{improvements_str}",
        ]

        if self.warnings:
            warnings_str = "\n".join(f"  - {w}" for w in self.warnings)
            parts.append(f"\n**Warnings**:\n{warnings_str}")

        return "\n".join(parts)


_PLANNER_SYSTEM_PROMPT = """\
You are a GPU kernel optimization planner. Your role is to analyze a kernel's
performance profile and produce a SINGLE, focused optimization plan.

CRITICAL RULES:
1. ONE-METHOD RULE: Propose exactly ONE optimization method per plan.
   Do not combine multiple methods. Each method gets its own round.
2. CONCRETE ACTIONS: Each step in the modification plan must describe
   a specific code change (e.g., "Replace 4 scalar loads at line 15
   with 2 float2 vectorized loads").
3. EVIDENCE-BASED: Every recommendation must cite specific metrics
   from the profiling data.
4. HISTORY-AWARE: Do not propose a method that was already tried
   in a prior round unless you can articulate a concrete delta.

Output a JSON object with these fields:
{
    "primary_method": "string — the ONE optimization method to apply",
    "bottleneck_summary": "string — one-line bottleneck description",
    "modification_plan": ["step 1...", "step 2...", ...],
    "evidence": ["metric: value — interpretation", ...],
    "expected_improvements": {"metric_name": "expected direction", ...},
    "headroom": "high | medium | low",
    "confidence": 0.0-1.0,
    "warnings": ["any caveats or risks"]
}
"""


class Planner:
    """Produces structured, single-method optimization plans.

    Sits between Critic (bottleneck identification) and Generator
    (kernel code production) in the agent pipeline.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def plan(
        self,
        kernel_code: str,
        critic_feedback: str,
        hardware: str,
        profile_data: str | None = None,
        optimization_history: list[dict] | None = None,
        allowed_methods: list[str] | None = None,
    ) -> PlanDiagnostic:
        """Generate a structured optimization plan.

        Parameters
        ----------
        kernel_code : str
            Current kernel source code.
        critic_feedback : str
            Formatted output from the Critic agent.
        hardware : str
            Target GPU (e.g., "H100", "A100-80GB").
        profile_data : str, optional
            NCU profiling metrics (inline or from Modal).
        optimization_history : list[dict], optional
            Prior rounds' method names and results for deduplication.
        allowed_methods : list[str], optional
            Methods permitted by the deterministic gate.

        Returns
        -------
        PlanDiagnostic
            A structured, single-method optimization plan.
        """
        # Build the instruction prompt
        instruction_parts = [
            f"## Current Kernel\n```python\n{kernel_code}\n```\n",
            f"## Critic Analysis\n{critic_feedback}\n",
            f"## Target Hardware: {hardware}\n",
        ]

        if profile_data:
            instruction_parts.append(f"## Profiling Data\n{profile_data}\n")

        if optimization_history:
            history_str = "\n".join(
                f"  Round {h.get('round', '?')}: {h.get('method_name', 'unknown')} "
                f"-> {h.get('speedup', 0.0):.2f}x"
                for h in optimization_history
            )
            instruction_parts.append(
                f"## Prior Optimization Attempts\n"
                f"Do NOT repeat these methods unless you have a concrete delta:\n{history_str}\n"
            )

        if allowed_methods:
            instruction_parts.append(
                f"## Allowed Methods (from deterministic gate)\n"
                f"You MUST choose from: {', '.join(allowed_methods)}\n"
            )

        instruction = "\n".join(instruction_parts)

        # Prepend system prompt to instruction (LLMProvider.generate_structured
        # does not accept a separate system_prompt parameter)
        full_prompt = _PLANNER_SYSTEM_PROMPT + "\n\n" + instruction

        logger.info("Planner: calling LLM for optimization plan")
        response = await self._llm.generate_structured(
            full_prompt,
            response_format={"type": "json_object"},
        )

        return self._parse_response(response)

    def _parse_response(self, response: str) -> PlanDiagnostic:
        """Parse the LLM JSON response into a PlanDiagnostic."""
        # Try to extract JSON from markdown fences if present
        text = response.strip()
        if "```" in text:
            import re as _re
            match = _re.search(r"```(?:json)?\s*\n?(.*?)```", text, _re.DOTALL)
            if match:
                text = match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Planner: failed to parse JSON, returning fallback")
            return PlanDiagnostic(
                primary_method="unknown",
                bottleneck_summary="Parse error — LLM did not return valid JSON",
                modification_plan=["Review kernel manually"],
                evidence=[],
                expected_improvements={},
                headroom="medium",
            )

        # Type guards for list/dict fields
        modification_plan = data.get("modification_plan", [])
        if not isinstance(modification_plan, list):
            modification_plan = [str(modification_plan)]

        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]

        expected_improvements = data.get("expected_improvements", {})
        if not isinstance(expected_improvements, dict):
            expected_improvements = {}

        warnings = data.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = [str(warnings)]

        try:
            confidence = float(data.get("confidence", 0.0))
        except (ValueError, TypeError):
            confidence = 0.0

        return PlanDiagnostic(
            primary_method=data.get("primary_method", "unknown"),
            bottleneck_summary=data.get("bottleneck_summary", ""),
            modification_plan=modification_plan,
            evidence=evidence,
            expected_improvements=expected_improvements,
            headroom=data.get("headroom", "medium"),
            confidence=confidence,
            warnings=warnings,
        )
