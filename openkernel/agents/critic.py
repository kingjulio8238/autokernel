"""Critic agent — analyzes kernel performance and diagnoses bottlenecks.

The Critic takes a kernel's source code, its evaluation result, hardware info,
and backend type, then produces a structured :class:`CriticDiagnosis` describing
what is limiting performance and what to try next.
"""

from __future__ import annotations

import logging
from pathlib import Path

from openkernel.eval.types import CriticDiagnosis, EvalResult
from openkernel.llm.provider import LLMProvider
from openkernel.llm.structured import parse_critic_diagnosis

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"


class Critic:
    """Analyzes kernel performance via an LLM and profiler data."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._template = self._load_template()

    async def analyze(
        self,
        kernel_code: str,
        eval_result: EvalResult,
        hardware: str,
        backend: str,
    ) -> CriticDiagnosis:
        """Produce a diagnosis for a kernel based on its eval result.

        Parameters
        ----------
        kernel_code : str
            The kernel source code that was evaluated.
        eval_result : EvalResult
            The evaluation result including profile data.
        hardware : str
            Target GPU description.
        backend : str
            "triton" or "cuda".

        Returns
        -------
        CriticDiagnosis
            Structured analysis with bottleneck type, specific issue,
            recommendation, and estimated headroom.
        """
        profile = eval_result.profile

        prompt = _safe_format(
            self._template,
            kernel_code=kernel_code,
            status=eval_result.status.value,
            speedup=f"{eval_result.speedup:.2f}",
            runtime_us=f"{eval_result.runtime_us:.1f}",
            ref_runtime_us=f"{eval_result.ref_runtime_us:.1f}",
            bandwidth_utilization=f"{profile.bandwidth_utilization * 100:.1f}",
            compute_utilization=f"{profile.compute_utilization * 100:.1f}",
            cache_efficiency=f"{profile.cache_efficiency * 100:.1f}",
            occupancy=f"{profile.occupancy:.2f}",
            top_stalls=", ".join(profile.top_stalls) if profile.top_stalls else "N/A",
            hardware=hardware,
            backend=backend,
        )

        logger.info("Critic: calling LLM for diagnosis")
        response = await self._llm.generate_structured(
            prompt,
            response_format={"type": "json_object"},
        )

        diagnosis = parse_critic_diagnosis(response)
        logger.info(
            "Critic: %s — %s (confidence=%.2f)",
            diagnosis.bottleneck_type.value,
            diagnosis.specific_issue[:80],
            diagnosis.confidence,
        )
        return diagnosis

    def format_feedback(self, diagnosis: CriticDiagnosis, speedup: float) -> str:
        """Format a diagnosis into a string suitable for the generator prompt.

        This is used as the ``critic_feedback`` argument to the backend's
        ``get_generator_prompt`` in subsequent iterations.
        """
        return (
            f"Previous speedup: {speedup:.2f}x\n"
            f"Bottleneck: {diagnosis.bottleneck_type.value}\n"
            f"Specific issue: {diagnosis.specific_issue}\n"
            f"Recommendation: {diagnosis.recommendation}\n"
            f"Estimated headroom: {diagnosis.estimated_headroom:.2f}x\n"
            f"Confidence: {diagnosis.confidence:.2f}"
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _load_template() -> str:
        """Load the critic prompt template from disk."""
        path = _PROMPTS_DIR / "critic_v1.md"
        if not path.exists():
            logger.warning("Critic prompt template not found at %s; using inline fallback", path)
            return _FALLBACK_TEMPLATE
        return path.read_text()


def _safe_format(template: str, **kwargs: str) -> str:
    """Format a template string, leaving unknown {placeholders} intact."""
    import re

    def _replacer(match: re.Match) -> str:
        key = match.group(1)
        if key in kwargs:
            return kwargs[key]
        return match.group(0)

    return re.sub(r"\{(\w+)\}", _replacer, template)


_FALLBACK_TEMPLATE = """\
You are a GPU performance analysis expert. Analyze the following kernel:

Kernel code:
{kernel_code}

Benchmark: {status}, {speedup}x speedup, {runtime_us}us (ref {ref_runtime_us}us)
Bandwidth: {bandwidth_utilization}%, Compute: {compute_utilization}%, Cache: {cache_efficiency}%
Occupancy: {occupancy}, Top stalls: {top_stalls}
Hardware: {hardware}, Backend: {backend}

Return a JSON object with: bottleneck_type, roofline_position, specific_issue,
recommendation, estimated_headroom, confidence.
"""
