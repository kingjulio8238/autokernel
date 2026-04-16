"""Structured output parsing utilities.

Handles the messy reality of LLM responses: markdown code fences, trailing
prose around JSON, and varied formatting of kernel code blocks.
"""

from __future__ import annotations

import json
import logging
import re

from openkernel.eval.types import BottleneckType, CriticDiagnosis

logger = logging.getLogger(__name__)


def extract_kernel_code(response: str) -> str:
    """Extract the first Python code block from an LLM response.

    Looks for fenced code blocks (```python ... ``` or ``` ... ```).
    If no fenced block is found, returns the full response stripped of
    leading/trailing whitespace (best-effort fallback).
    """
    # Try ```python first, then bare ```
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback: if the response looks like it's all code (has "class ModelNew"),
    # return it directly.
    if "class ModelNew" in response:
        return response.strip()

    logger.warning("No code block found in LLM response; returning raw text")
    return response.strip()


def parse_json_response(response: str) -> dict:
    """Robustly extract a JSON object from an LLM response.

    Handles:
    - Raw JSON
    - JSON wrapped in ```json ... ``` fences
    - JSON wrapped in ``` ... ``` fences
    - JSON embedded in prose (finds first { ... } block)
    - Trailing commas (common LLM mistake)

    Raises ``ValueError`` if no valid JSON can be extracted.
    """
    # Strategy 1: Try direct parse (response is pure JSON)
    text = response.strip()
    parsed = _try_parse(text)
    if parsed is not None:
        return parsed

    # Strategy 2: Extract from ```json ... ``` or ``` ... ``` fences
    for pattern in [r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            parsed = _try_parse(match.group(1).strip())
            if parsed is not None:
                return parsed

    # Strategy 3: Find the outermost { ... } in the response
    brace_start = text.find("{")
    if brace_start != -1:
        # Walk from the end to find the matching closing brace
        brace_end = text.rfind("}")
        if brace_end > brace_start:
            candidate = text[brace_start : brace_end + 1]
            parsed = _try_parse(candidate)
            if parsed is not None:
                return parsed

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}...")


def parse_critic_diagnosis(response: str) -> CriticDiagnosis:
    """Parse a Critic agent response into a :class:`CriticDiagnosis`.

    The critic is prompted to return structured JSON.  This function handles
    the common failure modes (wrapped in markdown, extra prose, etc.).
    """
    try:
        data = parse_json_response(response)
    except ValueError:
        logger.warning("Failed to parse Critic JSON; returning default diagnosis")
        return CriticDiagnosis(
            specific_issue="Could not parse critic response",
            recommendation="Retry with a different approach",
            confidence=0.0,
        )

    # Map bottleneck_type string to enum
    raw_bottleneck = data.get("bottleneck_type", "unknown")
    try:
        bottleneck = BottleneckType(raw_bottleneck)
    except ValueError:
        bottleneck = BottleneckType.UNKNOWN

    return CriticDiagnosis(
        bottleneck_type=bottleneck,
        roofline_position=float(data.get("roofline_position", 0.0)),
        specific_issue=str(data.get("specific_issue", "")),
        recommendation=str(data.get("recommendation", "")),
        estimated_headroom=float(data.get("estimated_headroom", 0.0)),
        confidence=float(data.get("confidence", 0.0)),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _try_parse(text: str) -> dict | None:
    """Attempt to parse *text* as JSON, with light cleanup for common LLM quirks."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None
