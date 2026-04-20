"""Generate 3 concrete next optimization suggestions after every run.

Creates the recursive loop:
  optimize -> analyze -> suggest 3 approaches -> user picks -> optimize again

Uses LLM when available, falls back to rule-based suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.text import Text


@dataclass
class NextStep:
    number: int  # 1, 2, or 3
    title: str  # "Vectorize loads with float4"
    approach: str  # "Use float4 loads to improve memory bandwidth..."
    expected_gain: str  # "~1.3x improvement expected"
    skill_id: str | None  # matching skill to auto-load, if any


async def generate_next_steps_llm(
    session_data: dict,
    advisor_state: dict | None = None,
    model_config=None,
) -> list[NextStep]:
    """Generate 3 next-step suggestions using LLM.

    Sends current session context (best speedup, bottleneck, what's been tried)
    to the LLM and asks for 3 concrete optimization suggestions.
    """
    from openkernel.config import ModelConfig
    from openkernel.llm.provider import LLMProvider

    from kernel_code.compaction import compact_session

    config = model_config or ModelConfig()
    llm = LLMProvider(config)

    context = compact_session(session_data, max_tokens=1500)

    prompt = f"""You are a GPU kernel optimization expert. Based on the optimization session below, suggest exactly 3 concrete next optimization approaches to try.

SESSION CONTEXT:
{context}

For each suggestion, provide:
1. A short title (5-10 words)
2. A brief approach description (1-2 sentences)
3. Expected gain estimate (e.g., "~1.3x improvement")

Format as a numbered list:
1. TITLE: ...
   APPROACH: ...
   EXPECTED: ...
2. ...
3. ...

Focus on approaches NOT yet tried. Prioritize based on the current bottleneck."""

    try:
        response = await llm.generate(prompt)
        return _parse_suggestions(response)
    except Exception:
        return generate_next_steps_rule_based(session_data, advisor_state)


def generate_next_steps_rule_based(
    session_data: dict,
    advisor_state: dict | None = None,
) -> list[NextStep]:
    """Rule-based fallback when LLM is unavailable."""
    iterations = session_data.get("iterations", [])
    if not iterations:
        return [
            NextStep(
                1,
                "Start with basic tiled kernel",
                "Write a basic tiled kernel as baseline",
                "~1.0x baseline",
                "triton_gemm",
            ),
            NextStep(
                2,
                "Try vectorized elementwise",
                "Use float4 loads for bandwidth",
                "~1.2x expected",
                "triton_elementwise",
            ),
            NextStep(
                3,
                "Profile the reference first",
                "Understand the bottleneck before optimizing",
                "diagnostic",
                None,
            ),
        ]

    # Analyze what's been tried and what's left
    tried_categories: set[str] = set()
    last_bottleneck = "unknown"
    for it in iterations:
        intent = it.get("intent", "").lower()
        profile = it.get("profile", {})
        if profile.get("bottleneck_type"):
            last_bottleneck = profile["bottleneck_type"]
        for cat in [
            "tiling",
            "vectorize",
            "shared",
            "tensor",
            "fusion",
            "pipeline",
            "reduction",
            "split",
        ]:
            if cat in intent:
                tried_categories.add(cat)

    suggestions: list[NextStep] = []
    n = 1

    if last_bottleneck == "memory_bound":
        if "vectorize" not in tried_categories:
            suggestions.append(
                NextStep(
                    n,
                    "Vectorize loads with float4",
                    "Use float4 loads for better memory bandwidth utilization",
                    "~1.3x improvement",
                    "triton_elementwise",
                )
            )
            n += 1
        if "shared" not in tried_categories:
            suggestions.append(
                NextStep(
                    n,
                    "Add shared memory tiling",
                    "Cache frequently accessed data in shared memory to reduce DRAM traffic",
                    "~1.2x improvement",
                    "triton_reduction",
                )
            )
            n += 1
        if "fusion" not in tried_categories:
            suggestions.append(
                NextStep(
                    n,
                    "Fuse adjacent operations",
                    "Eliminate intermediate memory writes between operators",
                    "~1.5x improvement",
                    "fusion_patterns",
                )
            )
            n += 1
    elif last_bottleneck == "compute_bound":
        if "tensor" not in tried_categories:
            suggestions.append(
                NextStep(
                    n,
                    "Use tensor core instructions",
                    "Leverage tl.dot for tensor core MMA operations",
                    "~1.4x improvement",
                    "triton_gemm",
                )
            )
            n += 1
        if "pipeline" not in tried_categories:
            suggestions.append(
                NextStep(
                    n,
                    "Add software pipelining",
                    "Overlap memory loads with computation",
                    "~1.2x improvement",
                    None,
                )
            )
            n += 1
        if "tiling" not in tried_categories or True:  # always suggest tuning
            suggestions.append(
                NextStep(
                    n,
                    "Tune tile sizes",
                    "Try larger BLOCK_M/N for better data reuse",
                    "~1.1x improvement",
                    None,
                )
            )
            n += 1
    else:
        suggestions.append(
            NextStep(
                n,
                "Profile the kernel first",
                "Run profiler to identify the bottleneck type",
                "diagnostic",
                None,
            )
        )
        n += 1
        suggestions.append(
            NextStep(
                n,
                "Try vectorized loads",
                "Start with float4 for bandwidth improvement",
                "~1.2x improvement",
                "triton_elementwise",
            )
        )
        n += 1
        suggestions.append(
            NextStep(
                n,
                "Switch backend",
                "Try CUDA if Triton is hitting ceiling, or vice versa",
                "variable",
                None,
            )
        )
        n += 1

    return suggestions[:3]


def _parse_suggestions(response: str) -> list[NextStep]:
    """Parse LLM response into NextStep objects."""
    import re

    suggestions: list[NextStep] = []
    # Simple parsing: look for numbered items
    blocks = re.split(r"\n\d+\.", response)
    for i, block in enumerate(blocks[1:4], 1):  # skip preamble, take 3
        lines = block.strip().split("\n")
        title = lines[0].strip().lstrip(". ") if lines else f"Suggestion {i}"
        # Extract TITLE/APPROACH/EXPECTED if formatted
        approach = ""
        expected = "improvement expected"
        for line in lines:
            if "APPROACH:" in line.upper():
                approach = line.split(":", 1)[1].strip()
            elif "EXPECTED:" in line.upper():
                expected = line.split(":", 1)[1].strip()
        if not approach and len(lines) > 1:
            approach = lines[1].strip().lstrip(". ")

        # Try to match a skill
        title_lower = title.lower()
        skill_id = None
        if "vectoriz" in title_lower or "float4" in title_lower:
            skill_id = "triton_elementwise_vectorized"
        elif "softmax" in title_lower or "reduction" in title_lower:
            skill_id = "triton_online_reduction"
        elif "tensor" in title_lower or "gemm" in title_lower:
            skill_id = "triton_tiled_gemm"
        elif "fus" in title_lower:
            skill_id = "fusion_patterns"

        suggestions.append(
            NextStep(i, title[:60], approach[:120], expected[:40], skill_id)
        )

    # Pad with rule-based if parsing found fewer than 3
    while len(suggestions) < 3:
        n = len(suggestions) + 1
        suggestions.append(
            NextStep(
                n,
                f"Alternative approach {n}",
                "Try a different optimization technique",
                "variable",
                None,
            )
        )

    return suggestions[:3]


def _strip_markdown(text: str) -> str:
    """Strip common markdown formatting and LLM artifacts from output."""
    import re
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # *italic*
    text = re.sub(r"`(.*?)`", r"\1", text)  # `code`
    # Strip common LLM prefixes
    text = re.sub(r"^TITLE:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^APPROACH:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^EXPECTED:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def format_next_steps(
    steps: list[NextStep], console: Console | None = None
) -> Text:
    """Format the 3 suggestions for display."""
    result = Text()
    result.append("\n  \u2500\u2500 Next Steps \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n", style="bold white")

    for step in steps:
        title = _strip_markdown(step.title)
        result.append(f"  {step.number}. ", style="bold #d77757")
        result.append(title, style="bold white")
        if step.expected_gain:
            result.append(f"  ({step.expected_gain})", style="#999999")
        result.append("\n")
        if step.approach:
            approach = _strip_markdown(step.approach)
            result.append(f"     \u23bf  {approach}\n", style="#999999")
        if step.skill_id:
            result.append(f"     \u23bf  /skill:{step.skill_id}\n", style="#d77757")

    result.append(
        "\n  Type 1, 2, or 3 to start\n",
        style="#777777",
    )
    return result
