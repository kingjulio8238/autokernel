"""LLM-powered Q&A for kernel optimization results.

Provides a ShellAI class that answers natural-language questions about
optimization sessions using the openkernel LLMProvider.  Context is assembled
from session JSON data (iterations, profiling, kernel code) and sent to the
model alongside the user's question.
"""

from __future__ import annotations

import logging

from openkernel.config import ModelConfig
from openkernel.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert GPU kernel optimization assistant. The user has run
an optimization session and is asking about the results. Answer
concisely and technically. Reference specific iterations by number.
When suggesting next steps, be specific about what optimization
technique to try and why."""

# Hard budget for the context portion of the prompt (leaves room for the
# system prompt, question, and model response within Groq free-tier limits).
_MAX_CONTEXT_CHARS = 12_000  # ~3000 tokens


def format_session_context(session_data: dict) -> str:
    """Format session data into a concise prompt context string.

    Prioritises (in order):
      1. Summary stats (always)
      2. Recent iterations (last 10)
      3. Best kernel snippet (first 15 lines)
      4. Latest profiling data

    The output is kept under ~3000 tokens so that the full prompt
    (system + context + question) fits within 4000 tokens.
    """
    parts: list[str] = []

    # ------------------------------------------------------------------
    # 1. Summary stats (always included)
    # ------------------------------------------------------------------
    iterations: list[dict] = session_data.get("iterations", [])
    kept = [it for it in iterations if it.get("decision") == "keep" or it.get("status") == "keep"]
    errors = [it for it in iterations if it.get("status") in ("compile_error", "incorrect", "error")]
    discarded = [it for it in iterations if it.get("decision") == "discard" or it.get("status") == "discard"]

    best_speedup = session_data.get("best_speedup", 0.0)
    # Also compute from iterations in case the top-level field is stale
    for it in iterations:
        sp = it.get("speedup", 0.0)
        if sp and sp > best_speedup:
            best_speedup = sp

    summary = (
        f"=== Optimization Session Summary ===\n"
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
    # 2. Top 5 kept optimizations
    # ------------------------------------------------------------------
    if kept:
        sorted_kept = sorted(kept, key=lambda it: it.get("speedup", 0.0), reverse=True)[:5]
        lines = ["=== Top Kept Optimizations ==="]
        for it in sorted_kept:
            lines.append(
                f"  Iter #{it.get('iteration', '?')}: "
                f"{it.get('speedup', 0.0):.2f}x — {it.get('intent', 'unknown')}"
            )
        parts.append("\n".join(lines))

    # ------------------------------------------------------------------
    # 3. Recent iterations (last 10) — full detail
    # ------------------------------------------------------------------
    recent = iterations[-10:] if iterations else []
    if recent:
        lines = ["=== Recent Iterations ==="]
        for it in recent:
            status = it.get("status", "unknown")
            speedup = it.get("speedup", 0.0)
            intent = it.get("intent", "")
            decision = it.get("decision", "")
            error = it.get("error")
            line = (
                f"  #{it.get('iteration', '?')}: "
                f"status={status}, speedup={speedup:.2f}x, "
                f"decision={decision}, intent=\"{intent}\""
            )
            if error:
                # Truncate long errors
                err_short = error[:120] + "..." if len(error) > 120 else error
                line += f"\n    error: {err_short}"
            lines.append(line)
        parts.append("\n".join(lines))

    # ------------------------------------------------------------------
    # 4. Latest profiling data
    # ------------------------------------------------------------------
    latest_profile = _latest_profile(iterations)
    if latest_profile:
        lines = ["=== Latest Profiling ==="]
        lines.append(f"  Bottleneck: {latest_profile.get('bottleneck_type', 'unknown')}")
        for key in ("bandwidth_util", "compute_util", "cache_efficiency", "occupancy",
                     "roofline_position", "estimated_headroom"):
            val = latest_profile.get(key)
            if val is not None:
                label = key.replace("_", " ").title()
                if isinstance(val, float) and val <= 1.0:
                    lines.append(f"  {label}: {val:.0%}")
                else:
                    lines.append(f"  {label}: {val}")
        stalls = latest_profile.get("top_stalls")
        if stalls:
            lines.append(f"  Top stalls: {', '.join(stalls[:3])}")
        parts.append("\n".join(lines))

    # ------------------------------------------------------------------
    # 5. Best kernel code snippet (first 15 lines)
    # ------------------------------------------------------------------
    best_code = _best_kernel_snippet(iterations, best_speedup)
    if best_code:
        code_lines = best_code.strip().splitlines()[:15]
        parts.append("=== Best Kernel (first 15 lines) ===\n" + "\n".join(code_lines))

    # ------------------------------------------------------------------
    # Assemble and trim to budget
    # ------------------------------------------------------------------
    context = "\n\n".join(parts)
    if len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS] + "\n... (truncated)"
    return context


def _latest_profile(iterations: list[dict]) -> dict | None:
    """Return the profile dict from the most recent correct iteration, or the last one."""
    for it in reversed(iterations):
        profile = it.get("profile")
        if profile and it.get("status") not in ("compile_error", "error"):
            return profile
    # Fallback: any last profile
    if iterations:
        return iterations[-1].get("profile")
    return None


def _best_kernel_snippet(iterations: list[dict], best_speedup: float) -> str | None:
    """Return the kernel code snippet from the iteration with the best speedup."""
    for it in iterations:
        if abs(it.get("speedup", 0.0) - best_speedup) < 0.01:
            snippet = it.get("kernel_code_snippet")
            if snippet:
                return snippet
    # Fallback: last kept iteration with a snippet
    for it in reversed(iterations):
        if it.get("decision") == "keep" and it.get("kernel_code_snippet"):
            return it["kernel_code_snippet"]
    return None


class ShellAI:
    """LLM-powered Q&A for optimization results.

    Usage::

        ai = ShellAI()                       # uses default ModelConfig
        answer = await ai.answer(question, session_data)
    """

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        if model_config is None:
            model_config = ModelConfig()
        self._provider = LLMProvider(model_config)

    async def answer(self, question: str, session_context: dict) -> str:
        """Answer a natural language question about optimization results.

        Args:
            question: The user's natural language question.
            session_context: Session data dict (as produced by mock_data or
                the live engine bridge).

        Returns:
            A plain-text answer string.  On LLM failure, returns a
            human-readable error message instead of raising.
        """
        context = format_session_context(session_context)
        prompt = self._build_prompt(context, question)

        try:
            response = await self._provider.generate(prompt)
            return response.strip()
        except Exception as exc:
            logger.error("ShellAI LLM call failed: %s", exc)
            return (
                f"[AI error] Could not generate a response: {exc}\n"
                "Check your API key and network connection. "
                "You can set GROQ_API_KEY or MINIMAX_API_KEY in your environment."
            )

    @staticmethod
    def _build_prompt(context: str, question: str) -> str:
        """Assemble the full prompt from system instructions, context, and question."""
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"--- SESSION DATA ---\n{context}\n\n"
            f"--- USER QUESTION ---\n{question}"
        )
