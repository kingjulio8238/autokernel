"""Agentic loop for the kernel optimization shell.

Replaces the one-shot Q&A pattern with an iterative tool-use loop:
natural language input -> LLM decides what tool/action to take -> execute ->
feed result back -> LLM decides next action -> loop until final answer.

The LLM responds with either plain text (final answer) or a JSON tool call
like ``{"tool": "explain_iteration", "args": {"iteration": 9}}``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openkernel.config import ModelConfig
from openkernel.llm.provider import LLMProvider

from kernel_code.shell_ai import format_session_context

logger = logging.getLogger(__name__)

# Maximum number of tool calls the agent may make per user turn.
_MAX_TOOL_CALLS = 3

# ------------------------------------------------------------------
# Tool definitions (sent to the LLM so it knows what it can call)
# ------------------------------------------------------------------

TOOLS = [
    {
        "name": "show_best_kernel",
        "description": "Display the current best optimized kernel with syntax highlighting",
    },
    {
        "name": "show_results",
        "description": "Show summary of optimization results (iterations, speedup, cost)",
    },
    {
        "name": "explain_iteration",
        "description": "Explain what a specific iteration did and why it succeeded/failed",
        "parameters": {"iteration": "int - the iteration number to explain"},
    },
    {
        "name": "suggest_next",
        "description": (
            "Suggest the next optimization technique to try based on current "
            "bottleneck and profiling data"
        ),
    },
    {
        "name": "search_skills",
        "description": "Search the optimization skill library for relevant patterns",
        "parameters": {"query": "str - what to search for"},
    },
    {
        "name": "compare_iterations",
        "description": "Compare two iterations side by side",
        "parameters": {"iter_a": "int", "iter_b": "int"},
    },
]


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

def _build_system_prompt() -> str:
    """Build the system prompt including tool descriptions."""
    tool_block = json.dumps(TOOLS, indent=2)
    return f"""\
You are an expert GPU kernel optimization assistant embedded in an interactive
shell.  The user has run (or is running) an optimization session and is asking
questions or requesting actions.

You have access to the following tools:

{tool_block}

RESPONSE FORMAT
---------------
If you can answer the user directly without calling a tool, respond with plain
text only.

If you need to call a tool, respond with EXACTLY one JSON object on its own
(no surrounding text), for example:

{{"tool": "explain_iteration", "args": {{"iteration": 9}}}}

For tools with no parameters, omit "args":

{{"tool": "show_results"}}

After each tool call you will receive the tool's output and may either answer
the user or make another tool call (up to {_MAX_TOOL_CALLS} total per turn).

GUIDELINES
- Be concise and technical.
- Reference specific iteration numbers.
- When suggesting next steps, be specific about what optimization technique
  to try and why, grounded in profiling data.
- Prefer calling a tool to get precise data over guessing from the summary.
"""


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------

def _tool_show_best_kernel(session_context: dict) -> str:
    """Return the best kernel's code and metadata as a string."""
    iterations = session_context.get("iterations", [])
    best_speedup = session_context.get("best_speedup", 0.0)

    best = None
    for it in iterations:
        if it.get("status") == "keep":
            sp = it.get("speedup", 0.0)
            if sp >= best_speedup or (best is None and sp > 0):
                best = it
                best_speedup = sp

    if best is None:
        return "No best kernel found. Run /optimize first."

    code = best.get("kernel_code_snippet", "(code not available)")
    return (
        f"Best kernel — iteration #{best.get('iteration')}, "
        f"speedup {best.get('speedup', 0):.2f}x, "
        f"intent: {best.get('intent', 'unknown')}\n\n{code}"
    )


def _tool_show_results(session_context: dict) -> str:
    """Return a concise results summary."""
    iterations = session_context.get("iterations", [])
    if not iterations:
        return "No optimization results yet. Run /optimize first."

    kept = [it for it in iterations if it.get("status") == "keep"]
    discarded = [it for it in iterations if it.get("status") == "discard"]
    errors = [it for it in iterations if it.get("status") in ("compile_error", "error", "incorrect")]
    best_speedup = session_context.get("best_speedup", 0.0)
    for it in iterations:
        sp = it.get("speedup", 0.0)
        if sp > best_speedup:
            best_speedup = sp

    lines = [
        f"Total iterations: {len(iterations)}",
        f"Kept: {len(kept)}  |  Discarded: {len(discarded)}  |  Errors: {len(errors)}",
        f"Best speedup: {best_speedup:.2f}x",
        f"Estimated cost: ${len(iterations) * 0.02:.2f}",
    ]

    if kept:
        sorted_kept = sorted(kept, key=lambda it: it.get("speedup", 0.0), reverse=True)
        lines.append("\nTop kept iterations:")
        for it in sorted_kept[:5]:
            lines.append(
                f"  #{it.get('iteration')}: {it.get('speedup', 0):.2f}x — {it.get('intent', '')}"
            )

    return "\n".join(lines)


def _tool_explain_iteration(session_context: dict, iteration: int) -> str:
    """Explain what a specific iteration did."""
    iterations = session_context.get("iterations", [])
    target = None
    for it in iterations:
        if it.get("iteration") == iteration:
            target = it
            break

    if target is None:
        return (
            f"Iteration #{iteration} not found. "
            f"Available iterations: 1-{len(iterations)}"
        )

    lines = [
        f"Iteration #{iteration}",
        f"  Status:  {target.get('status', 'unknown')}",
        f"  Speedup: {target.get('speedup', 0):.2f}x",
        f"  Intent:  {target.get('intent', 'unknown')}",
    ]

    if target.get("runtime_us"):
        lines.append(f"  Runtime: {target['runtime_us']:.1f} us")
    if target.get("ref_runtime_us"):
        lines.append(f"  Ref runtime: {target['ref_runtime_us']:.1f} us")

    profile = target.get("profile", {})
    if profile:
        lines.append("  Profile:")
        lines.append(f"    Bottleneck:    {profile.get('bottleneck_type', 'unknown')}")
        lines.append(f"    Bandwidth:     {profile.get('bandwidth_util', 0):.0%}")
        lines.append(f"    Compute:       {profile.get('compute_util', 0):.0%}")
        lines.append(f"    Cache eff.:    {profile.get('cache_efficiency', 0):.0%}")
        lines.append(f"    Occupancy:     {profile.get('occupancy', 0):.2f}")
        stalls = profile.get("top_stalls")
        if stalls:
            lines.append(f"    Top stalls:    {', '.join(stalls[:3])}")

    if target.get("error"):
        err = target["error"]
        if len(err) > 200:
            err = err[:200] + "..."
        lines.append(f"  Error: {err}")

    code = target.get("kernel_code_snippet", "")
    if code:
        code_lines = code.strip().splitlines()[:10]
        lines.append("  Code (first 10 lines):")
        for cl in code_lines:
            lines.append(f"    {cl}")

    return "\n".join(lines)


def _tool_suggest_next(session_context: dict) -> str:
    """Suggest the next optimization based on profiling data."""
    iterations = session_context.get("iterations", [])
    if not iterations:
        return "No optimization data yet. Run /optimize first."

    # Find the latest valid profile
    latest_profile = None
    latest_iter = None
    for it in reversed(iterations):
        profile = it.get("profile")
        if profile and it.get("status") not in ("compile_error", "error"):
            latest_profile = profile
            latest_iter = it
            break

    if latest_profile is None:
        return "No profiling data available to make a suggestion."

    bottleneck = latest_profile.get("bottleneck_type", "unknown")
    bw = latest_profile.get("bandwidth_util", 0)
    cu = latest_profile.get("compute_util", 0)
    ce = latest_profile.get("cache_efficiency", 0)
    occ = latest_profile.get("occupancy", 0)
    headroom = latest_profile.get("estimated_headroom", "unknown")

    lines = [
        f"Analysis based on iteration #{latest_iter.get('iteration')} profile:",
        f"  Bottleneck: {bottleneck}",
        f"  Bandwidth util: {bw:.0%}  |  Compute util: {cu:.0%}",
        f"  Cache efficiency: {ce:.0%}  |  Occupancy: {occ:.2f}",
        f"  Estimated headroom: {headroom}",
        "",
    ]

    # Bottleneck-specific suggestions
    if bottleneck == "memory":
        suggestions = [
            "Try memory coalescing — ensure threads in a warp access consecutive addresses.",
            "Increase shared memory usage to reduce global memory traffic.",
            "If bandwidth utilization is low, look for redundant loads/stores.",
        ]
    elif bottleneck == "compute":
        suggestions = [
            "Try instruction-level parallelism — unroll inner loops.",
            "Consider using tensor cores / warp-level matrix ops if applicable.",
            "Reduce register pressure to improve occupancy.",
        ]
    elif bottleneck == "latency":
        suggestions = [
            "Increase occupancy to hide latency — reduce register/shared memory usage.",
            "Try prefetching data into shared memory.",
            "Check for warp divergence in conditional branches.",
        ]
    else:
        suggestions = [
            "Review the top stall reasons and address them directly.",
            "Try a different tiling strategy.",
            "Consider fusing adjacent operations.",
        ]

    if ce < 0.6:
        suggestions.insert(0, f"Cache efficiency is low ({ce:.0%}) — consider tiling or blocking.")
    if occ < 0.5:
        suggestions.insert(0, f"Occupancy is low ({occ:.2f}) — reduce register/smem usage.")

    lines.append("Suggestions:")
    for i, s in enumerate(suggestions[:3], 1):
        lines.append(f"  {i}. {s}")

    return "\n".join(lines)


def _tool_search_skills(session_context: dict, query: str) -> str:
    """Search the optimization skill library for relevant patterns.

    Currently a simple keyword match against iteration intents and known
    optimization categories.  A real implementation would query a vector
    store or the HF Hub skill-library repo.
    """
    iterations = session_context.get("iterations", [])
    query_lower = query.lower()

    # Keyword categories
    _SKILL_CATEGORIES = {
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

    matches = []

    # Search skill categories
    for cat, desc in _SKILL_CATEGORIES.items():
        if query_lower in cat or cat in query_lower:
            matches.append(f"[{cat}] {desc}")

    # Search iteration intents
    intent_matches = []
    for it in iterations:
        intent = it.get("intent", "").lower()
        if query_lower in intent:
            intent_matches.append(
                f"  Iter #{it.get('iteration')}: {it.get('intent')} "
                f"(status={it.get('status')}, speedup={it.get('speedup', 0):.2f}x)"
            )

    if not matches and not intent_matches:
        return f"No skills found matching '{query}'. Try: tiling, vectorization, coalescing, fusion, shared memory, loop unrolling."

    lines = []
    if matches:
        lines.append("Matching skill patterns:")
        for m in matches:
            lines.append(f"  {m}")
    if intent_matches:
        lines.append("\nPast iterations using similar techniques:")
        lines.extend(intent_matches)

    return "\n".join(lines)


def _tool_compare_iterations(session_context: dict, iter_a: int, iter_b: int) -> str:
    """Compare two iterations side by side."""
    iterations = session_context.get("iterations", [])

    a = b = None
    for it in iterations:
        num = it.get("iteration")
        if num == iter_a:
            a = it
        if num == iter_b:
            b = it

    missing = []
    if a is None:
        missing.append(str(iter_a))
    if b is None:
        missing.append(str(iter_b))
    if missing:
        return (
            f"Iteration(s) #{', #'.join(missing)} not found. "
            f"Available: 1-{len(iterations)}"
        )

    def _fmt(it: dict) -> list[str]:
        lines = [
            f"  Status:    {it.get('status', 'unknown')}",
            f"  Speedup:   {it.get('speedup', 0):.2f}x",
            f"  Intent:    {it.get('intent', 'unknown')}",
        ]
        if it.get("runtime_us"):
            lines.append(f"  Runtime:   {it['runtime_us']:.1f} us")
        profile = it.get("profile", {})
        if profile:
            lines.append(f"  Bottleneck: {profile.get('bottleneck_type', 'unknown')}")
            lines.append(f"  Bandwidth:  {profile.get('bandwidth_util', 0):.0%}")
            lines.append(f"  Compute:    {profile.get('compute_util', 0):.0%}")
            lines.append(f"  Cache eff.: {profile.get('cache_efficiency', 0):.0%}")
            lines.append(f"  Occupancy:  {profile.get('occupancy', 0):.2f}")
        return lines

    lines = [f"=== Iteration #{iter_a} vs #{iter_b} ===", ""]
    lines.append(f"--- #{iter_a} ---")
    lines.extend(_fmt(a))
    lines.append("")
    lines.append(f"--- #{iter_b} ---")
    lines.extend(_fmt(b))

    # Deltas
    sp_a = a.get("speedup", 0)
    sp_b = b.get("speedup", 0)
    delta = sp_b - sp_a
    lines.append("")
    lines.append(
        f"Delta: #{iter_b} is {'+' if delta >= 0 else ''}{delta:.2f}x "
        f"{'faster' if delta > 0 else 'slower' if delta < 0 else 'same'} than #{iter_a}"
    )

    return "\n".join(lines)


# Dispatch table: tool name -> (function, list of parameter names)
_TOOL_DISPATCH: dict[str, tuple[Any, list[str]]] = {
    "show_best_kernel": (_tool_show_best_kernel, []),
    "show_results": (_tool_show_results, []),
    "explain_iteration": (_tool_explain_iteration, ["iteration"]),
    "suggest_next": (_tool_suggest_next, []),
    "search_skills": (_tool_search_skills, ["query"]),
    "compare_iterations": (_tool_compare_iterations, ["iter_a", "iter_b"]),
}


def _execute_tool(name: str, args: dict[str, Any], session_context: dict) -> str:
    """Execute a tool by name and return its string result."""
    entry = _TOOL_DISPATCH.get(name)
    if entry is None:
        return f"Unknown tool: {name}. Available: {', '.join(_TOOL_DISPATCH)}"

    func, param_names = entry
    kwargs: dict[str, Any] = {}
    for p in param_names:
        if p not in args:
            return f"Missing required parameter '{p}' for tool '{name}'."
        kwargs[p] = args[p]

    try:
        return func(session_context, **kwargs)
    except Exception as exc:
        logger.error("Tool %s raised: %s", name, exc)
        return f"Tool error: {exc}"


# ------------------------------------------------------------------
# JSON tool-call parser
# ------------------------------------------------------------------

# Matches a JSON object that contains a "tool" key.  Intentionally
# permissive — it grabs the first {...} block on its own line(s).
_TOOL_CALL_RE = re.compile(
    r'```(?:json)?\s*(\{.*?\})\s*```'   # fenced code block
    r'|'
    r'(\{[^{}]*"tool"\s*:[^{}]*\})',     # bare JSON object with "tool" key
    re.DOTALL,
)


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """Try to extract a tool-call JSON object from the LLM response.

    Returns a dict with "tool" (str) and optional "args" (dict), or None
    if the response is a plain-text answer.
    """
    stripped = text.strip()

    # Fast path: the entire response is a JSON object
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if "tool" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Regex fallback: find embedded JSON
    match = _TOOL_CALL_RE.search(text)
    if match:
        raw = match.group(1) or match.group(2)
        try:
            obj = json.loads(raw)
            if "tool" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None


# ------------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------------

class AgentLoop:
    """Agentic loop that lets the LLM decide what actions to take.

    Usage::

        loop = AgentLoop(session_context=session_data)
        answer = await loop.run("what should I try next?")
    """

    def __init__(
        self,
        session_context: dict,
        model_config: ModelConfig | None = None,
    ) -> None:
        if model_config is None:
            model_config = ModelConfig()
        self._provider = LLMProvider(model_config)
        self._session_context = session_context
        self._system_prompt = _build_system_prompt()

    @property
    def provider(self) -> LLMProvider:
        """Expose the LLM provider for cost/token tracking."""
        return self._provider

    def update_context(self, session_context: dict) -> None:
        """Replace the session context (e.g. after a new optimization run)."""
        self._session_context = session_context

    async def run(self, user_input: str) -> str:
        """Process user input through the agentic loop.

        1. Send user input + session context + available tools to LLM.
        2. LLM responds with either:
           a. A direct text answer (no tool needed).
           b. A tool call (function name + arguments).
        3. If tool call: execute the tool, append result, send back to LLM.
        4. Loop until LLM gives a final text answer (max _MAX_TOOL_CALLS
           tool calls per turn).

        Returns:
            The final text answer from the LLM.
        """
        context_str = format_session_context(self._session_context)

        # Build the initial prompt
        prompt_parts = [
            self._system_prompt,
            "--- SESSION DATA ---",
            context_str,
            "--- USER ---",
            user_input,
        ]

        tool_calls_made = 0

        for _ in range(_MAX_TOOL_CALLS + 1):  # +1 for the final text answer
            prompt = "\n\n".join(prompt_parts)

            try:
                response = await self._provider.generate(prompt)
            except Exception as exc:
                logger.error("AgentLoop LLM call failed: %s", exc)
                return (
                    f"[AI error] Could not generate a response: {exc}\n"
                    "Check your API key and network connection."
                )

            response = response.strip()

            # Try to parse a tool call
            tool_call = _parse_tool_call(response)

            if tool_call is None:
                # Plain text answer — we're done.
                return response

            if tool_calls_made >= _MAX_TOOL_CALLS:
                # Exceeded budget — return whatever text the LLM gave
                # alongside the tool call, or a fallback.
                return response

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            logger.info("AgentLoop: calling tool %s(%s)", tool_name, tool_args)

            tool_result = _execute_tool(
                tool_name, tool_args, self._session_context
            )
            tool_calls_made += 1

            # Append the tool interaction to the conversation
            prompt_parts.append(f"--- TOOL CALL ---\n{json.dumps(tool_call)}")
            prompt_parts.append(f"--- TOOL RESULT ---\n{tool_result}")
            prompt_parts.append(
                "Based on the tool result above, either call another tool or "
                "give a final answer to the user."
            )

        # Should not reach here, but just in case
        return "I was unable to complete the request within the tool call limit."
