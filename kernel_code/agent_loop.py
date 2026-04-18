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
from collections.abc import AsyncIterator
from typing import Any

from rich.console import Console
from openkernel.config import ModelConfig
from openkernel.llm.provider import LLMProvider

from kernel_code.compaction import compact_session, should_compact
from kernel_code.progress import AgentProgress
from kernel_code.shell_ai import format_session_context
from kernel_code.tools.registry import ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)

# Maximum number of tool calls the agent may make per user turn.
_MAX_TOOL_CALLS = 3

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

def _build_system_prompt(registry: ToolRegistry) -> str:
    """Build the system prompt including tool descriptions from the registry."""
    tool_block = registry.tools_for_llm_prompt()
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


def _execute_tool(
    name: str,
    args: dict[str, Any],
    session_context: dict,
    registry: ToolRegistry,
) -> str:
    """Execute a registered tool by name and return its string result."""
    tool = registry.get_tool(name)
    if tool is None:
        available = ", ".join(t.name for t in registry.list_tools())
        return f"Unknown tool: {name}. Available: {available}"

    try:
        return tool.execute(session_context, **args)
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
        console: Console | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        if model_config is None:
            model_config = ModelConfig()
        self._provider = LLMProvider(model_config)
        self._session_context = session_context
        self._registry = registry if registry is not None else create_default_registry()
        self._system_prompt = _build_system_prompt(self._registry)
        self._progress = AgentProgress(console=console)

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
        # Use compacted context for long sessions to stay within token budget
        if should_compact(self._session_context):
            context_str = compact_session(self._session_context)
        else:
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

            # Show which tool is being called
            args_preview = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if tool_args else ""
            self._progress.calling_tool(tool_name, args_preview)

            tool_result = _execute_tool(
                tool_name, tool_args, self._session_context, self._registry
            )
            tool_calls_made += 1

            # Show a brief preview of the result
            result_first_line = tool_result.split("\n", 1)[0] if tool_result else "(empty)"
            self._progress.tool_result(tool_name, result_first_line)

            # Append the tool interaction to the conversation
            prompt_parts.append(f"--- TOOL CALL ---\n{json.dumps(tool_call)}")
            prompt_parts.append(f"--- TOOL RESULT ---\n{tool_result}")
            prompt_parts.append(
                "Based on the tool result above, either call another tool or "
                "give a final answer to the user."
            )

        # Should not reach here, but just in case
        return "I was unable to complete the request within the tool call limit."

    async def run_stream(self, user_input: str) -> AsyncIterator[str]:
        """Like run() but yields final response tokens for streaming display.

        Tool calls are handled internally (not streamed) since they need
        JSON parsing.  Only the final text answer is streamed token by token.

        Yields:
            Token strings as they arrive from the LLM.
        """
        # Use compacted context for long sessions to stay within token budget
        if should_compact(self._session_context):
            context_str = compact_session(self._session_context)
        else:
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

            # On the last possible iteration (or when we expect a final
            # answer), try streaming.  For tool-call rounds, we use the
            # non-streaming generate() since we need the full JSON.

            if tool_calls_made >= _MAX_TOOL_CALLS:
                # Final round — stream it
                async for token in self._provider.generate_stream(prompt):
                    yield token
                return

            # Attempt a non-streaming call to check for tool calls
            try:
                response = await self._provider.generate(prompt)
            except Exception as exc:
                logger.error("AgentLoop LLM call failed: %s", exc)
                yield (
                    f"[AI error] Could not generate a response: {exc}\n"
                    "Check your API key and network connection."
                )
                return

            response = response.strip()

            # Try to parse a tool call
            tool_call = _parse_tool_call(response)

            if tool_call is None:
                # Plain text answer on a non-final round.
                # Re-do this as a streaming call for the nice UX.
                async for token in self._provider.generate_stream(prompt):
                    yield token
                return

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            logger.info("AgentLoop: calling tool %s(%s)", tool_name, tool_args)

            # Show which tool is being called
            args_preview = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if tool_args else ""
            self._progress.calling_tool(tool_name, args_preview)

            tool_result = _execute_tool(
                tool_name, tool_args, self._session_context, self._registry
            )
            tool_calls_made += 1

            # Show a brief preview of the result
            result_first_line = tool_result.split("\n", 1)[0] if tool_result else "(empty)"
            self._progress.tool_result(tool_name, result_first_line)

            # Append the tool interaction to the conversation
            prompt_parts.append(f"--- TOOL CALL ---\n{json.dumps(tool_call)}")
            prompt_parts.append(f"--- TOOL RESULT ---\n{tool_result}")
            prompt_parts.append(
                "Based on the tool result above, either call another tool or "
                "give a final answer to the user."
            )

        # Should not reach here, but just in case
        yield "I was unable to complete the request within the tool call limit."
