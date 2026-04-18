"""Tool registry: central catalogue of all tools available to the agent loop.

Usage::

    from kernel_code.tools.registry import create_default_registry

    registry = create_default_registry()
    prompt_section = registry.tools_for_llm_prompt()
    tool = registry.get_tool("profile_kernel")
    result = tool.execute(session_context, iteration=3)
"""

from __future__ import annotations

import json
from typing import Any

from kernel_code.tools import KernelTool


class ToolRegistry:
    """Thread-safe registry that maps tool names to ``KernelTool`` instances."""

    def __init__(self) -> None:
        self._tools: dict[str, KernelTool] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, tool: KernelTool) -> None:
        """Register a tool (overwrites if name already exists)."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> KernelTool | None:
        """Look up a tool by name. Returns ``None`` if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[KernelTool]:
        """Return every registered tool (insertion order)."""
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # LLM prompt formatting
    # ------------------------------------------------------------------

    def tools_for_llm_prompt(self) -> str:
        """Format all tools as a JSON-style prompt string for the LLM."""
        specs: list[dict[str, Any]] = []
        for tool in self._tools.values():
            entry: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
            }
            if tool.parameters:
                entry["parameters"] = tool.parameters
            specs.append(entry)
        return json.dumps(specs, indent=2)


# ------------------------------------------------------------------
# Default registry factory
# ------------------------------------------------------------------

def create_default_registry() -> ToolRegistry:
    """Create a registry pre-loaded with all kernel-specific tools."""
    from kernel_code.tools import (
        profile_kernel,
        evaluate_kernel,
        generate_kernel,
        compare_kernels,
        search_skills,
        show_roofline,
        suggest_optimization,
        explain_iteration,
        read_file,
        write_file,
        edit_kernel,
    )

    registry = ToolRegistry()

    registry.register(KernelTool(
        name="profile_kernel",
        description="Analyse profiler data from the session, classify bottleneck, return summary",
        parameters={"iteration": "int (optional) - iteration number; defaults to latest"},
        execute=profile_kernel.execute,
        permission="auto",
        category="analysis",
    ))

    registry.register(KernelTool(
        name="evaluate_kernel",
        description="Get eval results for a specific iteration or the best kernel",
        parameters={"iteration": "int (optional) - iteration number; omit for best"},
        execute=evaluate_kernel.execute,
        permission="auto",
        category="analysis",
    ))

    registry.register(KernelTool(
        name="generate_kernel",
        description="Describe what optimisation the Generator would try next (does not actually generate)",
        parameters={},
        execute=generate_kernel.execute,
        permission="auto",
        category="optimization",
    ))

    registry.register(KernelTool(
        name="compare_kernels",
        description="Compare two iterations side by side (speedup, profiling metrics, key differences)",
        parameters={"iter_a": "int - first iteration number", "iter_b": "int - second iteration number"},
        execute=compare_kernels.execute,
        permission="auto",
        category="analysis",
    ))

    registry.register(KernelTool(
        name="search_skills",
        description="Search the optimization skill library by keyword, return matching skills with templates",
        parameters={"query": "str - what to search for"},
        execute=search_skills.execute,
        permission="auto",
        category="analysis",
    ))

    registry.register(KernelTool(
        name="show_roofline",
        description="Format roofline analysis (arithmetic intensity, peak BW, peak compute, where the kernel sits)",
        parameters={"iteration": "int (optional) - iteration number; defaults to latest"},
        execute=show_roofline.execute,
        permission="auto",
        category="analysis",
    ))

    registry.register(KernelTool(
        name="suggest_optimization",
        description="Based on current bottleneck and what has been tried, suggest the next optimisation technique",
        parameters={},
        execute=suggest_optimization.execute,
        permission="auto",
        category="optimization",
    ))

    registry.register(KernelTool(
        name="explain_iteration",
        description="Detailed explanation of a specific iteration (what it did, why it succeeded/failed)",
        parameters={"iteration": "int - the iteration number to explain"},
        execute=explain_iteration.execute,
        permission="auto",
        category="analysis",
    ))

    registry.register(KernelTool(
        name="read_file",
        description="Read a file and return its contents. Restricted to .py, .json, .yaml, .md, .toml files. Max 200 lines.",
        parameters={"path": "str - path to the file to read"},
        execute=read_file.execute,
        permission="auto",
        category="management",
    ))

    registry.register(KernelTool(
        name="write_file",
        description="Write content to a file. Restricted to *_optimized.py, *_results.md, or files inside .kernel-code/.",
        parameters={"path": "str - destination file path", "content": "str - content to write"},
        execute=write_file.execute,
        permission="ask",
        category="management",
    ))

    registry.register(KernelTool(
        name="edit_kernel",
        description="Apply a find/replace edit to the current best kernel code. Useful for making targeted modifications to the optimization.",
        parameters={"find": "str - exact substring to find in the kernel code", "replace": "str - replacement string"},
        execute=edit_kernel.execute,
        permission="ask",
        category="optimization",
    ))

    # ------------------------------------------------------------------
    # Backward-compatible aliases for the old tool names
    # ------------------------------------------------------------------
    # The original TOOLS list used these names.  Register them as thin
    # wrappers so any persisted references or muscle-memory still works.

    registry.register(KernelTool(
        name="show_best_kernel",
        description="Display the current best optimized kernel with syntax highlighting",
        parameters={},
        execute=evaluate_kernel.execute,  # no iteration arg -> returns best
        permission="auto",
        category="analysis",
    ))

    registry.register(KernelTool(
        name="show_results",
        description="Show summary of optimization results (iterations, speedup, cost)",
        parameters={},
        execute=_show_results_compat,
        permission="auto",
        category="management",
    ))

    registry.register(KernelTool(
        name="suggest_next",
        description="Suggest the next optimization technique to try based on current bottleneck and profiling data",
        parameters={},
        execute=suggest_optimization.execute,
        permission="auto",
        category="optimization",
    ))

    registry.register(KernelTool(
        name="compare_iterations",
        description="Compare two iterations side by side",
        parameters={"iter_a": "int", "iter_b": "int"},
        execute=compare_kernels.execute,
        permission="auto",
        category="analysis",
    ))

    return registry


# ------------------------------------------------------------------
# Compat helpers for old tool names
# ------------------------------------------------------------------

def _show_results_compat(session_context: dict, **kwargs: Any) -> str:
    """Backward-compatible show_results (no separate module needed)."""
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
                f"  #{it.get('iteration')}: {it.get('speedup', 0):.2f}x -- {it.get('intent', '')}"
            )

    return "\n".join(lines)
