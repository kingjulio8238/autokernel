"""Tool registry system for the kernel optimization agent loop.

Provides a formal ``KernelTool`` dataclass that describes each tool the LLM
can invoke (name, description, parameter spec, execute function, permission
level, and category).  Individual tool modules live alongside this package and
are wired together by :func:`registry.create_default_registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class KernelTool:
    """Descriptor for a single tool available to the agent loop."""

    name: str
    description: str  # for LLM context
    parameters: dict[str, str]  # simplified param descriptions for LLM
    execute: Callable[..., str]  # (session_context: dict, **args) -> str
    permission: str = "auto"  # "auto" | "ask"
    category: str = "analysis"  # "optimization" | "analysis" | "management"
