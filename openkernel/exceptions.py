"""openkernel exception hierarchy.

All openkernel-specific exceptions inherit from :class:`OpenKernelError` so
callers can catch a single base type.
"""

from __future__ import annotations


class OpenKernelError(Exception):
    """Base exception for all openkernel errors."""


class ConfigurationError(OpenKernelError):
    """Missing API keys, invalid configuration values, or missing infrastructure."""


class EvalError(OpenKernelError):
    """Modal evaluation failures, compilation errors, or runtime crashes."""


class GenerationError(OpenKernelError):
    """LLM generation failures (timeouts, rate limits, malformed output)."""


class ProfilingError(OpenKernelError):
    """Profiler failures (NCU, Proton, roofline analysis)."""


class KernelBenchError(OpenKernelError):
    """Problem loading failures (missing problems, corrupt data)."""


class BudgetExceededError(OpenKernelError):
    """Raised when optimization exceeds the configured cost budget."""

    def __init__(self, budget: float, spent: float, message: str = ""):
        self.budget = budget
        self.spent = spent
        super().__init__(
            message or f"Budget exceeded: ${spent:.2f} spent of ${budget:.2f} limit"
        )
