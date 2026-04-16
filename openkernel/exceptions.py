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
