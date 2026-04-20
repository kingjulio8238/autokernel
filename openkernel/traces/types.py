"""Trace types for capturing optimization runs (feeds kernelgen-1 training pipeline).

Follows Cursor/Composer 2 best practices: capture everything, filter for quality later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IterationTrace:
    """Single iteration within an optimization run."""

    iteration: int
    intent: str  # optimization intent from world model

    # LLM interaction
    generator_prompt: str = ""
    generator_response: str = ""
    critic_prompt: str | None = None
    critic_response: str | None = None

    # Generated code
    kernel_code: str = ""

    # Evaluation
    eval_status: str = ""  # correct | compile_error | incorrect | error
    speedup: float = 0.0
    runtime_us: float = 0.0
    ref_runtime_us: float = 0.0

    # Profiling
    profile_data: dict = field(default_factory=dict)
    bottleneck_type: str | None = None
    critic_diagnosis: str | None = None

    # Decision
    decision: str = ""  # keep | discard | retry

    # Cost
    tokens_used: int = 0
    llm_cost_usd: float = 0.0
    modal_cost_usd: float = 0.0
    latency_seconds: float = 0.0


@dataclass
class OptimizationTrace:
    """Full trace of a kernel optimization session."""

    # Session metadata
    session_id: str = ""
    timestamp: str = ""
    problem_id: str = ""  # KernelBench level+id or custom
    problem_source: str = ""  # "kernelbench" | "custom" | "production"
    hardware: str = ""  # "H100" | "A100" | "L40S"
    backend: str = ""  # "triton" | "cuda"
    model_id: str = ""  # which LLM was used
    openkernel_version: str = ""

    # Results
    final_speedup: float = 0.0
    final_correct: bool = False
    total_iterations: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_time_seconds: float = 0.0

    # Iteration-level detail
    iterations: list[IterationTrace] = field(default_factory=list)

    # Strategy-level detail
    strategies_tried: list[str] = field(default_factory=list)
    strategies_succeeded: list[str] = field(default_factory=list)
    skills_retrieved: list[str] = field(default_factory=list)
    skills_created: list[str] = field(default_factory=list)
