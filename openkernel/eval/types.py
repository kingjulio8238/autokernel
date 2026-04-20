"""Shared contract types for the openkernel evaluation pipeline.

These dataclasses define the API boundary that all components code against:
- EvalResult: returned by the eval engine (Modal)
- ProfileData: hardware profiling metrics
- CriticDiagnosis: structured output from the Critic agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvalStatus(str, Enum):
    CORRECT = "correct"
    COMPILE_ERROR = "compile_error"
    INCORRECT = "incorrect"
    ERROR = "error"


class BottleneckType(str, Enum):
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    LATENCY_BOUND = "latency_bound"
    UNKNOWN = "unknown"


@dataclass
class ProfileData:
    """Hardware profiling metrics from Proton (Triton), torch.profiler (CUDA), or analytical."""

    bottleneck_type: BottleneckType = BottleneckType.UNKNOWN
    roofline_position: float = 0.0  # 0-1, how close to theoretical ceiling
    cache_efficiency: float = 0.0  # L2 hit rate estimate
    occupancy: float = 0.0  # achieved / theoretical occupancy
    bandwidth_utilization: float = 0.0  # achieved / peak memory bandwidth
    compute_utilization: float = 0.0  # achieved / peak compute throughput
    top_stalls: list[str] = field(default_factory=list)  # top warp stall reasons
    raw_metrics: dict = field(default_factory=dict)  # full profiler output


@dataclass
class EvalResult:
    """Result of evaluating a kernel against a reference implementation."""

    status: EvalStatus
    correct: bool = False
    speedup: float = 0.0  # ref_runtime / kernel_runtime
    runtime_us: float = 0.0  # kernel runtime in microseconds
    ref_runtime_us: float = 0.0  # reference runtime in microseconds
    profile: ProfileData = field(default_factory=ProfileData)
    error: str | None = None  # error message if status != correct
    eval_seconds: float = 0.0  # wall-clock time for evaluation


@dataclass
class CriticDiagnosis:
    """Structured diagnosis from the Critic agent after analyzing profiler data."""

    bottleneck_type: BottleneckType = BottleneckType.UNKNOWN
    roofline_position: float = 0.0  # 0-1
    specific_issue: str = ""  # "L2 hit rate 45% — strided access with 256-byte gaps"
    recommendation: str = ""  # "Restructure to coalesced access with BLOCK_K=64 tiles"
    estimated_headroom: float = 0.0  # estimated remaining speedup possible
    confidence: float = 0.0  # 0-1, how confident in the diagnosis
