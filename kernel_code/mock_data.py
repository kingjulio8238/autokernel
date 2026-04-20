"""Mock data generator for kernel code TUI and dashboard development.

Generates realistic optimization session data matching the openkernel EvalResult
and ProfileData contracts, written as JSON cache files to cache/sessions/.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from pathlib import Path


# Project root — cache lives at repo root, not inside the kernel_code package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SESSIONS_DIR = _PROJECT_ROOT / "cache" / "sessions"


def _rand(low: float, high: float) -> float:
    return round(random.uniform(low, high), 3)


_INTENTS = [
    "baseline (reference wrapper)",
    "naive tiling with BLOCK_SIZE=128",
    "shared memory for input reuse",
    "vectorized float4 loads",
    "online softmax reduction (single-pass)",
    "register blocking for partial sums",
    "warp-level shuffle reductions",
    "increased tile size BLOCK_SIZE=256",
    "fused multiply-add chain",
    "coalesced memory access pattern",
    "loop unrolling with pragma",
    "persistent kernel approach",
    "split-K parallel reduction",
    "double buffering shared memory",
    "bank-conflict-free shared memory layout",
    "mixed precision (fp16 accumulation)",
    "thread coarsening (2 elements/thread)",
    "software pipelining",
    "predicated loads for boundary handling",
    "auto-tune block dimensions",
    "cooperative groups for sync",
    "tensor core utilization via wmma",
]

_KERNEL_SNIPPETS = [
    """\
@triton.jit
def kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = x * 2.0  # placeholder
    tl.store(Y + offs, y, mask=mask)""",
    """\
@triton.jit
def kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    # shared memory tiling
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc += x
    tl.store(Y + offs, acc, mask=mask)""",
    """\
@triton.jit
def fused_softmax(X, Y, stride, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + pid * stride + offs, mask=offs < N, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, axis=0)
    y = exp_x / sum_exp
    tl.store(Y + pid * stride + offs, y, mask=offs < N)""",
]

_BOTTLENECK_TYPES = ["memory_bound", "compute_bound", "latency_bound"]


def _make_iteration(
    iteration: int,
    best_speedup: float,
    ref_runtime_us: float,
) -> tuple[dict, float]:
    """Generate a single iteration with realistic progression.

    Returns (iteration_dict, new_best_speedup).
    """
    # Early iterations: compile errors and incorrect results
    if iteration <= 2:
        if random.random() < 0.4:
            return {
                "iteration": iteration,
                "speedup": 0.0,
                "status": "compile_error",
                "runtime_us": 0.0,
                "ref_runtime_us": ref_runtime_us,
                "profile": {
                    "bandwidth_util": 0.0,
                    "compute_util": 0.0,
                    "cache_efficiency": 0.0,
                    "occupancy": 0.0,
                    "bottleneck_type": "unknown",
                },
                "kernel_code_snippet": _KERNEL_SNIPPETS[0],
                "intent": _INTENTS[iteration] if iteration < len(_INTENTS) else "retry",
                "decision": "error",
                "error": "CompilationError: undeclared variable in kernel",
            }, best_speedup

        if random.random() < 0.3:
            return {
                "iteration": iteration,
                "speedup": 0.0,
                "status": "incorrect",
                "runtime_us": _rand(80, 200),
                "ref_runtime_us": ref_runtime_us,
                "profile": {
                    "bandwidth_util": _rand(0.1, 0.3),
                    "compute_util": _rand(0.1, 0.2),
                    "cache_efficiency": _rand(0.2, 0.4),
                    "occupancy": _rand(0.2, 0.5),
                    "bottleneck_type": "memory_bound",
                },
                "kernel_code_snippet": _KERNEL_SNIPPETS[0],
                "intent": _INTENTS[iteration] if iteration < len(_INTENTS) else "retry",
                "decision": "discard",
                "error": "Correctness check failed: max abs diff 0.0312 > tolerance 0.001",
            }, best_speedup

    # From iteration 3 onward: mostly correct, improving speedup
    # Logarithmic improvement curve that stalls around iteration 15+
    progress = 1 - math.exp(-0.15 * (iteration - 2))
    target_ceiling = _rand(2.0, 3.5)  # theoretical max for this session
    noise = _rand(-0.15, 0.15)

    # Base speedup from progress curve
    raw_speedup = 1.0 + (target_ceiling - 1.0) * progress + noise

    # Occasional regression
    if random.random() < 0.25:
        raw_speedup *= _rand(0.6, 0.9)

    raw_speedup = max(0.5, round(raw_speedup, 2))
    runtime_us = round(ref_runtime_us / raw_speedup, 1) if raw_speedup > 0 else 0.0

    # Determine status
    if raw_speedup > best_speedup:
        status = "keep"
        decision = "keep"
        new_best = raw_speedup
    else:
        status = "discard"
        decision = "discard"
        new_best = best_speedup

    # Profile data — improves with speedup
    bw_util = min(0.95, _rand(0.3, 0.5) + 0.3 * progress)
    compute_util = min(0.90, _rand(0.2, 0.4) + 0.25 * progress)
    cache_eff = min(0.92, _rand(0.3, 0.5) + 0.2 * progress)
    occupancy = min(0.95, _rand(0.4, 0.6) + 0.2 * progress)

    bottleneck = random.choice(_BOTTLENECK_TYPES)

    # Roofline position improves with progress (how close to theoretical ceiling)
    roofline_pos = min(0.95, _rand(0.3, 0.5) + 0.3 * progress)
    # Estimated headroom decreases as we approach the ceiling
    estimated_headroom = max(1.0, round(target_ceiling / max(raw_speedup, 0.5), 1))

    intent_idx = min(iteration, len(_INTENTS) - 1)
    snippet_idx = min(iteration % len(_KERNEL_SNIPPETS), len(_KERNEL_SNIPPETS) - 1)

    return {
        "iteration": iteration,
        "speedup": raw_speedup,
        "status": status,
        "runtime_us": runtime_us,
        "ref_runtime_us": ref_runtime_us,
        "profile": {
            "bandwidth_util": bw_util,
            "compute_util": compute_util,
            "cache_efficiency": cache_eff,
            "occupancy": occupancy,
            "bottleneck_type": bottleneck,
            "roofline_position": roofline_pos,
            "estimated_headroom": estimated_headroom,
        },
        "kernel_code_snippet": _KERNEL_SNIPPETS[snippet_idx],
        "intent": _INTENTS[intent_idx],
        "decision": decision,
        "error": None,
    }, new_best


def generate_mock_session(
    num_iterations: int = 20,
    session_id: str | None = None,
) -> Path:
    """Generate a mock optimization session and write it to cache/sessions/.

    Args:
        num_iterations: Number of optimization iterations to generate.
        session_id: Optional session ID. Auto-generated if not provided.

    Returns:
        Path to the written JSON file.
    """
    if session_id is None:
        session_id = uuid.uuid4().hex[:12]

    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    ref_runtime_us = _rand(80.0, 300.0)
    best_speedup = 1.0
    iterations = []

    for i in range(1, num_iterations + 1):
        iteration_data, best_speedup = _make_iteration(i, best_speedup, ref_runtime_us)
        iterations.append(iteration_data)

    session = {
        "session_id": session_id,
        "problem": "L1#23 softmax",
        "hardware": "H100",
        "backend": "triton",
        "model": "claude-sonnet-4-20250514",
        "ref_runtime_us": ref_runtime_us,
        "best_speedup": best_speedup,
        "num_iterations": num_iterations,
        "iterations": iterations,
    }

    out_path = _SESSIONS_DIR / f"{session_id}.json"
    out_path.write_text(json.dumps(session, indent=2))
    return out_path


if __name__ == "__main__":
    path = generate_mock_session()
    print(f"Generated mock session: {path}")
