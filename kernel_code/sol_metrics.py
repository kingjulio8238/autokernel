"""Speed-of-Light (SOL) score computation.

SOL score contextualizes kernel performance against theoretical hardware
limits, following Cursor/NVIDIA's SOL-ExecBench methodology:
- 0.0 = no useful work
- 0.5 = optimized PyTorch baseline performance
- 1.0 = theoretical hardware limit (peak TFLOPS or peak bandwidth)

The score is computed on a logarithmic scale so that improvements
near the ceiling are weighted more heavily than improvements near
the baseline.
"""

from __future__ import annotations

import math
from kernel_code.optimization_gate import GPU_SPECS


def compute_sol_score(
    kernel_runtime_us: float,
    ref_runtime_us: float,
    total_flops: int = 0,
    total_bytes: int = 0,
    gpu_type: str = "L40S",
) -> float:
    """Compute SOL score for a kernel.

    The SOL score maps kernel performance onto a 0-1 scale where:
    - 0.5 = reference (baseline) performance
    - 1.0 = theoretical hardware ceiling
    - >0.5 means faster than baseline
    - <0.5 means slower than baseline

    Uses roofline model: theoretical minimum runtime is determined
    by the bottleneck (compute or memory bandwidth).

    Args:
        kernel_runtime_us: Measured kernel runtime in microseconds.
        ref_runtime_us: Reference (baseline) runtime in microseconds.
        total_flops: Total floating-point operations (from profiling).
        total_bytes: Total bytes transferred (from profiling).
        gpu_type: GPU name for hardware specs lookup.

    Returns:
        SOL score in [0.0, 1.0] range.
    """
    if kernel_runtime_us <= 0 or ref_runtime_us <= 0:
        return 0.0

    specs = GPU_SPECS.get(gpu_type)
    if not specs:
        # Fallback: use simple speedup-based score (0.5 = baseline)
        speedup = ref_runtime_us / kernel_runtime_us
        return min(1.0, 0.5 * speedup)

    # Compute theoretical minimum runtime using roofline model
    theoretical_min_us = _compute_theoretical_min_us(
        total_flops, total_bytes, specs
    )

    if theoretical_min_us <= 0:
        # No profiling data — use speedup-relative scoring
        speedup = ref_runtime_us / kernel_runtime_us
        return min(1.0, 0.5 * speedup)

    # Map kernel runtime to SOL score on logarithmic scale
    # baseline (ref_runtime) → 0.5, theoretical_min → 1.0
    # Uses log scale so improvements near ceiling matter more
    if kernel_runtime_us >= ref_runtime_us:
        # Slower than or equal to baseline → score <= 0.5
        if ref_runtime_us > 0:
            ratio = ref_runtime_us / kernel_runtime_us  # <= 1.0
            return max(0.0, 0.5 * ratio)
        return 0.0

    if kernel_runtime_us <= theoretical_min_us:
        # At or beyond theoretical limit → 1.0
        return 1.0

    # Between baseline and theoretical limit: logarithmic interpolation
    # log(ref / kernel) / log(ref / theoretical) maps [ref, theoretical] → [0, 1]
    # Then shift to [0.5, 1.0]
    log_progress = math.log(ref_runtime_us / kernel_runtime_us)
    log_total = math.log(ref_runtime_us / theoretical_min_us)

    if log_total <= 0:
        return 0.5

    normalized = log_progress / log_total  # 0.0 at baseline, 1.0 at ceiling
    return 0.5 + 0.5 * min(1.0, normalized)


def _compute_theoretical_min_us(
    total_flops: int,
    total_bytes: int,
    specs: dict,
) -> float:
    """Compute theoretical minimum runtime using roofline model.

    The minimum is determined by whichever bottleneck is tighter:
    compute (TFLOPS) or memory bandwidth (GB/s).

    Returns:
        Theoretical minimum runtime in microseconds, or 0.0 if
        insufficient data.
    """
    if total_flops <= 0 and total_bytes <= 0:
        return 0.0

    peak_tflops = specs.get("peak_tflops_fp16", 0)
    peak_bw_gb_s = specs.get("bandwidth_gb_s", 0)

    compute_min_us = 0.0
    memory_min_us = 0.0

    if total_flops > 0 and peak_tflops > 0:
        # TFLOPS = 10^12 FLOPS, so peak_flops_per_us = peak_tflops * 10^6
        peak_flops_per_us = peak_tflops * 1e6
        compute_min_us = total_flops / peak_flops_per_us

    if total_bytes > 0 and peak_bw_gb_s > 0:
        # GB/s = 10^9 bytes/s, so peak_bytes_per_us = peak_bw_gb_s * 10^3
        peak_bytes_per_us = peak_bw_gb_s * 1e3
        memory_min_us = total_bytes / peak_bytes_per_us

    # Roofline: theoretical min is the larger (tighter) bottleneck
    if compute_min_us > 0 and memory_min_us > 0:
        return max(compute_min_us, memory_min_us)
    return compute_min_us or memory_min_us


def compute_bandwidth_sol(
    total_bytes: int,
    runtime_us: float,
    gpu_type: str = "L40S",
) -> float:
    """Compute memory bandwidth SOL (% of peak bandwidth achieved).

    Args:
        total_bytes: Total bytes transferred.
        runtime_us: Kernel runtime in microseconds.
        gpu_type: GPU name.

    Returns:
        Percentage of peak bandwidth (0-100).
    """
    if total_bytes <= 0 or runtime_us <= 0:
        return 0.0
    specs = GPU_SPECS.get(gpu_type)
    if not specs:
        return 0.0
    peak_bw_gb_s = specs.get("bandwidth_gb_s", 0)
    if peak_bw_gb_s <= 0:
        return 0.0
    achieved_gb_s = (total_bytes / 1e9) / (runtime_us / 1e6)
    return min(100.0, (achieved_gb_s / peak_bw_gb_s) * 100)


def compute_compute_sol(
    total_flops: int,
    runtime_us: float,
    gpu_type: str = "L40S",
) -> float:
    """Compute compute SOL (% of peak TFLOPS achieved).

    Args:
        total_flops: Total floating-point operations.
        runtime_us: Kernel runtime in microseconds.
        gpu_type: GPU name.

    Returns:
        Percentage of peak compute (0-100).
    """
    if total_flops <= 0 or runtime_us <= 0:
        return 0.0
    specs = GPU_SPECS.get(gpu_type)
    if not specs:
        return 0.0
    peak_tflops = specs.get("peak_tflops_fp16", 0)
    if peak_tflops <= 0:
        return 0.0
    achieved_tflops = (total_flops / 1e12) / (runtime_us / 1e6)
    return min(100.0, (achieved_tflops / peak_tflops) * 100)


def format_sol_summary(
    sol_score: float,
    bw_sol: float = 0.0,
    compute_sol: float = 0.0,
    speedup: float = 0.0,
) -> str:
    """Format SOL metrics into a human-readable summary line.

    Example: "SOL: 0.72 (BW: 45.2%, Compute: 12.3%, Speedup: 1.85x)"
    """
    parts = [f"SOL: {sol_score:.2f}"]
    if bw_sol > 0:
        parts.append(f"BW: {bw_sol:.1f}%")
    if compute_sol > 0:
        parts.append(f"Compute: {compute_sol:.1f}%")
    if speedup > 0:
        parts.append(f"Speedup: {speedup:.2f}x")
    return parts[0] + " (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else parts[0]
