"""torch.profiler + CUDA events profiler for CUDA kernels.

Provides kernel-level metrics without requiring admin permissions.
Works on Modal containers out of the box.

Public API:
    profile_cuda(kernel_source, reference_source) -> ProfileData

Metrics collected:
  - Per-kernel GPU time and call counts
  - Memory bandwidth estimation (from memory events)
  - SM utilization estimation (from kernel occupancy)
  - CUDA memory allocation tracking
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile
from typing import Any

from openkernel.eval.types import BottleneckType, ProfileData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# H100 hardware constants (will be parameterized by GPU type later)
# ---------------------------------------------------------------------------
_HW_SPECS = {
    "H100": {
        "peak_bandwidth_bytes_s": 3.35e12,  # 3.35 TB/s HBM3
        "peak_flops_fp16": 989e12,  # 989 TFLOPS FP16 tensor core
        "peak_flops_fp32": 67e12,  # 67 TFLOPS FP32
        "sm_count": 132,
        "max_threads_per_sm": 2048,
    },
    "A100-80GB": {
        "peak_bandwidth_bytes_s": 2.0e12,  # 2 TB/s HBM2e
        "peak_flops_fp16": 312e12,
        "peak_flops_fp32": 19.5e12,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
    },
    "A100-40GB": {
        "peak_bandwidth_bytes_s": 1.6e12,
        "peak_flops_fp16": 312e12,
        "peak_flops_fp32": 19.5e12,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
    },
    "L40S": {
        "peak_bandwidth_bytes_s": 864e9,  # 864 GB/s GDDR6X
        "peak_flops_fp16": 362e12,
        "peak_flops_fp32": 91e12,
        "sm_count": 142,
        "max_threads_per_sm": 1536,
    },
}


def profile_cuda(
    kernel_source: str,
    reference_source: str,
    hardware: str = "H100",
    num_warmup: int = 3,
    num_profile_runs: int = 5,
) -> ProfileData:
    """Profile a CUDA kernel using torch.profiler and return structured metrics.

    Args:
        kernel_source: Python source defining ModelNew.
        reference_source: Python source defining Model + get_inputs.
        hardware: GPU type for peak-spec calculations.
        num_warmup: Number of warmup iterations before profiling.
        num_profile_runs: Number of profiled iterations.

    Returns:
        ProfileData with torch.profiler metrics.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    hw = _HW_SPECS.get(hardware, _HW_SPECS["H100"])

    # Load modules
    tmpdir = tempfile.mkdtemp(prefix="torch_profile_")
    ref_path = os.path.join(tmpdir, "reference.py")
    kernel_path = os.path.join(tmpdir, "kernel.py")

    with open(ref_path, "w") as f:
        f.write(reference_source)
    with open(kernel_path, "w") as f:
        f.write(kernel_source)

    sys.path.insert(0, tmpdir)
    try:
        ref_spec = importlib.util.spec_from_file_location("_tp_ref", ref_path)
        ref_mod = importlib.util.module_from_spec(ref_spec)
        ref_spec.loader.exec_module(ref_mod)

        kernel_spec = importlib.util.spec_from_file_location("_tp_kernel", kernel_path)
        kernel_mod = importlib.util.module_from_spec(kernel_spec)
        kernel_spec.loader.exec_module(kernel_mod)
    except Exception as exc:
        logger.error("Failed to load modules for profiling: %s", exc)
        return ProfileData(
            bottleneck_type=BottleneckType.UNKNOWN,
            raw_metrics={"error": str(exc)},
        )
    finally:
        sys.path.remove(tmpdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel_model = kernel_mod.ModelNew().to(device).eval()
    get_inputs = ref_mod.get_inputs

    # Warmup
    for _ in range(num_warmup):
        inputs = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in get_inputs()
        ]
        with torch.no_grad():
            kernel_model(*inputs)
    torch.cuda.synchronize()

    # Profile with torch.profiler
    all_events: list[Any] = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        for _ in range(num_profile_runs):
            inputs = [
                inp.to(device) if isinstance(inp, torch.Tensor) else inp
                for inp in get_inputs()
            ]
            with torch.no_grad():
                kernel_model(*inputs)
            torch.cuda.synchronize()

    # Analyze profiler output
    return _analyze_profile(prof, hw, num_profile_runs)


def _analyze_profile(
    prof: Any,
    hw: dict[str, Any],
    num_runs: int,
) -> ProfileData:
    """Analyze torch.profiler output and extract structured metrics."""
    events = prof.key_averages()

    # Separate CUDA kernel events from CPU events
    cuda_kernels = []
    total_cuda_time_us = 0.0
    total_flops = 0
    total_cpu_time_us = 0.0

    for evt in events:
        cuda_time = getattr(evt, "self_cuda_time_total", 0) or 0
        cpu_time = getattr(evt, "self_cpu_time_total", 0) or 0
        flops = getattr(evt, "flops", 0) or 0

        total_cpu_time_us += cpu_time

        if cuda_time > 0:
            cuda_kernels.append({
                "name": evt.key,
                "cuda_time_us": cuda_time,
                "cpu_time_us": cpu_time,
                "count": evt.count,
                "flops": flops,
                "input_shapes": getattr(evt, "input_shapes", None),
            })
            total_cuda_time_us += cuda_time
            total_flops += flops

    # Sort by CUDA time descending
    cuda_kernels.sort(key=lambda k: k["cuda_time_us"], reverse=True)

    # Calculate utilization metrics
    total_cuda_time_s = (total_cuda_time_us / 1e6) / max(num_runs, 1)  # per-run average
    peak_flops = hw.get("peak_flops_fp32", 67e12)
    peak_bw = hw.get("peak_bandwidth_bytes_s", 3.35e12)

    # Compute utilization: achieved FLOPS / peak
    compute_util = 0.0
    if total_cuda_time_s > 0 and total_flops > 0:
        achieved_flops = (total_flops / max(num_runs, 1)) / total_cuda_time_s
        compute_util = min(achieved_flops / peak_flops, 1.0)

    # Memory bandwidth utilization — estimate from CUDA memory events
    memory_bytes = _estimate_memory_bytes(prof)
    bandwidth_util = 0.0
    if total_cuda_time_s > 0 and memory_bytes > 0:
        achieved_bw = (memory_bytes / max(num_runs, 1)) / total_cuda_time_s
        bandwidth_util = min(achieved_bw / peak_bw, 1.0)

    # Occupancy estimation from kernel names and heuristics
    occupancy = _estimate_occupancy(cuda_kernels, hw)

    # Bottleneck classification
    bottleneck = _classify_bottleneck(compute_util, bandwidth_util, total_cuda_time_s)

    # Roofline position: geometric mean of primary utilizations
    roofline_pos = 0.0
    if compute_util > 0 and bandwidth_util > 0:
        roofline_pos = (compute_util * bandwidth_util) ** 0.5
    elif compute_util > 0:
        roofline_pos = compute_util
    elif bandwidth_util > 0:
        roofline_pos = bandwidth_util

    raw_metrics = {
        "total_cuda_time_us": total_cuda_time_us,
        "total_cpu_time_us": total_cpu_time_us,
        "total_flops": total_flops,
        "num_cuda_kernels": len(cuda_kernels),
        "num_profile_runs": num_runs,
        "top_kernels": cuda_kernels[:10],
        "memory_bytes_estimated": memory_bytes,
        "hardware": hw,
    }

    return ProfileData(
        bottleneck_type=bottleneck,
        roofline_position=round(roofline_pos, 4),
        cache_efficiency=0.0,  # Not available without NCU
        occupancy=round(occupancy, 4),
        bandwidth_utilization=round(bandwidth_util, 4),
        compute_utilization=round(compute_util, 4),
        top_stalls=[],  # Not available without NCU
        raw_metrics=raw_metrics,
    )


def _estimate_memory_bytes(prof: Any) -> int:
    """Estimate total memory bytes transferred from profiler events.

    Uses memory allocation events and tensor shapes when available.
    """
    total_bytes = 0

    for evt in prof.key_averages():
        # self_cuda_memory_usage gives allocation delta, not throughput
        # We use input shapes to estimate read bytes
        shapes = getattr(evt, "input_shapes", None)
        if shapes and hasattr(evt, "self_cuda_time_total"):
            if evt.self_cuda_time_total > 0:
                for shape in shapes:
                    if isinstance(shape, (list, tuple)) and shape:
                        # Assume FP32 (4 bytes) for each element
                        numel = 1
                        for dim in shape:
                            if isinstance(dim, int) and dim > 0:
                                numel *= dim
                        total_bytes += numel * 4  # read
                        total_bytes += numel * 4  # write (conservative)

    return total_bytes


def _estimate_occupancy(
    cuda_kernels: list[dict[str, Any]],
    hw: dict[str, Any],
) -> float:
    """Estimate achieved occupancy from kernel information.

    Without NCU, we can only make rough estimates. Returns 0.0 if
    insufficient information is available.
    """
    if not cuda_kernels:
        return 0.0

    # Heuristic: more distinct CUDA kernels in a single forward pass
    # generally means better SM utilization
    num_kernels = sum(k["count"] for k in cuda_kernels)
    sm_count = hw.get("sm_count", 132)

    # Very rough heuristic — will be replaced by NCU data when available
    if num_kernels >= sm_count:
        return 0.7  # likely good occupancy
    elif num_kernels >= sm_count // 2:
        return 0.5
    elif num_kernels >= 4:
        return 0.3
    else:
        return 0.15  # single large kernel, unknown occupancy


def _classify_bottleneck(
    compute_util: float,
    bandwidth_util: float,
    total_cuda_time_s: float,
) -> BottleneckType:
    """Classify the primary performance bottleneck."""
    if compute_util == 0.0 and bandwidth_util == 0.0:
        if total_cuda_time_s < 1e-5:
            return BottleneckType.LATENCY_BOUND
        return BottleneckType.UNKNOWN

    # Clear compute bound
    if compute_util > 0.5 and compute_util > bandwidth_util * 1.5:
        return BottleneckType.COMPUTE_BOUND

    # Clear memory bound
    if bandwidth_util > 0.3 and bandwidth_util > compute_util * 1.5:
        return BottleneckType.MEMORY_BOUND

    # Both low: latency bound (kernel launch overhead dominates)
    if compute_util < 0.15 and bandwidth_util < 0.15:
        return BottleneckType.LATENCY_BOUND

    # Ambiguous — classify by the higher utilization
    if compute_util >= bandwidth_util:
        return BottleneckType.COMPUTE_BOUND
    return BottleneckType.MEMORY_BOUND
