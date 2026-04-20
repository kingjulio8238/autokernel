"""Triton Proton profiler integration.

Proton is Triton's built-in profiler that provides instruction-level metrics
without requiring admin/root permissions. It works on Modal containers out
of the box since it's part of the Triton package.

Public API:
    profile_triton(kernel_source, reference_source) -> ProfileData

Proton metrics include:
  - Time spent per Triton kernel (instruction-level breakdown)
  - Memory throughput (bytes read/written)
  - Compute throughput (FLOP/s)
  - Hardware counters (occupancy, L2 hit rate, etc.)
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


def profile_triton(
    kernel_source: str,
    reference_source: str,
) -> ProfileData:
    """Profile a Triton kernel using Proton and return structured metrics.

    Loads the kernel and reference as modules, runs the kernel under Proton
    profiling, and extracts hardware metrics.

    Args:
        kernel_source: Python source defining ModelNew (with Triton kernel).
        reference_source: Python source defining Model + get_inputs.

    Returns:
        ProfileData populated with Proton metrics.
    """
    try:
        import triton.profiler as proton  # noqa: F401
    except ImportError:
        logger.warning("triton.profiler (Proton) not available. Trying proton package.")
        try:
            import proton  # noqa: F401
        except ImportError:
            raise ImportError(
                "Proton is not installed. Install triton >= 3.0 for built-in Proton support."
            )

    # Load modules
    tmpdir = tempfile.mkdtemp(prefix="proton_profile_")
    ref_path = os.path.join(tmpdir, "reference.py")
    kernel_path = os.path.join(tmpdir, "kernel.py")

    with open(ref_path, "w") as f:
        f.write(reference_source)
    with open(kernel_path, "w") as f:
        f.write(kernel_source)

    sys.path.insert(0, tmpdir)
    try:
        ref_spec = importlib.util.spec_from_file_location("_proton_ref", ref_path)
        ref_mod = importlib.util.module_from_spec(ref_spec)
        ref_spec.loader.exec_module(ref_mod)

        kernel_spec = importlib.util.spec_from_file_location("_proton_kernel", kernel_path)
        kernel_mod = importlib.util.module_from_spec(kernel_spec)
        kernel_spec.loader.exec_module(kernel_mod)
    finally:
        sys.path.remove(tmpdir)

    return _run_proton_profile(ref_mod, kernel_mod, tmpdir)


def _run_proton_profile(
    ref_mod: Any,
    kernel_mod: Any,
    tmpdir: str,
) -> ProfileData:
    """Execute the kernel under Proton profiling and extract metrics."""
    import json

    import torch

    device = torch.device("cuda")
    kernel_model = kernel_mod.ModelNew().to(device).eval()
    get_inputs = ref_mod.get_inputs

    # Proton session-based profiling
    proton_output_path = os.path.join(tmpdir, "proton_trace")

    try:
        # Try the triton.profiler API (Triton >= 3.0)
        try:
            from triton.profiler import proton as proton_api
        except ImportError:
            import proton as proton_api

        session_id = proton_api.start(name="openkernel_profile", backend="cupti")

        # Warmup
        for _ in range(3):
            inputs = [
                inp.to(device) if isinstance(inp, torch.Tensor) else inp
                for inp in get_inputs()
            ]
            with torch.no_grad():
                kernel_model(*inputs)
        torch.cuda.synchronize()

        # Profiled run
        proton_api.activate(session_id)
        for _ in range(5):
            inputs = [
                inp.to(device) if isinstance(inp, torch.Tensor) else inp
                for inp in get_inputs()
            ]
            with torch.no_grad():
                kernel_model(*inputs)
        torch.cuda.synchronize()
        proton_api.deactivate(session_id)

        # Finalize and get metrics
        proton_api.finalize()

        # Parse Proton output
        return _parse_proton_output(proton_output_path, proton_api)

    except Exception as exc:
        logger.warning("Proton profiling failed: %s. Falling back to basic metrics.", exc)
        return _fallback_basic_profile(kernel_model, get_inputs, device)


def _parse_proton_output(output_path: str, proton_api: Any) -> ProfileData:
    """Parse Proton's output into ProfileData.

    Proton outputs a JSON trace with per-kernel metrics including:
    - duration (ns)
    - bytes read/written (memory throughput)
    - flops (compute throughput)
    - sm_active / sm_occupancy
    """
    raw_metrics: dict[str, Any] = {}

    # Try to read the Proton JSON output
    json_path = output_path + ".json"
    hatchet_path = output_path + ".hatchet"

    for candidate in [json_path, hatchet_path, output_path]:
        if os.path.exists(candidate):
            try:
                with open(candidate) as f:
                    raw_metrics = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                continue

    # Extract metrics from Proton's output format
    # Proton stores metrics in a tree structure; we flatten the top-level kernel
    metrics = _extract_proton_metrics(raw_metrics)

    # Classify bottleneck
    bottleneck = _classify_bottleneck(metrics)

    return ProfileData(
        bottleneck_type=bottleneck,
        roofline_position=metrics.get("roofline_position", 0.0),
        cache_efficiency=metrics.get("l2_hit_rate", 0.0),
        occupancy=metrics.get("occupancy", 0.0),
        bandwidth_utilization=metrics.get("bandwidth_utilization", 0.0),
        compute_utilization=metrics.get("compute_utilization", 0.0),
        top_stalls=metrics.get("top_stalls", []),
        raw_metrics=raw_metrics,
    )


def _extract_proton_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract structured metrics from Proton's raw JSON output.

    Proton's output format varies by version. This function normalizes
    across known formats.
    """
    metrics: dict[str, Any] = {
        "roofline_position": 0.0,
        "l2_hit_rate": 0.0,
        "occupancy": 0.0,
        "bandwidth_utilization": 0.0,
        "compute_utilization": 0.0,
        "top_stalls": [],
    }

    if not raw:
        return metrics

    # Proton stores data in nested dicts: frame -> children -> metrics
    # Walk the tree to find kernel-level metrics
    def _walk(node: Any, depth: int = 0) -> None:
        if isinstance(node, dict):
            # Look for hardware counter metrics
            for key in [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "l2_hit_rate",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "achieved_occupancy",
            ]:
                if key in node:
                    val = node[key]
                    if "sm__throughput" in key:
                        metrics["compute_utilization"] = max(
                            metrics["compute_utilization"], float(val) / 100.0
                        )
                    elif "dram__throughput" in key:
                        metrics["bandwidth_utilization"] = max(
                            metrics["bandwidth_utilization"], float(val) / 100.0
                        )
                    elif "l2_hit_rate" in key:
                        metrics["l2_hit_rate"] = max(
                            metrics["l2_hit_rate"], float(val)
                        )
                    elif "occupancy" in key or "warps_active" in key:
                        metrics["occupancy"] = max(
                            metrics["occupancy"], float(val) / 100.0
                        )

            # Proton custom metrics: bytes, flops, duration
            bytes_total = node.get("bytes", 0) or 0
            flops = node.get("flops", 0) or 0
            duration_ns = node.get("duration", 0) or 0

            if duration_ns > 0:
                duration_s = duration_ns / 1e9
                # H100: ~3.35 TB/s HBM bandwidth, ~989 TFLOPS FP16
                if bytes_total > 0:
                    bw_achieved = bytes_total / duration_s  # bytes/s
                    # Use H100 peak as reference (parameterize later)
                    peak_bw = 3.35e12  # 3.35 TB/s
                    metrics["bandwidth_utilization"] = max(
                        metrics["bandwidth_utilization"],
                        min(bw_achieved / peak_bw, 1.0),
                    )
                if flops > 0:
                    flops_achieved = flops / duration_s
                    peak_flops = 989e12  # 989 TFLOPS FP16
                    metrics["compute_utilization"] = max(
                        metrics["compute_utilization"],
                        min(flops_achieved / peak_flops, 1.0),
                    )

            # Recurse into children
            for key, val in node.items():
                if isinstance(val, (dict, list)):
                    _walk(val, depth + 1)
        elif isinstance(node, list):
            for item in node:
                _walk(item, depth + 1)

    _walk(raw)

    # Compute roofline position (geometric mean of utilizations)
    cu = metrics["compute_utilization"]
    bu = metrics["bandwidth_utilization"]
    if cu > 0 and bu > 0:
        metrics["roofline_position"] = (cu * bu) ** 0.5
    elif cu > 0:
        metrics["roofline_position"] = cu
    elif bu > 0:
        metrics["roofline_position"] = bu

    return metrics


def _classify_bottleneck(metrics: dict[str, Any]) -> BottleneckType:
    """Classify the primary bottleneck based on profiler metrics."""
    cu = metrics.get("compute_utilization", 0.0)
    bu = metrics.get("bandwidth_utilization", 0.0)

    if cu == 0.0 and bu == 0.0:
        return BottleneckType.UNKNOWN

    # If compute utilization >> bandwidth utilization -> compute bound
    if cu > 0.5 and cu > bu * 1.5:
        return BottleneckType.COMPUTE_BOUND
    # If bandwidth utilization >> compute utilization -> memory bound
    if bu > 0.3 and bu > cu * 1.5:
        return BottleneckType.MEMORY_BOUND
    # If both are low, likely latency bound (kernel launch overhead, etc.)
    if cu < 0.2 and bu < 0.2:
        return BottleneckType.LATENCY_BOUND

    # Mixed — classify by the higher one
    if cu >= bu:
        return BottleneckType.COMPUTE_BOUND
    return BottleneckType.MEMORY_BOUND


def _fallback_basic_profile(
    model: Any,
    get_inputs: Any,
    device: Any,
) -> ProfileData:
    """Fallback: collect basic timing metrics without Proton."""
    import torch

    inputs = [
        inp.to(device) if isinstance(inp, torch.Tensor) else inp
        for inp in get_inputs()
    ]

    # Use torch.profiler as a fallback
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                inputs = [
                    inp.to(device) if isinstance(inp, torch.Tensor) else inp
                    for inp in get_inputs()
                ]
                model(*inputs)

    cuda_events = [
        e for e in prof.key_averages()
        if hasattr(e, "self_cuda_time_total") and e.self_cuda_time_total > 0
    ]

    raw_metrics = {}
    if cuda_events:
        total_cuda = sum(e.self_cuda_time_total for e in cuda_events)
        raw_metrics["total_cuda_time_us"] = total_cuda
        raw_metrics["num_cuda_kernels"] = len(cuda_events)

    return ProfileData(
        bottleneck_type=BottleneckType.UNKNOWN,
        raw_metrics=raw_metrics,
    )
