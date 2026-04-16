"""Analytical roofline estimator — no GPU required.

Estimates compute/memory bounds by analyzing the reference implementation's
operations and comparing against hardware peak specifications. This is used
for pre-screening: sorting problems by estimated optimization headroom before
spending GPU time.

Public API:
    estimate_roofline(reference_source, hardware) -> ProfileData

This module works entirely on CPU by:
1. Parsing the reference code to instantiate the Model
2. Running a forward pass with dummy inputs to trace operations
3. Counting FLOPs and memory accesses using torch.utils.flop_counter
4. Computing operational intensity (FLOPs/byte)
5. Classifying as compute-bound or memory-bound against the roofline
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
# Hardware specifications for roofline analysis
# ---------------------------------------------------------------------------

_HW_SPECS: dict[str, dict[str, float]] = {
    "H100": {
        "peak_flops_fp32": 67e12,  # 67 TFLOPS FP32
        "peak_flops_fp16": 989e12,  # 989 TFLOPS FP16 tensor core
        "peak_bandwidth": 3.35e12,  # 3.35 TB/s HBM3
        "ridge_point_fp32": 67e12 / 3.35e12,  # ~20 FLOP/byte
        "ridge_point_fp16": 989e12 / 3.35e12,  # ~295 FLOP/byte
    },
    "A100-80GB": {
        "peak_flops_fp32": 19.5e12,
        "peak_flops_fp16": 312e12,
        "peak_bandwidth": 2.0e12,
        "ridge_point_fp32": 19.5e12 / 2.0e12,  # ~9.75
        "ridge_point_fp16": 312e12 / 2.0e12,  # ~156
    },
    "A100-40GB": {
        "peak_flops_fp32": 19.5e12,
        "peak_flops_fp16": 312e12,
        "peak_bandwidth": 1.6e12,
        "ridge_point_fp32": 19.5e12 / 1.6e12,  # ~12.2
        "ridge_point_fp16": 312e12 / 1.6e12,  # ~195
    },
    "L40S": {
        "peak_flops_fp32": 91e12,
        "peak_flops_fp16": 362e12,
        "peak_bandwidth": 864e9,
        "ridge_point_fp32": 91e12 / 864e9,  # ~105
        "ridge_point_fp16": 362e12 / 864e9,  # ~419
    },
}


def estimate_roofline(
    reference_source: str,
    hardware: str = "H100",
) -> ProfileData:
    """Estimate roofline position analytically without GPU.

    Analyzes the reference implementation to estimate FLOPs and memory
    accesses, then computes operational intensity and classifies the
    kernel as compute-bound or memory-bound.

    Args:
        reference_source: Python source defining Model + get_inputs.
        hardware: GPU type for peak specifications.

    Returns:
        ProfileData with analytical estimates.
    """
    hw = _HW_SPECS.get(hardware, _HW_SPECS["H100"])

    # Load the reference module
    tmpdir = tempfile.mkdtemp(prefix="roofline_")
    ref_path = os.path.join(tmpdir, "reference.py")
    with open(ref_path, "w") as f:
        f.write(reference_source)

    sys.path.insert(0, tmpdir)
    try:
        spec = importlib.util.spec_from_file_location("_roofline_ref", ref_path)
        ref_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_mod)
    except Exception as exc:
        logger.warning("Failed to load reference for roofline: %s", exc)
        return ProfileData(
            bottleneck_type=BottleneckType.UNKNOWN,
            raw_metrics={"error": f"Module load failed: {exc}"},
        )
    finally:
        sys.path.remove(tmpdir)

    # Count FLOPs and memory
    try:
        flop_count, memory_bytes, raw_analysis = _analyze_reference(ref_mod)
    except Exception as exc:
        logger.warning("FLOPs analysis failed: %s", exc)
        return ProfileData(
            bottleneck_type=BottleneckType.UNKNOWN,
            raw_metrics={"error": f"Analysis failed: {exc}"},
        )

    # Compute operational intensity
    if memory_bytes > 0:
        operational_intensity = flop_count / memory_bytes  # FLOPs per byte
    else:
        operational_intensity = float("inf")

    # Classify using roofline
    ridge_point = hw["ridge_point_fp32"]
    peak_flops = hw["peak_flops_fp32"]
    peak_bw = hw["peak_bandwidth"]

    if operational_intensity < ridge_point:
        # Memory-bound region: performance limited by bandwidth
        bottleneck = BottleneckType.MEMORY_BOUND
        # Achievable FLOPS = operational_intensity * peak_bandwidth
        achievable_flops = operational_intensity * peak_bw
        roofline_position = min(achievable_flops / peak_flops, 1.0) if peak_flops > 0 else 0.0
        # Bandwidth utilization would be high if perfectly optimized
        bandwidth_util = 0.8  # optimistic estimate for PyTorch baseline
        compute_util = roofline_position * 0.5  # estimated actual
    else:
        # Compute-bound region: performance limited by compute throughput
        bottleneck = BottleneckType.COMPUTE_BOUND
        roofline_position = 0.5  # PyTorch baseline typically ~50% of peak
        compute_util = 0.5
        bandwidth_util = 0.3

    # Estimate optimization headroom
    # PyTorch eager typically achieves 30-70% of what's theoretically possible
    pytorch_efficiency = 0.4  # conservative estimate
    headroom = 1.0 / pytorch_efficiency if pytorch_efficiency > 0 else 2.5

    raw_metrics = {
        "flop_count": flop_count,
        "memory_bytes": memory_bytes,
        "operational_intensity": operational_intensity,
        "ridge_point_fp32": ridge_point,
        "estimated_headroom": round(headroom, 2),
        "hardware": hardware,
        **raw_analysis,
    }

    return ProfileData(
        bottleneck_type=bottleneck,
        roofline_position=round(roofline_position, 4),
        cache_efficiency=0.0,  # cannot estimate without GPU
        occupancy=0.0,  # cannot estimate without GPU
        bandwidth_utilization=round(bandwidth_util, 4),
        compute_utilization=round(compute_util, 4),
        top_stalls=[],
        raw_metrics=raw_metrics,
    )


def _analyze_reference(ref_mod: Any) -> tuple[int, int, dict[str, Any]]:
    """Analyze the reference module to count FLOPs and memory bytes.

    Uses torch.utils.flop_counter when available, falls back to
    shape-based estimation.

    Returns:
        (flop_count, memory_bytes, raw_analysis_dict)
    """
    import torch

    model = ref_mod.Model()
    get_inputs = ref_mod.get_inputs

    # Generate inputs on CPU for analysis
    inputs = get_inputs()

    # Calculate input/output memory
    input_bytes = _count_tensor_bytes(inputs)

    # Try torch.utils.flop_counter (PyTorch >= 2.1)
    flop_count = 0
    raw_analysis: dict[str, Any] = {}

    try:
        from torch.utils.flop_counter import FlopCounterMode

        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                output = model(*inputs)

        flop_count = flop_counter.get_total_flops()
        raw_analysis["flop_source"] = "torch.utils.flop_counter"

    except (ImportError, AttributeError):
        # Fallback: estimate FLOPs from operation shapes
        logger.info("torch.utils.flop_counter not available, using shape-based estimation")
        flop_count, raw_analysis = _estimate_flops_from_shapes(model, inputs)
        raw_analysis["flop_source"] = "shape_estimation"

    # Estimate output size
    try:
        with torch.no_grad():
            output = model(*inputs)
        output_bytes = _count_tensor_bytes(
            output if isinstance(output, (list, tuple)) else [output]
        )
    except Exception:
        output_bytes = input_bytes  # conservative fallback

    # Total memory bytes: inputs read + outputs written
    # For most kernels, intermediate tensors are also read/written,
    # but for a first-order estimate we use 2x (read + write) of max(input, output)
    memory_bytes = input_bytes + output_bytes

    raw_analysis.update({
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_memory_bytes": memory_bytes,
    })

    return flop_count, memory_bytes, raw_analysis


def _count_tensor_bytes(tensors: Any) -> int:
    """Count total bytes across a collection of tensors."""
    import torch

    total = 0
    if isinstance(tensors, torch.Tensor):
        total += tensors.nelement() * tensors.element_size()
    elif isinstance(tensors, (list, tuple)):
        for t in tensors:
            total += _count_tensor_bytes(t)
    return total


def _estimate_flops_from_shapes(
    model: Any,
    inputs: list[Any],
) -> tuple[int, dict[str, Any]]:
    """Estimate FLOPs from tensor shapes using operation-specific formulas.

    Covers common operations: matmul, conv2d, elementwise, reduction.
    """
    import torch

    flop_count = 0
    ops_found: list[str] = []

    # Trace the model to find operations
    try:
        # Use torch.jit.trace for operation analysis
        with torch.no_grad():
            traced = torch.jit.trace(model, inputs)

        # Walk the traced graph for known operations
        graph = traced.graph
        for node in graph.nodes():
            kind = node.kind()
            if "matmul" in kind.lower() or "mm" in kind.lower():
                # Matrix multiply: 2*M*N*K FLOPs
                input_shapes = [
                    list(inp.type().sizes()) for inp in node.inputs()
                    if hasattr(inp.type(), "sizes")
                ]
                if len(input_shapes) >= 2:
                    a_shape = input_shapes[0]
                    b_shape = input_shapes[1]
                    if len(a_shape) >= 2 and len(b_shape) >= 2:
                        m, k = a_shape[-2], a_shape[-1]
                        n = b_shape[-1]
                        # Batch dimensions
                        batch = 1
                        for d in a_shape[:-2]:
                            batch *= d
                        flop_count += batch * 2 * m * k * n
                        ops_found.append(f"matmul({a_shape}x{b_shape})")

            elif "conv" in kind.lower():
                ops_found.append(f"conv({kind})")
                # Conservative: assume 2x input elements
                for inp in node.inputs():
                    if hasattr(inp.type(), "sizes"):
                        sizes = inp.type().sizes()
                        numel = 1
                        for s in sizes:
                            numel *= s
                        flop_count += numel * 2

            elif any(op in kind.lower() for op in ["add", "mul", "sub", "div", "relu", "gelu"]):
                # Elementwise: 1 FLOP per element
                for inp in node.inputs():
                    if hasattr(inp.type(), "sizes"):
                        sizes = inp.type().sizes()
                        numel = 1
                        for s in sizes:
                            numel *= s
                        flop_count += numel
                        ops_found.append(f"{kind}({sizes})")
                        break

    except Exception as exc:
        logger.debug("JIT trace failed for FLOP estimation: %s", exc)
        # Last resort: assume 2 FLOPs per input element
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                flop_count += inp.nelement() * 2

    return flop_count, {"operations_detected": ops_found}
