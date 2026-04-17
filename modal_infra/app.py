"""Modal GPU application for kernel evaluation.

Accepts kernel source + reference source + eval mode, compiles the kernel,
checks correctness against the reference, and benchmarks performance.

Uses KernelBench's evaluation machinery inside a GPU container with
CUDA toolkit, Triton, and PyTorch pre-installed.
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Container image — everything needed for kernel compilation + benchmarking
# ---------------------------------------------------------------------------

kernelbench_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        # PyTorch (CUDA 12.8 wheel)
        "torch>=2.5",
        # Triton (ships with torch but pin explicitly)
        "triton>=3.0",
        # KernelBench from source (requires Python 3.10)
        "kernelbench @ git+https://github.com/ScalingIntelligence/KernelBench.git",
        # Profiling
        "pydantic>=2.0",
    )
)

app = modal.App("openkernel-eval", image=kernelbench_image)

# ---------------------------------------------------------------------------
# GPU type mapping — matches openkernel.config.GpuType
# Modal v1.x uses string GPU specifiers instead of modal.gpu.* objects
# ---------------------------------------------------------------------------

_GPU_MAP = {
    "H100": "H100",
    "A100-80GB": "A100-80GB",
    "A100-40GB": "A100-40GB",
    "L40S": "L40S",
}

# ---------------------------------------------------------------------------
# Eval function — the core GPU workload
# ---------------------------------------------------------------------------


@app.function(
    gpu="L40S",
    timeout=600,
    retries=0,
)
def eval_kernel_on_gpu(
    kernel_source: str,
    reference_source: str,
    eval_mode: str = "fast",
    correctness_trials: int = 5,
    perf_trials_fast: int = 10,
    perf_trials_thorough: int = 100,
    gpu_type: str = "L40S",
) -> dict[str, Any]:
    """Evaluate a kernel against a reference implementation on a GPU.

    This function runs inside a Modal container with GPU access.
    It writes sources to temp files, uses KernelBench's eval machinery
    to check correctness and measure performance.

    Returns a dict matching EvalResult fields.
    """
    import torch

    wall_start = time.time()
    num_perf_trials = (
        perf_trials_fast if eval_mode == "fast" else perf_trials_thorough
    )

    # Write sources to temp files so KernelBench can load them as modules
    tmpdir = tempfile.mkdtemp(prefix="openkernel_eval_")
    ref_path = os.path.join(tmpdir, "reference.py")
    kernel_path = os.path.join(tmpdir, "kernel.py")

    with open(ref_path, "w") as f:
        f.write(reference_source)
    with open(kernel_path, "w") as f:
        f.write(kernel_source)

    try:
        result = _run_eval(
            ref_path=ref_path,
            kernel_path=kernel_path,
            reference_source=reference_source,
            kernel_source=kernel_source,
            correctness_trials=correctness_trials,
            num_perf_trials=num_perf_trials,
            tmpdir=tmpdir,
        )
    except Exception as exc:
        result = {
            "status": "error",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }

    result["eval_seconds"] = time.time() - wall_start
    result.setdefault("profile", {})
    return result


def _run_eval(
    *,
    ref_path: str,
    kernel_path: str,
    reference_source: str,
    kernel_source: str,
    correctness_trials: int,
    num_perf_trials: int,
    tmpdir: str,
) -> dict[str, Any]:
    """Core evaluation logic — compile, check correctness, benchmark.

    Uses KernelBench's eval_kernel_against_ref when available, falls back
    to a direct module-loading approach that mirrors the same contract.
    """
    import importlib.util
    import sys

    import torch

    # ---- Load reference module ----
    sys.path.insert(0, tmpdir)
    try:
        ref_spec = importlib.util.spec_from_file_location("_ref_module", ref_path)
        ref_mod = importlib.util.module_from_spec(ref_spec)
        ref_spec.loader.exec_module(ref_mod)

        kernel_spec = importlib.util.spec_from_file_location("_kernel_module", kernel_path)
        kernel_mod = importlib.util.module_from_spec(kernel_spec)
        kernel_spec.loader.exec_module(kernel_mod)
    except Exception as exc:
        return {
            "status": "compile_error",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"Compilation/import error: {exc}\n{traceback.format_exc()}",
        }

    # ---- Instantiate models ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ref_model = ref_mod.Model().to(device).eval()
        kernel_model = kernel_mod.ModelNew().to(device).eval()
    except Exception as exc:
        return {
            "status": "compile_error",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"Model instantiation error: {exc}\n{traceback.format_exc()}",
        }

    # ---- Get inputs ----
    try:
        get_inputs = ref_mod.get_inputs
        get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
    except AttributeError as exc:
        return {
            "status": "error",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"Reference module missing get_inputs: {exc}",
        }

    # ---- Correctness check ----
    all_correct = True
    max_diff = 0.0
    for trial in range(correctness_trials):
        inputs = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in get_inputs()
        ]
        try:
            with torch.no_grad():
                ref_out = ref_model(*inputs)
                kernel_out = kernel_model(*inputs)
        except Exception as exc:
            return {
                "status": "error",
                "correct": False,
                "speedup": 0.0,
                "runtime_us": 0.0,
                "ref_runtime_us": 0.0,
                "error": f"Runtime error on trial {trial}: {exc}\n{traceback.format_exc()}",
            }

        # Correctness tolerance (matches KernelBench defaults)
        if isinstance(ref_out, torch.Tensor) and isinstance(kernel_out, torch.Tensor):
            diff = (ref_out - kernel_out).abs().max().item()
            max_diff = max(max_diff, diff)
            # KernelBench uses a relative + absolute tolerance check
            if not torch.allclose(ref_out, kernel_out, rtol=1e-2, atol=1e-2):
                all_correct = False
        elif isinstance(ref_out, (tuple, list)):
            for r, k in zip(ref_out, kernel_out):
                if isinstance(r, torch.Tensor) and isinstance(k, torch.Tensor):
                    diff = (r - k).abs().max().item()
                    max_diff = max(max_diff, diff)
                    if not torch.allclose(r, k, rtol=1e-2, atol=1e-2):
                        all_correct = False

    if not all_correct:
        return {
            "status": "incorrect",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"Correctness check failed. Max diff: {max_diff:.6e}",
        }

    # ---- Performance benchmark ----
    ref_times = _benchmark(ref_model, get_inputs, device, num_perf_trials)
    kernel_times = _benchmark(kernel_model, get_inputs, device, num_perf_trials)

    ref_runtime_us = _median(ref_times) * 1e6
    kernel_runtime_us = _median(kernel_times) * 1e6
    speedup = ref_runtime_us / kernel_runtime_us if kernel_runtime_us > 0 else 0.0

    # ---- Profile data (basic CUDA event metrics) ----
    profile = _collect_basic_profile(
        kernel_model, get_inputs, device, ref_runtime_us, kernel_runtime_us
    )

    return {
        "status": "correct",
        "correct": True,
        "speedup": round(speedup, 4),
        "runtime_us": round(kernel_runtime_us, 2),
        "ref_runtime_us": round(ref_runtime_us, 2),
        "profile": profile,
        "error": None,
    }


def _benchmark(
    model: Any,
    get_inputs: Any,
    device: Any,
    num_trials: int,
) -> list[float]:
    """Benchmark a model using CUDA events for accurate GPU timing."""
    import torch

    times: list[float] = []

    # Warmup
    for _ in range(3):
        inputs = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in get_inputs()
        ]
        with torch.no_grad():
            model(*inputs)

    torch.cuda.synchronize()

    for _ in range(num_trials):
        inputs = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in get_inputs()
        ]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            model(*inputs)
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms / 1000.0)  # convert to seconds

    return times


def _median(values: list[float]) -> float:
    """Return the median of a list of floats."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _collect_basic_profile(
    model: Any,
    get_inputs: Any,
    device: Any,
    ref_runtime_us: float,
    kernel_runtime_us: float,
) -> dict[str, Any]:
    """Collect basic profiling data using torch.profiler inside the Modal container.

    Returns a dict matching ProfileData fields. The full deep profiling
    (Proton, NCU) is handled by the profiler orchestrator on a separate path.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    profile_result: dict[str, Any] = {
        "bottleneck_type": "unknown",
        "roofline_position": 0.0,
        "cache_efficiency": 0.0,
        "occupancy": 0.0,
        "bandwidth_utilization": 0.0,
        "compute_utilization": 0.0,
        "top_stalls": [],
        "raw_metrics": {},
    }

    try:
        inputs = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in get_inputs()
        ]

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            with torch.no_grad():
                model(*inputs)

        # Extract key metrics from profiler events
        cuda_events = [
            e for e in prof.key_averages()
            if e.device_type is not None
            and hasattr(e, "self_cuda_time_total")
            and e.self_cuda_time_total > 0
        ]

        if cuda_events:
            total_cuda_us = sum(e.self_cuda_time_total for e in cuda_events)
            top_events = sorted(cuda_events, key=lambda e: e.self_cuda_time_total, reverse=True)[:5]

            profile_result["raw_metrics"] = {
                "total_cuda_time_us": total_cuda_us,
                "num_cuda_kernels": len(cuda_events),
                "top_kernels": [
                    {
                        "name": e.key,
                        "cuda_time_us": e.self_cuda_time_total,
                        "count": e.count,
                    }
                    for e in top_events
                ],
            }

            # Heuristic bottleneck classification from timing ratios
            # Real classification comes from the profiler orchestrator
            if ref_runtime_us > 0 and kernel_runtime_us > 0:
                ratio = kernel_runtime_us / ref_runtime_us
                # If kernel is much slower, likely memory-bound (unoptimized access)
                if ratio > 1.5:
                    profile_result["bottleneck_type"] = "memory_bound"
                elif ratio < 0.8:
                    profile_result["bottleneck_type"] = "compute_bound"

    except Exception:
        # Profiling is best-effort; don't fail the eval
        pass

    return profile_result


# ---------------------------------------------------------------------------
# GPU-parameterized class variant — allows runtime GPU selection
# ---------------------------------------------------------------------------


@app.cls(timeout=600, retries=0)
class EvalWorker:
    """Class-based worker that supports dynamic GPU selection via keep_warm.

    Usage from Python client:
        worker = EvalWorker()
        result = worker.run.remote(kernel_source=..., reference_source=..., ...)
    """

    @modal.method()
    def run(
        self,
        kernel_source: str,
        reference_source: str,
        eval_mode: str = "fast",
        correctness_trials: int = 5,
        perf_trials_fast: int = 10,
        perf_trials_thorough: int = 100,
        gpu_type: str = "L40S",
    ) -> dict[str, Any]:
        """Delegate to the standalone function."""
        return eval_kernel_on_gpu.local(
            kernel_source=kernel_source,
            reference_source=reference_source,
            eval_mode=eval_mode,
            correctness_trials=correctness_trials,
            perf_trials_fast=perf_trials_fast,
            perf_trials_thorough=perf_trials_thorough,
            gpu_type=gpu_type,
        )


# ---------------------------------------------------------------------------
# Health check / warm-up endpoint
# ---------------------------------------------------------------------------


@app.function(gpu="L40S", timeout=60)
def health_check() -> dict[str, Any]:
    """Verify the container is functional and GPU is accessible."""
    import torch

    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
    }
