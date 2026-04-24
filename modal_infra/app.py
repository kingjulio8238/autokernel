"""Modal GPU application for kernel evaluation.

Accepts kernel source + reference source + eval mode, compiles the kernel,
checks correctness against the reference, and benchmarks performance.

Uses KernelBench's evaluation machinery inside a GPU container with
CUDA toolkit, Triton, and PyTorch pre-installed.
"""

from __future__ import annotations

import multiprocessing
import os
import queue as _queue
import shutil
import tempfile
import threading
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


_CUDA_FATAL_MARKERS = (
    "illegalAddress",
    "illegal memory access",
    "CUDA error",
    "CUDA_ERROR",
    "cudaErrorIllegal",
    "an illegal memory access was encountered",
    "device-side assert triggered",
    "misaligned address",
)


def _classify_cuda_error(err_str: str) -> bool:
    """True if the error string looks like a fatal CUDA-context poisoning."""
    if not err_str:
        return False
    return any(marker in err_str for marker in _CUDA_FATAL_MARKERS)


def _maybe_schedule_container_death(result: Any) -> None:
    """Schedule os._exit(1) ~2s after return if result is a cuda_error.

    The Modal container parent process persists across calls even though each
    eval runs in a fresh spawn subprocess. A CUDA illegal-memory-access in a
    child can leave the driver-level device state wedged, so subsequent
    unrelated evals in the same container see sticky CUDA errors.

    Scheduling os._exit(1) via a 2s Timer lets Modal finish serializing the
    result back to the client before the container dies. Modal transparently
    rotates to a fresh container for the next call. Hot path (no cuda_error)
    is untouched, so container reuse still amortizes cold-start cost.

    Gated by env var OPENKERNEL_DIE_ON_CUDA_ERROR (default on; set to "0"
    to disable for debugging).
    """
    if os.environ.get("OPENKERNEL_DIE_ON_CUDA_ERROR", "1") == "0":
        return
    if not (isinstance(result, dict) and result.get("status") == "cuda_error"):
        return
    threading.Timer(2.0, lambda: os._exit(1)).start()


# Typographic Unicode chars that LLMs frequently emit inside generated code
# (from markdown-ish reasoning, copy-pasted prose, etc.). Python's parser
# rejects any of these on sight, which costs us rounds across ALL problem
# types. Normalizing them to ASCII equivalents recovers ~15–20% of otherwise-
# failing rounds without changing code semantics. Mapping is conservative:
# only chars that have a clear ASCII intent (quotes, dashes, spaces, bullets).
_UNICODE_CODE_REPLACEMENTS = {
    "‘": "'",   # LEFT SINGLE QUOTATION MARK
    "’": "'",   # RIGHT SINGLE QUOTATION MARK
    "“": '"',   # LEFT DOUBLE QUOTATION MARK
    "”": '"',   # RIGHT DOUBLE QUOTATION MARK
    "–": "-",   # EN DASH
    "—": "-",   # EM DASH
    "‐": "-",   # HYPHEN
    "‑": "-",   # NON-BREAKING HYPHEN
    "‒": "-",   # FIGURE DASH
    "−": "-",   # MINUS SIGN
    "…": "...", # HORIZONTAL ELLIPSIS
    " ": " ",   # NON-BREAKING SPACE
    "•": "#",   # BULLET (replace with comment prefix so a "bullet" line doesn't blow up)
    "·": "#",   # MIDDLE DOT
    "﻿": "",    # BYTE ORDER MARK
}


def _sanitize_kernel_source(src: str) -> str:
    """Replace typographic Unicode chars with ASCII equivalents.

    LLMs often echo markdown-style prose into code (smart quotes, bullets,
    em-dashes). Python's parser fails immediately with
    ``SyntaxError: invalid character '…'`` even though the surrounding code
    is well-formed. Normalizing is safe — the replacements preserve semantic
    intent (e.g. ``"`` for smart double-quote, ``-`` for em-dash).
    """
    if not src:
        return src
    for bad, good in _UNICODE_CODE_REPLACEMENTS.items():
        if bad in src:
            src = src.replace(bad, good)
    return src


def _eval_kernel_core(
    kernel_source: str,
    reference_source: str,
    eval_mode: str = "fast",
    problem_format: str = "auto",
    correctness_trials: int = 3,
    perf_trials_fast: int = 10,
    perf_trials_thorough: int = 100,
    gpu_type: str = "L40S",
    _child_start_perf: float | None = None,
) -> dict[str, Any]:
    """Core eval logic — compile, check correctness, benchmark.

    Intended to run inside an isolated subprocess via `_eval_subprocess_worker`.
    Do NOT call directly from the Modal handler — use `_eval_kernel_impl`
    which provides subprocess isolation so a poisoned CUDA context in the
    child cannot contaminate sibling evals in the same container.
    """
    # Per-call Triton JIT cache isolation (Change B, child side).
    # Parent passes a fresh per-call dir via params; set it before torch/triton
    # import so the JIT cache can't leak state between unrelated evals.
    triton_cache_dir = os.environ.get("TRITON_CACHE_DIR")

    # Phase timers — each entry is (label, seconds). Exposed via result dict
    # so time_phases.py can render a breakdown. Low overhead (~µs).
    phases: list[tuple[str, float]] = []
    _t_import_start = time.perf_counter()
    import torch  # noqa: F401 — imported for side effect (fresh CUDA init)
    phases.append(("child: torch import", time.perf_counter() - _t_import_start))
    if _child_start_perf is not None:
        phases.insert(0, (
            "child: subprocess spawn + param receive",
            _t_import_start - _child_start_perf,
        ))

    wall_start = time.time()
    num_perf_trials = (
        perf_trials_fast if eval_mode == "fast" else perf_trials_thorough
    )

    # Sanitize typographic Unicode before the Python parser sees the source.
    # Applies to BOTH kernel and reference — the reference is authored by us
    # but we sanitize it too so "sanitize everything that hits exec_module"
    # is the single invariant callers can rely on.
    kernel_source = _sanitize_kernel_source(kernel_source)
    reference_source = _sanitize_kernel_source(reference_source)

    # Write sources to temp files so KernelBench can load them as modules
    tmpdir = tempfile.mkdtemp(prefix="openkernel_eval_")
    try:
        ref_path = os.path.join(tmpdir, "reference.py")
        kernel_path = os.path.join(tmpdir, "kernel.py")

        with open(ref_path, "w") as f:
            f.write(reference_source)
        with open(kernel_path, "w") as f:
            f.write(kernel_source)

        # Auto-detect format if needed
        if problem_format == "auto":
            if "kernel_function" in kernel_source and "class ModelNew" not in kernel_source:
                problem_format = "gpumode"
            else:
                problem_format = "kernelbench"

        try:
            if problem_format == "gpumode":
                result = _run_eval_gpumode(
                    ref_path=ref_path,
                    kernel_path=kernel_path,
                    correctness_trials=correctness_trials,
                    num_perf_trials=num_perf_trials,
                    tmpdir=tmpdir,
                    gpu_type=gpu_type,
                )
            else:
                result = _run_eval(
                    ref_path=ref_path,
                    kernel_path=kernel_path,
                    reference_source=reference_source,
                    kernel_source=kernel_source,
                    correctness_trials=correctness_trials,
                    num_perf_trials=num_perf_trials,
                    tmpdir=tmpdir,
                    gpu_type=gpu_type,
                    phases=phases,
                )
        except Exception as exc:
            err_str = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            result = {
                "status": "cuda_error" if _classify_cuda_error(err_str) else "error",
                "correct": False,
                "speedup": 0.0,
                "runtime_us": 0.0,
                "ref_runtime_us": 0.0,
                "error": err_str,
            }

        result["eval_seconds"] = time.time() - wall_start
        result.setdefault("profile", {})
        result["phases"] = phases
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if triton_cache_dir:
            shutil.rmtree(triton_cache_dir, ignore_errors=True)


def _eval_subprocess_worker(in_q, out_q) -> None:
    """Child-process entrypoint: pull params from in_q, push result to out_q.

    Runs in a fresh `multiprocessing.get_context('spawn')` process so the CUDA
    context is brand-new. If a generated kernel corrupts CUDA (illegal memory
    access, etc.), only this process dies — the Modal container parent stays
    alive and the next eval gets a fresh subprocess.
    """
    # Child-side phase timers — capture subprocess-entry wall time so we can
    # attribute the "from spawn() to first useful work" window to the
    # interpreter boot + torch import, separate from the actual eval work.
    _child_start = time.perf_counter()
    try:
        params = in_q.get(timeout=60)
    except Exception as exc:
        out_q.put({
            "status": "process_crashed",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"subprocess failed to receive params: {exc}",
        })
        return

    params["_child_start_perf"] = _child_start
    try:
        result = _eval_kernel_core(**params)
    except BaseException as exc:  # catch SystemExit too
        err_str = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        result = {
            "status": "cuda_error" if _classify_cuda_error(err_str) else "error",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": err_str,
        }

    try:
        out_q.put(result)
    except Exception:
        # If the queue is broken (parent already reaped us), nothing we can do.
        pass


def _eval_kernel_impl(
    kernel_source: str,
    reference_source: str,
    eval_mode: str = "fast",
    problem_format: str = "auto",
    correctness_trials: int = 3,
    perf_trials_fast: int = 10,
    perf_trials_thorough: int = 100,
    gpu_type: str = "L40S",
) -> dict[str, Any]:
    """Evaluate a kernel in an isolated subprocess on the GPU.

    Fresh CUDA context per call — if the child dies from illegal memory
    access or any other CUDA-fatal crash, the parent survives and returns
    a structured error. Upstream treats `cuda_error` / `process_crashed`
    / `timeout` as infra retries, not as kernel-quality signals.
    """
    wall_start = time.time()

    params = {
        "kernel_source": kernel_source,
        "reference_source": reference_source,
        "eval_mode": eval_mode,
        "problem_format": problem_format,
        "correctness_trials": correctness_trials,
        "perf_trials_fast": perf_trials_fast,
        "perf_trials_thorough": perf_trials_thorough,
        "gpu_type": gpu_type,
    }

    # `spawn` gives a brand-new interpreter → fresh CUDA context every call.
    # `fork` would inherit the poisoned parent context, which defeats the point.
    try:
        ctx = multiprocessing.get_context("spawn")
    except ValueError as exc:
        # Spawn unavailable in this environment — surface loudly rather than
        # silently falling back to fork (which would reintroduce the bug).
        return {
            "status": "error",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"multiprocessing spawn context unavailable: {exc}",
            "eval_seconds": time.time() - wall_start,
            "profile": {},
        }

    # Per-call Triton JIT cache dir (Change B, parent side).
    # Set in os.environ so the spawn child inherits it on import. Using a
    # unique per-call dir prevents JIT cache pollution from a prior kernel's
    # failed compile/run leaking into this one. Child rmtrees this dir in a
    # finally block after eval completes.
    triton_cache_dir = os.path.join(
        tempfile.gettempdir(),
        f"triton_cache_{os.getpid()}_{time.time_ns()}",
    )
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

    in_q = ctx.Queue()
    out_q = ctx.Queue()
    proc = ctx.Process(target=_eval_subprocess_worker, args=(in_q, out_q))
    proc.start()
    in_q.put(params)

    # Parent-side timeout: the Modal function itself has timeout=600, so give
    # the child a bit less so the parent still has time to terminate cleanly
    # and return a structured timeout dict.
    child_timeout_s = 540.0

    result: dict[str, Any] | None = None
    try:
        result = out_q.get(timeout=child_timeout_s)
    except _queue.Empty:
        result = {
            "status": "timeout",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"eval subprocess timed out after {child_timeout_s:.0f}s",
        }
    except Exception as exc:
        result = {
            "status": "process_crashed",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"subprocess queue error: {exc}",
        }

    # We already have the result (or a synthesized timeout). Join the child
    # so it cannot outlive the parent call. The child's Queue feeder thread
    # may still be draining after the put(), so give it a brief grace period
    # before terminating — otherwise we routinely SIGTERM a healthy child and
    # misread the negative exitcode as a CUDA crash.
    got_result_from_child = isinstance(result, dict) and result.get("status") != "timeout"
    if got_result_from_child:
        proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    if proc.is_alive():
        try:
            proc.kill()
        except Exception:
            pass
        proc.join(timeout=5)

    exitcode = proc.exitcode

    if not isinstance(result, dict):
        # Child produced nothing — hard crash / signal kill. Treat as infra.
        result = {
            "status": "process_crashed",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"subprocess produced no result (exitcode={exitcode})",
        }
    elif not got_result_from_child:
        # Timeout branch — already has status=timeout, leave as-is.
        pass
    else:
        # Child returned a result dict. Only upgrade to cuda_error when the
        # CHILD itself raised and the message screams CUDA — a parent-driven
        # SIGTERM after successful put is NOT a CUDA crash.
        if (result.get("status") == "error"
                and _classify_cuda_error(result.get("error", ""))):
            result["status"] = "cuda_error"

    result.setdefault("eval_seconds", time.time() - wall_start)
    result.setdefault("profile", {})

    # Best-effort queue cleanup
    try:
        in_q.close()
        out_q.close()
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# GPU-specific Modal functions — each deploys with a different GPU type.
# Clients select the correct function via gpu_type at runtime.
# Note: KernelBench leaderboard standardizes on L40S. Results on other GPUs
# are not directly comparable to the leaderboard.
# ---------------------------------------------------------------------------

@app.function(gpu="L40S", timeout=600, retries=0, scaledown_window=300)
def eval_kernel_on_gpu(
    kernel_source: str, reference_source: str, eval_mode: str = "fast",
    problem_format: str = "auto", correctness_trials: int = 3,
    perf_trials_fast: int = 10, perf_trials_thorough: int = 100,
    gpu_type: str = "L40S",
) -> dict[str, Any]:
    """Evaluate kernel on L40S (default, KernelBench standard)."""
    result = _eval_kernel_impl(
        kernel_source, reference_source, eval_mode, problem_format,
        correctness_trials, perf_trials_fast, perf_trials_thorough, gpu_type,
    )
    _maybe_schedule_container_death(result)
    return result


@app.function(gpu="H100", timeout=600, retries=0, scaledown_window=300)
def eval_kernel_h100(
    kernel_source: str, reference_source: str, eval_mode: str = "fast",
    problem_format: str = "auto", correctness_trials: int = 3,
    perf_trials_fast: int = 10, perf_trials_thorough: int = 100,
    gpu_type: str = "H100",
) -> dict[str, Any]:
    """Evaluate kernel on H100."""
    result = _eval_kernel_impl(
        kernel_source, reference_source, eval_mode, problem_format,
        correctness_trials, perf_trials_fast, perf_trials_thorough, gpu_type,
    )
    _maybe_schedule_container_death(result)
    return result


@app.function(gpu="A100-80GB", timeout=600, retries=0, scaledown_window=300)
def eval_kernel_a100_80gb(
    kernel_source: str, reference_source: str, eval_mode: str = "fast",
    problem_format: str = "auto", correctness_trials: int = 3,
    perf_trials_fast: int = 10, perf_trials_thorough: int = 100,
    gpu_type: str = "A100-80GB",
) -> dict[str, Any]:
    """Evaluate kernel on A100-80GB."""
    result = _eval_kernel_impl(
        kernel_source, reference_source, eval_mode, problem_format,
        correctness_trials, perf_trials_fast, perf_trials_thorough, gpu_type,
    )
    _maybe_schedule_container_death(result)
    return result


@app.function(gpu="A100", timeout=600, retries=0, scaledown_window=300)
def eval_kernel_a100_40gb(
    kernel_source: str, reference_source: str, eval_mode: str = "fast",
    problem_format: str = "auto", correctness_trials: int = 3,
    perf_trials_fast: int = 10, perf_trials_thorough: int = 100,
    gpu_type: str = "A100-40GB",
) -> dict[str, Any]:
    """Evaluate kernel on A100-40GB."""
    result = _eval_kernel_impl(
        kernel_source, reference_source, eval_mode, problem_format,
        correctness_trials, perf_trials_fast, perf_trials_thorough, gpu_type,
    )
    _maybe_schedule_container_death(result)
    return result


# Map GPU type strings to Modal function names for runtime lookup.
# Keep in sync with kernel_code/gpu_functions.py (canonical source).
# Duplicated here because Modal containers cannot import kernel_code.
_GPU_FUNCTION_MAP = {
    "L40S": "eval_kernel_on_gpu",
    "H100": "eval_kernel_h100",
    "A100-80GB": "eval_kernel_a100_80gb",
    "A100-40GB": "eval_kernel_a100_40gb",
}


def _run_eval(
    *,
    ref_path: str,
    kernel_path: str,
    reference_source: str,
    kernel_source: str,
    correctness_trials: int,
    num_perf_trials: int,
    tmpdir: str,
    gpu_type: str = "L40S",
    phases: list | None = None,
) -> dict[str, Any]:
    """Core evaluation logic — compile, check correctness, benchmark.

    Uses KernelBench's eval_kernel_against_ref when available, falls back
    to a direct module-loading approach that mirrors the same contract.
    """
    import importlib.util
    import sys

    import torch

    # Phase-timer helper. Caller may pass an external ``phases`` list so
    # the child can return a single contiguous phase history; otherwise
    # we collect locally and discard.
    _p = phases if phases is not None else []
    def _phase(label: str, t0: float) -> None:
        _p.append((label, time.perf_counter() - t0))

    # ---- Load reference + kernel modules ----
    _t0 = time.perf_counter()
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

    _phase("eval: exec ref+kernel modules", _t0)

    # ---- Get inputs + Model ctor args ----
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

    # ---- Instantiate models (pass get_init_inputs() to ctor) ----
    # Parameterized KernelBench Model classes (e.g. Conv2D) require ctor kwargs
    # declared via get_init_inputs(). Previously this was ignored, causing
    # "Model.__init__() missing N positional args" failures on all L2+ problems.
    _t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CUDA init happens lazily on first cuda op; force it here so we can
    # attribute the cost to its own phase instead of hiding inside "exec".
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _phase("eval: CUDA context init", _t0)

    _t0 = time.perf_counter()
    try:
        init_inputs = get_init_inputs()
    except Exception as exc:
        return {
            "status": "error",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"get_init_inputs() failed: {exc}\n{traceback.format_exc()}",
        }

    def _instantiate_model(model_cls):
        # init_inputs may be: dict → kwargs; list/tuple → positional; None/empty → no args
        if isinstance(init_inputs, dict):
            return model_cls(**init_inputs)
        if isinstance(init_inputs, (list, tuple)) and len(init_inputs) > 0:
            return model_cls(*init_inputs)
        return model_cls()

    try:
        ref_model = _instantiate_model(ref_mod.Model).to(device).eval()
        kernel_model = _instantiate_model(kernel_mod.ModelNew).to(device).eval()
    except Exception as exc:
        return {
            "status": "compile_error",
            "correct": False,
            "speedup": 0.0,
            "runtime_us": 0.0,
            "ref_runtime_us": 0.0,
            "error": f"Model instantiation error: {exc}\n{traceback.format_exc()}",
        }
    _phase("eval: instantiate models", _t0)

    # ---- Correctness check ----
    _t0 = time.perf_counter()
    _inputs_gen_time = 0.0
    all_correct = True
    max_diff = 0.0
    for trial in range(correctness_trials):
        _t_in = time.perf_counter()
        inputs = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in get_inputs()
        ]
        _inputs_gen_time += time.perf_counter() - _t_in
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
    _phase(
        f"eval: correctness ({correctness_trials} trials, "
        f"{_inputs_gen_time:.1f}s in get_inputs)",
        _t0,
    )

    # ---- Performance benchmark ----
    # Pre-materialize one GPU-resident input set and reuse across all
    # warmup + timed iterations of BOTH ref and kernel benchmarks. For a
    # 1GB ReLU input this cuts ~1s × 26 = ~26s off the eval by skipping
    # redundant ``torch.randn(…)`` + ``.to(cuda)`` calls. Input sits in
    # HBM across benchmarks, so there's no cold-cache artifact — it's
    # already orders of magnitude larger than any GPU cache.
    _t0 = time.perf_counter()
    bench_inputs = [
        inp.to(device) if isinstance(inp, torch.Tensor) else inp
        for inp in get_inputs()
    ]
    _phase("eval: materialize bench inputs (1×)", _t0)

    _t0 = time.perf_counter()
    ref_times = _benchmark(
        ref_model, get_inputs, device, num_perf_trials, inputs=bench_inputs
    )
    _phase(f"eval: ref benchmark ({num_perf_trials} trials)", _t0)

    _t0 = time.perf_counter()
    kernel_times = _benchmark(
        kernel_model, get_inputs, device, num_perf_trials, inputs=bench_inputs
    )
    _phase(f"eval: kernel benchmark ({num_perf_trials} trials)", _t0)

    ref_runtime_us = _median(ref_times) * 1e6
    kernel_runtime_us = _median(kernel_times) * 1e6
    speedup = ref_runtime_us / kernel_runtime_us if kernel_runtime_us > 0 else 0.0

    # ---- Profile data (basic CUDA event metrics) ----
    _t0 = time.perf_counter()
    profile = _collect_basic_profile(
        kernel_model, get_inputs, device, ref_runtime_us, kernel_runtime_us,
        gpu_type=gpu_type, inputs=bench_inputs,
    )
    _phase("eval: profile collection", _t0)

    return {
        "status": "correct",
        "correct": True,
        "speedup": round(speedup, 4),
        "runtime_us": round(kernel_runtime_us, 2),
        "ref_runtime_us": round(ref_runtime_us, 2),
        "profile": profile,
        "error": None,
    }


def _run_eval_gpumode(
    *,
    ref_path: str,
    kernel_path: str,
    correctness_trials: int,
    num_perf_trials: int,
    tmpdir: str,
    gpu_type: str = "L40S",
) -> dict[str, Any]:
    """Eval for GPU Mode format: ref_kernel() + kernel_function()."""
    import importlib.util
    import sys
    import torch

    sys.path.insert(0, tmpdir)
    try:
        ref_spec = importlib.util.spec_from_file_location("_ref_mod", ref_path)
        ref_mod = importlib.util.module_from_spec(ref_spec)
        ref_spec.loader.exec_module(ref_mod)

        kernel_spec = importlib.util.spec_from_file_location("_kernel_mod", kernel_path)
        kernel_mod = importlib.util.module_from_spec(kernel_spec)
        kernel_spec.loader.exec_module(kernel_mod)
    except Exception as exc:
        return {
            "status": "compile_error", "correct": False, "speedup": 0.0,
            "runtime_us": 0.0, "ref_runtime_us": 0.0,
            "error": f"Import error: {exc}",
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get functions
    ref_kernel = getattr(ref_mod, "ref_kernel", None)
    kernel_function = getattr(kernel_mod, "kernel_function", None)
    generate_input = getattr(ref_mod, "generate_input", None)

    if ref_kernel is None or kernel_function is None:
        return {
            "status": "error", "correct": False, "speedup": 0.0,
            "runtime_us": 0.0, "ref_runtime_us": 0.0,
            "error": "Missing ref_kernel or kernel_function",
        }

    # Generate inputs — WORKLOAD_SPEC takes precedence, else auto-scale probe.
    workload_spec = getattr(ref_mod, "WORKLOAD_SPEC", None)
    try:
        if isinstance(workload_spec, dict):
            data, workload_size = _build_inputs_from_spec(
                generate_input, workload_spec, device
            )
            print(f"[workload] from WORKLOAD_SPEC: size={workload_size} kwargs={workload_spec}")
        else:
            if workload_spec is not None:
                print(
                    f"[workload] WORKLOAD_SPEC present but not a dict "
                    f"(got {type(workload_spec).__name__}); falling back to probe"
                )
            data, workload_size, ref_probe_us = _auto_scale_probe(
                generate_input, ref_kernel, device
            )
            print(
                f"[workload] auto-scale picked size={workload_size} "
                f"(ref={ref_probe_us:.1f}μs)"
            )
    except Exception as exc:
        return {
            "status": "error", "correct": False, "speedup": 0.0,
            "runtime_us": 0.0, "ref_runtime_us": 0.0,
            "error": f"generate_input failed: {exc}\n{traceback.format_exc()}",
        }

    # Detect calling convention: tuple (GPU Mode) or unpacked args (LLM-generated)
    def _call_kernel(fn, data):
        try:
            return fn(data)
        except TypeError:
            # LLM generated kernel_function(A, B, C) instead of kernel_function(data)
            if isinstance(data, tuple):
                return fn(*data)
            raise

    # Correctness check
    all_correct = True
    max_diff = 0.0
    for trial in range(correctness_trials):
        try:
            with torch.no_grad():
                ref_out = ref_kernel(data)
                kernel_out = _call_kernel(kernel_function, data)
        except Exception as exc:
            return {
                "status": "error", "correct": False, "speedup": 0.0,
                "runtime_us": 0.0, "ref_runtime_us": 0.0,
                "error": f"Runtime error on trial {trial}: {exc}",
            }

        if isinstance(ref_out, torch.Tensor) and isinstance(kernel_out, torch.Tensor):
            diff = (ref_out - kernel_out).abs().max().item()
            max_diff = max(max_diff, diff)
            if not torch.allclose(ref_out, kernel_out, rtol=1e-2, atol=1e-2):
                all_correct = False

    if not all_correct:
        return {
            "status": "incorrect", "correct": False, "speedup": 0.0,
            "runtime_us": 0.0, "ref_runtime_us": 0.0,
            "error": f"Correctness failed. Max diff: {max_diff:.6e}",
        }

    # Benchmark
    def _ref_call():
        with torch.no_grad():
            return ref_kernel(data)

    def _kernel_call():
        with torch.no_grad():
            return _call_kernel(kernel_function, data)

    ref_times = _benchmark_fn(_ref_call, num_perf_trials)
    kernel_times = _benchmark_fn(_kernel_call, num_perf_trials)

    ref_us = _median(ref_times) * 1e6
    kernel_us = _median(kernel_times) * 1e6
    speedup = ref_us / kernel_us if kernel_us > 0 else 0.0

    # Collect SOL-flavoured profile so downstream sees real compute/bandwidth
    # utilization and a bottleneck classification. We profile the REFERENCE
    # kernel because workload flops/bytes are invariant across implementations;
    # candidate runtime (kernel_us) is what we combine with them for SOL.
    gpumode_profile: dict[str, Any] = {}
    try:
        def _ref_model_adapter(*inputs):
            if len(inputs) == 1:
                return ref_kernel(inputs[0])
            return ref_kernel(inputs)

        def _gpumode_get_inputs():
            return [data]

        gpumode_profile = _collect_basic_profile(
            model=_ref_model_adapter,
            get_inputs=_gpumode_get_inputs,
            device=device,
            ref_runtime_us=ref_us,
            kernel_runtime_us=kernel_us,
            gpu_type=gpu_type,
        )
    except Exception as exc:
        # Profiling blew up entirely. Emit the honest shape: runtime-relative
        # sol_score is still meaningful (it's kernel_us/ref_us rescaled), but
        # compute/bandwidth utils are NOT — they're zero because we have no
        # flops/bytes, not because the kernel hit 0% of peak. Flag with
        # profile_available=False + bottleneck_type="unprofiled" so the
        # display renders "—" for utils instead of "0%".
        fallback_sol = _sol_compute_sol_score(kernel_us, ref_us, 0, 0, gpu_type)
        gpumode_profile = {
            "error": f"profiling failed: {exc}",
            "sol_score": fallback_sol,
            "compute_util": 0.0,
            "bandwidth_util": 0.0,
            "total_flops": 0,
            "total_bytes": 0,
            "runtime_us": kernel_us,
            "ref_runtime_us": ref_us,
            "bottleneck_type": "unprofiled",
            "profile_available": False,
        }

    return {
        "status": "correct", "correct": True,
        "speedup": round(speedup, 4),
        "runtime_us": round(kernel_us, 2),
        "ref_runtime_us": round(ref_us, 2),
        "workload_size": workload_size,
        "profile": gpumode_profile, "error": None,
    }


def _move_to_device(data: Any, device: Any) -> Any:
    """Move a tensor / tuple-of-tensors payload onto the target device."""
    import torch

    if isinstance(data, tuple):
        return tuple(
            x.to(device) if isinstance(x, torch.Tensor) else x for x in data
        )
    if isinstance(data, list):
        return [x.to(device) if isinstance(x, torch.Tensor) else x for x in data]
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def _call_generate_input(generate_input, size: int, extra_kwargs: dict | None = None):
    """Call generate_input respecting its signature.

    Backwards-compatible with the three legacy shapes:
      - generate_input(size)
      - generate_input(size, seed)           (positional)
      - generate_input(size, seed=42)        (keyword)
    When extra_kwargs are supplied (WORKLOAD_SPEC path), they are filtered
    to parameters the signature actually accepts so stray metadata keys
    like "notes" or "shape_hint" don't blow up the call.

    For signatures with REQUIRED params beyond `size` (e.g. GPU-MODE
    histogram's `contention`, Conv2D's `kernelsize/channels/batch`), a
    heuristic-defaults table fills in reasonable values so the harness
    can produce SOME workload even when no WORKLOAD_SPEC is declared.
    """
    import inspect

    sig = inspect.signature(generate_input)
    params = sig.parameters
    param_names = list(params.keys())
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    kwargs: dict = {}
    if extra_kwargs:
        for key, value in extra_kwargs.items():
            if key == "size":
                continue  # passed positionally
            if accepts_var_kw or key in param_names:
                kwargs[key] = value

    if "seed" in param_names and "seed" not in kwargs and not extra_kwargs:
        kwargs["seed"] = 42

    # Fill heuristic defaults for REQUIRED params without defaults beyond
    # (size, seed). Known gpumode/kernelbench parameter names mapped to
    # sane middling values. Without this, problems like histogram (which
    # requires `contention`) and conv2d (`kernelsize`/`channels`/`batch`)
    # TypeError out of generate_input when no WORKLOAD_SPEC is declared.
    _HEURISTIC_DEFAULTS: dict[str, Any] = {
        "seed": 42,
        "contention": 50,      # [0, 100] — middling atomic contention
        "kernelsize": 3,       # conv2d kernel size
        "kernel_size": 3,
        "channels": 16,        # conv channel count
        "batch": 4,            # batch size
        "n": 1024, "m": 1024, "k": 1024,  # matmul dims (fallback, WORKLOAD_SPEC should override)
    }
    for pname, pobj in params.items():
        if pname == "size" or pname in kwargs:
            continue
        if pobj.default is not inspect.Parameter.empty:
            continue  # has its own default
        if pname in _HEURISTIC_DEFAULTS:
            kwargs[pname] = _HEURISTIC_DEFAULTS[pname]

    try:
        return generate_input(size, **kwargs)
    except TypeError:
        # Signature is positional-only like generate_input(size, seed).
        # Fall back to the legacy branching.
        if "seed" in param_names:
            seed_val = kwargs.get("seed", 42)
            return generate_input(size, seed=seed_val)
        if len(param_names) >= 2:
            return generate_input(size, 42)
        return generate_input(size)


def _build_inputs_from_spec(generate_input, spec: dict, device: Any):
    """Build inputs from an explicit WORKLOAD_SPEC dict.

    Returns (data_on_device, size). Extra metadata keys in spec are
    filtered out against generate_input's signature.
    """
    size = int(spec.get("size", 1024))
    data = _call_generate_input(generate_input, size, extra_kwargs=spec)
    return _move_to_device(data, device), size


def _auto_scale_probe(generate_input, ref_kernel, device: Any):
    """Probe geometric sizes, pick the first whose ref runtime ≥ 200μs.

    If none hit the threshold, use the largest probe size. Returns
    (data_on_device, size, ref_runtime_us) where data is already on device.
    """
    import torch

    probe_sizes = [1024, 65536, 1_048_576]
    threshold_us = 200.0

    last_data = None
    last_size = probe_sizes[-1]
    last_ref_us = 0.0

    for size in probe_sizes:
        try:
            raw = _call_generate_input(generate_input, size)
        except Exception as exc:
            print(f"[workload] probe size={size} generate_input failed: {exc}")
            continue

        data = _move_to_device(raw, device)

        # Warmup
        try:
            with torch.no_grad():
                ref_kernel(data)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"[workload] probe size={size} warmup failed: {exc}")
            continue

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            with torch.no_grad():
                ref_kernel(data)
        except Exception as exc:
            print(f"[workload] probe size={size} measure failed: {exc}")
            continue
        end.record()
        torch.cuda.synchronize()
        ref_us = start.elapsed_time(end) * 1000.0  # ms → μs

        last_data = data
        last_size = size
        last_ref_us = ref_us

        if ref_us >= threshold_us:
            return data, size, ref_us

    if last_data is None:
        # Every probe failed — fall back to legacy hardcoded size so the
        # caller's except handler can at least surface a generate_input error.
        data = _move_to_device(
            _call_generate_input(generate_input, 1024), device
        )
        return data, 1024, 0.0

    return last_data, last_size, last_ref_us


def _benchmark_fn(fn, num_trials: int) -> list[float]:
    """Benchmark a callable using CUDA events."""
    import torch

    times = []
    # Warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    for _ in range(num_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)

    return times


def _benchmark(
    model: Any,
    get_inputs: Any,
    device: Any,
    num_trials: int,
    inputs: Any = None,
) -> list[float]:
    """Benchmark a model using CUDA events for accurate GPU timing.

    ``inputs``: if provided, reuse this pre-materialized GPU tensor list
    across all warmup + timed iterations instead of calling
    ``get_inputs()`` on every trial. Reuse is standard for GPU perf
    measurement — the input is >> L2 cache so there's no caching
    artifact, and allocator noise drops. Skipping the per-trial
    ``torch.randn(16384,16384)`` + ``.to(cuda)`` saves ~1s per call.
    """
    import torch

    times: list[float] = []

    if inputs is None:
        # Legacy path: re-materialize inputs per trial. Kept so callers
        # that want per-trial randomization can opt in by not passing
        # ``inputs``. Our production path always passes them.
        def _fresh():
            return [
                inp.to(device) if isinstance(inp, torch.Tensor) else inp
                for inp in get_inputs()
            ]
    else:
        def _fresh():
            return inputs

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model(*_fresh())

    torch.cuda.synchronize()

    for _ in range(num_trials):
        trial_inputs = _fresh()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            model(*trial_inputs)
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


# ---------------------------------------------------------------------------
# SOL metrics — inlined from kernel_code/sol_metrics.py + kernel_code/
# optimization_gate.py::GPU_SPECS. Duplicated because Modal containers
# cannot import kernel_code (see note at _GPU_FUNCTION_MAP above).
# Keep peaks in sync with kernel_code/optimization_gate.py:GPU_SPECS.
# ---------------------------------------------------------------------------

_SOL_GPU_SPECS = {
    "H100":      {"peak_tflops_fp16":  989, "bandwidth_gb_s": 3350},
    "A100-80GB": {"peak_tflops_fp16":  312, "bandwidth_gb_s": 2039},
    "A100-40GB": {"peak_tflops_fp16":  312, "bandwidth_gb_s": 1555},
    "L40S":      {"peak_tflops_fp16":  183, "bandwidth_gb_s":  864},
    "B200":      {"peak_tflops_fp16": 2250, "bandwidth_gb_s": 8000},
}


def _sol_theoretical_min_us(total_flops: int, total_bytes: int, specs: dict) -> float:
    if total_flops <= 0 and total_bytes <= 0:
        return 0.0
    peak_tflops = specs.get("peak_tflops_fp16", 0)
    peak_bw_gb_s = specs.get("bandwidth_gb_s", 0)
    compute_min_us = (total_flops / (peak_tflops * 1e6)) if total_flops > 0 and peak_tflops > 0 else 0.0
    memory_min_us = (total_bytes / (peak_bw_gb_s * 1e3)) if total_bytes > 0 and peak_bw_gb_s > 0 else 0.0
    if compute_min_us > 0 and memory_min_us > 0:
        return max(compute_min_us, memory_min_us)
    return compute_min_us or memory_min_us


def _sol_compute_sol_score(
    kernel_runtime_us: float,
    ref_runtime_us: float,
    total_flops: int,
    total_bytes: int,
    gpu_type: str,
) -> float:
    import math
    if kernel_runtime_us <= 0 or ref_runtime_us <= 0:
        return 0.0
    specs = _SOL_GPU_SPECS.get(gpu_type)
    if not specs:
        speedup = ref_runtime_us / kernel_runtime_us
        return min(1.0, 0.5 * speedup)
    theoretical_min_us = _sol_theoretical_min_us(total_flops, total_bytes, specs)
    if theoretical_min_us <= 0:
        speedup = ref_runtime_us / kernel_runtime_us
        return min(1.0, 0.5 * speedup)
    if kernel_runtime_us >= ref_runtime_us:
        ratio = ref_runtime_us / kernel_runtime_us
        return max(0.0, 0.5 * ratio)
    if kernel_runtime_us <= theoretical_min_us:
        return 1.0
    log_progress = math.log(ref_runtime_us / kernel_runtime_us)
    log_total = math.log(ref_runtime_us / theoretical_min_us)
    if log_total <= 0:
        return 0.5
    return 0.5 + 0.5 * min(1.0, log_progress / log_total)


def _sol_compute_bandwidth_sol(total_bytes: int, runtime_us: float, gpu_type: str) -> float:
    if total_bytes <= 0 or runtime_us <= 0:
        return 0.0
    specs = _SOL_GPU_SPECS.get(gpu_type)
    if not specs:
        return 0.0
    peak_bw_gb_s = specs.get("bandwidth_gb_s", 0)
    if peak_bw_gb_s <= 0:
        return 0.0
    achieved_gb_s = (total_bytes / 1e9) / (runtime_us / 1e6)
    return min(100.0, (achieved_gb_s / peak_bw_gb_s) * 100)


def _sol_compute_compute_sol(total_flops: int, runtime_us: float, gpu_type: str) -> float:
    if total_flops <= 0 or runtime_us <= 0:
        return 0.0
    specs = _SOL_GPU_SPECS.get(gpu_type)
    if not specs:
        return 0.0
    peak_tflops = specs.get("peak_tflops_fp16", 0)
    if peak_tflops <= 0:
        return 0.0
    achieved_tflops = (total_flops / 1e12) / (runtime_us / 1e6)
    return min(100.0, (achieved_tflops / peak_tflops) * 100)


def _collect_basic_profile(
    model: Any,
    get_inputs: Any,
    device: Any,
    ref_runtime_us: float,
    kernel_runtime_us: float,
    gpu_type: str = "L40S",
    inputs: Any = None,
) -> dict[str, Any]:
    """Collect basic profiling data using torch.profiler inside the Modal container.

    Returns a dict matching ProfileData fields. The full deep profiling
    (Proton, NCU) is handled by the profiler orchestrator on a separate path.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    # profile_available flips to True only when we successfully captured
    # CUDA events AND derived non-zero total_flops OR total_bytes. Anything
    # short of that leaves utilizations at 0.0 for an honest reason (we
    # didn't measure), which is different from "measured exactly 0%".
    # Downstream display / prompt code reads this flag to decide between
    # rendering "unprofiled" vs "0%".
    profile_result: dict[str, Any] = {
        "bottleneck_type": "unknown",
        "roofline_position": 0.0,
        "cache_efficiency": 0.0,
        "occupancy": 0.0,
        "bandwidth_utilization": 0.0,
        "compute_utilization": 0.0,
        "total_flops": 0,
        "total_bytes": 0,
        "operational_intensity": 0.0,
        "top_stalls": [],
        "raw_metrics": {},
        "sol_score": 0.0,
        "compute_util": 0.0,
        "bandwidth_util": 0.0,
        "runtime_us": kernel_runtime_us,
        "ref_runtime_us": ref_runtime_us,
        "profile_available": False,
    }

    def _iter_tensors(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                yield from _iter_tensors(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                yield from _iter_tensors(item)

    def _sum_bytes(obj) -> int:
        return sum(t.nelement() * t.element_size() for t in _iter_tensors(obj))

    try:
        if inputs is None:
            inputs = [
                inp.to(device) if isinstance(inp, torch.Tensor) else inp
                for inp in get_inputs()
            ]

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with torch.no_grad():
                model(*inputs)

        # Extract key metrics from profiler events
        events = prof.key_averages()
        cuda_events = [
            e for e in events
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

            # Extract FLOPs from profiler events
            total_flops = sum(
                e.flops for e in events
                if hasattr(e, "flops") and e.flops and e.flops > 0
            )
            profile_result["total_flops"] = total_flops

            # Estimate memory I/O from input/output tensor sizes.
            # self_cuda_memory_usage tracks allocations, not actual I/O,
            # so tensor-based estimation is more reliable for OI calculation.
            # GPU-MODE problems pass inputs as tuples (e.g. prefixsum
            # (input, output), vectoradd (a, b, c)) so flatten containers.
            input_bytes = _sum_bytes(inputs)
            total_bytes = input_bytes  # read
            # Run once to capture output size for write estimate
            try:
                with torch.no_grad():
                    out = model(*inputs)
                total_bytes += _sum_bytes(out)  # write
            except Exception:
                # Conservative fallback: assume output size ≈ input size
                total_bytes += input_bytes
            profile_result["total_bytes"] = total_bytes

            # We captured CUDA events AND have at least bytes or flops —
            # downstream utilizations computed below are real measurements.
            if total_flops > 0 or total_bytes > 0:
                profile_result["profile_available"] = True

            # Operational intensity: FLOPs / bytes
            if total_bytes > 0 and total_flops > 0:
                profile_result["operational_intensity"] = total_flops / total_bytes

            # Heuristic bottleneck classification from timing ratios
            if ref_runtime_us > 0 and kernel_runtime_us > 0:
                ratio = kernel_runtime_us / ref_runtime_us
                if ratio > 1.5:
                    profile_result["bottleneck_type"] = "memory_bound"
                elif ratio < 0.8:
                    profile_result["bottleneck_type"] = "compute_bound"

            # Refine bottleneck using OI if available
            oi = profile_result["operational_intensity"]
            if oi > 0:
                # Ridge point varies by GPU — use L40S as default
                ridge_oi = 106  # L40S: 91.6 TFLOPS / 864 GB/s ≈ 106 FLOP/byte
                if oi < ridge_oi * 0.3:
                    profile_result["bottleneck_type"] = "memory_bound"
                elif oi > ridge_oi * 0.7:
                    profile_result["bottleneck_type"] = "compute_bound"

    except Exception:
        # Profiling is best-effort; don't fail the eval
        pass

    # ---- SOL metrics (runtime-relative + peak-relative) ----
    total_flops = profile_result.get("total_flops", 0) or 0
    total_bytes = profile_result.get("total_bytes", 0) or 0

    sol_score = _sol_compute_sol_score(
        kernel_runtime_us, ref_runtime_us, total_flops, total_bytes, gpu_type,
    )
    compute_util = _sol_compute_compute_sol(total_flops, kernel_runtime_us, gpu_type)
    bandwidth_util = _sol_compute_bandwidth_sol(total_bytes, kernel_runtime_us, gpu_type)

    profile_result["sol_score"] = sol_score
    # `compute_util` / `bandwidth_util` are percentages in 0-100 (SOL-style).
    # Keep those as-is for the live display, but expose the canonical
    # `compute_utilization` / `bandwidth_utilization` keys as 0.0-1.0
    # fractions so prompt templates can multiply by 100 uniformly (same
    # convention as `cache_efficiency`). Spec: pipeline-plumbing task #1.
    profile_result["compute_util"] = compute_util
    profile_result["bandwidth_util"] = bandwidth_util
    profile_result["bandwidth_utilization"] = bandwidth_util / 100.0
    profile_result["compute_utilization"] = compute_util / 100.0

    # Hardware peaks (handy for prompt labels like "of peak on L40S").
    specs = _SOL_GPU_SPECS.get(gpu_type, {})
    profile_result["gpu_type"] = gpu_type
    profile_result["hardware_peak_tflops"] = specs.get("peak_tflops_fp16", 0)
    profile_result["hardware_peak_gbps"] = specs.get("bandwidth_gb_s", 0)

    # Bottleneck classification — spec key set is
    # {"compute_bound", "memory_bound", "latency_bound", "unknown",
    # "unprofiled"}. Downstream `BottleneckType` enum only knows the
    # first four and silently falls back to UNKNOWN on unrecognized
    # strings, so "unprofiled" reaches the live display (raw dict) and
    # degrades cleanly everywhere else.
    if compute_util == 0 and bandwidth_util == 0:
        # Previously this branch emitted "latency_bound" whenever
        # flops/bytes were zero, which was dishonest — zero flops/bytes
        # means the profiler FAILED to capture them (custom ops like
        # LayerNorm don't register FLOPs via torch.profiler). We can't
        # distinguish latency-bound from unprofiled from that signal
        # alone, so prefer the honest label.
        if total_flops == 0 and total_bytes == 0:
            profile_result["bottleneck_type"] = "unprofiled"
        else:
            profile_result["bottleneck_type"] = "unknown"
    elif compute_util < 20 and bandwidth_util < 20:
        # Both utilizations very low → likely launch/latency-bound
        # (too little work per launch to saturate SMs or DRAM).
        profile_result["bottleneck_type"] = "latency_bound"
    elif compute_util > bandwidth_util:
        profile_result["bottleneck_type"] = "compute_bound"
    elif bandwidth_util > compute_util:
        profile_result["bottleneck_type"] = "memory_bound"
    # Exactly balanced → keep the existing value (OI-based heuristic if it
    # ran, else the initial "unknown").

    return profile_result


# ---------------------------------------------------------------------------
# GPU-parameterized class variant — allows runtime GPU selection
# ---------------------------------------------------------------------------


@app.cls(timeout=600, retries=0, scaledown_window=300)
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
        correctness_trials: int = 3,
        perf_trials_fast: int = 10,
        perf_trials_thorough: int = 100,
        gpu_type: str = "L40S",
    ) -> dict[str, Any]:
        """Delegate to the implementation function."""
        result = _eval_kernel_impl(
            kernel_source=kernel_source,
            reference_source=reference_source,
            eval_mode=eval_mode,
            correctness_trials=correctness_trials,
            perf_trials_fast=perf_trials_fast,
            perf_trials_thorough=perf_trials_thorough,
            gpu_type=gpu_type,
        )
        _maybe_schedule_container_death(result)
        return result


# ---------------------------------------------------------------------------
# Health check / warm-up endpoint
# ---------------------------------------------------------------------------


@app.function(gpu="L40S", timeout=60, scaledown_window=300)
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
