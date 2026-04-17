"""Evaluation harness — Python wrapper around the Modal eval engine.

Public API:
    evaluate(kernel_source, reference_source, config) -> EvalResult

Calls the Modal GPU function, handles errors/timeouts, and populates
EvalResult + ProfileData from the remote response.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openkernel.config import Backend, EvalMode, OpenKernelConfig
from openkernel.eval.types import (
    BottleneckType,
    EvalResult,
    EvalStatus,
    ProfileData,
)

logger = logging.getLogger(__name__)


def evaluate(
    kernel_source: str,
    reference_source: str,
    config: OpenKernelConfig | None = None,
) -> EvalResult:
    """Evaluate a kernel against a reference implementation.

    Sends the kernel and reference source to Modal for GPU evaluation,
    then parses the result into a structured EvalResult.

    Args:
        kernel_source: Python source code defining ModelNew.
        reference_source: Python source code defining Model + get_inputs + get_init_inputs.
        config: OpenKernelConfig controlling eval mode, GPU type, timeouts, etc.

    Returns:
        EvalResult with status, correctness, speedup, timing, and profile data.
    """
    if config is None:
        config = OpenKernelConfig()

    wall_start = time.time()

    try:
        raw_result = _call_modal(kernel_source, reference_source, config)
    except TimeoutError:
        return EvalResult(
            status=EvalStatus.ERROR,
            correct=False,
            error=f"Modal evaluation timed out after {config.modal.timeout_seconds}s",
            eval_seconds=time.time() - wall_start,
        )
    except ConnectionError as exc:
        return EvalResult(
            status=EvalStatus.ERROR,
            correct=False,
            error=f"Modal connection error: {exc}",
            eval_seconds=time.time() - wall_start,
        )
    except Exception as exc:
        return EvalResult(
            status=EvalStatus.ERROR,
            correct=False,
            error=f"Unexpected eval error: {type(exc).__name__}: {exc}",
            eval_seconds=time.time() - wall_start,
        )

    return _parse_result(raw_result, wall_start)


def evaluate_local(
    kernel_source: str,
    reference_source: str,
    config: OpenKernelConfig | None = None,
) -> EvalResult:
    """Evaluate locally (requires local GPU). Useful for testing.

    Uses the same eval logic as Modal but runs in-process.
    """
    if config is None:
        config = OpenKernelConfig()

    wall_start = time.time()

    try:
        from modal_infra.app import eval_kernel_on_gpu

        raw_result = eval_kernel_on_gpu.local(
            kernel_source=kernel_source,
            reference_source=reference_source,
            eval_mode=config.eval_mode.value,
            correctness_trials=config.correctness_trials,
            perf_trials_fast=config.perf_trials_fast,
            perf_trials_thorough=config.perf_trials_thorough,
            gpu_type=config.modal.gpu_type.value,
        )
    except Exception as exc:
        return EvalResult(
            status=EvalStatus.ERROR,
            correct=False,
            error=f"Local eval error: {type(exc).__name__}: {exc}",
            eval_seconds=time.time() - wall_start,
        )

    return _parse_result(raw_result, wall_start)


def _call_modal(
    kernel_source: str,
    reference_source: str,
    config: OpenKernelConfig,
) -> dict[str, Any]:
    """Call the Modal GPU function and return the raw result dict.

    Raises:
        TimeoutError: If the Modal call exceeds the configured timeout.
        ConnectionError: If Modal is unreachable.
    """
    import modal

    eval_kernel_on_gpu = modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")

    logger.info(
        "Calling Modal eval: mode=%s gpu=%s",
        config.eval_mode.value,
        config.modal.gpu_type.value,
    )

    try:
        result = eval_kernel_on_gpu.remote(
            kernel_source=kernel_source,
            reference_source=reference_source,
            eval_mode=config.eval_mode.value,
            correctness_trials=config.correctness_trials,
            perf_trials_fast=config.perf_trials_fast,
            perf_trials_thorough=config.perf_trials_thorough,
            gpu_type=config.modal.gpu_type.value,
        )
    except Exception as exc:
        exc_str = str(exc).lower()
        if "timeout" in exc_str:
            raise TimeoutError(str(exc)) from exc
        if "connection" in exc_str or "unreachable" in exc_str:
            raise ConnectionError(str(exc)) from exc
        raise

    return result


def _parse_result(raw: dict[str, Any], wall_start: float) -> EvalResult:
    """Parse a raw Modal result dict into a structured EvalResult."""
    status_str = raw.get("status", "error")
    try:
        status = EvalStatus(status_str)
    except ValueError:
        status = EvalStatus.ERROR

    profile_data = _parse_profile(raw.get("profile", {}))

    return EvalResult(
        status=status,
        correct=raw.get("correct", False),
        speedup=raw.get("speedup", 0.0),
        runtime_us=raw.get("runtime_us", 0.0),
        ref_runtime_us=raw.get("ref_runtime_us", 0.0),
        profile=profile_data,
        error=raw.get("error"),
        eval_seconds=raw.get("eval_seconds", time.time() - wall_start),
    )


def _parse_profile(raw_profile: dict[str, Any]) -> ProfileData:
    """Parse a raw profile dict into a ProfileData dataclass."""
    bottleneck_str = raw_profile.get("bottleneck_type", "unknown")
    try:
        bottleneck = BottleneckType(bottleneck_str)
    except ValueError:
        bottleneck = BottleneckType.UNKNOWN

    return ProfileData(
        bottleneck_type=bottleneck,
        roofline_position=raw_profile.get("roofline_position", 0.0),
        cache_efficiency=raw_profile.get("cache_efficiency", 0.0),
        occupancy=raw_profile.get("occupancy", 0.0),
        bandwidth_utilization=raw_profile.get("bandwidth_utilization", 0.0),
        compute_utilization=raw_profile.get("compute_utilization", 0.0),
        top_stalls=raw_profile.get("top_stalls", []),
        raw_metrics=raw_profile.get("raw_metrics", {}),
    )
