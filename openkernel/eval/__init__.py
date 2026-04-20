"""Eval package — bridge between InnerLoop's EvalFn and the Modal harness.

Public API:
    create_eval_fn(config)       — async EvalFn wrapping Modal harness.evaluate()
    create_local_eval_fn(config) — async EvalFn wrapping harness.evaluate_local()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from openkernel.config import OpenKernelConfig
from openkernel.eval.types import EvalResult, EvalStatus

if TYPE_CHECKING:
    from kernel_code.file_cache import FileStateCache

# Re-export the EvalFn type for convenience.
EvalFn = Callable[[str, str], Awaitable[EvalResult]]

logger = logging.getLogger(__name__)


def _eval_result_to_dict(result: EvalResult) -> dict:
    """Serialize an EvalResult to a plain dict for caching."""
    return {
        "status": result.status.value,
        "correct": result.correct,
        "speedup": result.speedup,
        "runtime_us": result.runtime_us,
        "ref_runtime_us": result.ref_runtime_us,
        "error": result.error,
        "eval_seconds": result.eval_seconds,
        "profile": {
            "bottleneck_type": result.profile.bottleneck_type.value,
            "roofline_position": result.profile.roofline_position,
            "cache_efficiency": result.profile.cache_efficiency,
            "occupancy": result.profile.occupancy,
            "bandwidth_utilization": result.profile.bandwidth_utilization,
            "compute_utilization": result.profile.compute_utilization,
            "top_stalls": result.profile.top_stalls,
            "raw_metrics": result.profile.raw_metrics,
        },
    }


def _dict_to_eval_result(d: dict) -> EvalResult:
    """Deserialize a cached dict back into an EvalResult."""
    from openkernel.eval.types import BottleneckType, ProfileData

    profile_data = d.get("profile", {})
    try:
        bottleneck = BottleneckType(profile_data.get("bottleneck_type", "unknown"))
    except ValueError:
        bottleneck = BottleneckType.UNKNOWN

    profile = ProfileData(
        bottleneck_type=bottleneck,
        roofline_position=profile_data.get("roofline_position", 0.0),
        cache_efficiency=profile_data.get("cache_efficiency", 0.0),
        occupancy=profile_data.get("occupancy", 0.0),
        bandwidth_utilization=profile_data.get("bandwidth_utilization", 0.0),
        compute_utilization=profile_data.get("compute_utilization", 0.0),
        top_stalls=profile_data.get("top_stalls", []),
        raw_metrics=profile_data.get("raw_metrics", {}),
    )

    try:
        status = EvalStatus(d.get("status", "error"))
    except ValueError:
        status = EvalStatus.ERROR

    return EvalResult(
        status=status,
        correct=d.get("correct", False),
        speedup=d.get("speedup", 0.0),
        runtime_us=d.get("runtime_us", 0.0),
        ref_runtime_us=d.get("ref_runtime_us", 0.0),
        profile=profile,
        error=d.get("error"),
        eval_seconds=d.get("eval_seconds", 0.0),
    )


def create_eval_fn(
    config: OpenKernelConfig,
    file_cache: "FileStateCache | None" = None,
) -> EvalFn:
    """Create an async eval function wrapping the Modal harness.

    Returns an ``async (kernel_source, reference_source) -> EvalResult``
    callable that captures *config* in a closure.  Connection / timeout
    errors are caught inside ``harness.evaluate`` already, but we add a
    defensive outer layer so the inner loop never sees an unhandled
    exception from the eval side.

    If *file_cache* is provided, identical kernel+reference pairs return
    a cached result instead of making a Modal GPU call.
    """
    from openkernel.eval.harness import evaluate

    async def _eval_fn(kernel_source: str, reference_source: str) -> EvalResult:
        # --- Cache lookup (saves GPU cost) ---
        if file_cache is not None:
            cached = file_cache.get_cached_eval(kernel_source, reference_source)
            if cached is not None:
                logger.info("Eval cache hit -- skipping Modal call")
                return _dict_to_eval_result(cached)

        try:
            # harness.evaluate is synchronous (blocks on Modal RPC), so
            # run it in a thread to keep the event loop responsive.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, evaluate, kernel_source, reference_source, config
            )

            # --- Cache store ---
            if file_cache is not None and result.status == EvalStatus.CORRECT:
                file_cache.cache_eval(
                    kernel_source,
                    reference_source,
                    _eval_result_to_dict(result),
                )

            return result
        except Exception as exc:
            logger.error("Eval bridge error: %s", exc, exc_info=True)
            return EvalResult(
                status=EvalStatus.ERROR,
                correct=False,
                error=f"Eval bridge error: {type(exc).__name__}: {exc}",
            )

    return _eval_fn


def create_local_eval_fn(
    config: OpenKernelConfig,
    file_cache: "FileStateCache | None" = None,
) -> EvalFn:
    """Create an async eval function wrapping the local GPU harness.

    Same contract as :func:`create_eval_fn` but calls
    ``harness.evaluate_local`` instead of the Modal remote path.

    If *file_cache* is provided, identical kernel+reference pairs return
    a cached result instead of re-running the local evaluation.
    """
    from openkernel.eval.harness import evaluate_local

    async def _eval_fn(kernel_source: str, reference_source: str) -> EvalResult:
        # --- Cache lookup ---
        if file_cache is not None:
            cached = file_cache.get_cached_eval(kernel_source, reference_source)
            if cached is not None:
                logger.info("Local eval cache hit -- skipping GPU eval")
                return _dict_to_eval_result(cached)

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, evaluate_local, kernel_source, reference_source, config
            )

            # --- Cache store ---
            if file_cache is not None and result.status == EvalStatus.CORRECT:
                file_cache.cache_eval(
                    kernel_source,
                    reference_source,
                    _eval_result_to_dict(result),
                )

            return result
        except Exception as exc:
            logger.error("Local eval bridge error: %s", exc, exc_info=True)
            return EvalResult(
                status=EvalStatus.ERROR,
                correct=False,
                error=f"Local eval bridge error: {type(exc).__name__}: {exc}",
            )

    return _eval_fn
