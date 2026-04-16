"""Profiling orchestrator — dispatches to the correct backend-specific profiler.

Routes profiling requests based on kernel backend:
  - Triton kernels -> Proton profiler
  - CUDA kernels  -> torch.profiler
  - All kernels   -> analytical roofline (no GPU required)

Public API:
    profile(kernel_source, reference_source, backend, config) -> ProfileData
"""

from __future__ import annotations

import logging

from openkernel.config import Backend, OpenKernelConfig
from openkernel.eval.types import BottleneckType, ProfileData

logger = logging.getLogger(__name__)


def profile(
    kernel_source: str,
    reference_source: str,
    backend: Backend,
    config: OpenKernelConfig | None = None,
) -> ProfileData:
    """Profile a kernel and return structured hardware metrics.

    Dispatches to the appropriate backend-specific profiler. If the primary
    profiler fails, falls back to analytical roofline estimation.

    Args:
        kernel_source: Python source code defining ModelNew.
        reference_source: Python source code defining Model + get_inputs.
        backend: Which backend the kernel targets (Triton or CUDA).
        config: OpenKernelConfig controlling profiling behavior.

    Returns:
        ProfileData with bottleneck classification and utilization metrics.
    """
    if config is None:
        config = OpenKernelConfig()

    # Step 1: Analytical pre-screen (always runs, no GPU required)
    analytical_profile = ProfileData()
    if config.analytical_prescreen:
        try:
            from openkernel.eval.profilers.roofline import estimate_roofline

            analytical_profile = estimate_roofline(
                reference_source=reference_source,
                hardware=config.modal.gpu_type.value,
            )
            logger.info(
                "Roofline pre-screen: bottleneck=%s, roofline_position=%.2f",
                analytical_profile.bottleneck_type.value,
                analytical_profile.roofline_position,
            )
        except Exception as exc:
            logger.warning("Analytical roofline failed: %s", exc)

    # Step 2: Backend-specific profiler (requires GPU — runs on Modal)
    gpu_profile = _dispatch_profiler(kernel_source, reference_source, backend, config)

    # Step 3: Merge — GPU profiler results take precedence, analytical fills gaps
    merged = _merge_profiles(gpu_profile, analytical_profile)

    return merged


def _dispatch_profiler(
    kernel_source: str,
    reference_source: str,
    backend: Backend,
    config: OpenKernelConfig,
) -> ProfileData:
    """Dispatch to the correct backend-specific profiler."""
    if backend == Backend.TRITON:
        try:
            from openkernel.eval.profilers.proton_profiler import profile_triton

            result = profile_triton(
                kernel_source=kernel_source,
                reference_source=reference_source,
            )
            logger.info("Proton profiler completed successfully.")
            return result
        except ImportError:
            logger.warning("Proton profiler not available, falling back to torch.profiler")
        except Exception as exc:
            logger.warning("Proton profiler failed: %s, falling back to torch.profiler", exc)

        # Fallback: use torch.profiler even for Triton kernels
        try:
            from openkernel.eval.profilers.torch_profiler import profile_cuda

            return profile_cuda(
                kernel_source=kernel_source,
                reference_source=reference_source,
            )
        except Exception as exc:
            logger.warning("torch.profiler fallback also failed: %s", exc)
            return ProfileData()

    elif backend == Backend.CUDA:
        try:
            from openkernel.eval.profilers.torch_profiler import profile_cuda

            result = profile_cuda(
                kernel_source=kernel_source,
                reference_source=reference_source,
            )
            logger.info("torch.profiler completed successfully.")
            return result
        except Exception as exc:
            logger.warning("torch.profiler failed: %s", exc)
            return ProfileData()

    else:
        logger.warning("Unknown backend %s, skipping GPU profiling.", backend)
        return ProfileData()


def _merge_profiles(gpu: ProfileData, analytical: ProfileData) -> ProfileData:
    """Merge GPU profiler and analytical roofline results.

    GPU profiler data takes precedence when available. Analytical fills
    in any missing fields (zeros).
    """
    return ProfileData(
        bottleneck_type=(
            gpu.bottleneck_type
            if gpu.bottleneck_type != BottleneckType.UNKNOWN
            else analytical.bottleneck_type
        ),
        roofline_position=(
            gpu.roofline_position if gpu.roofline_position > 0.0
            else analytical.roofline_position
        ),
        cache_efficiency=(
            gpu.cache_efficiency if gpu.cache_efficiency > 0.0
            else analytical.cache_efficiency
        ),
        occupancy=(
            gpu.occupancy if gpu.occupancy > 0.0
            else analytical.occupancy
        ),
        bandwidth_utilization=(
            gpu.bandwidth_utilization if gpu.bandwidth_utilization > 0.0
            else analytical.bandwidth_utilization
        ),
        compute_utilization=(
            gpu.compute_utilization if gpu.compute_utilization > 0.0
            else analytical.compute_utilization
        ),
        top_stalls=gpu.top_stalls if gpu.top_stalls else analytical.top_stalls,
        raw_metrics={**analytical.raw_metrics, **gpu.raw_metrics},
    )
