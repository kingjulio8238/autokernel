"""openkernel — Self-recursive GPU kernel optimization engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openkernel.config import Backend, EvalMode, OpenKernelConfig
from openkernel.engine.orchestrator import OptimizationResult
from openkernel.eval.types import EvalResult
from openkernel.exceptions import (
    ConfigurationError,
    EvalError,
    GenerationError,
    KernelBenchError,
    OpenKernelError,
    ProfilingError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__version__ = "0.1.0"

__all__ = [
    "optimize",
    "evaluate",
    "Backend",
    "EvalMode",
    "EvalResult",
    "OpenKernelConfig",
    "OptimizationResult",
    "OpenKernelError",
    "ConfigurationError",
    "EvalError",
    "GenerationError",
    "ProfilingError",
    "KernelBenchError",
    "__version__",
]


async def optimize(
    reference_source: str,
    backend: str = "triton",
    config: OpenKernelConfig | None = None,
    eval_fn: Callable[[str, str], Awaitable[EvalResult]] | None = None,
) -> OptimizationResult:
    """Optimize a GPU kernel. Main entry point.

    Creates an engine via the factory, runs the orchestrator's optimization
    loop, and returns the best kernel found.

    Args:
        reference_source: Python source code of the reference kernel
            (must define ``Model``, ``get_inputs``, ``get_init_inputs``).
        backend: Target backend — ``"triton"`` or ``"cuda"``.
        config: Full configuration. Defaults are used when ``None``.
        eval_fn: Optional async evaluation function
            ``(kernel_source, reference_source) -> EvalResult``.
            If ``None``, the real Modal eval function is used.

    Returns:
        OptimizationResult with the best kernel, speedup, and full
        search-tree history.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from openkernel.engine.factory import create_engine

    if config is None:
        config = OpenKernelConfig()

    # Override backend in config if the caller specified one explicitly.
    if backend != config.backend.value:
        config = config.model_copy(update={"backend": Backend(backend)})

    try:
        config.validate_config()
    except ConfigurationError as exc:
        raise ConfigurationError(
            f"Cannot start optimization — configuration is invalid.\n{exc}"
        ) from exc

    engine = create_engine(config=config, eval_fn=eval_fn)

    # Orchestrator.optimize() is synchronous and internally runs async code
    # via InnerLoopAdapter. Run it in a thread so we don't block the caller's
    # event loop.
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = await loop.run_in_executor(
            pool,
            engine.optimize,
            reference_source,
            config.backend.value,
        )

    return result


async def evaluate(
    kernel_source: str,
    reference_source: str,
    config: OpenKernelConfig | None = None,
) -> EvalResult:
    """Evaluate a kernel against a reference implementation.

    Sends the kernel and reference source to the evaluation harness and
    returns correctness, speedup, and profiling data.

    Args:
        kernel_source: Python source code defining ``ModelNew``.
        reference_source: Python source code defining ``Model``,
            ``get_inputs``, and ``get_init_inputs``.
        config: Evaluation configuration. Defaults are used when ``None``.

    Returns:
        EvalResult with status, correctness, speedup, and profile data.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from openkernel.eval.harness import evaluate as harness_evaluate

    if config is None:
        config = OpenKernelConfig()

    if not config.check_modal():
        raise ConfigurationError(
            "Modal is not configured. Evaluation requires Modal for remote GPU execution.\n"
            "Run `modal setup` or set the MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
            "environment variables."
        )

    # harness.evaluate() is synchronous (blocks on Modal RPC), so run it
    # in a thread to keep the caller's event loop responsive.
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = await loop.run_in_executor(
            pool, harness_evaluate, kernel_source, reference_source, config
        )

    return result
