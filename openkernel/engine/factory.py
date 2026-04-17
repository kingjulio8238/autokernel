"""Factory for assembling a fully-wired Orchestrator.

This is the single place that stitches together the real LLM provider,
backend, Generator, Critic, and InnerLoop into an Orchestrator ready to run.

Usage::

    from openkernel.config import OpenKernelConfig
    from openkernel.engine.factory import create_engine

    engine = create_engine(OpenKernelConfig())
    result = engine.optimize(reference_code="...")
"""

from __future__ import annotations

import asyncio
import logging

from openkernel.agents.critic import Critic
from openkernel.agents.generator import Generator
from openkernel.backends.cuda_backend import CudaBackend
from openkernel.backends.triton_backend import TritonBackend
from openkernel.config import Backend, OpenKernelConfig
from openkernel.exceptions import ConfigurationError
from openkernel.engine.inner_loop import (
    EvalFn,
    InnerLoop,
    InnerRefinementResult,
    RefinementStatus,
)
from openkernel.engine.orchestrator import (
    InnerLoopInterface,
    Orchestrator,
    RefinementResult,
)
from openkernel.engine.world_model import IntentNode
from openkernel.llm.models import get_default_model
from openkernel.llm.provider import LLMProvider
from openkernel.traces.capture import TraceCapture

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter: bridges InnerLoop (async, EvalFn) -> InnerLoopInterface (sync)
# ---------------------------------------------------------------------------


class InnerLoopAdapter:
    """Adapts the real :class:`InnerLoop` to the :class:`InnerLoopInterface` protocol.

    Key differences handled:

    * **async -> sync**: ``InnerLoop.refine`` is async; the orchestrator calls
      ``refine`` synchronously. The adapter runs the coroutine via
      ``asyncio.get_event_loop().run_until_complete`` (or creates a loop).
    * **Parameter mapping**: The orchestrator passes ``(intent, reference_code,
      backend, config)`` whereas the real inner loop expects
      ``(intent, reference, eval_fn, current_best)``.
    * **Result translation**: ``InnerRefinementResult`` (enum status,
      ``final_diagnosis``, ``all_speedups``) is mapped to the orchestrator's
      ``RefinementResult`` (string status, ``critic_feedback``, ``iterations``).
    """

    def __init__(
        self,
        inner_loop: InnerLoop,
        eval_fn: EvalFn,
        critic: Critic,
    ) -> None:
        self._inner_loop = inner_loop
        self._eval_fn = eval_fn
        self._critic = critic
        self._current_best: float = 0.0

    def refine(
        self,
        intent: IntentNode,
        reference_code: str,
        backend: str,
        config: dict,
    ) -> RefinementResult:
        """Conform to :class:`InnerLoopInterface` by delegating to the real loop."""
        inner_result = self._run_async(
            self._inner_loop.refine(
                intent=intent,
                reference=reference_code,
                eval_fn=self._eval_fn,
                current_best=self._current_best,
            )
        )

        # Keep track of the running best for subsequent calls.
        if inner_result.best_speedup > self._current_best:
            self._current_best = inner_result.best_speedup

        return self._translate(inner_result)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine from a synchronous context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an already-running event loop (e.g. Jupyter).
            # Create a new loop in a thread to avoid "cannot run nested".
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    @staticmethod
    def _translate(inner: InnerRefinementResult) -> RefinementResult:
        """Map an ``InnerRefinementResult`` to the orchestrator's ``RefinementResult``."""
        # Map enum status to the string the orchestrator expects.
        status_map = {
            RefinementStatus.IMPROVED: "succeeded",
            RefinementStatus.STAGNATED: "succeeded",  # has a kernel, just not better
            RefinementStatus.FAILED: "failed",
        }
        status = status_map.get(inner.status, "error")

        # Build a critic_feedback string from the diagnosis if available.
        if inner.final_diagnosis is not None:
            d = inner.final_diagnosis
            critic_feedback = (
                f"Bottleneck: {d.bottleneck_type.value}. "
                f"Issue: {d.specific_issue} "
                f"Recommendation: {d.recommendation} "
                f"Headroom: {d.estimated_headroom:.2f}x, "
                f"Confidence: {d.confidence:.2f}"
            )
        else:
            critic_feedback = f"No diagnosis available (status: {inner.status.value})."

        return RefinementResult(
            status=status,
            best_kernel=inner.best_kernel,
            best_speedup=inner.best_speedup,
            iterations=inner.iterations_used,
            critic_feedback=critic_feedback,
            total_tokens=inner.total_tokens,
            total_cost_usd=inner.total_cost_usd,
        )


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def create_engine(
    config: OpenKernelConfig | None = None,
    eval_fn: EvalFn | None = None,
) -> Orchestrator:
    """Create a fully wired Orchestrator with real Generator, Critic, InnerLoop.

    Parameters
    ----------
    config : OpenKernelConfig, optional
        Top-level configuration.  Defaults to ``OpenKernelConfig()`` with
        all default values.
    eval_fn : EvalFn, optional
        Async evaluation function ``(kernel_code, reference) -> EvalResult``.
        If ``None``, the real Modal eval function is used via
        ``create_eval_fn(config)``.

    Returns
    -------
    Orchestrator
        A ready-to-run orchestrator backed by the real inner loop.
    """
    if config is None:
        config = OpenKernelConfig()

    # Validate configuration before wiring up components.
    try:
        config.validate_config()
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Unexpected error during config validation: {exc}") from exc

    # 1. LLM provider
    model_config = config.model or get_default_model()
    llm = LLMProvider(model_config)

    # 2. Backend
    if config.backend == Backend.CUDA:
        backend = CudaBackend()
    else:
        backend = TritonBackend()

    # 3. Agents
    generator = Generator(llm, backend)
    critic = Critic(llm)

    # 4. Inner loop
    inner_loop = InnerLoop(generator, critic, config)

    # 5. Eval function — default to real Modal eval
    if eval_fn is None:
        from openkernel.eval import create_eval_fn

        eval_fn = create_eval_fn(config)

    # 6. Adapter
    adapter = InnerLoopAdapter(inner_loop, eval_fn, critic)

    # 7. Trace capture (optional)
    trace_capture: TraceCapture | None = None
    if config.capture_traces:
        trace_capture = TraceCapture(config=config)

    # 8. Orchestrator config dict (the Orchestrator still expects a plain dict)
    orch_config = {
        "max_iterations": config.max_iterations,
        "stagnation_threshold": config.stagnation_threshold,
        "max_retries_per_intent": config.max_retries_per_intent,
        "model_id": model_config.model_id,
        "traces_dir": f"{config.traces_dir}/raw",
    }

    # 9. Assemble — pass the real LLM provider for tree operations
    orchestrator = Orchestrator(
        inner_loop=adapter,
        config=orch_config,
        llm=llm,
        trace_capture=trace_capture,
    )

    logger.info(
        "Engine created: model=%s, backend=%s, max_iterations=%d",
        model_config.model_id,
        config.backend.value,
        config.max_iterations,
    )

    return orchestrator
