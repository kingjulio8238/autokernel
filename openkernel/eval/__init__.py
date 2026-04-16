"""Eval package — bridge between InnerLoop's EvalFn and the Modal harness.

Public API:
    create_eval_fn(config)       — async EvalFn wrapping Modal harness.evaluate()
    create_local_eval_fn(config) — async EvalFn wrapping harness.evaluate_local()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from openkernel.config import OpenKernelConfig
from openkernel.eval.types import EvalResult, EvalStatus

# Re-export the EvalFn type for convenience.
EvalFn = Callable[[str, str], Awaitable[EvalResult]]

logger = logging.getLogger(__name__)


def create_eval_fn(config: OpenKernelConfig) -> EvalFn:
    """Create an async eval function wrapping the Modal harness.

    Returns an ``async (kernel_source, reference_source) -> EvalResult``
    callable that captures *config* in a closure.  Connection / timeout
    errors are caught inside ``harness.evaluate`` already, but we add a
    defensive outer layer so the inner loop never sees an unhandled
    exception from the eval side.
    """
    from openkernel.eval.harness import evaluate

    async def _eval_fn(kernel_source: str, reference_source: str) -> EvalResult:
        try:
            # harness.evaluate is synchronous (blocks on Modal RPC), so
            # run it in a thread to keep the event loop responsive.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, evaluate, kernel_source, reference_source, config
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


def create_local_eval_fn(config: OpenKernelConfig) -> EvalFn:
    """Create an async eval function wrapping the local GPU harness.

    Same contract as :func:`create_eval_fn` but calls
    ``harness.evaluate_local`` instead of the Modal remote path.
    """
    from openkernel.eval.harness import evaluate_local

    async def _eval_fn(kernel_source: str, reference_source: str) -> EvalResult:
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, evaluate_local, kernel_source, reference_source, config
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
