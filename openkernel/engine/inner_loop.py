"""Inner refinement loop — generate, evaluate, diagnose, retry.

This is the Caesar-style inner loop from the design doc. For a single
optimization intent it:

1. Generator produces kernel code
2. Eval function scores it (correctness + speedup)
3. If error -> feed error back to Generator, retry
4. If correct and better -> keep as best
5. Critic diagnoses from profile data
6. Generator produces improved version
7. Repeat up to K attempts
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

from openkernel.agents.critic import Critic
from openkernel.agents.generator import Generator
from openkernel.config import OpenKernelConfig
from openkernel.engine.world_model import IntentNode
from openkernel.eval.types import (
    CriticDiagnosis,
    EvalResult,
    EvalStatus,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------


class RefinementStatus(str, Enum):
    IMPROVED = "improved"
    STAGNATED = "stagnated"
    FAILED = "failed"


@dataclass
class InnerRefinementResult:
    """Outcome of running the inner loop for a single intent.

    This is the inner loop's native result type.  The orchestrator consumes a
    different :class:`~openkernel.engine.orchestrator.RefinementResult`;
    :class:`InnerLoopAdapter` in ``factory.py`` handles the translation.
    """

    status: RefinementStatus
    best_kernel: str = ""
    best_speedup: float = 0.0
    iterations_used: int = 0
    final_diagnosis: CriticDiagnosis | None = None
    all_speedups: list[float] = field(default_factory=list)


# Backward-compat alias — existing code that imported ``RefinementResult``
# from this module keeps working.
RefinementResult = InnerRefinementResult


# ------------------------------------------------------------------
# Eval function type
# ------------------------------------------------------------------

EvalFn = Callable[[str, str], Awaitable[EvalResult]]
"""Signature: (kernel_code, reference_code) -> EvalResult"""


# ------------------------------------------------------------------
# Inner Loop
# ------------------------------------------------------------------


class InnerLoop:
    """Refinement loop: generate -> eval -> critic -> generate improved."""

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        config: OpenKernelConfig,
    ) -> None:
        self._generator = generator
        self._critic = critic
        self._config = config

    async def refine(
        self,
        intent: IntentNode,
        reference: str,
        eval_fn: EvalFn,
        current_best: float = 0.0,
    ) -> InnerRefinementResult:
        """Run the inner refinement loop for a single optimization intent.

        Parameters
        ----------
        intent : IntentNode
            The optimization intent to pursue.
        reference : str
            PyTorch reference implementation source code.
        eval_fn : EvalFn
            Async callable ``(kernel_code, reference) -> EvalResult``.
            Decoupled from Modal so the loop is testable with mocks.
        current_best : float
            The best speedup achieved so far (across all intents).
            Used to determine if this attempt improved things.

        Returns
        -------
        RefinementResult
            Summary of the refinement attempt.
        """
        max_attempts = self._config.max_retries_per_intent
        hardware = self._config.modal.gpu_type.value
        backend = self._config.backend.value

        best_kernel = ""
        best_speedup = current_best
        critic_feedback: str | None = None
        last_diagnosis: CriticDiagnosis | None = None
        all_speedups: list[float] = []

        logger.info(
            "InnerLoop: starting refinement for intent %r (max %d attempts, current_best=%.2f)",
            intent.description[:60],
            max_attempts,
            current_best,
        )

        for attempt in range(1, max_attempts + 1):
            logger.info("InnerLoop: attempt %d/%d", attempt, max_attempts)

            # --- Generate ---------------------------------------------------
            try:
                kernel_code = await self._generator.generate(
                    reference=reference,
                    hardware=hardware,
                    intent=intent.description,
                    critic_feedback=critic_feedback,
                )
            except ValueError as exc:
                # Validation failure from generator — treat as a soft error,
                # feed the error message back and retry.
                logger.warning("InnerLoop: generation failed — %s", exc)
                critic_feedback = f"Previous generation failed validation: {exc}"
                all_speedups.append(0.0)
                continue

            # --- Evaluate ----------------------------------------------------
            eval_result = await eval_fn(kernel_code, reference)
            all_speedups.append(eval_result.speedup)

            if eval_result.status in (EvalStatus.COMPILE_ERROR, EvalStatus.ERROR):
                # Feed error back to generator for retry
                error_msg = eval_result.error or "Unknown error"
                critic_feedback = (
                    f"Previous kernel had {eval_result.status.value}: {error_msg}\n"
                    "Fix the error and regenerate."
                )
                logger.info("InnerLoop: eval error — %s", error_msg[:120])
                continue

            if eval_result.status == EvalStatus.INCORRECT:
                critic_feedback = (
                    "Previous kernel was INCORRECT — output did not match reference.\n"
                    f"Error details: {eval_result.error or 'torch.allclose failed'}\n"
                    "Ensure numerical correctness. Accumulate in higher precision if needed."
                )
                logger.info("InnerLoop: incorrect result")
                continue

            # --- Correct kernel —  check if it's better --------------------
            if eval_result.correct and eval_result.speedup > best_speedup:
                best_speedup = eval_result.speedup
                best_kernel = kernel_code
                logger.info(
                    "InnerLoop: new best! speedup=%.2fx (was %.2fx)",
                    eval_result.speedup,
                    current_best,
                )

            # --- Critic diagnosis -------------------------------------------
            try:
                diagnosis = await self._critic.analyze(
                    kernel_code=kernel_code,
                    eval_result=eval_result,
                    hardware=hardware,
                    backend=backend,
                )
                last_diagnosis = diagnosis
                critic_feedback = self._critic.format_feedback(
                    diagnosis, eval_result.speedup
                )
            except Exception as exc:
                logger.warning("InnerLoop: critic failed — %s", exc)
                critic_feedback = (
                    f"Previous speedup: {eval_result.speedup:.2f}x. "
                    "Try a different optimization approach."
                )

        # --- Determine outcome ----------------------------------------------
        if best_speedup > current_best and best_kernel:
            status = RefinementStatus.IMPROVED
        elif best_kernel:
            status = RefinementStatus.STAGNATED
        else:
            status = RefinementStatus.FAILED

        result = RefinementResult(
            status=status,
            best_kernel=best_kernel,
            best_speedup=best_speedup,
            iterations_used=len(all_speedups),
            final_diagnosis=last_diagnosis,
            all_speedups=all_speedups,
        )

        logger.info(
            "InnerLoop: finished — %s, best_speedup=%.2fx, iterations=%d",
            status.value,
            best_speedup,
            result.iterations_used,
        )
        return result
