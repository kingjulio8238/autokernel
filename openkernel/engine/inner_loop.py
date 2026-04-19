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
    last_error: str = ""  # last error/feedback when all attempts fail
    all_speedups: list[float] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0


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
        on_phase: Callable[[str], None] | None = None,
    ) -> None:
        self._generator = generator
        self._critic = critic
        self._config = config
        self._on_phase = on_phase  # callback: (status_message) -> None
        # Load hardware/backend context for injection into generator prompts
        self._skills_context = self._load_context(config)

    @staticmethod
    def _load_context(config: OpenKernelConfig) -> str:
        """Load hardware specs, backend reference, and pitfalls as skills context."""
        try:
            from kernel_code.kernel_config import (
                load_hardware_context,
                load_backend_context,
                load_pitfalls,
            )
            parts: list[str] = []
            hw = config.modal.gpu_type.value if config.modal else "L40S"
            be = config.backend.value if config.backend else "triton"
            hw_ctx = load_hardware_context(hw)
            if hw_ctx:
                parts.append(hw_ctx)
            be_ctx = load_backend_context(be)
            if be_ctx:
                parts.append(be_ctx)
            pit = load_pitfalls()
            if pit:
                parts.append(pit)
            return "\n\n".join(parts) if parts else ""
        except Exception:
            return ""

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

        # Snapshot LLM usage before this refinement cycle so we can compute
        # the delta afterwards.  Generator and critic may share the same
        # LLMProvider instance, so we de-duplicate.
        _llm_instances: set[int] = set()
        _llm_list = []
        for llm in (self._generator._llm, self._critic._llm):
            if id(llm) not in _llm_instances:
                _llm_instances.add(id(llm))
                _llm_list.append(llm)
        tokens_before = sum(llm.tokens_used for llm in _llm_list)
        cost_before = sum(llm.cost_usd for llm in _llm_list)

        logger.info(
            "InnerLoop: starting refinement for intent %r (max %d attempts, current_best=%.2f)",
            intent.description[:60],
            max_attempts,
            current_best,
        )

        def _phase(msg: str) -> None:
            if self._on_phase is not None:
                self._on_phase(msg)

        for attempt in range(1, max_attempts + 1):
            logger.info("InnerLoop: attempt %d/%d", attempt, max_attempts)

            # --- Generate ---------------------------------------------------
            _phase(f"Writing kernel: {intent.description[:50]} (attempt {attempt}/{max_attempts})")
            try:
                kernel_code = await self._generator.generate(
                    reference=reference,
                    hardware=hardware,
                    intent=intent.description,
                    critic_feedback=critic_feedback,
                    skills=self._skills_context,
                )
            except ValueError as exc:
                # Validation failure from generator — treat as a soft error,
                # feed the error message back and retry.
                logger.warning("InnerLoop: generation failed — %s", exc)
                critic_feedback = f"Previous generation failed validation: {exc}"
                all_speedups.append(0.0)
                continue

            # --- Evaluate ----------------------------------------------------
            _phase(f"Evaluating on {hardware} (correctness + benchmark)")
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
            _phase(f"Analyzing performance ({eval_result.speedup:.2f}x) — diagnosing bottleneck")
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

        # Compute LLM token/cost delta for this refinement cycle.
        tokens_after = sum(llm.tokens_used for llm in _llm_list)
        cost_after = sum(llm.cost_usd for llm in _llm_list)

        result = RefinementResult(
            status=status,
            best_kernel=best_kernel,
            best_speedup=best_speedup,
            iterations_used=len(all_speedups),
            final_diagnosis=last_diagnosis,
            last_error=critic_feedback or "",
            all_speedups=all_speedups,
            total_tokens=tokens_after - tokens_before,
            total_cost_usd=cost_after - cost_before,
        )

        logger.info(
            "InnerLoop: finished — %s, best_speedup=%.2fx, iterations=%d",
            status.value,
            best_speedup,
            result.iterations_used,
        )
        return result
