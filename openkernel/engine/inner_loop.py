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

import ast
import logging
import re
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


def _extract_kb_init_hint(reference: str) -> str:
    """Extract the KernelBench ``Model.__init__`` signature as a prompt hint.

    KB references define ``class Model(nn.Module)`` and a companion
    ``get_init_inputs()`` function; the harness calls
    ``ModelNew(*get_init_inputs())`` on the generated kernel, so ``ModelNew``
    must accept the SAME positional args. LLMs routinely forget this and
    emit ``def __init__(self):`` which immediately fails with
    "takes 1 positional argument but N were given".

    Returns a single-line hint like::

        Init-signature hint (KB Model): ModelNew.__init__(self, in_channels, out_channels, kernel_size, bias_shape) — mirror Model.__init__ exactly.

    Or empty string for non-KB references (e.g. GPU MODE ref_kernel),
    unparseable refs, or KB refs without get_init_inputs(). Never raises —
    callers can trust the return value to be a string.
    """
    if not reference or "class Model" not in reference:
        return ""
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return ""

    model_init_args: list[str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for body_node in node.body:
                if isinstance(body_node, ast.FunctionDef) and body_node.name == "__init__":
                    args = body_node.args
                    names = [a.arg for a in args.args]
                    # Drop "self" — ModelNew's __init__ needs the rest.
                    if names and names[0] == "self":
                        names = names[1:]
                    model_init_args = names
                    break
            break

    if not model_init_args:
        return ""

    sig = ", ".join(["self", *model_init_args])
    return (
        f"Init-signature hint (KernelBench Model): "
        f"ModelNew.__init__({sig}) must mirror Model.__init__ exactly — "
        f"the harness calls ModelNew(*get_init_inputs()). "
        f"A no-arg `def __init__(self):` crashes with "
        f"\"takes 1 positional argument but {len(model_init_args)} were given\"."
    )


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
        # Per-run retrieval cache keyed by reference source. Classification is
        # deterministic for a given reference, so one call per distinct
        # reference is sufficient across all rounds in a run.
        self._retrieval_cache: dict[str, dict] = {}
        # Lazy-loaded SkillLibrary — only instantiated when first needed.
        self._skill_library = None

    def _get_retrieval(self, reference: str, hardware: str, intent: str) -> dict:
        """Return classifier+skills+archspec bundle, cached per reference.

        The bundle shape::

            {
                "problem_context": str | None,
                "strategy_hints": list[str] | None,
                "skills_extra": str | None,   # appended to static skills_context
                "archspec": dict | None,
            }

        All retrieval failures degrade to ``None``/empty — generator
        invocation must never crash because a retrieval step failed.
        """
        cached = self._retrieval_cache.get(reference)
        if cached is not None:
            return cached

        bundle: dict = {
            "problem_context": None,
            "strategy_hints": None,
            "skills_extra": None,
            "archspec": None,
            "op_template": "",
        }

        # --- Classifier -----------------------------------------------------
        op_tag = ""
        op_type_value: str | None = None
        try:
            from kernel_code.problem_classifier import classify_problem  # lazy

            classif = classify_problem(reference)
            bundle["problem_context"] = classif.to_context_string()
            bundle["strategy_hints"] = list(classif.strategy_hints) or None
            op_tag = classif.op_type.value
            op_type = getattr(classif, "op_type", None)
            op_type_value = getattr(op_type, "value", op_type) if op_type else None
        except Exception as exc:
            logger.warning("InnerLoop: classify_problem failed — %s", exc)

        # --- KB init-signature hint ----------------------------------------
        # Parameterized KB models (Conv2D, Linear, GroupNorm, …) repeatedly
        # failed with "ModelNew.__init__() takes 1 positional argument but N
        # were given" because the LLM emitted a no-arg ``def __init__(self):``.
        # The reference already encodes the required signature via
        # ``get_init_inputs()``; splice it into problem_context as an
        # unambiguous signature hint. Applies to ALL KB problems — not
        # problem-specific.
        try:
            init_hint = _extract_kb_init_hint(reference)
            if init_hint:
                pc = bundle["problem_context"] or ""
                bundle["problem_context"] = (pc + "\n" + init_hint).strip()
        except Exception as exc:
            logger.warning("InnerLoop: init-hint extraction failed — %s", exc)

        # --- Op-type template ----------------------------------------------
        try:
            from openkernel.backends.base import load_op_template  # lazy

            bundle["op_template"] = load_op_template(op_type_value)
        except Exception as exc:
            logger.warning("InnerLoop: load_op_template failed — %s", exc)

        # --- Skill library -------------------------------------------------
        try:
            if self._skill_library is None:
                from openkernel.memory.skill_library import SkillLibrary  # lazy

                lib = SkillLibrary()
                lib.load()
                self._skill_library = lib
            query = f"{op_tag} {intent}".strip() or intent
            matches = self._skill_library.search_skills(query, top_k=3)
            if matches:
                bundle["skills_extra"] = type(self._skill_library).to_context_string(matches)
        except Exception as exc:
            logger.warning("InnerLoop: skill search failed — %s", exc)

        # --- Archspec -------------------------------------------------------
        try:
            from openkernel.backends.base import _hardware_archspec  # lazy

            bundle["archspec"] = _hardware_archspec(hardware)
        except Exception as exc:
            logger.warning("InnerLoop: archspec lookup failed — %s", exc)

        self._retrieval_cache[reference] = bundle
        return bundle

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

        # Retrieval happens ONCE per (reference, intent) and is cached on self
        # across rounds to avoid redundant LLM-irrelevant work.
        retrieval = self._get_retrieval(reference, hardware, intent.description)
        # Merge static hardware/backend/pitfalls context with any dynamic
        # skills found for this problem. Static context has been the default
        # since before skill retrieval; keep it unless we actively replace it.
        skills_combined = self._skills_context
        if retrieval["skills_extra"]:
            skills_combined = (
                f"{self._skills_context}\n\n{retrieval['skills_extra']}"
                if self._skills_context
                else retrieval["skills_extra"]
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
                    skills=skills_combined,
                    problem_context=retrieval["problem_context"],
                    strategy_hints=retrieval["strategy_hints"],
                    archspec=retrieval["archspec"],
                    op_template=retrieval.get("op_template") or None,
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
