"""Generator agent — produces optimized kernel code from context.

The Generator assembles a prompt via the backend, calls the LLM, extracts the
kernel code from the response, and validates it before returning.
"""

from __future__ import annotations

import logging

from openkernel.backends.base import BackendBase
from openkernel.llm.provider import LLMProvider
from openkernel.llm.structured import extract_kernel_code

logger = logging.getLogger(__name__)


class Generator:
    """Generates optimized kernel code using an LLM and a backend-specific prompt."""

    def __init__(self, llm: LLMProvider, backend: BackendBase) -> None:
        self._llm = llm
        self._backend = backend

    async def generate(
        self,
        reference: str,
        hardware: str,
        intent: str,
        critic_feedback: str | None = None,
        skills: str | None = None,
        problem_context: str | None = None,
        strategy_hints: list[str] | None = None,
        archspec: dict | None = None,
        op_template: str | None = None,
    ) -> str:
        """Generate a kernel implementation.

        Parameters
        ----------
        reference : str
            PyTorch reference implementation source code.
        hardware : str
            Target GPU description.
        intent : str
            Optimization intent (what strategy to pursue).
        critic_feedback : str, optional
            Formatted critic diagnosis from a prior iteration.
        skills : str, optional
            Relevant optimization skills from the skill library.
        problem_context : str, optional
            Formatted classifier output (tier, op type, bottleneck).
        strategy_hints : list[str], optional
            Actionable hints derived from the classifier.
        archspec : dict, optional
            Structured hardware archspec to inject into the prompt.
        op_template : str, optional
            Canonical op-type skeleton body picked by the classifier.

        Returns
        -------
        str
            The generated kernel source code.

        Raises
        ------
        ValueError
            If the generated code fails backend validation after extraction.
        """
        # 1. Assemble prompt
        prompt = self._backend.get_generator_prompt(
            reference=reference,
            hardware=hardware,
            intent=intent,
            critic_feedback=critic_feedback,
            skills=skills,
            problem_context=problem_context,
            strategy_hints=strategy_hints,
            archspec=archspec,
            op_template=op_template,
        )

        # 2. Call LLM
        logger.info("Generator: calling LLM (intent=%r)", intent[:80])
        response = await self._llm.generate(prompt)

        # 3. Extract code from response
        code = extract_kernel_code(response)

        # 4. Validate
        valid, message = self._backend.validate_kernel(code)
        if not valid:
            logger.warning("Generator: validation failed — %s", message)
            raise ValueError(f"Generated kernel failed validation: {message}")

        logger.info(
            "Generator: produced %d-line kernel (tokens so far: %d)",
            code.count("\n") + 1,
            self._llm.tokens_used,
        )
        return code
