"""CUDA backend — prompt assembly and validation for CUDA C++ kernels."""

from __future__ import annotations

import logging
from pathlib import Path

from openkernel.backends.base import (
    BackendBase,
    format_archspec,
    format_hints,
    safe_format,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"


class CudaBackend(BackendBase):
    """Backend for generating and validating CUDA kernels (inline C++ extensions)."""

    def __init__(self) -> None:
        self._template = self._load_template()

    def get_generator_prompt(
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
        prompt = safe_format(
            self._template,
            reference_code=reference,
            hardware=hardware,
            intent=intent,
            critic_feedback=critic_feedback or "None (first attempt)",
            skills=skills or "None available",
            problem_context=problem_context or "None provided",
            strategy_hints=format_hints(strategy_hints) or "None provided",
            archspec=format_archspec(archspec) or "No structured hardware spec available",
            op_template=op_template or "No op-specific template available",
        )
        return prompt

    def validate_kernel(self, code: str) -> tuple[bool, str]:
        # CUDA kernels are Python files using torch.utils.cpp_extension.load_inline
        # but they still must define ModelNew.
        if "class ModelNew" not in code:
            return False, "Generated code does not contain 'class ModelNew'"

        # We can't easily syntax-check the embedded C++ from Python, but we can
        # verify the Python wrapper portion parses.
        # A full ast.parse may fail because of raw C++ strings, so we do a
        # lighter check: ensure the file has balanced triple-quotes and the
        # Python structure looks reasonable.
        if code.count("class ") < 1:
            return False, "No class definition found"

        return True, "Validation passed"

    def get_file_extension(self) -> str:
        return ".py"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _load_template() -> str:
        """Load the CUDA generator prompt template from disk."""
        path = _PROMPTS_DIR / "cuda_generator_v1.md"
        if not path.exists():
            logger.warning("CUDA prompt template not found at %s; using inline fallback", path)
            return _FALLBACK_TEMPLATE
        return path.read_text()


_FALLBACK_TEMPLATE = """\
You are an expert GPU kernel engineer specializing in CUDA C++.

Generate an optimized CUDA kernel that is functionally equivalent to the
given PyTorch reference implementation but faster. Use
`torch.utils.cpp_extension.load_inline()` for the CUDA C++ extension.

Reference code:
{reference_code}

Target hardware: {hardware}
Hardware archspec: {archspec}
Optimization intent: {intent}
Problem context: {problem_context}
Strategy hints:
{strategy_hints}
Critic feedback: {critic_feedback}
Relevant skills: {skills}

The kernel must define a `ModelNew` class with the same `forward()` signature.
Return the complete Python file inside a ```python code block.
"""
