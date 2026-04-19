"""Triton backend — prompt assembly and validation for Triton kernels."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from openkernel.backends.base import BackendBase

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"


class TritonBackend(BackendBase):
    """Backend for generating and validating Triton kernels."""

    def __init__(self) -> None:
        self._template = self._load_template()

    def get_generator_prompt(
        self,
        reference: str,
        hardware: str,
        intent: str,
        critic_feedback: str | None = None,
        skills: str | None = None,
    ) -> str:
        # Use _safe_format to avoid KeyError on extra placeholders in the
        # prompt template (e.g. the refinement section's {speedup}, {bottleneck_type}).
        prompt = _safe_format(
            self._template,
            reference_code=reference,
            hardware=hardware,
            intent=intent,
            critic_feedback=critic_feedback or "None (first attempt)",
            skills=skills or "None available",
        )
        return prompt

    def validate_kernel(self, code: str) -> tuple[bool, str]:
        # Check for required class definition
        if "class ModelNew" not in code:
            return False, "Generated code does not contain 'class ModelNew'"

        # Check nn.Module inheritance
        if "nn.Module" not in code and "torch.nn.Module" not in code:
            return False, "ModelNew must inherit from nn.Module (class ModelNew(nn.Module))"

        # Check super().__init__()
        if "super()" not in code and "super(ModelNew" not in code:
            return False, "ModelNew.__init__ must call super().__init__()"

        # Check for forward method
        if "def forward" not in code:
            return False, "ModelNew must define a forward() method"

        # Check Python syntax
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, f"Python syntax error: {exc}"

        return True, "Validation passed"

    def get_file_extension(self) -> str:
        return ".py"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _load_template() -> str:
        """Load the Triton generator prompt template from disk."""
        path = _PROMPTS_DIR / "triton_generator_v1.md"
        if not path.exists():
            logger.warning("Triton prompt template not found at %s; using inline fallback", path)
            return _FALLBACK_TEMPLATE
        return path.read_text()


# Minimal fallback in case the prompt file is missing (e.g. during testing)
def _safe_format(template: str, **kwargs: str) -> str:
    """Format a template string, leaving unknown {placeholders} intact.

    This avoids KeyError when prompt templates contain documentation
    placeholders (e.g. the refinement section) that aren't meant to be
    filled by the backend.
    """
    import re

    def _replacer(match: re.Match) -> str:
        key = match.group(1)
        if key in kwargs:
            return kwargs[key]
        return match.group(0)  # leave the placeholder as-is

    return re.sub(r"\{(\w+)\}", _replacer, template)


_FALLBACK_TEMPLATE = """\
You are an expert GPU kernel engineer specializing in Triton.

Generate an optimized Triton kernel that is functionally equivalent to the
given PyTorch reference implementation but faster.

Reference code:
{reference_code}

Target hardware: {hardware}
Optimization intent: {intent}
Critic feedback: {critic_feedback}
Relevant skills: {skills}

CRITICAL REQUIREMENTS:
- Define `ModelNew(torch.nn.Module)` that inherits from torch.nn.Module
- __init__ MUST call super(ModelNew, self).__init__()
- forward() MUST have the same signature as Model.forward() in the reference
- Output must be a complete self-contained Python file with all imports
- The eval harness calls ModelNew().to(device) — this MUST work

Return the complete Python file inside a ```python code block.
"""
