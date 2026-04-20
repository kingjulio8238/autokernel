"""Abstract backend interface.

Each backend (Triton, CUDA) knows how to:
- Assemble a generator prompt from context variables
- Validate generated kernel code (syntactic sanity checks)
- Declare its file extension
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BackendBase(ABC):
    """Abstract base class for kernel generation backends."""

    @abstractmethod
    def get_generator_prompt(
        self,
        reference: str,
        hardware: str,
        intent: str,
        critic_feedback: str | None = None,
        skills: str | None = None,
    ) -> str:
        """Build the full generator prompt for the LLM.

        Parameters
        ----------
        reference : str
            The PyTorch reference implementation source code.
        hardware : str
            Target GPU description (e.g. "NVIDIA H100 80GB").
        intent : str
            The optimization intent from the world model.
        critic_feedback : str, optional
            Formatted critic diagnosis from a previous iteration.
        skills : str, optional
            Relevant skills retrieved from the skill library.

        Returns
        -------
        str
            The assembled prompt ready to send to the LLM.
        """

    @abstractmethod
    def validate_kernel(self, code: str) -> tuple[bool, str]:
        """Run lightweight validation on generated kernel code.

        Returns
        -------
        tuple[bool, str]
            ``(is_valid, message)`` — if invalid, *message* describes the issue.
        """

    @abstractmethod
    def get_file_extension(self) -> str:
        """Return the file extension for kernels of this backend (e.g. '.py')."""
