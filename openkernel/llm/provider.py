"""BYOM LLM provider interface via litellm.

Supports any model accessible through litellm (Anthropic, OpenAI, Google, local, etc.).
Tracks cumulative token usage and estimated cost across all calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm

from openkernel.config import ModelConfig

logger = logging.getLogger(__name__)

# Estimated cost per million tokens (input/output) for common models.
# litellm has its own cost tracking, but we keep a fallback table.
_COST_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # Default — MiniMax M2.5 (via OpenAI-compatible API)
    "openai/MiniMax-M2.5": (0.30, 1.20),
    "openai/MiniMax-M2.7": (0.30, 1.20),
    "MiniMax-M2.5": (0.30, 1.20),
    "MiniMax-M2.7": (0.30, 1.20),
    # Groq (free tier / very cheap)
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "groq/llama-3.1-8b-instant": (0.05, 0.08),
    "groq/qwen-qwq-32b": (0.20, 0.20),
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34),
    # Open-source premium
    "z-ai/glm-5.1": (1.40, 4.40),
    "moonshotai/kimi-k2.5": (0.50, 2.80),
    "qwen3.5-397b": (0.20, 0.80),
    # Frontier
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "gpt-4o": (2.5, 10.0),
    "o3": (10.0, 40.0),
}

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


class LLMProvider:
    """Unified async LLM interface with retry logic, token tracking, and cost estimation."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost: float = 0.0
        self._max_retries: int = 3
        self._retry_base_delay: float = 1.0  # seconds

        # Resolve the API key: explicit config > provider-specific env var > None.
        # When using openai/ prefix with a custom api_base (e.g., MiniMax),
        # litellm looks for OPENAI_API_KEY, but the actual key is in
        # MINIMAX_API_KEY. We resolve it here and pass it explicitly.
        import os
        _PROVIDER_ENV_VARS = {
            "minimax": "MINIMAX_API_KEY",
            "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        if config.api_key:
            self._api_key = config.api_key
        else:
            env_var = _PROVIDER_ENV_VARS.get(config.provider, "")
            self._api_key = os.environ.get(env_var) if env_var else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, prompt: str) -> str:
        """Generate a completion from a plain text prompt.

        Retries up to 3 times with exponential backoff on transient errors.
        """
        return await self._call_with_retries(prompt)

    async def generate_structured(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate a completion, optionally requesting structured JSON output.

        If *response_format* is provided it is passed through to the model
        (e.g. ``{"type": "json_object"}``).  The raw text response is
        returned — callers are responsible for parsing.
        """
        return await self._call_with_retries(prompt, response_format=response_format)

    @property
    def tokens_used(self) -> int:
        """Total tokens consumed (prompt + completion) across all calls."""
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def cost_usd(self) -> float:
        """Estimated cumulative USD cost across all calls."""
        return self._total_cost

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _call_with_retries(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._single_call(prompt, response_format)
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = self._retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt,
                        self._max_retries,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "LLM call failed after %d attempts: %s",
                        self._max_retries,
                        exc,
                    )

        raise RuntimeError(
            f"LLM call failed after {self._max_retries} retries"
        ) from last_error

    async def _single_call(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }

        if self._api_key:
            kwargs["api_key"] = self._api_key

        if self._config.api_base:
            kwargs["api_base"] = self._config.api_base

        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await litellm.acompletion(**kwargs)

        # --- Track usage ---------------------------------------------------
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        # Estimate cost
        self._total_cost += self._estimate_cost(prompt_tokens, completion_tokens)

        # --- Extract text --------------------------------------------------
        text: str = response.choices[0].message.content or ""
        return text

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate USD cost for a single call."""
        model = self._config.model_id
        if model in _COST_PER_M_TOKENS:
            input_rate, output_rate = _COST_PER_M_TOKENS[model]
        else:
            # Fallback: assume a mid-range model
            input_rate, output_rate = 3.0, 15.0

        cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
        return cost
