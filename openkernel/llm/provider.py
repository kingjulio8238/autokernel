"""BYOM LLM provider interface.

Uses the Groq SDK directly for Groq models (no litellm middleman).
Falls back to litellm for all other providers (Anthropic, OpenAI, MiniMax, etc.).
Tracks cumulative token usage and estimated cost across all calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from openkernel.config import ModelConfig

logger = logging.getLogger(__name__)

# Estimated cost per million tokens (input/output) for common models.
_COST_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # Default — MiniMax M2.5 (via OpenAI-compatible API)
    "openai/MiniMax-M2.5": (0.30, 1.20),
    "openai/MiniMax-M2.7": (0.30, 1.20),
    "MiniMax-M2.5": (0.30, 1.20),
    "MiniMax-M2.7": (0.30, 1.20),
    # Groq (free tier / very cheap)
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "groq/llama-3.1-8b-instant": (0.05, 0.08),
    "groq/qwen/qwen3-32b": (0.20, 0.20),
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

# Provider → env var for API key resolution
_PROVIDER_ENV_VARS = {
    "minimax": "MINIMAX_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


class LLMProvider:
    """Unified async LLM interface with retry logic, token tracking, and cost estimation.

    Routes Groq models through the Groq SDK directly.
    All other providers go through litellm.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost: float = 0.0
        self._max_retries: int = 5
        self._retry_base_delay: float = 1.0

        self._is_groq = config.provider == "groq"

        # API key resolution
        if config.api_key:
            self._api_key = config.api_key
        else:
            env_var = _PROVIDER_ENV_VARS.get(config.provider, "")
            self._api_key = os.environ.get(env_var) if env_var else None

        # Groq: SDK client created lazily on first call
        if self._is_groq:
            self._groq_client = None  # created in _ensure_groq_client()
            # Strip "groq/" prefix for the Groq API
            self._groq_model = config.model_id
            if self._groq_model.startswith("groq/"):
                self._groq_model = self._groq_model[5:]
        else:
            # litellm for everything else
            import litellm
            litellm.suppress_debug_info = True
            # API base (only MiniMax needs a custom one)
            _PROVIDER_API_BASE = {
                "minimax": "https://api.minimax.io/v1",
            }
            if config.api_base:
                self._api_base = config.api_base
            else:
                self._api_base = _PROVIDER_API_BASE.get(config.provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, prompt: str) -> str:
        """Generate a completion from a plain text prompt."""
        return await self._call_with_retries(prompt)

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Generate a completion with streaming. Yields token strings as they arrive."""
        if self._is_groq:
            async for token in self._groq_stream(prompt):
                yield token
        else:
            async for token in self._litellm_stream(prompt):
                yield token

    async def generate_structured(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate a completion, optionally requesting structured JSON output."""
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
    # Retry wrapper
    # ------------------------------------------------------------------

    async def _call_with_retries(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                if self._is_groq:
                    return await self._groq_call(prompt, response_format)
                return await self._litellm_call(prompt, response_format)
            except Exception as exc:
                last_error = exc
                exc_str = str(exc).lower()

                # Don't retry on non-transient errors
                if any(k in exc_str for k in (
                    "decommissioned", "invalid_api_key", "authentication",
                    "not found", "bad request", "invalid_request",
                )):
                    logger.error("LLM call failed (non-retryable): %s", exc)
                    break

                if attempt < self._max_retries:
                    if "rate_limit" in exc_str or "429" in exc_str:
                        delay = 20.0
                    else:
                        delay = self._retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, self._max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "LLM call failed after %d attempts: %s",
                        self._max_retries, exc,
                    )

        raise RuntimeError(
            f"LLM call failed after {self._max_retries} retries: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Groq SDK (direct)
    # ------------------------------------------------------------------

    def _ensure_groq_client(self):
        """Lazily create the Groq client on first use."""
        if self._groq_client is None:
            # Re-check env in case key was set after provider init (e.g. /models)
            api_key = self._api_key or os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY not set. Run /models and select a Groq model to configure it."
                )
            from groq import AsyncGroq
            self._groq_client = AsyncGroq(api_key=api_key)
        return self._groq_client

    async def _groq_call(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await self._ensure_groq_client().chat.completions.create(**kwargs)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost += self._estimate_cost(prompt_tokens, completion_tokens)

        return response.choices[0].message.content or ""

    async def _groq_stream(self, prompt: str) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = {
            "model": self._groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": True,
        }

        stream = await self._ensure_groq_client().chat.completions.create(**kwargs)
        full_text = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                yield delta

        estimated_tokens = len(full_text) // 4
        self._total_completion_tokens += estimated_tokens
        self._total_cost += self._estimate_cost(0, estimated_tokens)

    # ------------------------------------------------------------------
    # litellm (everything else)
    # ------------------------------------------------------------------

    async def _litellm_call(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await litellm.acompletion(**kwargs)

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost += self._estimate_cost(prompt_tokens, completion_tokens)

        return response.choices[0].message.content or ""

    async def _litellm_stream(self, prompt: str) -> AsyncIterator[str]:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": True,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        response = await litellm.acompletion(**kwargs)
        full_text = ""
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                yield delta

        estimated_tokens = len(full_text) // 4
        self._total_completion_tokens += estimated_tokens
        self._total_cost += self._estimate_cost(0, estimated_tokens)

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate USD cost for a single call."""
        model = self._config.model_id
        if model in _COST_PER_M_TOKENS:
            input_rate, output_rate = _COST_PER_M_TOKENS[model]
        else:
            input_rate, output_rate = 3.0, 15.0
        return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
