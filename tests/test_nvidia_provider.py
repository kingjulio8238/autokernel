"""Tests for NvidiaProvider (NIM, OpenAI-compatible)."""

from __future__ import annotations

import os

import pytest


# Expected NIM model slugs (must match available_models.py registration).
_EXPECTED_NVIDIA_MODELS = [
    "deepseek-ai/deepseek-v3.2",
    "moonshotai/kimi-k2-thinking",
    "moonshotai/kimi-k2-instruct",
    "minimaxai/minimax-m2.7",
    "openai/gpt-oss-120b",
    "thudm/glm-5-air",
    "meta/llama-3.3-70b-instruct",
]


def test_nvidia_provider_sets_correct_base_url() -> None:
    from kernel_agent.ka_utils.providers.nvidia_provider import NvidiaProvider

    # Instantiation triggers _initialize_client, which needs no key to succeed
    # (client is only built when NVIDIA_API_KEY is present). Asserting the
    # static attributes on the instance either way.
    provider = NvidiaProvider()
    assert provider.base_url == "https://integrate.api.nvidia.com/v1"
    assert provider.api_key_env == "NVIDIA_API_KEY"
    assert provider.name == "nvidia"
    assert provider.rpm_limit == 40


def test_nvidia_models_registered() -> None:
    from kernel_agent.ka_utils.providers.available_models import AVAILABLE_MODELS
    from kernel_agent.ka_utils.providers.nvidia_provider import NvidiaProvider

    registered = {m.name: m for m in AVAILABLE_MODELS}

    for slug in _EXPECTED_NVIDIA_MODELS:
        assert slug in registered, f"NIM model missing from registry: {slug}"
        assert NvidiaProvider in registered[slug].provider_classes, (
            f"NvidiaProvider not registered for model: {slug}"
        )


def test_nvidia_api_key_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernel_code.settings import KernelCodeSettings, inject_api_keys

    # Ensure a clean slate for NVIDIA_API_KEY — monkeypatch auto-restores.
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    settings = KernelCodeSettings()
    settings.nvidia_api_key = "nvapi-test"

    inject_api_keys(settings)

    assert os.environ["NVIDIA_API_KEY"] == "nvapi-test"


def test_llmprovider_routes_nvidia_through_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: LLMProvider must prefix nvidia model ids with ``openai/``
    and set the NIM api_base, otherwise LiteLLM raises
    ``BadRequestError: LLM Provider NOT provided``.

    This is the failure mode that previously broke ``kernel_code.meta_reflect``.
    """
    from openkernel.config import ModelConfig
    from openkernel.llm.provider import LLMProvider

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    provider = LLMProvider(
        ModelConfig(provider="nvidia", model_id="deepseek-ai/deepseek-v3.2")
    )

    assert provider._api_key == "nvapi-test"
    assert provider._api_base == "https://integrate.api.nvidia.com/v1"
    assert provider._litellm_model_id() == "openai/deepseek-ai/deepseek-v3.2"

    # Already-prefixed ids must not be double-prefixed.
    prefixed = LLMProvider(
        ModelConfig(provider="nvidia", model_id="openai/deepseek-ai/deepseek-v3.2")
    )
    assert prefixed._litellm_model_id() == "openai/deepseek-ai/deepseek-v3.2"


def test_config_validates_nvidia_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OpenKernelConfig.validate_config`` must know about the nvidia provider
    so the default config (provider=nvidia) passes validation whenever
    ``NVIDIA_API_KEY`` is set."""
    from openkernel.config import OpenKernelConfig

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    # Default config uses provider="nvidia"; validation should now succeed.
    OpenKernelConfig().validate_config()


@pytest.mark.skipif(
    os.environ.get("NVIDIA_API_KEY") is None,
    reason="requires NVIDIA_API_KEY",
)
def test_nvidia_live_integration() -> None:
    """Live smoke test against the real NIM endpoint.

    Uses meta/llama-3.3-70b-instruct (cheapest) with a tiny completion.
    Skipped automatically when NVIDIA_API_KEY is not set.
    """
    from kernel_agent.ka_utils.providers.nvidia_provider import NvidiaProvider

    provider = NvidiaProvider()
    assert provider.is_available(), "client not initialized despite NVIDIA_API_KEY set"

    response = provider.get_response(
        "meta/llama-3.3-70b-instruct",
        [{"role": "user", "content": "Say 'ok' and nothing else."}],
        max_tokens=8,
        temperature=0.0,
    )
    assert response.content
    assert response.provider == "nvidia"
