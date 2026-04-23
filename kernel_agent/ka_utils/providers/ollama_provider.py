"""Ollama provider — connects to local Ollama server for self-hosted models.

Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1.
This provider is used for self-hosted models like KernelLLM
(facebook/KernelLLM), CodeLlama, and other local models.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull the model: ollama pull hf.co/facebook/KernelLLM
    3. Set in kernel-code: /config set default_model KernelLLM
"""

import os

from .openai_base import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Provider for local Ollama-hosted models.

    Ollama doesn't require an API key. A dummy key is set automatically
    since the OpenAI client library requires one to initialize.
    """

    def __init__(self):
        # Ollama doesn't need a real API key — set a dummy so the
        # parent's _initialize_client() can create the OpenAI client
        if not os.environ.get("OLLAMA_API_KEY"):
            os.environ["OLLAMA_API_KEY"] = "ollama"
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
        super().__init__(api_key_env="OLLAMA_API_KEY", base_url=base_url)

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        if self.client is None:
            return False
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return "ollama"

    def get_response(self, model_name: str, messages: list[dict[str, str]], **kwargs):
        """Strip 'ollama/' prefix before sending to Ollama server."""
        return super().get_response(self._strip_prefix(model_name), messages, **kwargs)

    def get_multiple_responses(self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs):
        """Ollama doesn't support n>1 — make sequential calls instead."""
        clean_name = self._strip_prefix(model_name)
        if n <= 1:
            return [self.get_response(clean_name, messages, **kwargs)]
        # Ollama returns 405 on n>1, so we call sequentially
        return [self.get_response(clean_name, messages, **kwargs) for _ in range(n)]

    @staticmethod
    def _strip_prefix(model_name: str) -> str:
        """Strip 'ollama/' prefix — Ollama expects bare model names."""
        if model_name.startswith("ollama/"):
            return model_name[7:]
        return model_name

    def _build_api_params(self, model_name: str, messages: list[dict[str, str]], **kwargs):
        """Override to strip params Ollama doesn't support."""
        params = {
            "model": self._strip_prefix(model_name),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": min(kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)),
        }
        # Ollama ignores n, reasoning_effort, max_completion_tokens
        return params

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Ollama models typically support 8K-32K context."""
        if "kernelllm" in model_name.lower():
            return 8192
        return 8192
