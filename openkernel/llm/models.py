"""Recommended models registry.

Loads the curated model list from ``data/models/recommended.json`` and provides
helpers for selecting models by ID or falling back to a sensible default.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from openkernel.config import ModelConfig

logger = logging.getLogger(__name__)

# Path is resolved relative to the repo root via the data/ directory.
# We walk up from this file: llm/models.py -> llm -> openkernel -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RECOMMENDED_PATH = _REPO_ROOT / "data" / "models" / "recommended.json"


def load_recommended_models() -> list[dict[str, Any]]:
    """Load the recommended models list from the bundled JSON file,
    plus any models available on the local Ollama server.

    Returns a list of dicts — each dict contains ``id``, ``provider``,
    ``name``, ``strengths``, ``cost_tier``, and ``recommended_for``.
    """
    models: list[dict[str, Any]] = []

    if _RECOMMENDED_PATH.exists():
        with open(_RECOMMENDED_PATH) as f:
            data = json.load(f)
        models.extend(data.get("models", []))

    # Auto-discover local Ollama models
    ollama_models = _discover_ollama_models()
    if ollama_models:
        # Add Ollama models that aren't already in the list
        existing_ids = {m.get("id") for m in models}
        for om in ollama_models:
            if om["id"] not in existing_ids:
                models.append(om)

    return models


def _discover_ollama_models() -> list[dict[str, Any]]:
    """Query local Ollama server for available models.

    Returns model entries in the same format as recommended.json.
    Returns empty list if Ollama is not running.
    """
    import os
    try:
        import urllib.request
        import urllib.error

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())

        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size_gb = m.get("size", 0) / 1e9
            param_size = m.get("details", {}).get("parameter_size", "")
            quant = m.get("details", {}).get("quantization_level", "")
            family = m.get("details", {}).get("family", "")

            # Build display name
            short_name = name.split("/")[-1].split(":")[0] if "/" in name else name.split(":")[0]
            display = f"{short_name} ({quant})" if quant else short_name

            strengths = []
            if "kernel" in name.lower():
                strengths.append("Fine-tuned for Triton kernel generation")
            if param_size:
                strengths.append(f"Parameters: {param_size}")
            if size_gb > 0:
                strengths.append(f"Size: {size_gb:.1f} GB on disk")
            strengths.append(f"Local — no API cost, runs on your machine")

            models.append({
                "id": name,
                "name": display,
                "provider": "ollama",
                "env_key": "OLLAMA_API_KEY",
                "cost_tier": "free",
                "cost_per_m_input": 0,
                "cost_per_m_output": 0,
                "context_window": 8192,
                "strengths": strengths,
            })

        return models
    except Exception:
        return []


def get_default_model() -> ModelConfig:
    """Return a :class:`ModelConfig` for the default recommended model.

    The default is read from ``recommended.json``'s ``"default"`` key.  If the
    file is missing or the key is absent, falls back to Claude Sonnet 4.
    """
    fallback_id = "deepseek-ai/deepseek-v3.2"
    fallback_provider = "nvidia"
    fallback_api_base = "https://integrate.api.nvidia.com/v1"

    if not _RECOMMENDED_PATH.exists():
        return ModelConfig(
            provider=fallback_provider,
            model_id=fallback_id,
            api_base=fallback_api_base,
        )

    with open(_RECOMMENDED_PATH) as f:
        data = json.load(f)

    default_id = data.get("default", fallback_id)
    models = data.get("models", [])

    for m in models:
        if m.get("id") == default_id:
            return _model_config_from_entry(m)

    # default key didn't match any model entry — still use it
    return ModelConfig(provider=fallback_provider, model_id=default_id, api_base=fallback_api_base)


def get_model_by_id(model_id: str) -> ModelConfig | None:
    """Look up a model by its ``model_id`` in the recommended list.

    Returns ``None`` if no match is found.
    """
    models = load_recommended_models()

    for m in models:
        if m.get("id") == model_id:
            return _model_config_from_entry(m)

    return None


def _model_config_from_entry(entry: dict[str, Any]) -> ModelConfig:
    """Build a ModelConfig from a recommended.json model entry."""
    return ModelConfig(
        provider=entry.get("provider", "openai"),
        model_id=entry["id"],
        api_base=entry.get("api_base"),
    )
