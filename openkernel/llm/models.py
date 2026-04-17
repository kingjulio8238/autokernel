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
    """Load the recommended models list from the bundled JSON file.

    Returns a list of dicts — each dict contains ``id``, ``provider``,
    ``name``, ``strengths``, ``cost_tier``, and ``recommended_for``.
    """
    if not _RECOMMENDED_PATH.exists():
        logger.warning("Recommended models file not found at %s", _RECOMMENDED_PATH)
        return []

    with open(_RECOMMENDED_PATH) as f:
        data = json.load(f)

    return data.get("models", [])


def get_default_model() -> ModelConfig:
    """Return a :class:`ModelConfig` for the default recommended model.

    The default is read from ``recommended.json``'s ``"default"`` key.  If the
    file is missing or the key is absent, falls back to Claude Sonnet 4.
    """
    fallback_id = "minimax/MiniMax-M2.5"
    fallback_provider = "minimax"
    fallback_api_base = "https://api.minimax.io/v1"

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
