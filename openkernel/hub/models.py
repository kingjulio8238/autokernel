"""Download and list models hosted on Hugging Face Hub.

This module supports the future *kernelgen-1* model and any custom
fine-tuned models hosted under the ``openkernel`` organisation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from openkernel.hub.client import HubClient

logger = logging.getLogger(__name__)

# Default model IDs published by the openkernel project.
_DEFAULT_MODEL_IDS: list[str] = [
    "openkernel/kernelgen-1",
    "openkernel/kernelgen-1-triton",
    "openkernel/kernelgen-1-cuda",
]


def download_model(
    model_id: str,
    client: HubClient,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a model from Hugging Face Hub.

    Uses ``snapshot_download`` with ``repo_type="model"`` so the full
    model directory (config, weights, tokenizer, etc.) is fetched.

    Parameters
    ----------
    model_id:
        Repository ID on HF Hub, e.g. ``"openkernel/kernelgen-1"``.
    client:
        An authenticated :class:`HubClient`.
    cache_dir:
        Optional local cache directory.  When *None* the default
        ``huggingface_hub`` cache location is used.

    Returns
    -------
    Path
        Directory containing the downloaded model files.
    """
    logger.info("Downloading model %s ...", model_id)
    kwargs: dict = {
        "repo_id": model_id,
        "repo_type": "model",
    }
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(cache_dir)

    # Use the underlying HfApi token for auth.
    import os

    from huggingface_hub import snapshot_download

    token = client.config.token or os.environ.get("HF_TOKEN")
    kwargs["token"] = token

    path = snapshot_download(**kwargs)
    logger.info("Model %s downloaded to %s.", model_id, path)
    return Path(path)


def list_available_models(client: HubClient) -> list[str]:
    """List openkernel model repositories available on HF Hub.

    Queries the Hub for model repos under the configured organisation.
    Falls back to the static ``_DEFAULT_MODEL_IDS`` list when the Hub
    query fails (e.g. no network, no auth).

    Parameters
    ----------
    client:
        An authenticated :class:`HubClient`.

    Returns
    -------
    list[str]
        Repository IDs of available models.
    """
    api = client._get_api()
    org = client.config.org
    try:
        models = list(api.list_models(author=org))
        if models:
            return [m.modelId for m in models]
    except Exception:
        logger.debug(
            "Failed to query HF Hub for models under '%s'; returning defaults.", org
        )

    return list(_DEFAULT_MODEL_IDS)
