"""Hugging Face Hub integration for openkernel.

Provides upload/download of traces, optimized kernels, skill libraries,
benchmark results, and models to/from HF Hub datasets and model repos.
"""

from openkernel.hub.client import HubClient
from openkernel.hub.datasets import (
    download_skill_library,
    upload_results,
    upload_skill_library,
    upload_traces,
)
from openkernel.hub.kernels import (
    download_kernel,
    list_kernels,
    upload_kernel,
)
from openkernel.hub.models import (
    download_model,
    list_available_models,
)

__all__ = [
    # Client
    "HubClient",
    # Datasets
    "upload_traces",
    "download_skill_library",
    "upload_results",
    "upload_skill_library",
    # Kernels
    "upload_kernel",
    "download_kernel",
    "list_kernels",
    # Models
    "download_model",
    "list_available_models",
]
