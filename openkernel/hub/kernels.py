"""Upload, download, and list optimized kernels on HF Hub.

Follows the layout described in docs/data-and-integrations.md::

    openkernel/optimized-kernels/
    └── kernelbench/
        ├── L1/
        │   ├── problem_001/
        │   │   ├── triton/
        │   │   │   ├── best_kernel.py
        │   │   │   └── metadata.json
        │   │   └── cuda/
        │   │       ├── best_kernel.cu
        │   │       └── metadata.json
        │   └── ...
        ├── L2/
        ├── L3/
        └── L4/
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from openkernel.hub.client import HubClient

logger = logging.getLogger(__name__)

# Map backend name to file extension for the kernel source file.
_KERNEL_EXTENSIONS: dict[str, str] = {
    "triton": ".py",
    "cuda": ".cu",
}


def _kernel_prefix(problem_id: str, backend: str) -> str:
    """Build the HF repo path prefix for a kernel.

    *problem_id* is expected to look like ``"L1/problem_001"`` or a plain
    ``"problem_001"`` (in which case no level directory is added).
    """
    # Normalise: strip leading/trailing slashes
    problem_id = problem_id.strip("/")
    return f"kernelbench/{problem_id}/{backend}"


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------


def upload_kernel(
    kernel_code: str,
    problem_id: str,
    backend: str,
    metadata: dict,
    client: HubClient,
) -> None:
    """Upload an optimized kernel to the ``optimized-kernels`` dataset.

    Parameters
    ----------
    kernel_code:
        Source code of the kernel.
    problem_id:
        E.g. ``"L1/problem_001"`` — level and problem identifier.
    backend:
        ``"triton"`` or ``"cuda"``.
    metadata:
        Dict with speedup, hardware, model, iterations, etc.
    client:
        An authenticated :class:`HubClient`.
    """
    repo_id = client.config.kernels_repo
    client.ensure_repo_exists(repo_id, repo_type="dataset", private=False)

    ext = _KERNEL_EXTENSIONS.get(backend, ".py")
    prefix = _kernel_prefix(problem_id, backend)
    kernel_filename = f"best_kernel{ext}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write kernel source
        kernel_path = Path(tmpdir) / kernel_filename
        kernel_path.write_text(kernel_code, encoding="utf-8")
        client.upload_file(repo_id, kernel_path, f"{prefix}/{kernel_filename}")

        # Write metadata
        meta_path = Path(tmpdir) / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        client.upload_file(repo_id, meta_path, f"{prefix}/metadata.json")

    logger.info("Uploaded kernel for %s (%s) to %s.", problem_id, backend, repo_id)


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------


def download_kernel(
    problem_id: str,
    backend: str,
    client: HubClient,
    *,
    local_dir: str | Path | None = None,
) -> str | None:
    """Download the best kernel source for a given problem + backend.

    Returns the kernel source code as a string, or *None* if the kernel
    is not found on HF Hub.

    Parameters
    ----------
    problem_id:
        E.g. ``"L1/problem_001"``.
    backend:
        ``"triton"`` or ``"cuda"``.
    client:
        An authenticated :class:`HubClient`.
    local_dir:
        Optional directory for caching the download.  A temp directory is
        used when not specified.
    """
    repo_id = client.config.kernels_repo
    ext = _KERNEL_EXTENSIONS.get(backend, ".py")
    prefix = _kernel_prefix(problem_id, backend)
    filename = f"{prefix}/best_kernel{ext}"

    try:
        target_dir = Path(local_dir) if local_dir else Path(tempfile.mkdtemp())
        path = client.download_file(repo_id, filename, target_dir)
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        # huggingface_hub raises EntryNotFoundError (subclass of HfHubHTTPError)
        # when the file doesn't exist.  Catch broadly so callers get None
        # instead of an explosion when the kernel simply hasn't been uploaded.
        logger.debug("Could not download kernel %s: %s", filename, exc)
        return None


# ------------------------------------------------------------------
# Listing
# ------------------------------------------------------------------


def list_kernels(level: int, client: HubClient) -> list[dict]:
    """List all uploaded kernels for a given KernelBench level.

    Returns a list of dicts with keys ``problem_id``, ``backend``,
    and ``path`` for each kernel found.

    Parameters
    ----------
    level:
        KernelBench level (1-4).
    client:
        An authenticated :class:`HubClient`.
    """
    repo_id = client.config.kernels_repo
    prefix = f"kernelbench/L{level}"
    files = client.list_files(repo_id, path=prefix)

    kernels: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for filepath in files:
        parts = filepath.split("/")
        # Expected: kernelbench/L{n}/problem_xxx/backend/best_kernel.ext
        if len(parts) < 5:
            continue
        problem_id = f"L{level}/{parts[2]}"
        backend = parts[3]
        key = (problem_id, backend)
        if key in seen:
            continue
        seen.add(key)
        kernels.append(
            {
                "problem_id": problem_id,
                "backend": backend,
                "path": filepath,
            }
        )

    return kernels
