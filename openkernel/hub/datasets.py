"""Upload and download datasets (traces, results, skills) to/from HF Hub.

Follows the layout described in docs/data-and-integrations.md:

    openkernel/optimization-traces   — Parquet traces (private)
    openkernel/kernelbench-results   — sweep results  (public)
    openkernel/skill-library         — skill JSON files (public)
"""

from __future__ import annotations

import logging
from pathlib import Path

from openkernel.hub.client import HubClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Traces
# ------------------------------------------------------------------


def upload_traces(
    traces_dir: str | Path,
    client: HubClient,
    *,
    path_in_repo: str = "traces/raw",
) -> None:
    """Upload local Parquet traces to the ``optimization-traces`` dataset.

    Expects the directory structure produced by
    :func:`openkernel.traces.storage.save_trace`::

        traces/raw/
        ├── 2026-04/
        │   ├── session_abc123.parquet
        │   └── ...
        └── ...

    Parameters
    ----------
    traces_dir:
        Root of the local traces directory (e.g. ``"traces/raw"``).
    client:
        An authenticated :class:`HubClient`.
    path_in_repo:
        Destination path inside the HF dataset repo.
    """
    traces_dir = Path(traces_dir)
    if not traces_dir.is_dir():
        raise NotADirectoryError(f"Traces directory not found: {traces_dir}")

    repo_id = client.config.traces_repo
    client.ensure_repo_exists(repo_id, repo_type="dataset", private=True)

    parquet_files = sorted(traces_dir.rglob("*.parquet"))
    if not parquet_files:
        logger.warning("No .parquet files found under %s — nothing to upload.", traces_dir)
        return

    for pf in parquet_files:
        relative = pf.relative_to(traces_dir)
        dest = f"{path_in_repo}/{relative}" if path_in_repo else str(relative)
        client.upload_file(repo_id, pf, dest)

    logger.info("Uploaded %d trace file(s) to %s.", len(parquet_files), repo_id)


# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------


def upload_results(
    results_dir: str | Path,
    client: HubClient,
    *,
    path_in_repo: str = "sweeps",
) -> None:
    """Upload KernelBench sweep results to the ``kernelbench-results`` dataset.

    Parameters
    ----------
    results_dir:
        Directory containing result Parquet or JSON files.
    client:
        An authenticated :class:`HubClient`.
    path_in_repo:
        Destination path inside the HF dataset repo.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results directory not found: {results_dir}")

    repo_id = client.config.results_repo
    client.ensure_repo_exists(repo_id, repo_type="dataset", private=False)

    client.upload_directory(repo_id, results_dir, path_in_repo)
    logger.info("Uploaded results from %s to %s:%s.", results_dir, repo_id, path_in_repo)


# ------------------------------------------------------------------
# Skill library
# ------------------------------------------------------------------


def upload_skill_library(
    skills_dir: str | Path,
    client: HubClient,
    *,
    path_in_repo: str = "skills",
) -> None:
    """Upload the local skill library (JSON files) to HF Hub.

    Parameters
    ----------
    skills_dir:
        Directory containing skill ``*.json`` files (e.g. ``"data/skills"``).
    client:
        An authenticated :class:`HubClient`.
    path_in_repo:
        Destination path inside the HF dataset repo.
    """
    skills_dir = Path(skills_dir)
    if not skills_dir.is_dir():
        raise NotADirectoryError(f"Skills directory not found: {skills_dir}")

    repo_id = client.config.skills_repo
    client.ensure_repo_exists(repo_id, repo_type="dataset", private=False)

    client.upload_directory(repo_id, skills_dir, path_in_repo)
    logger.info("Uploaded skill library from %s to %s:%s.", skills_dir, repo_id, path_in_repo)


def download_skill_library(
    client: HubClient,
    output_dir: str | Path = "data/skills",
) -> Path:
    """Download the latest skill library from HF Hub.

    Parameters
    ----------
    client:
        An authenticated :class:`HubClient`.
    output_dir:
        Local directory to write the downloaded skills to.

    Returns
    -------
    Path
        The directory containing the downloaded skill files.
    """
    output_dir = Path(output_dir)
    repo_id = client.config.skills_repo
    logger.info("Downloading skill library from %s to %s.", repo_id, output_dir)
    return client.download_directory(repo_id, output_dir, repo_type="dataset")
