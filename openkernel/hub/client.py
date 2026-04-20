"""Low-level Hugging Face Hub client.

Wraps *huggingface_hub* (HfApi, hf_hub_download, snapshot_download) and
provides convenience helpers used by the higher-level dataset / kernel /
model modules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from openkernel.config import HubConfig

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import (
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

logger = logging.getLogger(__name__)


def _require_hf() -> None:
    """Raise a helpful error when huggingface_hub is missing."""
    if not _HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for HF Hub integration but is not installed.\n"
            "Install it with:  pip install 'huggingface-hub>=0.25'"
        )


class HubClient:
    """Thin wrapper around the Hugging Face Hub API.

    Parameters
    ----------
    config : HubConfig | None
        Hub configuration.  When *None* a default ``HubConfig()`` is created
        which reads ``HF_TOKEN`` from the environment.
    """

    def __init__(self, config: HubConfig | None = None) -> None:
        _require_hf()
        self.config = config or HubConfig()
        self._api: HfApi | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_api(self) -> HfApi:
        """Lazily initialise and return the :class:`HfApi` instance."""
        if self._api is None:
            token = self.config.token or os.environ.get("HF_TOKEN")
            self._api = HfApi(token=token)
        return self._api

    # ------------------------------------------------------------------
    # Repository management
    # ------------------------------------------------------------------

    def ensure_repo_exists(
        self,
        repo_id: str,
        repo_type: str = "dataset",
        private: bool = True,
    ) -> None:
        """Create the repo on HF Hub if it does not already exist."""
        api = self._get_api()
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            logger.debug("Repo %s already exists.", repo_id)
        except RepositoryNotFoundError:
            logger.info("Creating repo %s (type=%s, private=%s)", repo_id, repo_type, private)
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                exist_ok=True,
            )
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                logger.error(
                    "Authentication failed.  Set HF_TOKEN or pass token via HubConfig."
                )
            raise

    # ------------------------------------------------------------------
    # Upload helpers
    # ------------------------------------------------------------------

    def upload_file(
        self,
        repo_id: str,
        local_path: str | Path,
        path_in_repo: str,
        repo_type: str = "dataset",
    ) -> None:
        """Upload a single file to a HF Hub repository."""
        api = self._get_api()
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        logger.info("Uploading %s -> %s:%s", local_path, repo_id, path_in_repo)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )

    def upload_directory(
        self,
        repo_id: str,
        local_dir: str | Path,
        path_in_repo: str,
        repo_type: str = "dataset",
    ) -> None:
        """Upload an entire directory to a HF Hub repository."""
        api = self._get_api()
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise NotADirectoryError(f"Local directory not found: {local_dir}")
        logger.info("Uploading directory %s -> %s:%s", local_dir, repo_id, path_in_repo)
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str | Path,
        repo_type: str = "dataset",
    ) -> Path:
        """Download a single file from a HF Hub repository.

        Returns the local path to the downloaded file.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            repo_type=repo_type,
            token=self.config.token or os.environ.get("HF_TOKEN"),
        )
        return Path(path)

    def download_directory(
        self,
        repo_id: str,
        local_dir: str | Path,
        repo_type: str = "dataset",
    ) -> Path:
        """Download the full repository snapshot.

        Returns the local directory containing the downloaded files.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            repo_type=repo_type,
            token=self.config.token or os.environ.get("HF_TOKEN"),
        )
        return Path(path)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_files(
        self,
        repo_id: str,
        path: str = "",
        repo_type: str = "dataset",
    ) -> list[str]:
        """List files in a HF Hub repository (optionally under *path*)."""
        api = self._get_api()
        try:
            files: list[str] = []
            for entry in api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=path or None,
                repo_type=repo_type,
            ):
                # RepoFile has rfilename; RepoFolder has rfilename too
                rfilename = getattr(entry, "rfilename", None) or getattr(entry, "path", "")
                if rfilename:
                    files.append(rfilename)
            return files
        except RepositoryNotFoundError:
            logger.warning("Repository %s not found.", repo_id)
            return []
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                logger.error("Authentication failed when listing %s.", repo_id)
            raise
