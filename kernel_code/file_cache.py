"""File-state caching for kernel-code.

Tracks file content hashes and modification times so unchanged files
can skip redundant processing.  Also caches expensive EvalResult dicts
keyed by (kernel_source + reference_source) content hash, saving real
GPU costs when the same kernel code is evaluated twice.

Usage::

    cache = FileStateCache()
    if cache.has_file_changed("reference.py"):
        # re-parse, re-evaluate, etc.
        cache.track_file("reference.py")
    cached = cache.get_cached_eval(kernel_code, reference_code)
    if cached is not None:
        return cached  # skip Modal call
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Tracked state for a single file."""

    content_hash: str
    mtime: float
    cached_data: dict[str, Any] = field(default_factory=dict)


class FileStateCache:
    """Caches file state and derived data to avoid redundant processing."""

    def __init__(self, cache_dir: str | Path = ".kernel-code/cache") -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._cache_dir = Path(cache_dir)
        self._eval_cache: dict[str, dict] = {}  # content_hash -> EvalResult dict
        self._load()

    # ------------------------------------------------------------------
    # Content hashing
    # ------------------------------------------------------------------

    def hash_content(self, content: str) -> str:
        """SHA-256 hash of *content*, truncated to 16 hex chars."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # File tracking
    # ------------------------------------------------------------------

    def track_file(self, path: str | Path) -> str:
        """Track a file and return its content hash.

        Reads the file, computes its hash, and records the entry so that
        subsequent :meth:`has_file_changed` calls can detect modifications.
        """
        p = Path(path).resolve()
        content = p.read_text()
        content_hash = self.hash_content(content)
        mtime = p.stat().st_mtime
        self._entries[str(p)] = CacheEntry(
            content_hash=content_hash, mtime=mtime
        )
        self._save()
        return content_hash

    def has_file_changed(self, path: str | Path) -> bool:
        """Check if a file has changed since it was last tracked.

        Returns ``True`` if the file has never been tracked, its mtime
        differs, or its content hash differs.
        """
        p = Path(path).resolve()
        key = str(p)
        entry = self._entries.get(key)
        if entry is None:
            return True  # never tracked

        try:
            stat = p.stat()
        except OSError:
            return True  # file disappeared

        # Fast path: mtime unchanged -> assume content unchanged
        if stat.st_mtime == entry.mtime:
            return False

        # Slow path: mtime changed, verify content hash
        try:
            content = p.read_text()
        except OSError:
            return True
        return self.hash_content(content) != entry.content_hash

    # ------------------------------------------------------------------
    # Generic key-value cache
    # ------------------------------------------------------------------

    def get_cached(self, key: str) -> Any | None:
        """Get cached derived data by key."""
        # Look across all entries for a cached_data match
        for entry in self._entries.values():
            if key in entry.cached_data:
                return entry.cached_data[key]
        return None

    def set_cached(self, key: str, value: Any) -> None:
        """Store cached derived data.

        Attaches the data to a synthetic ``__global__`` entry so it
        persists independently of any tracked file.
        """
        global_key = "__global__"
        if global_key not in self._entries:
            self._entries[global_key] = CacheEntry(content_hash="", mtime=0.0)
        self._entries[global_key].cached_data[key] = value
        self._save()

    # ------------------------------------------------------------------
    # Kernel eval caching
    # ------------------------------------------------------------------

    def get_cached_eval(self, kernel_code: str, reference_code: str) -> dict | None:
        """Check if this exact kernel+reference combo was already evaluated.

        Returns the cached EvalResult dict or ``None``.
        """
        cache_key = self.hash_content(kernel_code + ":::" + reference_code)
        return self._eval_cache.get(cache_key)

    def cache_eval(self, kernel_code: str, reference_code: str, result: dict) -> None:
        """Cache an eval result keyed by kernel+reference content hash."""
        cache_key = self.hash_content(kernel_code + ":::" + reference_code)
        self._eval_cache[cache_key] = result
        self._save()

    @property
    def eval_cache_size(self) -> int:
        """Number of cached eval results."""
        return len(self._eval_cache)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load cache from disk."""
        cache_file = self._cache_dir / "file_state.json"
        if not cache_file.exists():
            return
        try:
            data = json.loads(cache_file.read_text())
            # Restore file entries
            for path_key, entry_data in data.get("entries", {}).items():
                self._entries[path_key] = CacheEntry(
                    content_hash=entry_data["content_hash"],
                    mtime=entry_data["mtime"],
                    cached_data=entry_data.get("cached_data", {}),
                )
            # Restore eval cache
            self._eval_cache = data.get("eval_cache", {})
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Corrupt cache file %s -- starting fresh", cache_file)
            self._entries.clear()
            self._eval_cache.clear()

    def _save(self) -> None:
        """Persist cache to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / "file_state.json"

        data = {
            "entries": {
                path_key: {
                    "content_hash": entry.content_hash,
                    "mtime": entry.mtime,
                    "cached_data": entry.cached_data,
                }
                for path_key, entry in self._entries.items()
            },
            "eval_cache": self._eval_cache,
        }

        try:
            cache_file.write_text(json.dumps(data, indent=2))
        except OSError:
            logger.warning("Failed to write cache file: %s", cache_file)
