"""Tool: write_file -- write content to a file (restricted paths)."""

from __future__ import annotations

import os
from typing import Any


def _is_allowed_path(path: str) -> bool:
    """Return True if the path is in the write-allowed set.

    Allowed targets:
        - Files ending in ``_optimized.py``
        - Files ending in ``_results.md``
        - Any file inside a ``.kernel-code/`` directory
    """
    basename = os.path.basename(path)
    if basename.endswith("_optimized.py"):
        return True
    if basename.endswith("_results.md"):
        return True

    # Check if any component of the path is .kernel-code
    parts = os.path.normpath(path).split(os.sep)
    if ".kernel-code" in parts:
        return True

    return False


def execute(session_context: dict, **kwargs: Any) -> str:
    """Write content to a file.

    Restricted to: ``*_optimized.py``, ``*_results.md``, or files inside
    ``.kernel-code/``.

    Required kwargs:
        path (str): destination file path.
        content (str): content to write.
    """
    path = kwargs.get("path", "")
    content = kwargs.get("content", "")

    if not path:
        return "Error: 'path' parameter is required."
    if not content:
        return "Error: 'content' parameter is required."

    # Resolve relative paths
    if not os.path.isabs(path):
        cwd = session_context.get("working_dir", os.getcwd())
        path = os.path.join(cwd, path)

    if not _is_allowed_path(path):
        return (
            "Error: write not allowed to this path. "
            "Only *_optimized.py, *_results.md, or files inside .kernel-code/ are permitted."
        )

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        return f"Error writing file: {exc}"

    num_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"Written {num_lines} lines to {path}"
