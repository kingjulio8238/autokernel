"""Tool: read_file -- read a file and return its contents."""

from __future__ import annotations

import os
from typing import Any

_ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".md", ".toml"}
_MAX_LINES = 200


def execute(session_context: dict, **kwargs: Any) -> str:
    """Read a file and return its contents.

    Restricted to: .py, .json, .yaml, .yml, .md, .toml files.  Max 200 lines.

    Required kwargs:
        path (str): path to the file to read.
    """
    path = kwargs.get("path", "")
    if not path:
        return "Error: 'path' parameter is required."

    # Validate extension
    _, ext = os.path.splitext(path)
    if ext.lower() not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        return (
            f"Error: file extension '{ext}' is not allowed. "
            f"Allowed extensions: {allowed}"
        )

    # Resolve relative paths against the working directory if available
    if not os.path.isabs(path):
        cwd = session_context.get("working_dir", os.getcwd())
        path = os.path.join(cwd, path)

    if not os.path.isfile(path):
        return f"Error: file not found: {path}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return f"Error reading file: {exc}"

    truncated = len(lines) > _MAX_LINES
    display_lines = lines[:_MAX_LINES]

    numbered = []
    for i, line in enumerate(display_lines, start=1):
        numbered.append(f"{i:>4} | {line.rstrip()}")

    result = "\n".join(numbered)
    if truncated:
        result += (
            f"\n\n[truncated] Showing first {_MAX_LINES} of {len(lines)} lines."
        )

    return result
