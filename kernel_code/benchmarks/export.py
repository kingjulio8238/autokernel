"""Figure export utilities for KernelBench charts.

Provides helpers to save Plotly figures as static images (PNG/SVG) and batch
export a collection of named figures to an output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import plotly.graph_objects as go


def export_figure(
    fig: go.Figure,
    path: str | Path,
    fmt: Literal["png", "svg"] = "png",
    *,
    width: int = 1200,
    height: int = 700,
    scale: float = 2.0,
) -> Path:
    """Save a Plotly figure as a static image.

    Args:
        fig: The Plotly Figure to export.
        path: Output file path.  If the path has no extension matching *fmt*,
            the correct extension is appended.
        fmt: Image format — ``"png"`` or ``"svg"``.
        width: Image width in pixels (before scale).
        height: Image height in pixels (before scale).
        scale: Resolution multiplier (2.0 = retina).

    Returns:
        The resolved output path.

    Raises:
        ValueError: If *fmt* is not ``"png"`` or ``"svg"``.
    """
    if fmt not in ("png", "svg"):
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'png' or 'svg'.")

    path = Path(path)
    if path.suffix.lstrip(".") != fmt:
        path = path.with_suffix(f".{fmt}")

    path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_image(
        str(path),
        format=fmt,
        width=width,
        height=height,
        scale=scale,
    )

    return path


def export_all(
    figures: dict[str, go.Figure],
    output_dir: str | Path,
    *,
    fmt: Literal["png", "svg"] = "png",
    width: int = 1200,
    height: int = 700,
    scale: float = 2.0,
) -> dict[str, Path]:
    """Batch-export a collection of named figures.

    Args:
        figures: Mapping of figure name (used as filename stem) to Plotly
            Figure.
        output_dir: Directory to write images into.
        fmt: Image format — ``"png"`` or ``"svg"``.
        width: Image width in pixels (before scale).
        height: Image height in pixels (before scale).
        scale: Resolution multiplier.

    Returns:
        Dict mapping figure name to the resolved output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}
    for name, fig in figures.items():
        # Sanitize filename: replace spaces/slashes with underscores.
        safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        out_path = export_figure(
            fig,
            output_dir / safe_name,
            fmt=fmt,
            width=width,
            height=height,
            scale=scale,
        )
        exported[name] = out_path

    return exported
