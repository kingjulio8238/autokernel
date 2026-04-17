"""Per-problem heatmap for KernelBench results.

Chart 5: Heatmap with rows = problems, columns = systems, colour = speedup.
Inspired by the KernelBench "Kernelsseum" leaderboard.
"""

from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go


def create_problem_heatmap(
    results_by_system: dict[str, dict[str, float]],
    *,
    problem_order: Sequence[str] | None = None,
) -> go.Figure:
    """Create the per-problem heatmap.

    Args:
        results_by_system: Nested dict — outer key is system name, inner key
            is problem name/ID, value is speedup achieved.  Use 0 or NaN for
            unsolved problems.
        problem_order: Optional explicit ordering of problems (row order).
            If not provided, problems are sorted alphabetically.

    Returns:
        A Plotly Figure (heatmap).
    """
    fig = go.Figure()

    if not results_by_system:
        fig.update_layout(title="Problem Heatmap (no data)")
        return fig

    systems = list(results_by_system.keys())

    # Collect all problem names across systems.
    all_problems: set[str] = set()
    for sys_results in results_by_system.values():
        all_problems.update(sys_results.keys())

    if problem_order is not None:
        problems = list(problem_order)
    else:
        problems = sorted(all_problems)

    # Build the Z matrix: rows = problems, cols = systems.
    z: list[list[float | None]] = []
    hover_text: list[list[str]] = []
    for prob in problems:
        row: list[float | None] = []
        hover_row: list[str] = []
        for sys in systems:
            val = results_by_system[sys].get(prob)
            if val is not None and val == val:  # not NaN
                row.append(val)
                hover_row.append(f"{sys}<br>{prob}<br>Speedup: {val:.2f}x")
            else:
                row.append(None)
                hover_row.append(f"{sys}<br>{prob}<br>Not solved")
        z.append(row)
        hover_text.append(hover_row)

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=systems,
            y=problems,
            colorscale=[
                [0.0, "#1e1b4b"],   # very dark indigo (low speedup)
                [0.25, "#3730a3"],   # indigo
                [0.5, "#2563eb"],    # blue
                [0.75, "#22c55e"],   # green (good)
                [1.0, "#facc15"],    # yellow (excellent)
            ],
            colorbar=dict(title="Speedup (x)"),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            zmin=0,
        )
    )

    fig.update_layout(
        title="Per-Problem Speedup Heatmap",
        xaxis_title="System",
        yaxis_title="Problem",
        template="plotly_dark",
        height=max(400, len(problems) * 22 + 120),
        margin=dict(l=200, r=20, t=60, b=60),
        yaxis_autorange="reversed",
    )

    return fig
