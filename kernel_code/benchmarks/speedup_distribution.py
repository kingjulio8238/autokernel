"""Speedup distribution chart for KernelBench results.

Chart 2: Violin/box plot showing the distribution of speedup values across
all problems for each system.
"""

from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go


def create_speedup_distribution(
    our_results: Sequence[float],
    *,
    label: str = "openkernel (ours)",
    comparisons: dict[str, Sequence[float]] | None = None,
) -> go.Figure:
    """Create a violin + box plot of speedup distributions.

    Args:
        our_results: Sequence of per-problem speedup values for our system.
        label: Display name for our system.
        comparisons: Optional dict mapping system name to a list of speedup
            values.  If provided, each system gets its own violin.

    Returns:
        A Plotly Figure.
    """
    fig = go.Figure()

    all_systems: dict[str, Sequence[float]] = {label: our_results}
    if comparisons:
        all_systems.update(comparisons)

    colors = [
        "#22c55e", "#3b82f6", "#a855f7", "#ef4444",
        "#eab308", "#06b6d4", "#f97316", "#ec4899",
    ]

    for idx, (name, speedups) in enumerate(all_systems.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Violin(
                y=list(speedups),
                name=name,
                box_visible=True,
                meanline_visible=True,
                line_color=color,
                fillcolor=color.replace(")", ", 0.25)").replace("rgb", "rgba")
                if color.startswith("rgb") else color,
                opacity=0.7,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Speedup: %{y:.2f}x<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Baseline at 1.0x
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="1.0x (baseline)",
        annotation_position="bottom left",
    )

    fig.update_layout(
        title="Speedup Distribution Across Problems",
        yaxis_title="Speedup (x)",
        yaxis_type="log",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=60, r=20, t=60, b=40),
        showlegend=True,
    )

    return fig
