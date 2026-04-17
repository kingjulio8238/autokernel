"""fast_p grouped bar chart for KernelBench results.

Chart 1: Headline comparison of fast@1, fast@1.5, fast@2 across systems.
Uses BASELINES from openkernel.kernelbench.compare for competitor numbers.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from openkernel.kernelbench.compare import BASELINES


# Consistent colours for the three fast_p thresholds.
_FAST_P_COLORS = {
    "fast_1": "#22c55e",    # green
    "fast_1_5": "#3b82f6",  # blue
    "fast_2": "#a855f7",    # purple
}

_FAST_P_LABELS = {
    "fast_1": "fast@1.0",
    "fast_1_5": "fast@1.5",
    "fast_2": "fast@2.0",
}


def create_fast_p_chart(
    our_results: dict[str, float],
    baselines: dict[str, dict[str, float]] | None = None,
) -> go.Figure:
    """Create the fast_p grouped bar chart.

    Args:
        our_results: Dict with keys ``fast_1``, ``fast_1_5``, ``fast_2``
            (values in 0-1 range, e.g. 0.95 for 95%).
        baselines: Optional override for baseline data.  Defaults to
            :data:`BASELINES` from ``openkernel.kernelbench.compare``.

    Returns:
        A Plotly Figure with grouped bars per system.
    """
    if baselines is None:
        baselines = BASELINES

    # Build ordered system list: ours first, then baselines.
    all_systems: dict[str, dict[str, Any]] = {
        "openkernel (ours)": our_results,
        **baselines,
    }

    system_names = list(all_systems.keys())

    fig = go.Figure()

    for metric_key, label in _FAST_P_LABELS.items():
        values = [
            all_systems[s].get(metric_key, 0.0) for s in system_names
        ]
        fig.add_trace(
            go.Bar(
                x=system_names,
                y=values,
                name=label,
                marker_color=_FAST_P_COLORS[metric_key],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{label}: " + "%{y:.1%}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="KernelBench fast@p Scores",
        xaxis_title="System",
        yaxis_title="Score (fraction of problems)",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05],
        template="plotly_dark",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=60, r=20, t=60, b=100),
        xaxis_tickangle=-25,
    )

    return fig
