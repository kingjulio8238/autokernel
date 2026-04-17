"""Cost-performance frontier chart for KernelBench results.

Chart 4: Scatter plot showing cost per kernel vs average speedup for each
system.  Highlights the Pareto-optimal frontier.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from openkernel.kernelbench.compare import BASELINES


def create_cost_frontier(
    our_results: dict[str, float],
    baselines: dict[str, dict[str, float]] | None = None,
) -> go.Figure:
    """Create the cost-performance frontier scatter chart.

    Args:
        our_results: Dict with keys ``cost_per_kernel`` (USD) and
            ``avg_speedup`` (float).  Additional optional keys are used
            for hover context.
        baselines: Optional override for baseline data.  Each value must
            contain ``cost_per_kernel`` and ``avg_speedup``.
            Defaults to :data:`BASELINES` (extended with cost estimates).

    Returns:
        A Plotly Figure.
    """
    if baselines is None:
        baselines = BASELINES

    fig = go.Figure()

    all_systems: dict[str, dict[str, Any]] = {
        "openkernel (ours)": our_results,
    }
    # Only include baselines that have cost data.
    for name, metrics in baselines.items():
        if "cost_per_kernel" in metrics and "avg_speedup" in metrics:
            all_systems[name] = metrics

    # If no baselines had cost data, still include them with geomean as proxy.
    if len(all_systems) == 1:
        for name, metrics in baselines.items():
            if "geomean" in metrics:
                all_systems[name] = {
                    "cost_per_kernel": metrics.get("cost_per_kernel", 0.0),
                    "avg_speedup": metrics["geomean"],
                }

    names = list(all_systems.keys())
    costs = [all_systems[n].get("cost_per_kernel", 0.0) for n in names]
    speedups = [all_systems[n].get("avg_speedup", all_systems[n].get("geomean", 0.0)) for n in names]

    # Highlight our system differently.
    colors = []
    sizes = []
    for name in names:
        if name == "openkernel (ours)":
            colors.append("#22c55e")
            sizes.append(16)
        else:
            colors.append("#6366f1")
            sizes.append(11)

    fig.add_trace(
        go.Scatter(
            x=costs,
            y=speedups,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1, color="white"),
            ),
            text=names,
            textposition="top center",
            textfont=dict(size=10, color="#d4d4d8"),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Cost/kernel: $%{x:.3f}<br>"
                "Avg speedup: %{y:.2f}x<br>"
                "<extra></extra>"
            ),
        )
    )

    # --- Pareto frontier ---
    # Sort by cost ascending, then extract Pareto-optimal points.
    indexed = sorted(range(len(costs)), key=lambda i: costs[i])
    pareto_x: list[float] = []
    pareto_y: list[float] = []
    best_speedup = -1.0
    for i in indexed:
        if speedups[i] > best_speedup:
            pareto_x.append(costs[i])
            pareto_y.append(speedups[i])
            best_speedup = speedups[i]

    if len(pareto_x) > 1:
        fig.add_trace(
            go.Scatter(
                x=pareto_x,
                y=pareto_y,
                mode="lines",
                line=dict(color="#22c55e", width=1.5, dash="dash"),
                name="Pareto Frontier",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.update_layout(
        title="Cost-Performance Frontier",
        xaxis_title="Cost per Kernel (USD)",
        yaxis_title="Average Speedup (x)",
        template="plotly_dark",
        showlegend=False,
        height=450,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig
