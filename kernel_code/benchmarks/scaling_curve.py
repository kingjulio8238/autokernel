"""Scaling curve chart for KernelBench results.

Chart 3: Line chart showing how fast@1 improves with more iteration budget.
X = iteration budget {5, 10, 20, 50, 100}, Y = fast_1 score.
"""

from __future__ import annotations

import plotly.graph_objects as go


def create_scaling_curve(
    results_at_budgets: dict[int, float],
    *,
    comparisons: dict[str, dict[int, float]] | None = None,
    label: str = "openkernel (ours)",
) -> go.Figure:
    """Create a scaling curve: fast@1 vs iteration budget.

    Args:
        results_at_budgets: Dict mapping iteration budget (e.g. 5, 10, 20, 50,
            100) to the fast@1 score achieved at that budget.
        comparisons: Optional dict mapping system name to the same budget->score
            mapping, for overlay.
        label: Display name for our system.

    Returns:
        A Plotly Figure.
    """
    fig = go.Figure()

    all_systems: dict[str, dict[int, float]] = {label: results_at_budgets}
    if comparisons:
        all_systems.update(comparisons)

    colors = [
        "#22c55e", "#3b82f6", "#a855f7", "#ef4444",
        "#eab308", "#06b6d4", "#f97316", "#ec4899",
    ]
    symbols = [
        "circle", "square", "diamond", "triangle-up",
        "triangle-down", "cross", "star", "hexagon",
    ]

    for idx, (name, budget_scores) in enumerate(all_systems.items()):
        budgets_sorted = sorted(budget_scores.keys())
        scores = [budget_scores[b] for b in budgets_sorted]
        color = colors[idx % len(colors)]
        symbol = symbols[idx % len(symbols)]

        fig.add_trace(
            go.Scatter(
                x=budgets_sorted,
                y=scores,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=9, color=color, symbol=symbol),
                name=name,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Budget: %{x} iters<br>"
                    "fast@1: %{y:.1%}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Scaling Curve: fast@1 vs Iteration Budget",
        xaxis_title="Iteration Budget",
        yaxis_title="fast@1 Score",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05],
        xaxis_type="log",
        xaxis_tickvals=[5, 10, 20, 50, 100],
        xaxis_ticktext=["5", "10", "20", "50", "100"],
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig
