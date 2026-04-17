"""Cost efficiency layout for the Dash dashboard.

Panel 9 (post-hoc): Shows cost-performance relationship across iterations.
  - Scatter: X = cumulative cost (USD), Y = speedup achieved
  - Cost-performance frontier highlighted
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd


def create_cost_efficiency_figure(df: pd.DataFrame) -> go.Figure:
    """Create the cost efficiency Plotly figure.

    Args:
        df: DataFrame with columns: iteration, speedup, cost, cumulative_best.
            ``cost`` is the per-iteration cost in USD.  If ``cumulative_cost``
            is not present it is computed from ``cost``.

    Returns:
        A Plotly Figure showing the cost-performance frontier.
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="Cost Efficiency (no data)")
        return fig

    df = df.copy()

    # Compute cumulative cost if needed.
    if "cumulative_cost" not in df.columns:
        df["cumulative_cost"] = df["cost"].cumsum()

    # Ensure cumulative_best exists.
    if "cumulative_best" not in df.columns:
        df["cumulative_best"] = df["speedup"].cummax()

    # --- Scatter of all iterations ---
    fig.add_trace(
        go.Scatter(
            x=df["cumulative_cost"],
            y=df["speedup"],
            mode="markers",
            marker=dict(
                size=8,
                color=df["iteration"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Iteration"),
                line=dict(width=0.5, color="white"),
            ),
            name="All Iterations",
            text=df["iteration"],
            hovertemplate=(
                "<b>Iter %{text}</b><br>"
                "Cumulative cost: $%{x:.3f}<br>"
                "Speedup: %{y:.2f}x<br>"
                "<extra></extra>"
            ),
        )
    )

    # --- Cost-performance frontier (Pareto front) ---
    # Points where cumulative_best improved (i.e., new best found).
    frontier = df[df["cumulative_best"] == df["speedup"]].copy()
    if not frontier.empty:
        # Sort by cost for proper line drawing.
        frontier = frontier.sort_values("cumulative_cost")
        fig.add_trace(
            go.Scatter(
                x=frontier["cumulative_cost"],
                y=frontier["cumulative_best"],
                mode="lines+markers",
                line=dict(color="#22c55e", width=2, shape="hv"),
                marker=dict(size=10, color="#22c55e", symbol="diamond"),
                name="Cost-Performance Frontier",
                hovertemplate=(
                    "<b>New best</b><br>"
                    "Cost: $%{x:.3f}<br>"
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
        title="Cost Efficiency",
        xaxis_title="Cumulative Cost (USD)",
        yaxis_title="Speedup (x)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig
