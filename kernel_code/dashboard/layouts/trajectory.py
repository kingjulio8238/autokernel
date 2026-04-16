"""Trajectory chart layout for the Dash dashboard.

Panel 1: Interactive speedup-over-iterations scatter chart.
  - Green filled circles for "keep"
  - Red X markers for "discard"
  - Yellow triangles for errors/incorrect
  - Bold step line for running best
  - Dashed baseline at 1.0x
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd


_STATUS_COLORS = {
    "keep": "#22c55e",       # green
    "discard": "#ef4444",    # red
    "compile_error": "#dc2626",
    "incorrect": "#eab308",  # yellow
    "error": "#dc2626",
}

_STATUS_SYMBOLS = {
    "keep": "circle",
    "discard": "x",
    "compile_error": "x",
    "incorrect": "triangle-up",
    "error": "x",
}


def create_trajectory_figure(df: pd.DataFrame) -> go.Figure:
    """Create the optimization trajectory Plotly figure.

    Args:
        df: DataFrame with columns: iteration, speedup, decision, intent, cumulative_best.

    Returns:
        A Plotly Figure.
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="Optimization Trajectory (no data)")
        return fig

    # Baseline at 1.0x
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="1.0x (PyTorch ref)",
        annotation_position="bottom left",
    )

    # Running best step line
    if "cumulative_best" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["iteration"],
                y=df["cumulative_best"],
                mode="lines",
                line=dict(color="#3b82f6", width=2, shape="hv"),
                name="Running Best",
                hoverinfo="skip",
            )
        )

    # Individual iteration markers, grouped by decision
    for decision in df["decision"].unique():
        mask = df["decision"] == decision
        subset = df[mask]

        color = _STATUS_COLORS.get(decision, "#6b7280")
        symbol = _STATUS_SYMBOLS.get(decision, "circle")

        fig.add_trace(
            go.Scatter(
                x=subset["iteration"],
                y=subset["speedup"],
                mode="markers",
                marker=dict(
                    color=color,
                    symbol=symbol,
                    size=10,
                    line=dict(width=1, color="white"),
                ),
                name=decision,
                text=subset["intent"],
                hovertemplate=(
                    "<b>Iter %{x}</b><br>"
                    "Speedup: %{y:.2f}x<br>"
                    "Intent: %{text}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Optimization Trajectory",
        xaxis_title="Iteration",
        yaxis_title="Speedup (x)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig
