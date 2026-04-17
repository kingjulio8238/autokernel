"""Optimization landscape layout for the Dash dashboard.

Panel 6: 3D scatter plot (Constellation-inspired) with configurable
axes mapping any numeric column to X, Y, Z, and color.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
from dash import html, dcc


_STATUS_COLORS = {
    "keep": "#22c55e",
    "discard": "#ef4444",
    "error": "#dc2626",
    "incorrect": "#eab308",
    "compile_error": "#dc2626",
}

# Columns eligible for axis mapping
_AXIS_CANDIDATES = [
    "iteration",
    "speedup",
    "runtime_us",
    "bandwidth_util",
    "compute_util",
    "cache_efficiency",
    "occupancy",
    "cost_estimate",
    "cumulative_best",
]


def _build_axis_dropdown(axis_id: str, label: str, default: str) -> html.Div:
    """Build a labeled dropdown for axis selection."""
    options = [{"label": c.replace("_", " ").title(), "value": c} for c in _AXIS_CANDIDATES]
    return html.Div(
        style={"display": "inline-block", "marginRight": "16px", "marginBottom": "8px"},
        children=[
            html.Label(
                label,
                style={"color": "#94a3b8", "fontSize": "12px", "marginRight": "6px"},
            ),
            dcc.Dropdown(
                id=axis_id,
                options=options,
                value=default,
                clearable=False,
                style={
                    "width": "160px",
                    "backgroundColor": "#1e293b",
                    "color": "#e2e8f0",
                    "fontSize": "12px",
                },
            ),
        ],
    )


def create_landscape_controls() -> html.Div:
    """Return the dropdown controls for the landscape panel.

    These are rendered above the graph; Dash callbacks use their values
    to rebuild the figure on change.
    """
    return html.Div(
        style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"},
        children=[
            _build_axis_dropdown("landscape-x", "X Axis:", "iteration"),
            _build_axis_dropdown("landscape-y", "Y Axis:", "speedup"),
            _build_axis_dropdown("landscape-z", "Z Axis:", "cost_estimate"),
        ],
    )


def create_landscape_figure(
    df: pd.DataFrame,
    x_col: str = "iteration",
    y_col: str = "speedup",
    z_col: str = "cost_estimate",
) -> go.Figure:
    """Create a 3D scatter plot of kernel variants.

    Args:
        df: DataFrame with numeric columns and 'decision' for color.
        x_col: Column for X axis.
        y_col: Column for Y axis.
        z_col: Column for Z axis.

    Returns:
        A Plotly Figure (Scatter3d).
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="Optimization Landscape (no data)", template="plotly_dark")
        return fig

    # Ensure requested columns exist, fallback to defaults
    for col, fallback in [(x_col, "iteration"), (y_col, "speedup"), (z_col, "cost_estimate")]:
        if col not in df.columns:
            if fallback in df.columns:
                if col == x_col:
                    x_col = fallback
                elif col == y_col:
                    y_col = fallback
                else:
                    z_col = fallback

    # Color by decision status
    colors = df["decision"].map(_STATUS_COLORS).fillna("#6b7280")

    # Build hover text
    hover_texts = []
    for _, row in df.iterrows():
        hover_texts.append(
            f"Iter {int(row.get('iteration', 0))}<br>"
            f"Speedup: {row.get('speedup', 0):.2f}x<br>"
            f"Status: {row.get('decision', '')}<br>"
            f"Intent: {row.get('intent', '')}<br>"
            f"BW Util: {row.get('bandwidth_util', 0):.0%}<br>"
            f"Compute Util: {row.get('compute_util', 0):.0%}"
        )

    fig.add_trace(
        go.Scatter3d(
            x=df[x_col],
            y=df[y_col],
            z=df[z_col],
            mode="markers",
            marker=dict(
                size=6,
                color=colors.tolist(),
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
                opacity=0.9,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    def _axis_title(col: str) -> str:
        return col.replace("_", " ").title()

    fig.update_layout(
        title="Optimization Landscape",
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title=_axis_title(x_col),
            yaxis_title=_axis_title(y_col),
            zaxis_title=_axis_title(z_col),
            bgcolor="#0f172a",
        ),
        showlegend=False,
    )

    return fig
