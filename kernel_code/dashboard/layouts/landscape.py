"""Optimization landscape layout for the Dash dashboard.

Panel 6: 3D scatter plot (Constellation-inspired) with configurable
axes mapping any numeric column to X, Y, Z, and color.

Supports continuous colormap mode (Constellation HSV gradient) in
addition to discrete status-based coloring.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
from dash import html, dcc

from kernel_code.dashboard.theme import STATUS_COLORS, COLORS, FONTS, PLOTLY_THEME
from kernel_code.dashboard.interactions import (
    CONSTELLATION_COLORSCALE,
    CONSTELLATION_COLORSCALE_REVERSED,
)


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
                label.upper(),
                style={
                    "color": COLORS["text_dim"],
                    "fontSize": "10px",
                    "fontFamily": FONTS["mono"],
                    "letterSpacing": "1.5px",
                    "marginRight": "6px",
                },
            ),
            dcc.Dropdown(
                id=axis_id,
                options=options,
                value=default,
                clearable=False,
                style={
                    "width": "160px",
                    "backgroundColor": COLORS["bg_card"],
                    "color": COLORS["text"],
                    "fontSize": "12px",
                    "fontFamily": FONTS["body"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "4px",
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
    color_by: str = "status",
    selected_iteration: int | None = None,
) -> go.Figure:
    """Create a 3D scatter plot of kernel variants.

    Args:
        df: DataFrame with numeric columns and 'decision' for color.
        x_col: Column for X axis.
        y_col: Column for Y axis.
        z_col: Column for Z axis.
        color_by: Column to use for marker color. 'status' uses discrete
            decision-based colors; any numeric column uses the Constellation
            continuous green-gold-red gradient.
        selected_iteration: If set, highlight this iteration with a larger
            marker (Constellation linked-selection pattern).

    Returns:
        A Plotly Figure (Scatter3d).
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(
            title="Optimization Landscape (no data)",
            **PLOTLY_THEME,
        )
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

    # Compute per-point sizes -- enlarge the selected iteration
    sizes = []
    for _, row in df.iterrows():
        if (
            selected_iteration is not None
            and int(row.get("iteration", -1)) == selected_iteration
        ):
            sizes.append(12)  # highlighted
        else:
            sizes.append(6)  # default

    # Determine marker coloring
    marker_kwargs: dict = dict(
        size=sizes,
        line=dict(width=1, color=COLORS["border"]),
        opacity=0.9,
    )

    if color_by == "status" or color_by not in df.columns:
        # Discrete coloring by decision status (original behavior)
        colors = df["decision"].map(STATUS_COLORS).fillna(COLORS["text_dim"])
        marker_kwargs["color"] = colors.tolist()
    else:
        # Continuous coloring by a numeric metric (Constellation pattern)
        color_values = pd.to_numeric(df[color_by], errors="coerce").fillna(0)
        # For speedup, use reversed scale (higher = greener)
        if color_by == "speedup":
            colorscale = CONSTELLATION_COLORSCALE_REVERSED
        else:
            colorscale = CONSTELLATION_COLORSCALE
        marker_kwargs["color"] = color_values.tolist()
        marker_kwargs["colorscale"] = colorscale
        marker_kwargs["showscale"] = True
        marker_kwargs["colorbar"] = dict(
            title=color_by.replace("_", " ").title(),
            tickfont=dict(color=COLORS["text_secondary"], size=10),
            titlefont=dict(color=COLORS["text_secondary"], size=11),
        )

    fig.add_trace(
        go.Scatter3d(
            x=df[x_col],
            y=df[y_col],
            z=df[z_col],
            mode="markers",
            marker=marker_kwargs,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    def _axis_title(col: str) -> str:
        return col.replace("_", " ").title()

    fig.update_layout(
        title="Optimization Landscape",
        **PLOTLY_THEME,
        height=500,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title=_axis_title(x_col),
            yaxis_title=_axis_title(y_col),
            zaxis_title=_axis_title(z_col),
            bgcolor=COLORS["bg"],
            xaxis=dict(
                gridcolor=COLORS["gridline"],
                backgroundcolor=COLORS["bg"],
                color=COLORS["text_secondary"],
            ),
            yaxis=dict(
                gridcolor=COLORS["gridline"],
                backgroundcolor=COLORS["bg"],
                color=COLORS["text_secondary"],
            ),
            zaxis=dict(
                gridcolor=COLORS["gridline"],
                backgroundcolor=COLORS["bg"],
                color=COLORS["text_secondary"],
            ),
        ),
        showlegend=False,
    )

    return fig
