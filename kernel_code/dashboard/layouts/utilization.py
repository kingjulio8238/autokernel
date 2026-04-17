"""Resource utilization layout for the Dash dashboard.

Panel 3: Horizontal bar gauges for bandwidth_util, compute_util,
cache_efficiency, and occupancy.  Color-coded green/yellow/red.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd


_METRICS = [
    {"col": "bandwidth_util", "label": "Bandwidth Util"},
    {"col": "compute_util", "label": "Compute Util"},
    {"col": "cache_efficiency", "label": "Cache Efficiency"},
    {"col": "occupancy", "label": "Occupancy"},
]


def _gauge_color(value: float) -> str:
    """Return green / yellow / red based on utilization threshold."""
    if value >= 0.8:
        return "#22c55e"  # green
    if value >= 0.5:
        return "#eab308"  # yellow
    return "#ef4444"  # red


def create_utilization_figure(df: pd.DataFrame) -> go.Figure:
    """Create a resource utilization gauge figure.

    Uses the latest iteration's values to render four horizontal bar gauges
    via a single horizontal bar chart with color-coded bars.

    Args:
        df: DataFrame with columns: bandwidth_util, compute_util,
            cache_efficiency, occupancy (all 0-1 floats).

    Returns:
        A Plotly Figure.
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="Resource Utilization (no data)", template="plotly_dark")
        return fig

    # Use the latest iteration
    latest = df.iloc[-1]

    labels = []
    values = []
    colors = []
    texts = []

    for metric in _METRICS:
        val = float(latest.get(metric["col"], 0.0) or 0.0)
        labels.append(metric["label"])
        values.append(val)
        colors.append(_gauge_color(val))
        texts.append(f"{val:.0%}")

    # Reverse so first metric is at top
    labels = labels[::-1]
    values = values[::-1]
    colors = colors[::-1]
    texts = texts[::-1]

    # Filled portion
    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=texts,
            textposition="inside",
            textfont=dict(color="white", size=14),
            name="Utilization",
            hovertemplate="<b>%{y}</b>: %{x:.0%}<extra></extra>",
        )
    )

    # Background (remaining to 1.0) for gauge effect
    remaining = [1.0 - v for v in values]
    fig.add_trace(
        go.Bar(
            y=labels,
            x=remaining,
            orientation="h",
            marker=dict(color="rgba(255,255,255,0.05)", line=dict(width=0)),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Resource Utilization",
        barmode="stack",
        template="plotly_dark",
        height=280,
        margin=dict(l=120, r=20, t=60, b=20),
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            showgrid=False,
            title="",
        ),
        yaxis=dict(showgrid=False, title=""),
        showlegend=False,
    )

    return fig
