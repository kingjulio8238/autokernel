"""Resource utilization layout for the Dash dashboard.

Panel 3: Horizontal bar gauges for bandwidth_util, compute_util,
cache_efficiency, and occupancy.  Color-coded green/amber/red.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from kernel_code.dashboard.theme import COLORS, apply_theme


_METRICS = [
    {"col": "bandwidth_util", "label": "Bandwidth Util"},
    {"col": "compute_util", "label": "Compute Util"},
    {"col": "cache_efficiency", "label": "Cache Efficiency"},
    {"col": "occupancy", "label": "Occupancy"},
]


def _gauge_color(value: float) -> str:
    """Return green / amber / red based on utilization threshold."""
    if value >= 0.8:
        return COLORS["green"]
    if value >= 0.5:
        return "#b7791f"  # warm amber
    return COLORS["red"]


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
        apply_theme(fig, title="Resource Utilization (no data)")
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
            textfont=dict(color=COLORS["bg_card"], size=14),
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
            marker=dict(color=COLORS["bg_muted"], line=dict(width=0)),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    apply_theme(
        fig,
        title="Resource Utilization",
        barmode="stack",
        height=280,
        margin=dict(l=120, r=20, t=60, b=20),
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            showgrid=False,
            title="",
            gridcolor=COLORS["gridline"],
            linecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            showgrid=False,
            title="",
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        showlegend=False,
    )

    return fig
