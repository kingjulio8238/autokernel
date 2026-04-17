"""Trajectory chart layout for the Dash dashboard.

Panel 1: Interactive speedup-over-iterations scatter chart.
  - Green filled circles for "keep"
  - Red X markers for "discard"
  - Amber triangles for errors/incorrect
  - Bold step line for running best
  - Dashed baseline at 1.0x
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from kernel_code.dashboard.theme import STATUS_COLORS, COLORS, FONTS, apply_theme


_STATUS_SYMBOLS = {
    "keep": "circle",
    "discard": "x",
    "compile_error": "x",
    "incorrect": "triangle-up",
    "error": "x",
}


def create_trajectory_figure(
    df: pd.DataFrame,
    selected_iteration: int | None = None,
    y_scale: str = "linear",
) -> go.Figure:
    """Create the optimization trajectory Plotly figure.

    Args:
        df: DataFrame with columns: iteration, speedup, decision, intent, cumulative_best.
        selected_iteration: If set, highlight this iteration with a larger marker
            and annotation (Constellation linked-selection pattern).
        y_scale: Y-axis scale type -- 'linear' or 'log'.

    Returns:
        A Plotly Figure.
    """
    fig = go.Figure()

    if df.empty:
        apply_theme(fig, title="Optimization Trajectory (no data)")
        return fig

    # Baseline at 1.0x
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color=COLORS["baseline"],
        annotation_text="1.0x (PyTorch ref)",
        annotation_position="bottom left",
        annotation_font_color=COLORS["text_dim"],
    )

    # Running best step line
    if "cumulative_best" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["iteration"],
                y=df["cumulative_best"],
                mode="lines",
                line=dict(color=COLORS["green"], width=2, shape="hv"),
                name="Running Best",
                hoverinfo="skip",
            )
        )

    # Individual iteration markers, grouped by decision
    for decision in df["decision"].unique():
        mask = df["decision"] == decision
        subset = df[mask]

        color = STATUS_COLORS.get(decision, COLORS["text_dim"])
        symbol = _STATUS_SYMBOLS.get(decision, "circle")

        # Compute per-point sizes -- enlarge the selected iteration
        sizes = []
        for _, row in subset.iterrows():
            if (
                selected_iteration is not None
                and int(row["iteration"]) == selected_iteration
            ):
                sizes.append(18)  # highlighted
            else:
                sizes.append(10)  # default

        fig.add_trace(
            go.Scatter(
                x=subset["iteration"],
                y=subset["speedup"],
                mode="markers",
                marker=dict(
                    color=color,
                    symbol=symbol,
                    size=sizes,
                    line=dict(width=1, color=COLORS["bg_card"]),
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

    # Add annotation for the selected iteration (linked selection)
    if selected_iteration is not None:
        sel_rows = df[df["iteration"] == selected_iteration]
        if not sel_rows.empty:
            sel_row = sel_rows.iloc[0]
            fig.add_annotation(
                x=sel_row["iteration"],
                y=sel_row["speedup"],
                text=f"Iter {selected_iteration} ({sel_row['speedup']:.2f}x)",
                showarrow=True,
                arrowhead=2,
                arrowcolor=COLORS["accent"],
                font=dict(
                    color=COLORS["accent"],
                    size=11,
                    family=FONTS["mono"],
                ),
                bgcolor=COLORS["bg_card"],
                bordercolor=COLORS["border"],
                borderwidth=1,
            )

    apply_theme(
        fig,
        title="Optimization Trajectory",
        xaxis_title="Iteration",
        yaxis_title="Speedup (x)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    # Apply scale toggle
    if y_scale in ("linear", "log"):
        fig.update_yaxes(type=y_scale)

    return fig
