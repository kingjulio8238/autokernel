"""Convergence analysis layout for the Dash dashboard.

Panel 8 (post-hoc): Shows how speedup accumulated over iterations.
  - Line chart: cumulative best speedup over iteration
  - Bar overlay: per-iteration delta (improvement gained each step)
  - Annotation marking where 90% of final speedup was achieved
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from kernel_code.dashboard.theme import COLORS, apply_theme


def create_convergence_figure(df: pd.DataFrame) -> go.Figure:
    """Create the convergence analysis Plotly figure.

    Args:
        df: DataFrame with columns: iteration, speedup, cumulative_best.
            ``cumulative_best`` is the running maximum speedup up to each
            iteration.  If missing it is derived from ``speedup``.

    Returns:
        A Plotly Figure with cumulative best line and per-iteration delta bars.
    """
    fig = go.Figure()

    if df.empty:
        apply_theme(fig, title="Convergence Analysis (no data)")
        return fig

    # Ensure cumulative_best exists.
    if "cumulative_best" not in df.columns:
        df = df.copy()
        df["cumulative_best"] = df["speedup"].cummax()

    # Per-iteration delta: how much the cumulative best improved.
    deltas = df["cumulative_best"].diff().fillna(df["cumulative_best"].iloc[0] - 1.0)
    # Clamp negative deltas (shouldn't happen but be safe).
    deltas = deltas.clip(lower=0.0)

    # --- Bar trace: per-iteration improvement delta ---
    fig.add_trace(
        go.Bar(
            x=df["iteration"],
            y=deltas,
            name="Per-Iteration Delta",
            marker_color=f"rgba(26, 26, 26, 0.15)",  # accent at low opacity
            yaxis="y2",
            hovertemplate=(
                "<b>Iter %{x}</b><br>"
                "Delta: +%{y:.3f}x<br>"
                "<extra></extra>"
            ),
        )
    )

    # --- Line trace: cumulative best speedup ---
    fig.add_trace(
        go.Scatter(
            x=df["iteration"],
            y=df["cumulative_best"],
            mode="lines+markers",
            line=dict(color=COLORS["green"], width=2),
            marker=dict(size=5, color=COLORS["green"]),
            name="Cumulative Best",
            hovertemplate=(
                "<b>Iter %{x}</b><br>"
                "Best speedup: %{y:.2f}x<br>"
                "<extra></extra>"
            ),
        )
    )

    # --- Annotate 90% of final speedup ---
    final_speedup = df["cumulative_best"].iloc[-1]
    baseline = 1.0  # PyTorch reference
    total_gain = final_speedup - baseline
    if total_gain > 0:
        threshold = baseline + 0.9 * total_gain
        reached = df[df["cumulative_best"] >= threshold]
        if not reached.empty:
            iter_90 = reached["iteration"].iloc[0]
            val_90 = reached["cumulative_best"].iloc[0]
            fig.add_vline(
                x=iter_90,
                line_dash="dot",
                line_color=COLORS["text_secondary"],
                annotation_text=f"90% of gain (iter {iter_90})",
                annotation_position="top right",
                annotation_font_color=COLORS["text_secondary"],
            )
            fig.add_trace(
                go.Scatter(
                    x=[iter_90],
                    y=[val_90],
                    mode="markers",
                    marker=dict(size=12, color=COLORS["text_secondary"], symbol="star"),
                    name="90% Threshold",
                    hovertemplate=(
                        f"<b>90% of final gain</b><br>"
                        f"Iter {iter_90}, {val_90:.2f}x<br>"
                        "<extra></extra>"
                    ),
                )
            )

    # Baseline at 1.0x
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color=COLORS["baseline"],
        annotation_text="1.0x (baseline)",
        annotation_position="bottom left",
        annotation_font_color=COLORS["text_dim"],
    )

    apply_theme(
        fig,
        title="Convergence Analysis",
        xaxis_title="Iteration",
        yaxis_title="Cumulative Best Speedup (x)",
        yaxis2=dict(
            title="Per-Iteration Delta",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
            tickfont=dict(color=COLORS["text_secondary"]),
            title_font=dict(color=COLORS["text_secondary"]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=60, t=60, b=40),
        barmode="overlay",
    )

    return fig
