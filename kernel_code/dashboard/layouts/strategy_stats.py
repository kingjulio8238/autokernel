"""Strategy effectiveness layout for the Dash dashboard.

Panel 10 (post-hoc): Grouped bar chart comparing strategies.
  - One group per strategy
  - Bars show average speedup
  - Color by problem type if the column is available
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from kernel_code.dashboard.theme import COLORS, apply_theme


# Warm, muted palette for problem-type groups (avoids neon)
_GROUP_COLORS = [
    COLORS["accent"],       # #1a1a1a
    COLORS["green"],        # #276749
    COLORS["red"],          # #c53030
    "#b7791f",              # warm amber
    COLORS["text_secondary"],  # #6b6b6b
    "#5a67d8",              # muted indigo
]


def create_strategy_stats_figure(df: pd.DataFrame) -> go.Figure:
    """Create the strategy effectiveness Plotly figure.

    Args:
        df: DataFrame with columns: strategy, speedup.
            Optional columns: problem_type (for colour breakdown).

    Returns:
        A Plotly Figure with grouped bars per strategy.
    """
    fig = go.Figure()

    if df.empty:
        apply_theme(fig, title="Strategy Effectiveness (no data)")
        return fig

    has_problem_type = "problem_type" in df.columns

    if has_problem_type:
        # Grouped bar: one colour per problem type within each strategy group.
        problem_types = sorted(df["problem_type"].dropna().unique())
        for idx, ptype in enumerate(problem_types):
            subset = df[df["problem_type"] == ptype]
            avg_by_strategy = subset.groupby("strategy")["speedup"].mean()
            color = _GROUP_COLORS[idx % len(_GROUP_COLORS)]
            fig.add_trace(
                go.Bar(
                    x=avg_by_strategy.index.tolist(),
                    y=avg_by_strategy.values.tolist(),
                    name=str(ptype),
                    marker_color=color,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        f"Type: {ptype}<br>"
                        "Avg speedup: %{y:.2f}x<br>"
                        "<extra></extra>"
                    ),
                )
            )
    else:
        # Single bar per strategy.
        avg_by_strategy = df.groupby("strategy")["speedup"].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=avg_by_strategy.index.tolist(),
                y=avg_by_strategy.values.tolist(),
                name="Avg Speedup",
                marker_color=COLORS["accent"],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Avg speedup: %{y:.2f}x<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Count annotations (number of samples per strategy).
    counts = df.groupby("strategy")["speedup"].count()
    for strategy, count in counts.items():
        avg = df[df["strategy"] == strategy]["speedup"].mean()
        fig.add_annotation(
            x=strategy,
            y=avg,
            text=f"n={count}",
            showarrow=False,
            yshift=12,
            font=dict(size=10, color=COLORS["text_dim"]),
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
        title="Strategy Effectiveness",
        xaxis_title="Strategy",
        yaxis_title="Average Speedup (x)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=20, t=60, b=80),
        xaxis_tickangle=-30,
    )

    return fig
