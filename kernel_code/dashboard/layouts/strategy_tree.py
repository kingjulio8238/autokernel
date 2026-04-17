"""Strategy tree layout for the Dash dashboard.

Panel 7: Treemap visualization of the optimization intent tree.
Color by status, size by speedup achieved.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from kernel_code.dashboard.theme import STATUS_COLORS, COLORS, FONTS, PLOTLY_THEME


def create_strategy_tree_figure(df: pd.DataFrame) -> go.Figure:
    """Create a treemap of optimization intents.

    Each iteration is a leaf node, grouped by decision status.
    Size encodes speedup (clamped to a minimum so failed attempts are visible).
    Color encodes decision status.

    Args:
        df: DataFrame with columns: iteration, intent, decision, speedup.

    Returns:
        A Plotly Figure (Treemap).
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="Strategy Tree (no data)", **PLOTLY_THEME)
        return fig

    # Build treemap data: root -> decision group -> individual iteration
    ids = []
    labels = []
    parents = []
    values = []
    colors = []
    hover_texts = []

    root_label = "All Strategies"
    ids.append(root_label)
    labels.append(root_label)
    parents.append("")
    values.append(0)
    colors.append(COLORS["bg_muted"])
    hover_texts.append("")

    # Group nodes by decision
    decision_groups = df["decision"].unique()
    for group in decision_groups:
        group_id = f"group-{group}"
        group_label = group.replace("_", " ").title()
        group_count = int((df["decision"] == group).sum())
        ids.append(group_id)
        labels.append(f"{group_label} ({group_count})")
        parents.append(root_label)
        values.append(0)
        colors.append(STATUS_COLORS.get(group, COLORS["text_dim"]))
        hover_texts.append(f"{group_label}: {group_count} iterations")

    # Leaf nodes: individual iterations
    for _, row in df.iterrows():
        iteration = int(row.get("iteration", 0))
        intent = str(row.get("intent", ""))
        decision = row.get("decision", "discard")
        speedup = float(row.get("speedup", 0.0))

        # Clamp size so zero-speedup entries are still visible
        size_val = max(speedup, 0.2)

        node_id = f"iter-{iteration}"
        parent_id = f"group-{decision}"

        # Truncate long intents for the label
        short_intent = intent[:30] + "..." if len(intent) > 30 else intent

        ids.append(node_id)
        labels.append(f"#{iteration}: {short_intent}")
        parents.append(parent_id)
        values.append(size_val)
        colors.append(STATUS_COLORS.get(decision, COLORS["text_dim"]))
        hover_texts.append(
            f"<b>Iter {iteration}</b><br>"
            f"Intent: {intent}<br>"
            f"Speedup: {speedup:.2f}x<br>"
            f"Status: {decision}"
        )

    fig.add_trace(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors,
                line=dict(width=1, color=COLORS["bg_card"]),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            textfont=dict(
                color=COLORS["bg_card"],
                size=11,
                family=FONTS["mono"],
            ),
            branchvalues="remainder",
        )
    )

    fig.update_layout(
        title="Strategy Tree",
        **PLOTLY_THEME,
        height=400,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig
