"""Experiment table layout for the Dash dashboard.

Panel 4: Sortable, filterable DataTable with color-coded rows.
"""

from __future__ import annotations

import pandas as pd
from dash import dash_table

from kernel_code.dashboard.theme import COLORS, FONTS


def create_experiment_table(
    df: pd.DataFrame,
    selected_iteration: int | None = None,
) -> dash_table.DataTable:
    """Create the experiment results DataTable.

    Args:
        df: DataFrame with columns: iteration, speedup, decision, intent,
            runtime_us, bandwidth_util, bottleneck_type.
        selected_iteration: If set, highlight the row for this iteration
            (Constellation linked-selection pattern).

    Returns:
        A Dash DataTable component.
    """
    display_columns = [
        {"name": "#", "id": "iteration", "type": "numeric"},
        {"name": "Speedup", "id": "speedup", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "Status", "id": "decision", "type": "text"},
        {"name": "Intent", "id": "intent", "type": "text"},
        {"name": "Runtime (us)", "id": "runtime_us", "type": "numeric", "format": {"specifier": ".1f"}},
        {"name": "BW Util", "id": "bandwidth_util", "type": "numeric", "format": {"specifier": ".0%"}},
        {"name": "Bottleneck", "id": "bottleneck_type", "type": "text"},
    ]

    # Filter to only columns that exist
    available_cols = set(df.columns) if not df.empty else set()
    columns = [c for c in display_columns if c["id"] in available_cols]

    # Build conditional styles for row coloring
    style_data_conditional = [
        # Keep rows: left green border
        {
            "if": {"filter_query": '{decision} = "keep"'},
            "borderLeft": f"3px solid {COLORS['green']}",
        },
        # Discard rows: left red border
        {
            "if": {"filter_query": '{decision} = "discard"'},
            "borderLeft": f"3px solid {COLORS['red']}",
        },
        # Error rows: left red border
        {
            "if": {"filter_query": '{decision} = "error"'},
            "borderLeft": f"3px solid {COLORS['red']}",
        },
    ]

    # Highlight the selected iteration row (linked selection)
    if selected_iteration is not None:
        style_data_conditional.append(
            {
                "if": {
                    "filter_query": f"{{iteration}} = {selected_iteration}",
                },
                "backgroundColor": COLORS["bg_muted"],
                "fontWeight": "600",
                "borderLeft": f"3px solid {COLORS['accent']}",
            }
        )

    return dash_table.DataTable(
        id="experiment-table",
        columns=columns,
        data=df.to_dict("records") if not df.empty else [],
        sort_action="native",
        filter_action="native",
        page_size=25,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLORS["bg"],
            "color": COLORS["text_dim"],
            "fontWeight": "600",
            "fontFamily": FONTS["mono"],
            "fontSize": "10px",
            "textTransform": "uppercase",
            "letterSpacing": "1.5px",
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "10px 8px",
        },
        style_cell={
            "backgroundColor": COLORS["bg_card"],
            "color": COLORS["text"],
            "border": "none",
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "8px",
            "textAlign": "left",
            "fontFamily": FONTS["body"],
            "fontSize": "13px",
        },
        style_data_conditional=style_data_conditional,
    )
