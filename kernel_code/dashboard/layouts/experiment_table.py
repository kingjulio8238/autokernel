"""Experiment table layout for the Dash dashboard.

Panel 4: Sortable, filterable DataTable with color-coded rows.
"""

from __future__ import annotations

import pandas as pd
from dash import dash_table


_DECISION_COLORS = {
    "keep": {"backgroundColor": "#14532d", "color": "#86efac"},
    "discard": {"backgroundColor": "#450a0a", "color": "#fca5a5"},
    "error": {"backgroundColor": "#450a0a", "color": "#fca5a5"},
}


def create_experiment_table(df: pd.DataFrame) -> dash_table.DataTable:
    """Create the experiment results DataTable.

    Args:
        df: DataFrame with columns: iteration, speedup, decision, intent,
            runtime_us, bandwidth_util, bottleneck_type.

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
    style_data_conditional = []
    for decision, colors in _DECISION_COLORS.items():
        style_data_conditional.append(
            {
                "if": {
                    "filter_query": f'{{decision}} = "{decision}"',
                },
                **colors,
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
            "backgroundColor": "#1e293b",
            "color": "#e2e8f0",
            "fontWeight": "bold",
            "borderBottom": "2px solid #475569",
        },
        style_cell={
            "backgroundColor": "#0f172a",
            "color": "#cbd5e1",
            "border": "1px solid #1e293b",
            "padding": "8px",
            "textAlign": "left",
            "fontFamily": "monospace",
        },
        style_data_conditional=style_data_conditional,
    )
