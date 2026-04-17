"""Constellation-inspired interactive features for the dashboard.

Provides shared state management for:
  1. Linked selection -- clicking a point highlights it across all panels
  2. Range filters -- filter experiments by any metric with dual filter controls
  3. Scale toggles -- linear/log axis scale switching
  4. Continuous colormap -- color landscape points by any numeric metric

Inspired by PufferLib Constellation's architecture:
  - Shared settings across figures (linked views)
  - Two-filter system on any key
  - Linear/log/logit scale toggles
  - Continuous HSV color gradient
"""

from __future__ import annotations

from dash import html, dcc

import pandas as pd

from kernel_code.dashboard.theme import COLORS, FONTS


# ---------------------------------------------------------------------------
# Constellation-inspired green -> gold -> red colorscale
# Mirrors Constellation's ColorFromHSV(120*(1+h), 0.8, 0.15) gradient
# ---------------------------------------------------------------------------
CONSTELLATION_COLORSCALE = [
    [0.0, "#276749"],   # green (good)
    [0.5, "#d69e2e"],   # gold (mid)
    [1.0, "#c53030"],   # red (bad)
]

# Reversed version: low = red (bad), high = green (good) -- for speedup
CONSTELLATION_COLORSCALE_REVERSED = [
    [0.0, "#c53030"],   # red (bad / low speedup)
    [0.5, "#d69e2e"],   # gold (mid)
    [1.0, "#276749"],   # green (good / high speedup)
]

# Metrics eligible for filtering and color-by
FILTERABLE_METRICS = [
    "speedup",
    "runtime_us",
    "cost_estimate",
    "iteration",
    "bandwidth_util",
    "compute_util",
    "cache_efficiency",
    "occupancy",
]

# Scale options for axis toggles
SCALE_OPTIONS = [
    {"label": "Linear", "value": "linear"},
    {"label": "Log", "value": "log"},
]


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------

def _dropdown_style() -> dict:
    """Consistent dropdown styling using the shared theme."""
    return {
        "width": "140px",
        "backgroundColor": COLORS["bg"],
        "color": COLORS["text"],
        "fontSize": "11px",
        "fontFamily": FONTS["mono"],
    }


def _input_style() -> dict:
    """Consistent numeric input styling using the shared theme."""
    return {
        "width": "72px",
        "backgroundColor": COLORS["bg"],
        "color": COLORS["text"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "4px",
        "padding": "4px 8px",
        "fontSize": "11px",
        "fontFamily": FONTS["mono"],
    }


def _label_style() -> dict:
    return {
        "color": COLORS["text_secondary"],
        "fontSize": "10px",
        "fontFamily": FONTS["mono"],
        "textTransform": "uppercase",
        "letterSpacing": "1px",
        "marginRight": "4px",
    }


def create_filter_control(index: int) -> html.Div:
    """Build a single filter control: metric dropdown + min + max inputs.

    Args:
        index: 1 or 2 (for dual-filter system).
    """
    metric_options = [
        {"label": m.replace("_", " ").title(), "value": m}
        for m in FILTERABLE_METRICS
    ]
    # Add a "None" option to disable the filter
    metric_options.insert(0, {"label": "-- none --", "value": ""})

    return html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "8px",
            "padding": "4px 0",
        },
        children=[
            html.Label(f"Filter {index}:", style=_label_style()),
            dcc.Dropdown(
                id=f"filter-metric-{index}",
                options=metric_options,
                value="",
                clearable=False,
                style=_dropdown_style(),
                placeholder="Select metric",
            ),
            html.Label("Min:", style=_label_style()),
            dcc.Input(
                id=f"filter-min-{index}",
                type="number",
                placeholder="min",
                style=_input_style(),
                debounce=True,
            ),
            html.Label("Max:", style=_label_style()),
            dcc.Input(
                id=f"filter-max-{index}",
                type="number",
                placeholder="max",
                style=_input_style(),
                debounce=True,
            ),
        ],
    )


def create_filter_bar() -> html.Div:
    """Build the dual-filter bar (Constellation two-filter pattern)."""
    return html.Div(
        id="filter-bar",
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "16px",
            "alignItems": "center",
            "padding": "10px 16px",
            "backgroundColor": COLORS["bg_card"],
            "borderRadius": "4px",
            "marginBottom": "16px",
            "border": f"1px solid {COLORS['border']}",
        },
        children=[
            html.Span(
                "FILTERS",
                style={
                    "color": COLORS["accent"],
                    "fontWeight": "600",
                    "fontSize": "10px",
                    "fontFamily": FONTS["mono"],
                    "letterSpacing": "1.5px",
                    "marginRight": "8px",
                },
            ),
            create_filter_control(1),
            html.Div(
                style={
                    "borderLeft": f"1px solid {COLORS['border']}",
                    "height": "28px",
                    "margin": "0 4px",
                },
            ),
            create_filter_control(2),
        ],
    )


def create_scale_toggle(axis_id: str, label: str, default: str = "linear") -> html.Div:
    """Build a scale toggle dropdown (linear / log).

    Args:
        axis_id: Unique ID for the dropdown (e.g. 'trajectory-y-scale').
        label: Display label (e.g. 'Y Scale').
        default: Default scale value.
    """
    return html.Div(
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "4px",
            "marginLeft": "12px",
        },
        children=[
            html.Label(label + ":", style=_label_style()),
            dcc.Dropdown(
                id=axis_id,
                options=SCALE_OPTIONS,
                value=default,
                clearable=False,
                style={
                    "width": "90px",
                    "backgroundColor": COLORS["bg"],
                    "color": COLORS["text"],
                    "fontSize": "11px",
                    "fontFamily": FONTS["mono"],
                },
            ),
        ],
    )


def create_color_by_dropdown() -> html.Div:
    """Build the 'color by' dropdown for the landscape panel."""
    options = [
        {"label": "Status (discrete)", "value": "status"},
        {"label": "Speedup", "value": "speedup"},
        {"label": "Cost", "value": "cost_estimate"},
        {"label": "Iteration", "value": "iteration"},
        {"label": "BW Util", "value": "bandwidth_util"},
        {"label": "Compute Util", "value": "compute_util"},
    ]
    return html.Div(
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "4px",
            "marginLeft": "12px",
        },
        children=[
            html.Label("Color By:", style=_label_style()),
            dcc.Dropdown(
                id="landscape-color-by",
                options=options,
                value="status",
                clearable=False,
                style=_dropdown_style(),
            ),
        ],
    )


def create_interaction_stores() -> list:
    """Return dcc.Store components for shared interactive state."""
    return [
        dcc.Store(id="selected-iteration", data=None),
        dcc.Store(id="filter-state", data={"filters": []}),
    ]


# ---------------------------------------------------------------------------
# Data filtering
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filter_state: dict | None) -> pd.DataFrame:
    """Apply range filters to a DataFrame.

    Args:
        df: Source DataFrame.
        filter_state: Dict with 'filters' list, each entry:
            {"metric": str, "min": float|None, "max": float|None}

    Returns:
        Filtered DataFrame (or original if no filters active).
    """
    if df.empty or not filter_state:
        return df

    filters = filter_state.get("filters", [])
    mask = pd.Series(True, index=df.index)

    for f in filters:
        metric = f.get("metric")
        if not metric or metric not in df.columns:
            continue

        col = pd.to_numeric(df[metric], errors="coerce")

        f_min = f.get("min")
        if f_min is not None:
            try:
                mask = mask & (col >= float(f_min))
            except (ValueError, TypeError):
                pass

        f_max = f.get("max")
        if f_max is not None:
            try:
                mask = mask & (col <= float(f_max))
            except (ValueError, TypeError):
                pass

    return df[mask].copy()
