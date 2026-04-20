"""Plotly Dash dashboard server for kernel code.

Serves on localhost:8050 with 10 panels:
  1. Optimization trajectory chart
  2. Roofline model
  3. Resource utilization gauges
  4. Experiment results table
  5. Code diff
  6. Optimization landscape (3D)
  7. Strategy tree (treemap)
  8. Convergence analysis (post-hoc)
  9. Cost efficiency frontier (post-hoc)
  10. Strategy statistics (post-hoc)

Auto-refresh every 5 seconds.

Constellation-inspired interactive features:
  - Linked selection across all panels (click a trajectory point)
  - Dual range filters on any numeric metric
  - Linear/log scale toggles on trajectory and roofline axes
  - Continuous colormap on landscape (color by any metric)
"""

from __future__ import annotations

from dash import Dash, html, dcc, Input, Output, no_update

from kernel_code.dashboard.data import load_session, load_session_metadata
from kernel_code.dashboard.theme import (
    COLORS,
    FONTS,
    GOOGLE_FONTS_LINK,
    apply_theme,
    card_style,
    section_header,
)
from kernel_code.dashboard.interactions import (
    apply_filters,
    create_filter_bar,
    create_scale_toggle,
    create_color_by_dropdown,
    create_interaction_stores,
)
from kernel_code.dashboard.layouts.trajectory import create_trajectory_figure
from kernel_code.dashboard.layouts.experiment_table import create_experiment_table
from kernel_code.dashboard.layouts.roofline import create_roofline_figure
from kernel_code.dashboard.layouts.utilization import create_utilization_figure
from kernel_code.dashboard.layouts.code_diff import create_code_diff_component
from kernel_code.dashboard.layouts.landscape import (
    create_landscape_controls,
    create_landscape_figure,
)
from kernel_code.dashboard.layouts.strategy_tree import create_strategy_tree_figure
from kernel_code.dashboard.layouts.convergence import create_convergence_figure
from kernel_code.dashboard.layouts.cost_efficiency import create_cost_efficiency_figure
from kernel_code.dashboard.layouts.strategy_stats import create_strategy_stats_figure


def create_dash_app(session_id: str) -> Dash:
    """Create and configure the Dash application.

    Args:
        session_id: Session ID to display.

    Returns:
        Configured Dash app instance.
    """
    app = Dash(
        __name__,
        title="kernel code dashboard",
        update_title=None,
    )

    # Load initial data
    try:
        metadata = load_session_metadata(session_id)
    except FileNotFoundError:
        metadata = {"session_id": session_id, "problem": "unknown", "hardware": "", "backend": ""}

    hardware = metadata.get("hardware", "H100") or "H100"

    # Reusable section wrapper with card styling
    def _section(title: str, children: list) -> html.Div:
        return html.Div(
            style=card_style(),
            children=[
                section_header(title),
                *children,
            ],
        )

    app.layout = html.Div(
        style={
            "backgroundColor": COLORS["bg"],
            "color": COLORS["text"],
            "fontFamily": FONTS["body"],
            "minHeight": "100vh",
            "padding": "20px",
        },
        children=[
            # Google Fonts
            GOOGLE_FONTS_LINK,
            # Header
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "20px",
                },
                children=[
                    html.H1(
                        "KERNEL CODE DASHBOARD",
                        style={
                            "color": COLORS["accent"],
                            "margin": "0",
                            "fontFamily": FONTS["mono"],
                            "fontSize": "20px",
                            "letterSpacing": "2px",
                            "textTransform": "uppercase",
                        },
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{metadata.get('hardware', '')} | "
                                f"{metadata.get('backend', '')} | "
                                f"{metadata.get('problem', '')}",
                                style={
                                    "color": COLORS["text_secondary"],
                                    "fontSize": "12px",
                                    "fontFamily": FONTS["mono"],
                                    "letterSpacing": "1px",
                                },
                            ),
                        ],
                    ),
                ],
            ),
            # Session info
            html.Div(
                style={
                    "marginBottom": "15px",
                    "color": COLORS["text_dim"],
                    "fontFamily": FONTS["mono"],
                    "fontSize": "11px",
                    "letterSpacing": "1px",
                },
                children=[
                    html.Span(f"SESSION: {session_id}"),
                ],
            ),
            # --- Filter Bar (Constellation dual-filter pattern) ---
            create_filter_bar(),
            # --- Row 1: Trajectory + Roofline ---
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                children=[
                    html.Div(
                        style={**card_style(), "flex": "1", "minWidth": "0"},
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "marginBottom": "4px",
                                },
                                children=[
                                    section_header("Optimization Trajectory"),
                                    # Scale toggle for trajectory Y axis
                                    create_scale_toggle(
                                        "trajectory-y-scale", "Y Scale", "linear"
                                    ),
                                ],
                            ),
                            dcc.Graph(id="trajectory-chart"),
                        ],
                    ),
                    html.Div(
                        style={**card_style(), "flex": "1", "minWidth": "0"},
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "flexWrap": "wrap",
                                    "marginBottom": "4px",
                                },
                                children=[
                                    section_header("Roofline Model"),
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                        },
                                        children=[
                                            # Scale toggles for roofline axes
                                            create_scale_toggle(
                                                "roofline-x-scale", "X Scale", "log"
                                            ),
                                            create_scale_toggle(
                                                "roofline-y-scale", "Y Scale", "log"
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Graph(id="roofline-chart"),
                        ],
                    ),
                ],
            ),
            # --- Row 2: Utilization + Strategy Tree ---
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                children=[
                    html.Div(
                        style={**card_style(), "flex": "1", "minWidth": "0"},
                        children=[
                            section_header("Resource Utilization"),
                            dcc.Graph(id="utilization-chart"),
                        ],
                    ),
                    html.Div(
                        style={**card_style(), "flex": "1", "minWidth": "0"},
                        children=[
                            section_header("Strategy Tree"),
                            dcc.Graph(id="strategy-tree-chart"),
                        ],
                    ),
                ],
            ),
            # --- Row 3: Code Diff ---
            _section("Code Diff", [html.Div(id="code-diff-container")]),
            # --- Row 4: Optimization Landscape (3D) ---
            _section(
                "Optimization Landscape",
                [
                    html.Div(
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "alignItems": "center",
                        },
                        children=[
                            create_landscape_controls(),
                            # Color-by dropdown (Constellation continuous colormap)
                            create_color_by_dropdown(),
                        ],
                    ),
                    dcc.Graph(id="landscape-chart"),
                ],
            ),
            # --- Row 5: Experiment Table ---
            _section("Experiment Log", [html.Div(id="experiment-table-container")]),
            # --- Row 6: Post-Hoc Analysis ---
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                children=[
                    html.Div(
                        style={**card_style(), "flex": "1", "minWidth": "0"},
                        children=[
                            section_header("Convergence Analysis"),
                            dcc.Graph(id="convergence-chart"),
                        ],
                    ),
                    html.Div(
                        style={**card_style(), "flex": "1", "minWidth": "0"},
                        children=[
                            section_header("Cost Efficiency"),
                            dcc.Graph(id="cost-efficiency-chart"),
                        ],
                    ),
                ],
            ),
            # --- Row 7: Strategy Stats ---
            _section("Strategy Analysis", [dcc.Graph(id="strategy-stats-chart")]),
            # Auto-refresh interval (every 5 seconds)
            dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
            # Stores
            dcc.Store(id="session-id-store", data=session_id),
            dcc.Store(id="hardware-store", data=hardware),
            # Constellation interaction stores (selected-iteration, filter-state)
            *create_interaction_stores(),
        ],
    )

    # ----------------------------------------------------------------
    # Callback: Linked Selection -- trajectory click -> selected-iteration
    # ----------------------------------------------------------------
    @app.callback(
        Output("selected-iteration", "data"),
        Input("trajectory-chart", "clickData"),
        prevent_initial_call=True,
    )
    def on_trajectory_click(click_data):
        """Store the clicked iteration number for cross-panel highlighting."""
        if click_data and click_data.get("points"):
            point = click_data["points"][0]
            iteration = point.get("x")
            if iteration is not None:
                return int(iteration)
        return no_update

    # ----------------------------------------------------------------
    # Callback: Filter State -- filter inputs -> filter-state store
    # ----------------------------------------------------------------
    @app.callback(
        Output("filter-state", "data"),
        [
            Input("filter-metric-1", "value"),
            Input("filter-min-1", "value"),
            Input("filter-max-1", "value"),
            Input("filter-metric-2", "value"),
            Input("filter-min-2", "value"),
            Input("filter-max-2", "value"),
        ],
    )
    def update_filter_state(
        metric1, min1, max1,
        metric2, min2, max2,
    ):
        """Build the filter-state dict from the two filter controls."""
        filters = []
        if metric1:
            filters.append({"metric": metric1, "min": min1, "max": max1})
        if metric2:
            filters.append({"metric": metric2, "min": min2, "max": max2})
        return {"filters": filters}

    # ----------------------------------------------------------------
    # Callback: Refresh all panels (reads filter + selection state)
    # ----------------------------------------------------------------
    @app.callback(
        [
            Output("trajectory-chart", "figure"),
            Output("roofline-chart", "figure"),
            Output("utilization-chart", "figure"),
            Output("strategy-tree-chart", "figure"),
            Output("code-diff-container", "children"),
            Output("experiment-table-container", "children"),
            Output("convergence-chart", "figure"),
            Output("cost-efficiency-chart", "figure"),
            Output("strategy-stats-chart", "figure"),
        ],
        [
            Input("refresh-interval", "n_intervals"),
            Input("session-id-store", "data"),
            Input("hardware-store", "data"),
            Input("selected-iteration", "data"),
            Input("filter-state", "data"),
            Input("trajectory-y-scale", "value"),
            Input("roofline-x-scale", "value"),
            Input("roofline-y-scale", "value"),
        ],
    )
    def update_dashboard(
        n_intervals: int,
        sid: str,
        hw: str,
        selected_iter,
        filter_state,
        traj_y_scale,
        roofline_x_scale,
        roofline_y_scale,
    ):
        """Refresh all dashboard panels with latest session data.

        Reads the shared Constellation stores (selected-iteration, filter-state,
        scale toggles) and passes them through to each panel builder.
        """
        try:
            df = load_session(sid)
        except FileNotFoundError:
            import plotly.graph_objects as go

            empty_fig = go.Figure()
            apply_theme(empty_fig, title="No session data found")
            no_data = html.Div(
                "No data available.",
                style={"color": COLORS["text_dim"], "padding": "20px"},
            )
            return (empty_fig,) * 4 + (no_data, no_data) + (empty_fig,) * 3

        # Apply range filters (Constellation dual-filter pattern)
        filtered_df = apply_filters(df, filter_state)

        # Normalize selected iteration
        sel_iter = int(selected_iter) if selected_iter is not None else None

        # Build panels with interactive state
        trajectory_fig = create_trajectory_figure(
            filtered_df,
            selected_iteration=sel_iter,
            y_scale=traj_y_scale or "linear",
        )
        roofline_fig = create_roofline_figure(
            filtered_df,
            hardware=hw,
            x_scale=roofline_x_scale or "log",
            y_scale=roofline_y_scale or "log",
            selected_iteration=sel_iter,
        )
        utilization_fig = create_utilization_figure(filtered_df)
        strategy_tree_fig = create_strategy_tree_figure(filtered_df)
        code_diff = create_code_diff_component(filtered_df)
        experiment_table = create_experiment_table(
            filtered_df,
            selected_iteration=sel_iter,
        )
        convergence_fig = create_convergence_figure(filtered_df)
        cost_efficiency_fig = create_cost_efficiency_figure(filtered_df)
        strategy_stats_fig = create_strategy_stats_figure(filtered_df)

        return (
            trajectory_fig,
            roofline_fig,
            utilization_fig,
            strategy_tree_fig,
            code_diff,
            experiment_table,
            convergence_fig,
            cost_efficiency_fig,
            strategy_stats_fig,
        )

    # ----------------------------------------------------------------
    # Callback: Landscape (axis + color-by + selection + filters)
    # ----------------------------------------------------------------
    @app.callback(
        Output("landscape-chart", "figure"),
        [
            Input("refresh-interval", "n_intervals"),
            Input("session-id-store", "data"),
            Input("landscape-x", "value"),
            Input("landscape-y", "value"),
            Input("landscape-z", "value"),
            Input("landscape-color-by", "value"),
            Input("selected-iteration", "data"),
            Input("filter-state", "data"),
        ],
    )
    def update_landscape(
        n_intervals: int,
        sid: str,
        x_col: str,
        y_col: str,
        z_col: str,
        color_by: str,
        selected_iter,
        filter_state,
    ):
        """Rebuild the 3D landscape when axes, color-by, or data change."""
        try:
            df = load_session(sid)
        except FileNotFoundError:
            import plotly.graph_objects as go

            empty_fig = go.Figure()
            apply_theme(empty_fig, title="No session data found")
            return empty_fig

        # Apply range filters
        filtered_df = apply_filters(df, filter_state)

        sel_iter = int(selected_iter) if selected_iter is not None else None

        return create_landscape_figure(
            filtered_df,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            color_by=color_by or "status",
            selected_iteration=sel_iter,
        )

    return app


def run_dashboard(session_id: str, port: int = 8050, debug: bool = False) -> None:
    """Convenience function to create and run the dashboard."""
    app = create_dash_app(session_id)
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    import sys

    from kernel_code.mock_data import generate_mock_session

    path = generate_mock_session()
    sid = path.stem
    print(f"Running dashboard for session: {sid}")
    run_dashboard(sid, debug=True)
