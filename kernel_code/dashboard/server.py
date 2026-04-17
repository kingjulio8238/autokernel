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
"""

from __future__ import annotations

from dash import Dash, html, dcc, Input, Output

from kernel_code.dashboard.data import load_session, load_session_metadata
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

    # Reusable section wrapper
    def _section(title: str, children: list) -> html.Div:
        return html.Div(
            style={"marginBottom": "24px"},
            children=[
                html.H3(title, style={"color": "#e2e8f0", "marginBottom": "8px"}),
                *children,
            ],
        )

    app.layout = html.Div(
        style={
            "backgroundColor": "#0f172a",
            "color": "#e2e8f0",
            "fontFamily": "monospace",
            "minHeight": "100vh",
            "padding": "20px",
        },
        children=[
            # Header
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"},
                children=[
                    html.H1(
                        "kernel code dashboard",
                        style={"color": "#3b82f6", "margin": "0"},
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{metadata.get('hardware', '')} | "
                                f"{metadata.get('backend', '')} | "
                                f"{metadata.get('problem', '')}",
                                style={"color": "#94a3b8", "fontSize": "14px"},
                            ),
                        ],
                    ),
                ],
            ),
            # Session info
            html.Div(
                style={"marginBottom": "15px", "color": "#64748b"},
                children=[
                    html.Span(f"Session: {session_id}"),
                ],
            ),
            # --- Row 1: Trajectory + Roofline ---
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                children=[
                    html.Div(
                        style={"flex": "1", "minWidth": "0"},
                        children=[dcc.Graph(id="trajectory-chart")],
                    ),
                    html.Div(
                        style={"flex": "1", "minWidth": "0"},
                        children=[dcc.Graph(id="roofline-chart")],
                    ),
                ],
            ),
            # --- Row 2: Utilization + Strategy Tree ---
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                children=[
                    html.Div(
                        style={"flex": "1", "minWidth": "0"},
                        children=[dcc.Graph(id="utilization-chart")],
                    ),
                    html.Div(
                        style={"flex": "1", "minWidth": "0"},
                        children=[dcc.Graph(id="strategy-tree-chart")],
                    ),
                ],
            ),
            # --- Row 3: Code Diff ---
            _section("Code Diff", [html.Div(id="code-diff-container")]),
            # --- Row 4: Optimization Landscape (3D) ---
            _section(
                "Optimization Landscape",
                [
                    create_landscape_controls(),
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
                        style={"flex": "1", "minWidth": "0"},
                        children=[dcc.Graph(id="convergence-chart")],
                    ),
                    html.Div(
                        style={"flex": "1", "minWidth": "0"},
                        children=[dcc.Graph(id="cost-efficiency-chart")],
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
        ],
    )

    # ---- Callback: refresh all panels ----
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
        ],
    )
    def update_dashboard(n_intervals: int, sid: str, hw: str):
        """Refresh all 10 dashboard panels with latest session data."""
        try:
            df = load_session(sid)
        except FileNotFoundError:
            import plotly.graph_objects as go

            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No session data found",
                template="plotly_dark",
            )
            no_data = html.Div("No data available.", style={"color": "#64748b"})
            return (empty_fig,) * 4 + (no_data, no_data) + (empty_fig,) * 3

        trajectory_fig = create_trajectory_figure(df)
        roofline_fig = create_roofline_figure(df, hardware=hw)
        utilization_fig = create_utilization_figure(df)
        strategy_tree_fig = create_strategy_tree_figure(df)
        code_diff = create_code_diff_component(df)
        experiment_table = create_experiment_table(df)
        convergence_fig = create_convergence_figure(df)
        cost_efficiency_fig = create_cost_efficiency_figure(df)
        strategy_stats_fig = create_strategy_stats_figure(df)

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

    # ---- Callback: landscape axis changes ----
    @app.callback(
        Output("landscape-chart", "figure"),
        [
            Input("refresh-interval", "n_intervals"),
            Input("session-id-store", "data"),
            Input("landscape-x", "value"),
            Input("landscape-y", "value"),
            Input("landscape-z", "value"),
        ],
    )
    def update_landscape(n_intervals: int, sid: str, x_col: str, y_col: str, z_col: str):
        """Rebuild the 3D landscape when axes or data change."""
        try:
            df = load_session(sid)
        except FileNotFoundError:
            import plotly.graph_objects as go

            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No session data found",
                template="plotly_dark",
            )
            return empty_fig

        return create_landscape_figure(df, x_col=x_col, y_col=y_col, z_col=z_col)

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
