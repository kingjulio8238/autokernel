"""Plotly Dash dashboard server for kernel code.

Serves on localhost:8050 with:
  - Optimization trajectory chart (Panel 1)
  - Experiment results table (Panel 4)
  - Auto-refresh every 5 seconds
"""

from __future__ import annotations

from dash import Dash, html, dcc, Input, Output

from kernel_code.dashboard.data import load_session, load_session_metadata
from kernel_code.dashboard.layouts.trajectory import create_trajectory_figure
from kernel_code.dashboard.layouts.experiment_table import create_experiment_table


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
            # Trajectory chart
            dcc.Graph(id="trajectory-chart"),
            html.Br(),
            # Experiment table
            html.H3("Experiment Log", style={"color": "#e2e8f0"}),
            html.Div(id="experiment-table-container"),
            # Auto-refresh interval (every 5 seconds)
            dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
            # Store the session_id for callbacks
            dcc.Store(id="session-id-store", data=session_id),
        ],
    )

    @app.callback(
        [
            Output("trajectory-chart", "figure"),
            Output("experiment-table-container", "children"),
        ],
        [
            Input("refresh-interval", "n_intervals"),
            Input("session-id-store", "data"),
        ],
    )
    def update_dashboard(n_intervals: int, sid: str):
        """Refresh dashboard panels with latest session data."""
        try:
            df = load_session(sid)
        except FileNotFoundError:
            import plotly.graph_objects as go

            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No session data found",
                template="plotly_dark",
            )
            return empty_fig, html.Div("No data available.")

        trajectory_fig = create_trajectory_figure(df)
        experiment_table = create_experiment_table(df)

        return trajectory_fig, experiment_table

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
