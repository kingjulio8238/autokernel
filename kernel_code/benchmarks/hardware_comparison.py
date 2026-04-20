"""Hardware comparison chart for KernelBench results.

Chart 6: Grouped bar chart showing performance across GPU types and backends.
Demonstrates that openkernel works across different hardware configurations.
"""

from __future__ import annotations

import plotly.graph_objects as go


def create_hardware_comparison(
    results_by_hardware: dict[str, dict[str, float]],
) -> go.Figure:
    """Create a grouped bar chart comparing GPU types and backends.

    Args:
        results_by_hardware: Nested dict where outer key is the backend name
            (e.g. "Triton", "CUDA") and inner key is the GPU type (e.g.
            "H100", "A100", "L40S"), value is average speedup.

    Returns:
        A Plotly Figure.
    """
    fig = go.Figure()

    if not results_by_hardware:
        fig.update_layout(title="Hardware Comparison (no data)")
        return fig

    # Collect all GPU types across backends.
    all_gpus: set[str] = set()
    for backend_results in results_by_hardware.values():
        all_gpus.update(backend_results.keys())
    gpu_types = sorted(all_gpus)

    colors = {
        "Triton": "#3b82f6",   # blue
        "CUDA": "#22c55e",     # green
    }
    # Fallback palette for unexpected backends.
    fallback_colors = ["#a855f7", "#ef4444", "#eab308", "#06b6d4", "#f97316"]

    for idx, (backend, gpu_results) in enumerate(results_by_hardware.items()):
        speedups = [gpu_results.get(gpu, 0.0) for gpu in gpu_types]
        color = colors.get(backend, fallback_colors[idx % len(fallback_colors)])

        fig.add_trace(
            go.Bar(
                x=gpu_types,
                y=speedups,
                name=backend,
                marker_color=color,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Backend: {backend}<br>"
                    "Avg speedup: %{y:.2f}x<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Baseline at 1.0x
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="1.0x (baseline)",
        annotation_position="bottom left",
    )

    fig.update_layout(
        title="Hardware Comparison: GPU Types x Backends",
        xaxis_title="GPU Type",
        yaxis_title="Average Speedup (x)",
        template="plotly_dark",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig
