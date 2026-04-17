"""Roofline model layout for the Dash dashboard.

Panel 2: Log-log scatter plot showing kernel variants against
hardware performance ceilings (peak compute + peak memory bandwidth).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pandas as pd


# Hardware specs: peak_tflops (FP16), peak_bw_tb_s (TB/s)
_HARDWARE_SPECS = {
    "H100": {"peak_tflops": 3958.0, "peak_bw_tb_s": 3.35},
    "A100": {"peak_tflops": 312.0, "peak_bw_tb_s": 2.0},
}


def create_roofline_figure(df: pd.DataFrame, hardware: str = "H100") -> go.Figure:
    """Create a roofline model figure.

    Args:
        df: DataFrame with columns: iteration, speedup, decision, runtime_us,
            ref_runtime_us, bandwidth_util, compute_util.
        hardware: Hardware identifier (H100 or A100).

    Returns:
        A Plotly Figure with log-log roofline plot.
    """
    specs = _HARDWARE_SPECS.get(hardware, _HARDWARE_SPECS["H100"])
    peak_gflops = specs["peak_tflops"] * 1000.0  # TFLOP/s -> GFLOP/s
    peak_bw_gb_s = specs["peak_bw_tb_s"] * 1000.0  # TB/s -> GB/s

    # Ridge point: where compute ceiling meets memory ceiling
    # peak_gflops = arithmetic_intensity * peak_bw_gb_s
    ridge_ai = peak_gflops / peak_bw_gb_s  # FLOP/byte

    fig = go.Figure()

    # Build ceiling lines across a wide range of arithmetic intensity
    ai_range = np.logspace(-1, 5, 500)
    # Memory ceiling: performance = AI * peak_bandwidth
    mem_ceiling = ai_range * peak_bw_gb_s
    # Compute ceiling: performance = peak_compute (flat)
    roofline = np.minimum(mem_ceiling, peak_gflops)

    fig.add_trace(
        go.Scatter(
            x=ai_range,
            y=roofline,
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            name="Roofline",
            hoverinfo="skip",
        )
    )

    # Dashed extensions to show both ceilings clearly
    fig.add_trace(
        go.Scatter(
            x=ai_range[ai_range < ridge_ai * 2],
            y=np.full(int(np.sum(ai_range < ridge_ai * 2)), peak_gflops),
            mode="lines",
            line=dict(color="#3b82f6", width=1, dash="dot"),
            name="Peak Compute",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Ridge point annotation
    fig.add_trace(
        go.Scatter(
            x=[ridge_ai],
            y=[peak_gflops],
            mode="markers+text",
            marker=dict(color="#f59e0b", size=12, symbol="diamond"),
            text=[f"Ridge ({ridge_ai:.1f} FLOP/B)"],
            textposition="top right",
            textfont=dict(color="#f59e0b", size=10),
            name="Ridge Point",
            hovertemplate=(
                f"<b>Ridge Point</b><br>"
                f"AI: {ridge_ai:.1f} FLOP/byte<br>"
                f"Peak: {peak_gflops:.0f} GFLOP/s<br>"
                f"<extra></extra>"
            ),
        )
    )

    # Plot kernel variants from data
    if not df.empty:
        _STATUS_COLORS = {
            "keep": "#22c55e",
            "discard": "#ef4444",
            "error": "#dc2626",
            "incorrect": "#eab308",
        }

        for _, row in df.iterrows():
            if row.get("runtime_us", 0) <= 0:
                continue

            # Estimate arithmetic intensity and achieved performance from profile data
            # AI approximation: higher compute_util / bandwidth_util -> higher AI
            bw_util = row.get("bandwidth_util", 0.0) or 0.0
            comp_util = row.get("compute_util", 0.0) or 0.0

            # Estimate achieved GFLOP/s from compute utilization
            achieved_gflops = max(comp_util * peak_gflops, 1.0)
            # Estimate achieved bandwidth from bandwidth utilization
            achieved_bw = max(bw_util * peak_bw_gb_s, 1.0)
            # Arithmetic intensity = achieved_gflops / achieved_bw
            ai = achieved_gflops / achieved_bw

            decision = row.get("decision", "discard")
            color = _STATUS_COLORS.get(decision, "#6b7280")

            fig.add_trace(
                go.Scatter(
                    x=[ai],
                    y=[achieved_gflops],
                    mode="markers",
                    marker=dict(color=color, size=10, line=dict(width=1, color="white")),
                    name=f"Iter {int(row.get('iteration', 0))}",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Iter {int(row.get('iteration', 0))}</b><br>"
                        f"AI: {ai:.1f} FLOP/byte<br>"
                        f"Perf: {achieved_gflops:.0f} GFLOP/s<br>"
                        f"BW util: {bw_util:.0%}<br>"
                        f"Compute util: {comp_util:.0%}<br>"
                        f"Speedup: {row.get('speedup', 0):.2f}x<br>"
                        f"<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"Roofline Model ({hardware})",
        xaxis_title="Arithmetic Intensity (FLOP/byte)",
        yaxis_title="Performance (GFLOP/s)",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add region labels
    fig.add_annotation(
        x=np.log10(ridge_ai / 10),
        y=np.log10(peak_gflops * 0.3),
        text="Memory Bound",
        showarrow=False,
        font=dict(color="#94a3b8", size=11),
        xref="x",
        yref="y",
    )
    fig.add_annotation(
        x=np.log10(ridge_ai * 10),
        y=np.log10(peak_gflops * 0.3),
        text="Compute Bound",
        showarrow=False,
        font=dict(color="#94a3b8", size=11),
        xref="x",
        yref="y",
    )

    return fig
