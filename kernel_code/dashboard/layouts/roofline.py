"""Roofline model layout for the Dash dashboard.

Panel 2: Log-log scatter plot showing kernel variants against
hardware performance ceilings (peak compute + peak memory bandwidth).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pandas as pd

from kernel_code.dashboard.theme import STATUS_COLORS, COLORS, apply_theme


# Hardware specs: peak_tflops (FP16), peak_bw_tb_s (TB/s)
_HARDWARE_SPECS = {
    "H100": {"peak_tflops": 3958.0, "peak_bw_tb_s": 3.35},
    "A100": {"peak_tflops": 312.0, "peak_bw_tb_s": 2.0},
}


def create_roofline_figure(
    df: pd.DataFrame,
    hardware: str = "H100",
    x_scale: str = "log",
    y_scale: str = "log",
    selected_iteration: int | None = None,
) -> go.Figure:
    """Create a roofline model figure.

    Args:
        df: DataFrame with columns: iteration, speedup, decision, runtime_us,
            ref_runtime_us, bandwidth_util, compute_util.
        hardware: Hardware identifier (H100 or A100).
        x_scale: X-axis scale type -- 'linear' or 'log' (default log).
        y_scale: Y-axis scale type -- 'linear' or 'log' (default log).
        selected_iteration: If set, highlight this iteration with a larger
            marker (Constellation linked-selection pattern).

    Returns:
        A Plotly Figure with roofline plot.
    """
    specs = _HARDWARE_SPECS.get(hardware, _HARDWARE_SPECS["H100"])
    peak_gflops = specs["peak_tflops"] * 1000.0  # TFLOP/s -> GFLOP/s
    peak_bw_gb_s = specs["peak_bw_tb_s"] * 1000.0  # TB/s -> GB/s

    # Ridge point: where compute ceiling meets memory ceiling
    ridge_ai = peak_gflops / peak_bw_gb_s  # FLOP/byte

    fig = go.Figure()

    # Build ceiling lines across a wide range of arithmetic intensity
    ai_range = np.logspace(-1, 5, 500)
    mem_ceiling = ai_range * peak_bw_gb_s
    roofline = np.minimum(mem_ceiling, peak_gflops)

    fig.add_trace(
        go.Scatter(
            x=ai_range,
            y=roofline,
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
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
            line=dict(color=COLORS["accent"], width=1, dash="dot"),
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
            marker=dict(color=COLORS["text_secondary"], size=12, symbol="diamond"),
            text=[f"Ridge ({ridge_ai:.1f} FLOP/B)"],
            textposition="top right",
            textfont=dict(color=COLORS["text_secondary"], size=10),
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
        for _, row in df.iterrows():
            if row.get("runtime_us", 0) <= 0:
                continue

            bw_util = row.get("bandwidth_util", 0.0) or 0.0
            comp_util = row.get("compute_util", 0.0) or 0.0

            achieved_gflops = max(comp_util * peak_gflops, 1.0)
            achieved_bw = max(bw_util * peak_bw_gb_s, 1.0)
            ai = achieved_gflops / achieved_bw

            decision = row.get("decision", "discard")
            color = STATUS_COLORS.get(decision, COLORS["text_dim"])

            # Enlarge marker for selected iteration (linked selection)
            iter_num = int(row.get("iteration", 0))
            marker_size = 16 if (selected_iteration is not None and iter_num == selected_iteration) else 10

            fig.add_trace(
                go.Scatter(
                    x=[ai],
                    y=[achieved_gflops],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=marker_size,
                        line=dict(width=1, color=COLORS["bg_card"]),
                    ),
                    name=f"Iter {iter_num}",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Iter {iter_num}</b><br>"
                        f"AI: {ai:.1f} FLOP/byte<br>"
                        f"Perf: {achieved_gflops:.0f} GFLOP/s<br>"
                        f"BW util: {bw_util:.0%}<br>"
                        f"Compute util: {comp_util:.0%}<br>"
                        f"Speedup: {row.get('speedup', 0):.2f}x<br>"
                        f"<extra></extra>"
                    ),
                )
            )

    # Apply scale toggles (default to log-log for roofline)
    effective_x_scale = x_scale if x_scale in ("linear", "log") else "log"
    effective_y_scale = y_scale if y_scale in ("linear", "log") else "log"

    apply_theme(
        fig,
        title=f"Roofline Model ({hardware})",
        xaxis_title="Arithmetic Intensity (FLOP/byte)",
        yaxis_title="Performance (GFLOP/s)",
        xaxis_type=effective_x_scale,
        yaxis_type=effective_y_scale,
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
        font=dict(color=COLORS["text_dim"], size=11),
        xref="x",
        yref="y",
    )
    fig.add_annotation(
        x=np.log10(ridge_ai * 10),
        y=np.log10(peak_gflops * 0.3),
        text="Compute Bound",
        showarrow=False,
        font=dict(color=COLORS["text_dim"], size=11),
        xref="x",
        yref="y",
    )

    return fig
