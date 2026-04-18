"""Rich summary card — single bordered panel with all optimization results.

Replaces scattered post-optimization print statements with one cohesive card.
"""
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

def render_optimization_summary(
    iterations: list[dict],
    best_speedup: float,
    best_iteration: int,
    kept_count: int,
    total_count: int,
    cost_summary: str = "",  # from CostTracker.format_summary()
    elapsed_seconds: float = 0.0,
    bottleneck_type: str = "",
    bottleneck_metrics: dict | None = None,  # {bandwidth_util, compute_util, etc.}
    skill_suggestions: list[dict] | None = None,
    dashboard_url: str = "",
    saved_path: str = "",
    console: Console | None = None,
) -> None:
    """Render a comprehensive optimization summary as a Rich Panel."""
    c = console or Console()

    content = Text()

    # ── Key Metrics ──
    content.append("\n")
    content.append("  Best Speedup    ", style="dim")
    content.append(f"{best_speedup:.2f}x", style="bold #4ade80")
    content.append(f"  (iter {best_iteration})\n", style="dim")

    content.append("  Kept / Total    ", style="dim")
    pct = (kept_count / total_count * 100) if total_count > 0 else 0
    content.append(f"{kept_count} / {total_count}", style="bold")
    content.append(f"  ({pct:.0f}%)\n", style="dim")

    if cost_summary:
        content.append("  Cost            ", style="dim")
        content.append(f"{cost_summary}\n", style="bold")

    if elapsed_seconds > 0:
        content.append("  Time            ", style="dim")
        mins = int(elapsed_seconds) // 60
        secs = int(elapsed_seconds) % 60
        content.append(f"{mins}m {secs:02d}s\n" if mins > 0 else f"{secs}s\n", style="bold")

    content.append("\n")

    # ── Trajectory Chart ──
    from kernel_code.ascii_charts import render_trajectory_chart
    speedups = [it.get("speedup", 0) for it in iterations]
    statuses = [it.get("decision", it.get("status", "")) for it in iterations]
    if speedups:
        chart = render_trajectory_chart(speedups, statuses, width=40, height=4)
        content.append("  Trajectory\n", style="bold dim")
        # Indent chart lines
        for line in chart.plain.split("\n"):
            if line.strip():
                content.append(f"  {line}\n")
        content.append("\n")

    # ── Heatmap ──
    from kernel_code.heatmap import render_iteration_heatmap
    if iterations:
        heatmap = render_iteration_heatmap(iterations, show_legend=False)
        content.append("  ")
        content.append_text(heatmap)
        content.append("\n")

    # ── Bottleneck ──
    if bottleneck_type:
        content.append("  Bottleneck      ", style="dim")
        bn_color = "#ef4444" if "memory" in bottleneck_type else "#fbbf24" if "compute" in bottleneck_type else "#c084fc"
        content.append(bottleneck_type.upper().replace("_", " "), style=f"bold {bn_color}")
        content.append("\n")

        if bottleneck_metrics:
            bw = bottleneck_metrics.get("bandwidth_util", 0)
            comp = bottleneck_metrics.get("compute_util", 0)
            if bw > 0:
                bar_len = int(bw * 10)
                content.append(f"  BW {int(bw*100)}% ", style="dim")
                content.append("▓" * bar_len + "░" * (10 - bar_len), style="#4ade80" if bw > 0.8 else "#fbbf24" if bw > 0.5 else "#ef4444")
                content.append(" → 85%\n", style="dim")
        content.append("\n")

    # ── Skill Suggestions ──
    if skill_suggestions:
        content.append("  Suggested       ", style="dim")
        for i, skill in enumerate(skill_suggestions[:2]):
            if i > 0:
                content.append("                  ", style="dim")
            content.append(f"/skill:{skill.get('skill_id', skill.get('name', ''))}\n", style="#22d3ee")
        content.append("\n")

    # ── Links ──
    if dashboard_url:
        content.append("  Dashboard       ", style="dim")
        content.append(f"{dashboard_url}\n", style="#22d3ee underline")
    if saved_path:
        content.append("  Best saved      ", style="dim")
        content.append(f"{saved_path}\n", style="bold")

    content.append("\n")

    # Render the panel
    panel = Panel(
        content,
        title="[bold]Optimization Complete[/bold]",
        border_style="#3d3a36",
        padding=(0, 1),
    )
    c.print(panel)
