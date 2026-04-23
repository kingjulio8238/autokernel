"""Rich summary card — single bordered panel with all optimization results.

Replaces scattered post-optimization print statements with one cohesive card.
All text uses bright colors visible on dark terminals — no dim/gray text.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def _section_header(title: str) -> Text:
    """Render a section header."""
    t = Text()
    t.append(f"  {title}\n", style="bold white")
    return t


def _metric_row(label: str, value: str, value_style: str = "bold white",
                extra: str = "", extra_style: str = "white") -> Text:
    """Render a key-value metric row with aligned columns."""
    t = Text()
    t.append(f"  {label:<18}", style="white")
    t.append(value, style=value_style)
    if extra:
        t.append(f"  {extra}", style=extra_style)
    t.append("\n")
    return t


def _gauge(label: str, value: float, target: float = 0.85, width: int = 20) -> Text:
    """Render a single utilization gauge — wider, brighter."""
    t = Text()
    pct = int(value * 100)
    bar_filled = int(value * width)
    bar_empty = width - bar_filled

    color = "#4ade80" if value >= 0.8 else "#fbbf24" if value >= 0.5 else "#ef4444"

    t.append(f"  {label:<6}", style="white")
    t.append("█" * bar_filled, style=color)
    t.append("░" * bar_empty, style="#555555")
    t.append(f" {pct}%", style=f"bold {color}")
    t.append(f" → {int(target * 100)}%\n", style="white")
    return t


def render_optimization_summary(
    iterations: list[dict],
    best_speedup: float,
    best_iteration: int,
    kept_count: int,
    total_count: int,
    cost_summary: str = "",
    elapsed_seconds: float = 0.0,
    bottleneck_type: str = "",
    bottleneck_metrics: dict | None = None,
    skill_suggestions: list[dict] | None = None,
    next_steps: list | None = None,
    dashboard_url: str = "",
    saved_path: str = "",
    best_sol: float = 0.0,
    console: Console | None = None,
) -> None:
    """Render a comprehensive optimization summary as a Rich Panel.

    Phase 2 dual-display: SOL is the primary headline, speedup is shown as a
    dim secondary row. When ``best_sol`` is falsy the panel falls back to
    speedup-only (legacy runs, SOL profiling unavailable).
    """
    c = console or Console()
    content = Text()

    # Auto-derive best SOL from the iterations list if the caller did not
    # pass one explicitly — keeps legacy call sites working.
    if not best_sol:
        for it in iterations:
            prof = it.get("profile") or {}
            s = prof.get("sol_score", 0.0) if isinstance(prof, dict) else 0.0
            if s > best_sol:
                best_sol = s

    # ═══ METRICS ═══
    content.append("\n")
    pct = (kept_count / total_count * 100) if total_count > 0 else 0
    if best_sol > 0:
        # SOL primary headline, speedup as dim secondary row.
        sol_pct = int(best_sol * 100)
        bn = bottleneck_type.replace("_", "-") if bottleneck_type else ""
        primary_extra = f"{sol_pct}% of peak" + (f" · {bn}" if bn else "")
        content.append_text(_metric_row(
            "Best SOL", f"{best_sol:.2f}", "bold #4ade80",
            primary_extra, "white",
        ))
        content.append_text(_metric_row(
            "Best Speedup", f"{best_speedup:.2f}x speedup vs reference",
            "#999999", f"(iter {best_iteration})", "#999999",
        ))
    else:
        # Fallback: SOL unavailable, show speedup alone with explicit note.
        content.append_text(_metric_row(
            "Best Speedup", f"{best_speedup:.2f}x", "bold #4ade80",
            f"(iter {best_iteration}) · SOL unknown", "white",
        ))
    content.append_text(_metric_row("Kept / Total", f"{kept_count} / {total_count}", "bold white", f"({pct:.0f}%)"))
    if cost_summary:
        content.append_text(_metric_row("LLM Cost", cost_summary, "bold white"))
    if elapsed_seconds > 0:
        mins = int(elapsed_seconds) // 60
        secs = int(elapsed_seconds) % 60
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
        content.append_text(_metric_row("Time", time_str, "bold white"))
    content.append("\n")

    # ═══ TRAJECTORY ═══
    from kernel_code.ascii_charts import render_trajectory_chart

    speedups = [it.get("speedup", 0) for it in iterations]
    statuses = [it.get("decision", it.get("status", "")) for it in iterations]
    if speedups:
        content.append_text(_section_header("Trajectory"))
        chart = render_trajectory_chart(speedups, statuses, width=50, height=5)
        for line in chart.plain.split("\n"):
            if line.strip():
                content.append(f"  {line}\n", style="white")
        content.append("\n")

    # ═══ HEATMAP ═══
    from kernel_code.heatmap import render_iteration_heatmap

    if iterations:
        heatmap = render_iteration_heatmap(iterations, show_legend=True)
        content.append("  ")
        content.append_text(heatmap)
        content.append("\n")

    # ═══ PROFILING ═══
    if bottleneck_type or bottleneck_metrics:
        content.append_text(_section_header("Profiling"))
        if bottleneck_type:
            bn_color = "#ef4444" if "memory" in bottleneck_type else "#fbbf24" if "compute" in bottleneck_type else "#c084fc"
            content.append("  Bottleneck      ", style="white")
            content.append(bottleneck_type.upper().replace("_", " ") + "\n", style=f"bold {bn_color}")

        if bottleneck_metrics:
            bw = bottleneck_metrics.get("bandwidth_util", 0)
            comp = bottleneck_metrics.get("compute_util", 0)
            cache = bottleneck_metrics.get("cache_efficiency", 0)
            occ = bottleneck_metrics.get("occupancy", 0)
            if bw > 0:
                content.append_text(_gauge("BW", bw, 0.85))
            if comp > 0:
                content.append_text(_gauge("Comp", comp, 0.80))
            if cache > 0:
                content.append_text(_gauge("L2", cache, 0.85))
            if occ > 0:
                content.append_text(_gauge("Occ", occ, 0.90))
        content.append("\n")

    # ═══ NEXT STEPS ═══
    if next_steps:
        from kernel_code.next_steps import format_next_steps
        content.append_text(format_next_steps(next_steps))
        content.append("\n")
    elif skill_suggestions:
        content.append_text(_section_header("Suggested"))
        for skill in skill_suggestions[:3]:
            sid = skill.get("skill_id", skill.get("name", ""))
            content.append(f"  /skill:{sid}\n", style="bold #22d3ee")
        content.append("\n")

    # ═══ LINKS ═══
    if dashboard_url or saved_path:
        content.append_text(_section_header("Links"))
        if dashboard_url:
            content.append_text(_metric_row("Dashboard", dashboard_url, "#22d3ee underline"))
        if saved_path:
            content.append_text(_metric_row("Best saved", saved_path, "bold white"))
    content.append("")

    if best_sol > 0:
        headline = f"SOL {best_sol:.2f} · {best_speedup:.2f}x"
    else:
        headline = f"{best_speedup:.2f}x · SOL unknown"
    panel = Panel(
        content,
        title="[bold white] Optimization Complete [/bold white]",
        subtitle=f"[white]{total_count} iterations • {kept_count} kept • {headline}[/white]",
        border_style="#4ade80",
        padding=(1, 3),
        width=min(c.width, 90),
    )
    c.print(panel)
