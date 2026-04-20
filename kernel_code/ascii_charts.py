"""ASCII chart rendering for optimization results in the terminal."""
from rich.text import Text

def render_trajectory_chart(
    speedups: list[float],
    statuses: list[str],  # "keep", "discard", "error", etc.
    width: int = 50,
    height: int = 6,
) -> Text:
    """Render a mini ASCII line chart of speedup over iterations.

    Uses Unicode block characters for the chart area.
    Colors: green for keep iterations, red for discard, yellow for error.
    Shows Y-axis labels (min/max speedup) and X-axis (iteration range).
    """
    if not speedups:
        return Text("[dim]No data to chart[/dim]")

    result = Text()

    min_val = min(speedups) if speedups else 0
    max_val = max(speedups) if speedups else 1
    if max_val == min_val:
        max_val = min_val + 1

    # Y-axis labels
    y_labels = [f"{max_val:.1f}x", f"{(max_val+min_val)/2:.1f}x", f"{min_val:.1f}x"]
    label_width = max(len(l) for l in y_labels)

    # Scale speedups to height
    scaled = []
    for s in speedups:
        row = int((s - min_val) / (max_val - min_val) * (height - 1))
        scaled.append(min(row, height - 1))

    # Sample if more iterations than width
    if len(scaled) > width:
        indices = [int(i * (len(scaled) - 1) / (width - 1)) for i in range(width)]
        display_speedups = [speedups[i] for i in indices]
        display_statuses = [statuses[i] for i in indices]
        display_scaled = [scaled[i] for i in indices]
    else:
        display_speedups = speedups
        display_statuses = statuses
        display_scaled = scaled

    # Render rows (top to bottom)
    for row in range(height - 1, -1, -1):
        # Y-axis label
        if row == height - 1:
            result.append(f"{y_labels[0]:>{label_width}} │", style="dim")
        elif row == height // 2:
            result.append(f"{y_labels[1]:>{label_width}} │", style="dim")
        elif row == 0:
            result.append(f"{y_labels[2]:>{label_width}} │", style="dim")
        else:
            result.append(f"{'':>{label_width}} │", style="dim")

        # Data points
        for i, (val, status) in enumerate(zip(display_scaled, display_statuses)):
            if val == row:
                color = "#4ade80" if status == "keep" else "#ef4444" if status in ("discard", "error") else "#fbbf24"
                result.append("●", style=color)
            elif val > row:
                result.append("│", style="dim")
            else:
                result.append(" ")

        result.append("\n")

    # X-axis
    result.append(f"{'':>{label_width}} └{'─' * len(display_scaled)}", style="dim")
    result.append(f"\n{'':>{label_width}}  1{'':>{len(display_scaled)-4}}{len(speedups)}", style="dim")

    return result


def render_bar_chart(
    labels: list[str],
    values: list[float],
    max_width: int = 30,
    color: str = "#4ade80",
) -> Text:
    """Render a horizontal bar chart."""
    result = Text()
    max_val = max(values) if values else 1
    max_label = max(len(l) for l in labels) if labels else 0

    for label, value in zip(labels, values):
        bar_len = int(value / max_val * max_width) if max_val > 0 else 0
        result.append(f"  {label:>{max_label}} ", style="dim")
        result.append("█" * bar_len, style=color)
        result.append(f" {value:.2f}\n")

    return result
