"""Enhanced context visualization with percentage bars and warnings."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

BLOCKS = [" ", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589", "\u2588"]


def progress_bar(ratio: float, width: int = 20) -> Text:
    """Render a Unicode progress bar with color based on ratio."""
    ratio = max(0, min(1, ratio))
    if ratio < 0.6:
        color = "#4ade80"  # green
    elif ratio < 0.8:
        color = "#fbbf24"  # yellow
    else:
        color = "#ef4444"  # red

    whole = int(ratio * width)
    frac_idx = int((ratio * width - whole) * (len(BLOCKS) - 1))

    bar = Text()
    bar.append("\u2588" * whole, style=color)
    if whole < width:
        bar.append(BLOCKS[frac_idx], style=color)
        bar.append("\u2591" * (width - whole - 1), style="dim")

    return bar


def render_context_breakdown(
    components: list[tuple[str, int]],  # (name, tokens)
    budget: int = 4096,
    compacted: bool = False,
    compaction_strategy: str = "balanced",
    console: Console | None = None,
) -> None:
    """Render a detailed context breakdown with percentage bars."""
    c = console or Console()

    table = Table(
        title="Context Window",
        show_header=True,
        title_style="bold",
        border_style="dim",
    )
    table.add_column("Component", style="bold", width=20)
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("Usage", width=24)
    table.add_column("%", justify="right", width=5)

    total = sum(t for _, t in components)

    for name, tokens in components:
        pct = tokens / budget if budget > 0 else 0
        bar = progress_bar(pct, width=20)
        table.add_row(name, f"{tokens:,}", bar, f"{pct*100:.0f}%")

    table.add_section()
    total_pct = total / budget if budget > 0 else 0
    total_bar = progress_bar(total_pct, width=20)
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total:,}[/bold]",
        total_bar,
        f"[bold]{total_pct*100:.0f}%[/bold]",
    )

    c.print(table)

    # Compaction status
    if compacted:
        c.print(f"[dim]Compaction: active ({compaction_strategy} strategy)[/dim]")

    # Warnings
    if total_pct > 0.9:
        c.print(
            "[red bold]\u26a0 Context at {:.0f}% \u2014 run /compact to free space[/red bold]".format(
                total_pct * 100
            )
        )
    elif total_pct > 0.8:
        c.print(
            "[yellow]Context at {:.0f}% \u2014 approaching limit[/yellow]".format(
                total_pct * 100
            )
        )

    c.print(f"[dim]Budget: {budget:,} tokens ({budget * 4:,} chars)[/dim]")
