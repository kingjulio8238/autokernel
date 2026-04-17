"""Optimization feed panel -- rich left-side panel replacing the old ChatPanel.

Shows a dense, information-rich feed of the current optimization session:
  - Header with product name and problem identifier
  - Running summary line (iters, kept, best, cost, elapsed)
  - Current intent and evaluation status
  - Critic diagnosis with bottleneck and utilization snapshot
  - Best kept kernel snippet (syntax highlighted)
  - Recent activity log (last 5 iterations, compact)
"""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static


# ---------------------------------------------------------------------------
# Mapping from bottleneck type to a terse suggestion
# ---------------------------------------------------------------------------
_BOTTLENECK_HINTS: dict[str, str] = {
    "memory_bound": "Try coalesced accesses or shared-memory tiling",
    "compute_bound": "Try warp-level shuffle for partial reductions",
    "latency_bound": "Try persistent kernel or double buffering",
    "unknown": "Collect profile data for diagnosis",
}


class OptimizationFeedPanel(Widget):
    """Rich optimization feed panel for the left side of the TUI."""

    DEFAULT_CSS = """
    OptimizationFeedPanel {
        width: 2fr;
        height: 1fr;
        background: #24231f;
        border: solid #3d3a36;
        color: #e0ddd8;
        padding: 0;
        overflow-y: auto;
    }

    OptimizationFeedPanel > VerticalScroll {
        height: 1fr;
        padding: 0 1;
    }

    OptimizationFeedPanel #feed-content {
        width: 1fr;
    }
    """

    def __init__(
        self,
        session_data: dict | None = None,
        visible_iterations: list[dict] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._session_data: dict = session_data or {}
        self._visible: list[dict] = visible_iterations or []
        self._start_time: float = time.monotonic()

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(self._build_content(), id="feed-content", markup=True)

    # ------------------------------------------------------------------
    # Public API -- called from app._update_all_panels()
    # ------------------------------------------------------------------

    def update_feed(
        self,
        session_data: dict,
        visible_iterations: list[dict],
    ) -> None:
        """Re-render the feed with fresh data."""
        self._session_data = session_data
        self._visible = visible_iterations
        try:
            content = self.query_one("#feed-content", Static)
            content.update(self._build_content())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _build_content(self) -> str:
        """Build the full panel content as a Rich-markup string."""
        parts: list[str] = []

        parts.append(self._render_header())
        parts.append("")
        parts.append(self._render_summary())
        parts.append("")

        if self._visible:
            parts.append(self._render_intent())
            parts.append("")
            parts.append(self._render_critic())
            parts.append("")
            parts.append(self._render_best_kernel())
            parts.append("")
            parts.append(self._render_recent_activity())
        else:
            parts.append("[dim]Waiting for first iteration...[/dim]")

        return "\n".join(parts)

    # -- Section renderers -----------------------------------------------

    def _render_header(self) -> str:
        problem = self._session_data.get("problem", "...")
        backend = self._session_data.get("backend", "...")
        lines = [
            "[bold]OPENKERNEL[/bold] [dim]v0.1[/dim]",
            f"[bold]{problem}[/bold] [dim]|[/dim] [italic]{backend}[/italic]",
        ]
        return "\n".join(lines)

    def _render_summary(self) -> str:
        total = len(self._visible)
        kept = sum(1 for it in self._visible if it.get("decision") == "keep")
        best = max((it.get("speedup", 0.0) for it in self._visible), default=0.0)
        cost = total * 0.02  # ~$0.02 per iteration

        elapsed_s = time.monotonic() - self._start_time
        mins = int(elapsed_s) // 60
        secs = int(elapsed_s) % 60
        elapsed_fmt = f"{mins}m {secs:02d}s"

        return (
            f"[bold]{total}[/bold] [dim]iters[/dim] [dim]\u2502[/dim] "
            f"[#4ade80]{kept}[/#4ade80] [dim]kept[/dim] [dim]\u2502[/dim] "
            f"[bold #4ade80]{best:.2f}x[/bold #4ade80] [dim]\u2502[/dim] "
            f"[dim]${cost:.2f}[/dim] [dim]\u2502[/dim] "
            f"[dim]{elapsed_fmt}[/dim]"
        )

    def _render_intent(self) -> str:
        latest = self._visible[-1] if self._visible else {}
        intent = latest.get("intent", "...")
        decision = latest.get("decision", "")
        hardware = self._session_data.get("hardware", "GPU")

        if decision == "":
            status_line = f"[#fbbf24]STATUS:[/#fbbf24] [dim]evaluating on {hardware}...[/dim]"
        elif decision == "keep":
            status_line = f"[#4ade80]STATUS:[/#4ade80] [bold #4ade80]KEPT[/bold #4ade80] [dim]-- new best[/dim]"
        elif decision == "error":
            error_msg = latest.get("error", "unknown error")
            # truncate long errors
            if len(error_msg) > 50:
                error_msg = error_msg[:47] + "..."
            status_line = f"[#f87171]STATUS:[/#f87171] [#f87171]ERROR[/#f87171] [dim]{error_msg}[/dim]"
        else:
            status_line = f"[#f87171]STATUS:[/#f87171] [dim]discarded[/dim]"

        return (
            f"[#fbbf24 bold]INTENT:[/#fbbf24 bold] [italic]{intent}[/italic]\n"
            f"{status_line}"
        )

    def _render_critic(self) -> str:
        latest = self._visible[-1] if self._visible else {}
        profile = latest.get("profile", {})

        bottleneck = profile.get("bottleneck_type", "unknown")
        bw = profile.get("bandwidth_util", 0.0)
        comp = profile.get("compute_util", 0.0)
        cache = profile.get("cache_efficiency", 0.0)
        occ = profile.get("occupancy", 0.0)

        bottleneck_label = bottleneck.upper().replace("_", " ")
        hint = _BOTTLENECK_HINTS.get(bottleneck, "Profile data insufficient")

        bw_pct = int(bw * 100)
        comp_pct = int(comp * 100)
        cache_pct = int(cache * 100)
        occ_pct = int(occ * 100)

        lines = [
            f"[#fbbf24 bold]BOTTLENECK:[/#fbbf24 bold] [bold]{bottleneck_label}[/bold]",
            (
                f"  [dim]BW[/dim] {self._pct_color(bw_pct)} [dim]\u2502[/dim] "
                f"[dim]Compute[/dim] {self._pct_color(comp_pct)} [dim]\u2502[/dim] "
                f"[dim]L2[/dim] {self._pct_color(cache_pct)} [dim]\u2502[/dim] "
                f"[dim]Occ[/dim] {self._pct_color(occ_pct)}"
            ),
            f"  [dim]\u2192[/dim] [italic]{hint}[/italic]",
        ]
        return "\n".join(lines)

    def _render_best_kernel(self) -> str:
        # Find the last kept iteration (best kernel)
        best_iter = None
        for it in reversed(self._visible):
            if it.get("decision") == "keep":
                best_iter = it
                break

        if best_iter is None:
            return "[dim]BEST KERNEL: none kept yet[/dim]"

        iteration = best_iter.get("iteration", "?")
        speedup = best_iter.get("speedup", 0.0)
        snippet = best_iter.get("kernel_code_snippet", "")

        # Truncate snippet to ~8 lines
        snippet_lines = snippet.strip().splitlines()
        if len(snippet_lines) > 8:
            snippet_lines = snippet_lines[:7] + ["    ..."]

        # Indent and dim the code
        code_lines = []
        for line in snippet_lines:
            # Highlight decorators and keywords
            styled = line
            if styled.strip().startswith("@"):
                styled = f"[cyan]{styled}[/cyan]"
            elif styled.strip().startswith("def "):
                styled = f"[green]{styled}[/green]"
            elif styled.strip().startswith("#"):
                styled = f"[dim]{styled}[/dim]"
            else:
                styled = f"[dim]{styled}[/dim]"
            code_lines.append(f"  {styled}")

        header = (
            f"[bold #4ade80]BEST KERNEL[/bold #4ade80] "
            f"[dim](iter {iteration}, {speedup:.2f}x):[/dim]"
        )
        return header + "\n" + "\n".join(code_lines)

    def _render_recent_activity(self) -> str:
        recent = self._visible[-5:]
        if not recent:
            return ""

        lines = ["[bold]RECENT[/bold]"]
        for it in reversed(recent):
            iteration = it.get("iteration", 0)
            speedup = it.get("speedup", 0.0)
            decision = it.get("decision", "")
            intent = it.get("intent", "")

            # Truncate intent for compact display
            if len(intent) > 35:
                intent = intent[:32] + "..."

            iter_str = f"[bold]#{iteration:<3}[/bold]"

            if decision == "keep":
                speedup_str = f"[#4ade80]{speedup:.2f}x[/#4ade80]"
                marker = "[#4ade80]\u2713 keep   [/#4ade80]"
            elif decision == "error":
                speedup_str = f"[#f87171]  -- [/#f87171]"
                marker = "[#f87171]\u2717 error  [/#f87171]"
            else:
                speedup_str = f"[#f87171]{speedup:.2f}x[/#f87171]"
                marker = "[#f87171]\u2717 discard[/#f87171]"

            lines.append(
                f"{iter_str} {speedup_str}  {marker} [dim]{intent}[/dim]"
            )
        return "\n".join(lines)

    # -- Helpers ---------------------------------------------------------

    @staticmethod
    def _pct_color(pct: int) -> str:
        """Return a colored percentage string: green >= 80, yellow >= 50, red < 50."""
        if pct >= 80:
            return f"[#4ade80]{pct}%[/#4ade80]"
        elif pct >= 50:
            return f"[#fbbf24]{pct}%[/#fbbf24]"
        else:
            return f"[#f87171]{pct}%[/#f87171]"
