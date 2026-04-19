"""Per-step progress reporting for optimization and agent loops.

Replaces generic "Thinking..." spinners with specific step indicators
so users always know *what* the system is doing.  Spinners are
stall-aware: they escalate colour and messaging when an operation takes
longer than expected (yellow after 10 s, red after 30 s).

Usage::

    from kernel_code.progress import OptimizationProgress, AgentProgress

    progress = OptimizationProgress(console)
    progress.start_iteration(1, "tiled matmul with shared memory")
    progress.compiling()
    progress.checking_correctness(10)
    progress.benchmarking(100)
    progress.kept(2.34, is_new_best=True)
"""

from __future__ import annotations

import threading
import time

from rich.console import Console
from rich.status import Status


# ---------------------------------------------------------------------------
# Stall-aware spinner
# ---------------------------------------------------------------------------

class StallAwareSpinner:
    """Spinner that escalates colour when operations take too long.

    Behaves like :class:`rich.status.Status` — supports both context-manager
    usage (``with spinner: ...``) and explicit ``start()``/``stop()`` calls.
    """

    def __init__(self, message: str, console: Console) -> None:
        self._console = console
        self._base_message = message
        self._start: float = 0.0
        self._status: Status = console.status(message, spinner="dots")
        self._timer: threading.Timer | None = None
        self._stopped = True  # not yet started

    # -- public interface (mirrors Status) -----------------------------------

    def start(self) -> None:
        self._start = time.monotonic()
        self._stopped = False
        self._status.start()
        self._schedule_update()

    def stop(self) -> None:
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._status.stop()

    def update(self, msg: str) -> None:  # pragma: no cover – passthrough
        self._status.update(msg)

    def __enter__(self) -> "StallAwareSpinner":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    # -- internal ------------------------------------------------------------

    def _schedule_update(self) -> None:
        if self._stopped:
            return
        elapsed = time.monotonic() - self._start
        if elapsed > 30:
            msg = (
                f"[red]{self._base_message} ({elapsed:.0f}s) "
                f"— possible stall, press Ctrl+C[/red]"
            )
        elif elapsed > 10:
            msg = (
                f"[yellow]{self._base_message} ({elapsed:.0f}s) "
                f"— taking longer than usual[/yellow]"
            )
        else:
            msg = f"[dim]{self._base_message} ({elapsed:.1f}s)[/dim]"
        self._status.update(msg)
        self._timer = threading.Timer(1.0, self._schedule_update)
        self._timer.daemon = True
        self._timer.start()


# ---------------------------------------------------------------------------
# Optimization progress
# ---------------------------------------------------------------------------

class OptimizationProgress:
    """Reports per-step progress during optimization.

    Each step gets its own :class:`StallAwareSpinner` so the user can see
    elapsed time and get a visual cue if a step is taking unusually long.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._current_spinner: StallAwareSpinner | None = None

    def _update_status(self, message: str) -> None:
        """Start a new stall-aware spinner for *message*.

        If a spinner is already active it is stopped first so each step
        gets its own independent elapsed-time counter.
        """
        if self._current_spinner is not None:
            self._current_spinner.stop()
        self._current_spinner = StallAwareSpinner(message, self._console)
        self._current_spinner.start()

    def _stop_status(self) -> None:
        """Stop the spinner if one is active."""
        if self._current_spinner is not None:
            self._current_spinner.stop()
            self._current_spinner = None

    def start_iteration(self, iteration: int, intent: str) -> None:
        """Called at the start of each optimization iteration."""
        self._stop_status()
        self._console.print(f"\n[dim]── Iteration {iteration} ──[/dim]")
        self._update_status(f"Generating kernel ({intent[:50]})")

    def compiling(self) -> None:
        """Show that the kernel is being compiled."""
        self._update_status("Compiling kernel on L40S")

    def checking_correctness(self, trials: int) -> None:
        """Show that correctness checks are running."""
        self._update_status(f"Checking correctness ({trials} trials)")

    def benchmarking(self, trials: int) -> None:
        """Show that benchmarking is running."""
        self._update_status(f"Benchmarking ({trials} trials)")

    def profiling(self) -> None:
        """Show that hardware profiling is running."""
        self._update_status("Profiling hardware metrics")

    def analyzing(self) -> None:
        """Show that the critic is analyzing results."""
        self._update_status("Critic analyzing bottleneck")

    def kept(self, speedup: float, is_new_best: bool) -> None:
        """Report that an iteration was kept."""
        self._stop_status()
        marker = " (new best!)" if is_new_best else ""
        self._console.print(
            f"  [green]\u2713[/green] [bold green]Kept: {speedup:.2f}x{marker}[/bold green]"
        )

    def discarded(self, speedup: float, best: float) -> None:
        """Report that an iteration was discarded."""
        self._stop_status()
        self._console.print(
            f"  [red]\u2717[/red] [dim]Discarded: {speedup:.2f}x < best {best:.2f}x[/dim]"
        )

    def error(self, error_type: str, message: str) -> None:
        """Report an error during an iteration."""
        self._stop_status()
        # Show first line of error clearly, rest dimmed
        lines = message.strip().split("\n")
        first = lines[0][:120]
        self._console.print(
            f"  [red]![/red] [red]{error_type}[/red]: [white]{first}[/white]"
        )
        for line in lines[1:3]:  # show up to 2 more lines for context
            self._console.print(f"    [white]{line[:120]}[/white]")

    def complete(self, best_speedup: float, iterations: int, kept: int, cost: float) -> None:
        """Report that the optimization run is complete."""
        self._stop_status()
        self._console.print(f"\n[bold]Optimization complete[/bold]")
        self._console.print(
            f"  Best: [bold green]{best_speedup:.2f}x[/bold green] "
            f"| {kept}/{iterations} kept | ${cost:.2f}"
        )


# ---------------------------------------------------------------------------
# Agent progress
# ---------------------------------------------------------------------------

class AgentProgress:
    """Reports progress during agentic loop turns."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def thinking(self) -> StallAwareSpinner:
        """Show while LLM is generating response.

        Returns a :class:`StallAwareSpinner` that supports both
        context-manager usage (``with progress.thinking(): ...``) and
        explicit ``start()``/``stop()`` calls — a drop-in replacement for
        :class:`rich.status.Status`.
        """
        return StallAwareSpinner("Thinking...", self._console)

    def calling_tool(self, tool_name: str, args: str = "") -> None:
        """Show when a tool is being called."""
        display_args = f"({args})" if args else ""
        self._console.print(
            f"  [cyan]\u2192[/cyan] [dim]Calling {tool_name}{display_args}[/dim]"
        )

    def tool_result(self, tool_name: str, preview: str) -> None:
        """Show tool result preview."""
        self._console.print(f"  [cyan]\u2190[/cyan] [dim]{preview[:100]}[/dim]")
