"""Main Textual App for kernel code.

Layout (matching docs/kernel-code-design.md):
  ┌─────────────────────┬──────────────────────────────┐
  │  Optimization Feed  │  Optimization Trajectory      │
  │  - header / summary │  (sparkline)                  │
  │  - intent & status  ├──────────────────────────────┤
  │  - critic diagnosis │  Profiling Summary             │
  │  - best kernel      │  (4 gauges + bottleneck)      │
  │  - recent activity  ├──────────────────────────────┤
  │                     │  Experiment Log               │
  │                     │  (colored table)              │
  ├─────────────────────┴──────────────────────────────┤
  │  Status Bar: GPU | Backend | Model | Iter | Cost   │
  └────────────────────────────────────────────────────┘

Loads session JSON from cache/sessions/ and refreshes on a 2-second timer.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Static

from kernel_code.tui.keybindings import APP_BINDINGS
from kernel_code.tui.panels.experiment_log import ExperimentLogPanel
from kernel_code.tui.panels.optimization_feed import OptimizationFeedPanel
from kernel_code.tui.panels.profiling import ProfilingPanel
from kernel_code.tui.panels.status_bar import StatusBar
from kernel_code.tui.panels.trajectory import TrajectoryPanel


class KernelCodeApp(App):
    """kernel code -- terminal-native GPU kernel optimization tool."""

    TITLE = "kernel code v0.1"
    CSS = """
    Screen {
        layout: vertical;
        background: #1a1a18;
    }

    #main-area {
        height: 1fr;
    }

    #optimization-feed {
        width: 2fr;
        min-width: 30;
    }

    #right-panels {
        width: 3fr;
        min-width: 40;
    }

    #trajectory-panel {
        height: auto;
        min-height: 6;
    }

    #profiling-panel {
        height: auto;
        min-height: 8;
    }

    #experiment-log-panel {
        height: 1fr;
    }
    """

    BINDINGS = APP_BINDINGS

    def __init__(self, session_path: Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._session_path = session_path
        self._session_data: dict = {}
        self._displayed_iterations = 0
        self._paused = False
        self._refresh_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            yield OptimizationFeedPanel(
                session_data=self._session_data,
                visible_iterations=self._session_data.get("iterations", [])[:self._displayed_iterations],
                id="optimization-feed",
            )
            with Vertical(id="right-panels"):
                yield TrajectoryPanel(id="trajectory-panel")
                yield ProfilingPanel(id="profiling-panel")
                yield ExperimentLogPanel(id="experiment-log-panel")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        """Load session data and start the refresh timer."""
        self._load_session()
        self._update_all_panels()
        # Refresh every 2 seconds to simulate live updates
        self._refresh_timer = self.set_interval(2.0, self._tick)

    def _load_session(self) -> None:
        """Load session data from the JSON file."""
        if self._session_path and self._session_path.exists():
            try:
                self._session_data = json.loads(self._session_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._session_data = {}

    def _tick(self) -> None:
        """Timer tick — re-read the cache file and display one more iteration.

        The JSON cache file may be updated by the live engine bridge between
        ticks.  Re-loading ensures we pick up new iterations as they appear.
        Auto-exits when all iterations have been displayed.
        """
        if self._paused:
            return

        # Re-read from disk so we pick up iterations written by the bridge
        self._load_session()

        iterations = self._session_data.get("iterations", [])
        total = len(iterations)

        if self._displayed_iterations < total:
            # Show one more iteration each tick (simulates live optimization)
            self._displayed_iterations += 1
            self._update_all_panels()
        elif total > 0 and self._displayed_iterations >= total:
            # All iterations displayed — auto-exit after a brief pause
            # so the user can see the final state
            if not hasattr(self, "_exit_scheduled"):
                self._exit_scheduled = True
                self.set_timer(3.0, self._auto_exit)

    def _auto_exit(self) -> None:
        """Exit the TUI after all iterations are displayed."""
        self.exit()

    def _update_all_panels(self) -> None:
        """Update all panels with current iteration data."""
        iterations = self._session_data.get("iterations", [])
        visible = iterations[: self._displayed_iterations]

        if not visible:
            return

        # Update trajectory
        try:
            trajectory = self.query_one("#trajectory-panel", TrajectoryPanel)
            trajectory.update_iterations(visible)
        except Exception:
            pass

        # Update profiling with latest iteration's profile
        latest = visible[-1]
        profile = latest.get("profile", {})
        try:
            profiling = self.query_one("#profiling-panel", ProfilingPanel)
            profiling.update_profile(profile)
        except Exception:
            pass

        # Update experiment log
        try:
            log = self.query_one("#experiment-log-panel", ExperimentLogPanel)
            log.update_iterations(visible)
        except Exception:
            pass

        # Update status bar
        try:
            status = self.query_one("#status-bar", StatusBar)
            cost = self._displayed_iterations * 0.02  # ~$0.02 per iteration
            max_iter = self._session_data.get("num_iterations")
            status.update_status(
                iteration=self._displayed_iterations,
                cost=cost,
                hardware=self._session_data.get("hardware", "H100"),
                backend=self._session_data.get("backend", "Triton"),
                max_iterations=max_iter,
            )
        except Exception:
            pass

        # Update optimization feed panel
        try:
            feed = self.query_one("#optimization-feed", OptimizationFeedPanel)
            feed.update_feed(self._session_data, visible)
        except Exception:
            pass

    # --- Actions bound to keybindings ---

    def action_open_dashboard(self) -> None:
        """Open the Plotly Dash dashboard in a browser."""
        session_id = self._session_data.get("session_id", "unknown")
        self.notify(f"Opening dashboard for session {session_id}...")
        webbrowser.open("http://localhost:8050")

    def action_kernel_diff(self) -> None:
        """Show kernel diff (placeholder)."""
        self.notify("Kernel diff view not yet implemented.")

    def action_roofline(self) -> None:
        """Show roofline model (placeholder)."""
        self.notify("Roofline view not yet implemented.")

    def action_pause_resume(self) -> None:
        """Toggle pause/resume of the optimization simulation."""
        self._paused = not self._paused
        state = "PAUSED" if self._paused else "RUNNING"
        self.notify(f"Optimization: {state}")
