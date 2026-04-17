"""Status bar panel — footer showing GPU, Backend, Model, Iter, Cost.

Matches the design mockup:
  H100 | Triton | claude-sonnet-4 | L1#23 | Iter 6 | $0.12
  [d]ashboard [k]diff [r]oofline [q]uit
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class StatusBar(Widget):
    """Footer status bar with hardware info and keybinding hints."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 2;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        hardware: str = "H100",
        backend: str = "Triton",
        model: str = "claude-sonnet-4-20250514",
        problem: str = "L1#23",
        iteration: int = 0,
        cost: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._hardware = hardware
        self._backend = backend
        self._model = model
        self._problem = problem
        self._iteration = iteration
        self._cost = cost

    def compose(self) -> ComposeResult:
        yield Static(self._format_status(), id="status-info")

    def update_status(
        self,
        iteration: int | None = None,
        cost: float | None = None,
        hardware: str | None = None,
        backend: str | None = None,
    ) -> None:
        if iteration is not None:
            self._iteration = iteration
        if cost is not None:
            self._cost = cost
        if hardware is not None:
            self._hardware = hardware
        if backend is not None:
            self._backend = backend
        try:
            info = self.query_one("#status-info", Static)
            info.update(self._format_status())
        except Exception:
            pass

    def _format_status(self) -> str:
        """Format the status bar content."""
        # Shorten model name for display
        model_short = self._model
        if "claude-sonnet" in model_short:
            model_short = "sonnet-4"
        elif "claude-opus" in model_short:
            model_short = "opus-4"
        elif "MiniMax" in model_short:
            model_short = "M2.5"
        elif "llama" in model_short.lower():
            model_short = "llama-70b"

        info_line = (
            f" {self._hardware} | {self._backend} | {model_short} "
            f"| {self._problem} | Iter {self._iteration} | ${self._cost:.2f}"
        )
        keys_line = " [d]ashboard  [k]ernel diff  [r]oofline  [q]uit"
        return f"{info_line}\n{keys_line}"
