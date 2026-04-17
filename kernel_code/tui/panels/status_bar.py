"""Status bar panel -- footer showing GPU, Backend, Model, Iter, Cost, Time, Tokens.

Format:
  L40S | triton | llama-70b | L1#23 | 25/50 | $0.50 | 1m 42s | 12.5K tok
  [d]ashboard  [k]ernel diff  [r]oofline  [q]uit
"""

from __future__ import annotations

import time

from rich.text import Text

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class StatusBar(Widget):
    """Footer status bar with hardware info, timing, and keybinding hints."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 2;
        background: #252220;
        border-top: solid #3d3835;
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
        max_iterations: int | None = None,
        cost: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._hardware = hardware
        self._backend = backend
        self._model = model
        self._problem = problem
        self._iteration = iteration
        self._max_iterations = max_iterations
        self._cost = cost
        self._start_time: float | None = None
        self._tokens: int | None = None

    def compose(self) -> ComposeResult:
        yield Static(self._format_status(), id="status-info")

    def update_status(
        self,
        iteration: int | None = None,
        cost: float | None = None,
        hardware: str | None = None,
        backend: str | None = None,
        max_iterations: int | None = None,
        tokens: int | None = None,
    ) -> None:
        if iteration is not None:
            self._iteration = iteration
            if self._start_time is None and iteration > 0:
                self._start_time = time.monotonic()
        if cost is not None:
            self._cost = cost
        if hardware is not None:
            self._hardware = hardware
        if backend is not None:
            self._backend = backend
        if max_iterations is not None:
            self._max_iterations = max_iterations
        if tokens is not None:
            self._tokens = tokens
        try:
            info = self.query_one("#status-info", Static)
            info.update(self._format_status())
        except Exception:
            pass

    def _shorten_model(self) -> str:
        """Shorten model name for display."""
        m = self._model
        if "claude-sonnet" in m:
            return "sonnet-4"
        elif "claude-opus" in m:
            return "opus-4"
        elif "MiniMax" in m:
            return "M2.5"
        elif "llama" in m.lower():
            return "llama-70b"
        return m

    def _format_elapsed(self) -> str:
        """Format elapsed time since first iteration."""
        if self._start_time is None:
            return "\u2014"
        elapsed = time.monotonic() - self._start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if minutes > 0:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def _format_tokens(self) -> str:
        """Format token count."""
        if self._tokens is None:
            return "\u2014"
        if self._tokens >= 1_000_000:
            return f"{self._tokens / 1_000_000:.1f}M tok"
        if self._tokens >= 1_000:
            return f"{self._tokens / 1_000:.1f}K tok"
        return f"{self._tokens} tok"

    def _format_iter(self) -> str:
        """Format iteration count, with max if known."""
        if self._max_iterations:
            return f"{self._iteration}/{self._max_iterations}"
        return str(self._iteration)

    def _format_status(self) -> Text:
        """Format the status bar content as Rich Text."""
        model_short = self._shorten_model()
        sep = " \u2502 "

        # --- Line 1: info ---
        line1 = Text()
        line1.append(" ")
        line1.append(self._hardware, style="bold #e8e0d8")
        line1.append(sep, style="#6b6360")
        line1.append(self._backend, style="#e8e0d8")
        line1.append(sep, style="#6b6360")
        line1.append(model_short, style="#e8e0d8")
        line1.append(sep, style="#6b6360")
        line1.append(self._problem, style="bold #e8e0d8")
        line1.append(sep, style="#6b6360")
        line1.append(self._format_iter(), style="bold #fbbf24")
        line1.append(sep, style="#6b6360")
        line1.append(f"${self._cost:.2f}", style="bold #4ade80")
        line1.append(sep, style="#6b6360")
        line1.append(self._format_elapsed(), style="#a09890")
        line1.append(sep, style="#6b6360")
        line1.append(self._format_tokens(), style="#a09890")

        # --- Line 2: keybindings ---
        line2 = Text()
        line2.append(" ")
        for key, label, last in [
            ("d", "ashboard", False),
            ("k", "ernel diff", False),
            ("r", "oofline", False),
            ("q", "uit", True),
        ]:
            line2.append(key, style="bold #e8e0d8 underline")
            line2.append(label, style="#a09890")
            if not last:
                line2.append("  ", style="")

        output = Text()
        output.append_text(line1)
        output.append("\n")
        output.append_text(line2)
        return output
