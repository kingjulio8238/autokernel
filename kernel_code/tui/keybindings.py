"""Keybinding definitions for the kernel code TUI.

Central registry of keyboard shortcuts used by the main App.
"""

from __future__ import annotations

from textual.binding import Binding

# All keybindings for the KernelCodeApp
APP_BINDINGS: list[Binding] = [
    Binding("q", "quit", "Quit", show=True, priority=True),
    Binding("d", "open_dashboard", "Dashboard", show=True),
    Binding("k", "kernel_diff", "Kernel Diff", show=True),
    Binding("r", "roofline", "Roofline", show=True),
    Binding("p", "pause_resume", "Pause/Resume", show=True),
]
