"""Tests for kernel_code.shell user-boundary behavior."""

from __future__ import annotations

import io

from rich.console import Console

from kernel_code.shell import KernelCodeShell


def _bare_shell() -> tuple[KernelCodeShell, io.StringIO]:
    """Construct a shell without running __init__ (which loads settings, hooks, etc.).

    The error path of `_smart_optimize` only touches `self._console`, so the
    rest can stay unset.
    """
    buf = io.StringIO()
    shell = KernelCodeShell.__new__(KernelCodeShell)
    shell._console = Console(file=buf, force_terminal=False, width=120)
    return shell, buf


def test_smart_optimize_ambiguous_bare_number_shows_friendly_message() -> None:
    shell, buf = _bare_shell()

    # Bare "0.8" → parse_goal raises ValueError; shell must catch and guide.
    shell._smart_optimize("0.8")

    output = buf.getvalue()
    assert "Ambiguous target" in output
    # Friendly guidance lists both grammars.
    assert "2x" in output
    assert "SOL" in output


def test_smart_optimize_ambiguous_target_without_unit_shows_friendly_message() -> None:
    shell, buf = _bare_shell()

    shell._smart_optimize("target 0.8")

    output = buf.getvalue()
    assert "Ambiguous target" in output
    assert "speedup" in output.lower()
    assert "sol" in output.lower()


def test_smart_optimize_ambiguous_input_does_not_raise() -> None:
    # Regression guard: prior to the fix, ValueError bubbled out of
    # _smart_optimize and crashed the REPL loop's local handler.
    shell, _ = _bare_shell()
    shell._smart_optimize("0.8")  # must not raise
