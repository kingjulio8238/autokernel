"""User-friendly error formatting for kernel-code.

Displays concise, actionable error messages via Rich panels and logs full
tracebacks to `.kernel-code/error.log` for debugging.

Usage::

    from kernel_code.errors import format_error

    try:
        risky_operation()
    except Exception as exc:
        format_error(exc, context="Command: /optimize", console=console)
"""

from __future__ import annotations

import logging
import re
import traceback
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)

# Error log file for full tracebacks
_ERROR_LOG = Path(".kernel-code/error.log")


def format_error(exc: Exception, context: str = "", console: Console | None = None) -> None:
    """Display a user-friendly error message. Log full traceback to error.log."""

    # Log full traceback for debugging
    _ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_ERROR_LOG, "a") as f:
        f.write(f"\n{'='*60}\n{context}\n")
        traceback.print_exc(file=f)

    # Format user-friendly message
    msg, fix = _classify_error(exc)

    panel_content = f"[bold red]{msg}[/bold red]"
    if fix:
        panel_content += f"\n[dim]Fix: {fix}[/dim]"
    panel_content += f"\n[dim]Full traceback logged to: {_ERROR_LOG}[/dim]"

    c = console or Console()
    c.print(Panel(panel_content, title="Error", border_style="red", padding=(0, 1)))


def _classify_error(exc: Exception) -> tuple[str, str]:
    """Classify an error and return (message, fix_suggestion)."""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__

    # Rate limiting
    if "rate_limit" in exc_str or "429" in exc_str:
        wait = _extract_wait_time(exc_str)
        return (
            "Rate limited by LLM provider.",
            f"Wait {wait}s and retry, or switch to a different model with /config set default_model MODEL",
        )

    # Authentication
    if "auth" in exc_str or "api_key" in exc_str or "unauthorized" in exc_str:
        return (
            "Authentication failed.",
            "Check your API key: export GROQ_API_KEY=... or export MINIMAX_API_KEY=...",
        )

    # Modal
    if "modal" in exc_str or "hydrated" in exc_str:
        return (
            "Modal GPU eval failed.",
            "Run: python modal_infra/deploy.py",
        )

    # Configuration
    if "configuration" in exc_type.lower() or "configurationerror" in exc_type.lower():
        return (str(exc), "Check .kernel-code/settings.yaml and KERNEL.md")

    # Timeout
    if "timeout" in exc_str:
        return (
            "Operation timed out.",
            "Try --eval-mode fast or increase timeout in config",
        )

    # Kernel validation
    if "modelnew" in exc_str or "validation" in exc_str:
        return (
            "Generated kernel failed validation.",
            "Kernel must define a ModelNew class. Try a different optimization approach.",
        )

    # Import errors
    if isinstance(exc, ImportError):
        module = str(exc).split("'")[1] if "'" in str(exc) else "unknown"
        return (
            f"Missing dependency: {module}",
            f"pip install {module}",
        )

    # Generic fallback
    return (
        f"{exc_type}: {str(exc)[:200]}",
        "Check .kernel-code/error.log for details",
    )


def _extract_wait_time(exc_str: str) -> int:
    """Try to extract wait time from rate limit error message."""
    match = re.search(r"(\d+)\.?\d*\s*s", exc_str)
    return int(match.group(1)) if match else 20
