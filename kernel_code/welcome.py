"""Welcome screen renderer — A2 hero card with monogram + MOTD.

Renders on every shell launch (first run or returning). Context-aware
MOTD shows tips, recent sessions, or release notes.

Usage::

    from kernel_code.welcome import render_welcome, detect_hw, pick_motd
    render_welcome(console, returning=False, hw=detect_hw(settings), motd=motd)
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_ACCENT = "magenta"
_DIM = "#999999"
_VERSION = "0.1.0"

try:
    _VERSION = importlib.metadata.version("openkernel")
except Exception:
    pass

# GPU specs lookup
_GPU_SPECS = {
    "L40S": {"compute": "91.6 TFLOPs fp32", "bandwidth": "864 GB/s", "sm": "sm_89"},
    "H100": {"compute": "989 TFLOPs fp16", "bandwidth": "3.35 TB/s", "sm": "sm_90"},
    "A100-80GB": {"compute": "312 TFLOPs fp16", "bandwidth": "2.0 TB/s", "sm": "sm_80"},
    "A100-40GB": {"compute": "312 TFLOPs fp16", "bandwidth": "1.6 TB/s", "sm": "sm_80"},
}


@dataclass
class HardwareInfo:
    gpu: str = "L40S"
    sm: str = "sm_89"
    compute: str = "91.6 TFLOPs fp32"
    bandwidth: str = "864 GB/s"
    backend: str = "triton"
    model: str = "gpt-4o"
    model_ok: bool = False


@dataclass
class Motd:
    header: str = ""
    body: str = ""


def detect_hw(settings=None) -> HardwareInfo:
    """Detect hardware info from settings."""
    import os

    gpu = "L40S"
    backend = "triton"
    model = "gpt-4o"
    model_ok = False

    if settings is not None:
        gpu = getattr(settings, "default_gpu", gpu)
        backend = getattr(settings, "default_backend", backend)
        model = getattr(settings, "default_model", model)

        # Check if model's API key is set
        provider = getattr(settings, "default_provider", "")
        from kernel_code.settings import _API_KEY_ENV_MAP
        env_key = _API_KEY_ENV_MAP.get(f"{provider}_api_key", "")
        model_ok = bool(os.environ.get(env_key)) or bool(getattr(settings, f"{provider}_api_key", None))

    specs = _GPU_SPECS.get(gpu, _GPU_SPECS["L40S"])

    return HardwareInfo(
        gpu=gpu,
        sm=specs["sm"],
        compute=specs["compute"],
        bandwidth=specs["bandwidth"],
        backend=backend,
        model=model,
        model_ok=model_ok,
    )


def pick_motd(returning: bool, last_version: str | None = None) -> Motd:
    """Pick the right MOTD for context."""
    if not returning:
        return Motd(
            header="welcome",
            body=(
                "new here? point [bold]optimize[/bold] at a kernel:\n"
                "  [bold]optimize @my_kernel.py 2x $5[/bold]\n"
                "or run [bold]/optimize --mock[/bold] to try without GPU"
            ),
        )

    # Check for recent sessions
    runs_dir = Path(__file__).resolve().parent.parent / ".kernel-code" / "runs"
    if runs_dir.is_dir():
        logs = sorted(runs_dir.glob("*.log"), reverse=True)
        if logs:
            recent = logs[0].name
            return Motd(
                header="recent",
                body=f"last run: {recent}\ntype [bold]optimize[/bold] to continue",
            )

    return Motd(
        header="welcome back",
        body="type [bold]optimize[/bold] to start, or [bold]/help[/bold] for commands",
    )


def render_welcome(
    console: Console,
    *,
    returning: bool = False,
    hw: HardwareInfo | None = None,
    motd: Motd | None = None,
) -> None:
    """Render the A2 hero welcome card."""
    hw = hw or HardwareInfo()
    motd = motd or pick_motd(returning)

    console.print()

    # Monogram + title
    title = Text()
    title.append("  KC", style=f"bold {_ACCENT}")
    title.append("  openkernel", style="bold white")
    title.append(f"  v{_VERSION}", style=_DIM)
    console.print(title)

    # Tagline
    console.print(f"  [{_DIM}]agent-driven triton & cuda kernel optimization[/{_DIM}]")
    console.print()

    # Specs table
    key_dot = f"[green]\u2713[/green]" if hw.model_ok else f"[red]\u2717[/red]"
    specs = Text()
    specs.append(f"  {key_dot} {hw.model}", style="white")
    specs.append(f"  \u00b7  ", style=_DIM)
    specs.append(f"{hw.gpu}", style="white")
    specs.append(f" ({hw.compute})", style=_DIM)
    specs.append(f"  \u00b7  ", style=_DIM)
    specs.append(f"{hw.backend}", style="white")
    console.print(specs)
    console.print()

    # MOTD card
    if motd.body:
        motd_text = Text.from_markup(motd.body)
        panel = Panel(
            motd_text,
            title=f"[{_DIM}]{motd.header}[/{_DIM}]",
            border_style=_ACCENT,
            padding=(0, 2),
            width=min(console.width, 70),
        )
        console.print(f"  ", end="")
        console.print(panel)

    console.print()
