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


def recent_runs_from_sessions(limit: int = 3) -> list[dict]:
    """Read recent runs from .kernel-code/runs/*.log (JSON SUMMARY block)."""
    import json as _json
    import re as _re

    runs_dir = Path(__file__).resolve().parent.parent / ".kernel-code" / "runs"
    if not runs_dir.is_dir():
        return []

    results = []
    for f in sorted(runs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit * 2]:
        try:
            content = f.read_text()
            # Extract JSON SUMMARY block from the log
            json_match = _re.search(r"JSON SUMMARY\n(\{.*?\})\n", content, _re.DOTALL)
            if json_match:
                data = _json.loads(json_match.group(1))
                ref = data.get("config", {}).get("file", data.get("config", {}).get("reference", ""))
                if ref:
                    p = Path(ref)
                    # For GPU Mode: use parent dir name (vectoradd_py)
                    # For others: use filename without extension
                    name = p.parent.name if p.name == "reference.py" and p.parent.name != "." else p.stem
                else:
                    name = f.stem.split("_", 3)[-1]
                speedup = data.get("best_speedup", 0.0)
                if name and speedup > 0:
                    results.append({"name": name, "speedup": speedup})
        except Exception:
            continue
    return results[:limit]


def pick_motd(
    returning: bool,
    last_version: str | None = None,
    current_version: str | None = None,
    recent_runs: list[dict] | None = None,
) -> Motd:
    """Pick the right MOTD for context.

    Precedence:
    1. Release-note card (version bumped)
    2. Recent-sessions card (returning with runs)
    3. "welcome back" empty state (returning, no runs)
    4. First-run tip (everything else)
    """
    # 1. Release-note card on version bump
    if last_version and current_version and last_version != current_version:
        return Motd(
            header=f"what's new \u00b7 v{current_version}",
            body=(
                f"+ updated from v{last_version}\n"
                f"+ see /help for new commands"
            ),
        )

    if not returning:
        return Motd(
            header="welcome",
            body=(
                "new here? point [bold]optimize[/bold] at a kernel:\n"
                "  [bold]optimize @my_kernel.py 2x $5[/bold]\n"
                "or run [bold]/optimize --mock[/bold] to try without GPU"
            ),
        )

    # 2. Recent-sessions card
    runs = recent_runs or recent_runs_from_sessions(limit=3)
    if runs:
        lines = []
        for r in runs[:3]:
            name = r.get("name", "?")[:20]
            speedup = r.get("speedup", 0.0)
            lines.append(f"{name:<20} \u2192 {speedup:.2f}\u00d7")
        return Motd(
            header="recent",
            body="\n".join(lines),
        )

    # 3. Welcome back empty state
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

    # KC Monogram
    monogram = [
        "\u2588\u2588\u2557  \u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2557",
        "\u2588\u2588\u2551 \u2588\u2588\u2554\u255d\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d",
        "\u2588\u2588\u2588\u2588\u2588\u2554\u255d \u2588\u2588\u2551     ",
        "\u2588\u2588\u2554\u2550\u2588\u2588\u2557 \u2588\u2588\u2551     ",
        "\u2588\u2588\u2551  \u2588\u2588\u2557\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2557",
        "\u255a\u2550\u255d  \u255a\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u255d",
    ]
    for line in monogram:
        console.print(f"  [{_ACCENT}]{line}[/{_ACCENT}]")

    # Title + version
    console.print(f"  [bold white]kernel code[/bold white] [{_DIM}]v{_VERSION}[/{_DIM}]")
    console.print(f"  [{_DIM}]agent-driven triton & cuda optimization[/{_DIM}]")

    # Specs table
    specs_table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    specs_table.add_column("key", style=_DIM, width=12)
    specs_table.add_column("value", style="white")
    specs_table.add_row("gpu", f"{hw.gpu} \u00b7 {hw.sm}")
    specs_table.add_row("compute", hw.compute)
    specs_table.add_row("bandwidth", hw.bandwidth)
    specs_table.add_row("backend", hw.backend)
    key_dot = "[green]\u25cf[/green]" if hw.model_ok else "[red]\u25cf[/red]"
    specs_table.add_row("model", f"{hw.model} {key_dot}")
    console.print(specs_table)
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

    # Action row
    actions = Text()
    actions.append("  ")
    actions.append("[1]", style=f"bold {_ACCENT}")
    actions.append("optimize ", style="white")
    actions.append("[2]", style=f"bold {_ACCENT}")
    actions.append("skills ", style="white")
    actions.append("[3]", style=f"bold {_ACCENT}")
    actions.append("dashboard ", style="white")
    actions.append("[?]", style=f"bold {_ACCENT}")
    actions.append("help", style="white")
    console.print(actions)

    console.print()
