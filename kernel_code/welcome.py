"""Welcome screen renderer — A2 hero card with monogram + MOTD.

Renders on every shell launch (first run or returning). Context-aware
MOTD shows tips, recent sessions, or release notes.

Usage::

    from kernel_code.welcome import render_welcome, detect_hw, pick_motd
    render_welcome(console, returning=False, hw=detect_hw(settings), motd=motd)
"""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ACCENT = "magenta"
_ACCENT = ACCENT  # alias
_DIM = "#999999"
_VERSION = "0.1.0"

try:
    _VERSION = importlib.metadata.version("openkernel")
except Exception:
    pass

_FIRST_RUN_TIP = (
    "new here? point [bold]optimize[/bold] at a kernel:\n"
    "  [bold]optimize @my_kernel.py 2x $5[/bold]\n"
    "or run [bold]/optimize --mock[/bold] to try without GPU"
)


def _getattr_or_item(obj, key: str, default=None):
    """Accept both dataclass-style and dict-style settings."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

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


def detect_hw(creds: dict | None = None, settings=None) -> HardwareInfo:
    """Detect hardware info from settings. Accepts dataclass or dict."""
    creds = creds or {}
    gpu_key = _getattr_or_item(settings, "default_gpu", "L40S")
    specs = _GPU_SPECS.get(gpu_key, _GPU_SPECS["L40S"])
    backend = _getattr_or_item(settings, "default_backend", "triton")

    model = _getattr_or_item(settings, "default_model", "") or ""
    model_ok = False
    if model:
        if "groq" in model.lower():
            model_ok = bool(creds.get("groq") or
                            os.environ.get("GROQ_API_KEY") or
                            _getattr_or_item(settings, "groq_api_key"))
        elif "openai" in model.lower() or "gpt" in model.lower():
            model_ok = bool(creds.get("openai") or
                            os.environ.get("OPENAI_API_KEY") or
                            _getattr_or_item(settings, "openai_api_key"))
        elif "anthropic" in model.lower() or "claude" in model.lower():
            model_ok = bool(creds.get("anthropic") or
                            os.environ.get("ANTHROPIC_API_KEY") or
                            _getattr_or_item(settings, "anthropic_api_key"))
        elif "minimax" in model.lower():
            model_ok = bool(os.environ.get("MINIMAX_API_KEY") or
                            _getattr_or_item(settings, "minimax_api_key"))
        else:
            model_ok = True
    else:
        model = "none detected"

    return HardwareInfo(
        gpu=gpu_key,
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
    *,
    returning: bool,
    last_version_seen: str | None = None,
    current_version: str = "0.1.0",
    recent_runs: list[dict] | None = None,
) -> Motd:
    """Pick the right MOTD. Precedence: version-bump > returning > first-run."""
    today = date.today().isoformat()

    # 1. Release-note card on version bump
    if last_version_seen and last_version_seen != current_version:
        return Motd(
            header=f"what's new \u00b7 v{current_version}",
            body=(
                f"[{ACCENT}]+[/{ACCENT}] updated from v{last_version_seen}\n"
                f"[{ACCENT}]+[/{ACCENT}] see [{ACCENT}]/help[/{ACCENT}] for details"
            ),
        )

    # 2. Returning-user card
    if returning:
        runs = recent_runs if recent_runs is not None else recent_runs_from_sessions(limit=3)
        if runs:
            lines = []
            for r in runs[:3]:
                name = r.get("name", "?")[:22]
                spd = r.get("speedup", 0.0)
                spd_str = f"[green]{spd:.2f}\u00d7[/green]" if spd else "[dim]\u2014[/dim]"
                lines.append(f"  {name:<22s} \u2192 {spd_str}")
            return Motd(header=f"recent \u00b7 {today}", body="\n".join(lines))
        return Motd(
            header=f"welcome back \u00b7 {today}",
            body=f"[dim]no runs yet \u2014 point [{ACCENT}]/optimize[/{ACCENT}] at a kernel.[/dim]",
        )

    # 3. First-run tip
    return Motd(header=f"motd \u00b7 {today}", body=_FIRST_RUN_TIP)


def render_welcome(
    console: Console,
    *,
    returning: bool = False,
    hw: HardwareInfo | None = None,
    motd: Motd | None = None,
    version: str | None = None,
) -> None:
    """Render the A2 hero welcome card."""
    hw = hw or HardwareInfo()
    motd = motd or pick_motd(returning=returning)
    ver = version or _VERSION

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
    console.print(f"  [bold white]kernel code[/bold white] [{_DIM}]v{ver}[/{_DIM}]")
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
