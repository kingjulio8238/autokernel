"""Welcome-screen renderer for kernel code.

Renders the A2 hero (monogram + MOTD card) — used on first run (from
onboarding.run_onboarding) and on every returning-user shell launch
(from shell.KernelCodeShell).
"""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

__all__ = [
    "HardwareInfo", "Motd",
    "detect_hw", "pick_motd",
    "recent_runs_from_sessions",
    "render_welcome",
]

ACCENT = "magenta"

_KC_MONOGRAM = [
    "\u2588\u2588\u2557  \u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2557",
    "\u2588\u2588\u2551 \u2588\u2588\u2554\u255d\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d",
    "\u2588\u2588\u2588\u2588\u2588\u2554\u255d \u2588\u2588\u2551     ",
    "\u2588\u2588\u2554\u2550\u2588\u2588\u2557 \u2588\u2588\u2551     ",
    "\u2588\u2588\u2551  \u2588\u2588\u2557\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2557",
    "\u255a\u2550\u255d  \u255a\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u255d",
]


_GPU_SPECS: dict[str, dict[str, str]] = {
    "H100":      {"name": "H100 SXM5", "sm": "sm_90", "compute": "989 TFLOPs fp16", "bw": "3.35 TB/s"},
    "A100-80GB": {"name": "A100 80GB", "sm": "sm_80", "compute": "312 TFLOPs fp16", "bw": "2.04 TB/s"},
    "A100-40GB": {"name": "A100 40GB", "sm": "sm_80", "compute": "312 TFLOPs fp16", "bw": "1.56 TB/s"},
    "L40S":      {"name": "L40S",      "sm": "sm_89", "compute": "733 TFLOPs fp16", "bw": "864 GB/s"},
    "RTX4090":   {"name": "RTX 4090",  "sm": "sm_89", "compute": "661 TFLOPs fp16", "bw": "1.01 TB/s"},
}


# ── Helpers ──────────────────────────────────────────────────

def _getattr_or_item(obj, key: str, default=None):
    """Accept both dataclass-style and dict-style settings."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ── HardwareInfo ─────────────────────────────────────────────

@dataclass(frozen=True)
class HardwareInfo:
    """Everything the specs column needs."""
    gpu: str = "L40S"
    sm: str = "sm_89"
    compute: str = "733 TFLOPs fp16"
    bandwidth: str = "864 GB/s"
    backend: str = "triton"
    model: str = "none detected"
    model_ok: bool = False


def detect_hw(creds: dict | None = None, settings=None) -> HardwareInfo:
    """Build HardwareInfo from credentials + settings."""
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
        gpu=specs["name"], sm=specs["sm"],
        compute=specs["compute"], bandwidth=specs["bw"],
        backend=backend, model=model, model_ok=model_ok,
    )


# ── Recent runs ──────────────────────────────────────────────

def recent_runs_from_sessions(limit: int = 3) -> list[dict]:
    """Read recent runs from .kernel-code/runs/*.log."""
    import json as _json
    import re as _re

    runs_dir = Path(__file__).resolve().parent.parent / ".kernel-code" / "runs"
    if not runs_dir.is_dir():
        return []

    results = []
    for f in sorted(runs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit * 2]:
        try:
            content = f.read_text()
            json_match = _re.search(r"JSON SUMMARY\n(\{.*?\})\n", content, _re.DOTALL)
            if json_match:
                data = _json.loads(json_match.group(1))
                ref = data.get("config", {}).get("file", data.get("config", {}).get("reference", ""))
                if ref:
                    p = Path(ref)
                    name = p.parent.name if p.name == "reference.py" and p.parent.name != "." else p.stem
                else:
                    name = f.stem.split("_", 3)[-1]
                speedup = data.get("best_speedup", 0.0)
                if name and speedup > 0:
                    results.append({"name": name, "speedup": speedup})
        except Exception:
            continue
    return results[:limit]


# ── MOTD ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Motd:
    header: str = ""
    body: str = ""


_FIRST_RUN_TIP = (
    f"new here? point [{ACCENT}]/optimize[/{ACCENT}] at a kernel file \u2014 "
    f"or run with [{ACCENT}]--mock[/{ACCENT}] to try it without a gpu."
)


def pick_motd(
    *,
    returning: bool,
    last_version_seen: str | None = None,
    current_version: str = "0.1.0",
    recent_runs: list[dict] | None = None,
) -> Motd:
    """Precedence: new-version > returning > first-run."""
    if last_version_seen and last_version_seen != current_version:
        return Motd(
            header=f"what's new \u00b7 v{current_version}",
            body=(f"[{ACCENT}]+[/{ACCENT}] updated from v{last_version_seen}\n"
                  f"[{ACCENT}]+[/{ACCENT}] see [{ACCENT}]/help[/{ACCENT}] for details"),
        )
    if returning:
        today = date.today().isoformat()
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
    return Motd(header=f"motd \u00b7 {date.today().isoformat()}", body=_FIRST_RUN_TIP)


# ── Renderer ─────────────────────────────────────────────────

def _monogram() -> Text:
    t = Text()
    for i, line in enumerate(_KC_MONOGRAM):
        t.append(line, style=f"bold {ACCENT}")
        if i < len(_KC_MONOGRAM) - 1:
            t.append("\n")
    return t


def _right_column(hw: HardwareInfo, version: str) -> Group:
    title = Text()
    title.append("kernel code", style="bold")
    title.append(f" v{version}", style="dim")
    tagline = Text("agent-driven triton & cuda optimization", style="dim")

    specs = Table.grid(padding=(0, 2), expand=False)
    specs.add_column(style="dim", no_wrap=True)
    specs.add_column()

    gpu_val = Text()
    gpu_val.append(hw.gpu, style="bold")
    gpu_val.append(" \u00b7 ", style="dim")
    gpu_val.append(hw.sm, style="dim")
    specs.add_row("gpu", gpu_val)
    specs.add_row("compute", Text(hw.compute, style="bold"))
    specs.add_row("bandwidth", Text(hw.bandwidth, style="bold"))

    backend_val = Text()
    if hw.backend == "triton":
        backend_val.append("triton", style=f"bold {ACCENT}")
        backend_val.append("  cuda", style="dim")
    else:
        backend_val.append("triton  ", style="dim")
        backend_val.append("cuda", style=f"bold {ACCENT}")
    specs.add_row("backend", backend_val)

    model_val = Text()
    model_val.append(hw.model, style="bold" if hw.model_ok else "dim")
    if hw.model_ok:
        model_val.append(" \u25cf", style="green")
    specs.add_row("model", model_val)

    return Group(title, tagline, Text(""), specs)


def _motd_panel(motd: Motd) -> Panel:
    body = Text.from_markup(motd.body)
    return Panel(body, title=f"[dim]{motd.header}[/dim]",
                 border_style=ACCENT, padding=(0, 1), expand=True)


def _actions_row(returning: bool) -> Text:
    t = Text()

    def act(key: str, label: str) -> None:
        t.append("[", style="dim")
        t.append(key, style=f"bold {ACCENT}")
        t.append("]", style="dim")
        t.append(label + "   ")

    if returning:
        act("1", "resume")
        act("2", "optimize")
        act("3", "skills")
        act("4", "dashboard")
    else:
        act("1", "optimize")
        act("2", "skills")
        act("3", "dashboard")
    act("?", "help")
    return t


def render_welcome(
    con: Console,
    *,
    returning: bool = False,
    hw: HardwareInfo | None = None,
    motd: Motd | None = None,
    version: str = "0.1.0",
) -> None:
    """Print the full welcome screen. Safe to call repeatedly."""
    hw = hw or HardwareInfo()
    motd = motd or pick_motd(returning=returning, current_version=version)

    top = Columns(
        [_monogram(), _right_column(hw, version)],
        padding=(0, 3), equal=False, expand=False,
    )
    motd_panel = _motd_panel(motd)
    actions = _actions_row(returning)
    prompt = Text()
    prompt.append("\u25b8 ", style=f"bold {ACCENT}")

    con.print()
    con.print(top)
    con.print()
    con.print(motd_panel)
    con.print()
    con.print(Rule(style="dim"))
    con.print(actions)
    con.print()
    con.print(prompt)
    con.print()
