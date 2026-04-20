"""First-run onboarding flow for kernel code.

Provides an interactive setup wizard that runs on the first invocation,
detecting credentials, creating default settings, and running a quick
demo optimization to build confidence.

Usage::

    from kernel_code.onboarding import needs_onboarding, run_onboarding

    if needs_onboarding():
        run_onboarding()
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def needs_onboarding() -> bool:
    """Check if this is a first run (no .kernel-code/ directory)."""
    return not Path(".kernel-code").exists()


def run_onboarding(console: Console | None = None) -> dict:
    """Interactive first-run setup wizard. Returns setup results dict.

    The flow:
        1. Welcome banner
        2. Detect credentials (Modal, Groq, MiniMax, HF Hub)
        3. Create .kernel-code/settings.yaml with detected defaults
        4. Optionally create KERNEL.md
        5. Run a quick mock demo (5 iterations, inline output)
        6. Show next-steps panel

    Never crashes -- all errors are caught and handled gracefully.
    Skippable via Ctrl+C at any prompt.
    """
    con = console or Console()
    results: dict = {
        "completed": False,
        "credentials": {},
        "settings_created": False,
        "kernel_md_created": False,
        "demo_ran": False,
    }

    try:
        # Step 1: Welcome — A2 hero card from welcome.py
        from kernel_code.welcome import render_welcome, detect_hw, pick_motd
        hw = detect_hw()  # pre-credential: fallback values
        motd = pick_motd(returning=False)
        render_welcome(con, returning=False, hw=hw, motd=motd)

        # Step 2: Detect credentials
        creds = _step_detect_credentials(con)
        results["credentials"] = creds

        # Step 3: Create settings
        settings_info = _step_create_settings(con, creds)
        results["settings_created"] = True
        results["settings_info"] = settings_info

        # Step 4: KERNEL.md (optional)
        kernel_md_created = _step_kernel_md(con, creds)
        results["kernel_md_created"] = kernel_md_created

        # Steps 5+6 absorbed by welcome screen.
        # Demo is opt-in via /optimize --mock after onboarding.

        results["completed"] = True

    except KeyboardInterrupt:
        con.print()
        con.print("[dim]Setup skipped. You can re-run anytime with /setup[/dim]")
        con.print()
        # Still ensure .kernel-code/ exists so onboarding doesn't repeat
        _ensure_project_dir()

    except Exception as exc:
        con.print(f"[yellow]Setup encountered an issue:[/yellow] {exc}")
        con.print("[dim]Continuing anyway. Re-run with /setup if needed.[/dim]")
        con.print()
        _ensure_project_dir()

    return results


# ---------------------------------------------------------------------------
# Step 1: Welcome
# ---------------------------------------------------------------------------


def _step_welcome(con: Console) -> None:
    """Display the welcome banner."""
    con.print()
    welcome = Panel(
        Text.from_markup(
            "[bold]Welcome to kernel code[/bold] v0.1\n"
            "Interactive GPU kernel optimization\n"
            "\n"
            "Let's get you set up."
        ),
        border_style="cyan",
        padding=(1, 3),
    )
    con.print(welcome)
    time.sleep(1.5)


# ---------------------------------------------------------------------------
# Step 2: Detect credentials
# ---------------------------------------------------------------------------


def _has_modal() -> bool:
    """Check if Modal is configured."""
    return bool(
        os.environ.get("MODAL_TOKEN_ID")
        or Path.home().joinpath(".modal.toml").is_file()
        or Path.home().joinpath(".modal").is_dir()
    )


def _has_groq() -> bool:
    """Check if Groq API key is set."""
    return bool(os.environ.get("GROQ_API_KEY"))


def _has_minimax() -> bool:
    """Check if MiniMax API key is set."""
    return bool(os.environ.get("MINIMAX_API_KEY"))


def _has_hf() -> bool:
    """Check if Hugging Face token is set."""
    return bool(os.environ.get("HF_TOKEN"))


def _pick_best_model(creds: dict) -> tuple[str, str]:
    """Pick the best available LLM model. Returns (model_id, description)."""
    if creds.get("groq"):
        return "groq/llama-3.3-70b-versatile", "groq/llama-3.3-70b-versatile (free)"
    if creds.get("minimax"):
        return "openai/MiniMax-M2.5", "openai/MiniMax-M2.5"
    # No LLM credential found -- default to Groq (user can set up later)
    return "groq/llama-3.3-70b-versatile", "none detected (demo mode only)"


def _pick_best_provider(creds: dict) -> str:
    """Pick the best available LLM provider name."""
    if creds.get("groq"):
        return "groq"
    if creds.get("minimax"):
        return "minimax"
    return "groq"


def _pick_gpu(creds: dict) -> tuple[str, str]:
    """Pick the best available GPU. Returns (gpu_type, description)."""
    if creds.get("modal"):
        return "L40S", "L40S via Modal ($2.00/hr)"
    return "L40S", "none detected (demo mode only)"


def _step_detect_credentials(con: Console) -> dict:
    """Detect available credentials and display results."""
    con.print()
    con.print("[bold]Checking your environment...[/bold]")
    con.print()

    creds = {
        "modal": _has_modal(),
        "groq": _has_groq(),
        "minimax": _has_minimax(),
        "hf": _has_hf(),
    }

    # Build credential status lines
    def _status(found: bool, found_detail: str, missing_detail: str) -> str:
        if found:
            return f"[green]\\[ok][/green] {found_detail}"
        return f"[red]\\[--][/red] {missing_detail}"

    con.print(f"  Modal:   {_status(creds['modal'], 'authenticated (L40S GPU available)', 'not configured')}")
    con.print(f"  Groq:    {_status(creds['groq'], 'API key found (free tier)', 'not configured')}")
    con.print(f"  MiniMax: {_status(creds['minimax'], 'API key found', 'not configured')}")
    con.print(f"  HF Hub:  {_status(creds['hf'], 'token found', 'not configured')}")

    con.print()

    # Summary: chosen model + GPU
    model_id, model_desc = _pick_best_model(creds)
    _gpu_type, gpu_desc = _pick_gpu(creds)

    con.print(f"  [bold]LLM:[/bold]  {model_desc}")
    con.print(f"  [bold]GPU:[/bold]  {gpu_desc}")
    con.print()

    # If no LLM at all, show setup hint
    has_any_llm = creds["groq"] or creds["minimax"]
    if not has_any_llm:
        con.print(
            "[yellow]No LLM API key detected.[/yellow] "
            "The demo will use mock data.\n"
            "  To set up Groq (free): [bold]export GROQ_API_KEY=your_key[/bold]\n"
            "  To set up MiniMax:     [bold]export MINIMAX_API_KEY=your_key[/bold]"
        )
        con.print()

    if not creds["modal"]:
        con.print(
            "[dim]GPU not configured. To set up Modal: [bold]modal setup[/bold][/dim]"
        )
        con.print()

    return creds


# ---------------------------------------------------------------------------
# Step 3: Create settings
# ---------------------------------------------------------------------------


def _ensure_project_dir() -> Path:
    """Ensure .kernel-code/ directory exists. Returns the path."""
    project_dir = Path(".kernel-code")
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def _step_create_settings(con: Console, creds: dict) -> dict:
    """Create .kernel-code/settings.yaml with detected defaults."""
    con.print("[bold]Creating .kernel-code/settings.yaml with detected defaults...[/bold]")

    model_id, _ = _pick_best_model(creds)
    provider = _pick_best_provider(creds)
    gpu_type, _ = _pick_gpu(creds)
    backend = "triton"

    settings_info = {
        "default_model": model_id,
        "default_provider": provider,
        "default_backend": backend,
        "default_gpu": gpu_type,
    }

    con.print(f"  default_model:   {model_id}")
    con.print(f"  default_backend: {backend}")
    con.print(f"  default_gpu:     {gpu_type}")

    # Write the settings file
    project_dir = _ensure_project_dir()
    settings_path = project_dir / "settings.yaml"

    try:
        import yaml  # type: ignore[import-untyped]

        settings_data = {
            "default_model": model_id,
            "default_provider": provider,
            "default_backend": backend,
            "default_gpu": gpu_type,
            "auto_confirm_under": 0.10,
            "auto_save": True,
            "capture_traces": True,
            "show_profiling": True,
            "show_trajectory": True,
        }

        with open(settings_path, "w") as f:
            yaml.dump(settings_data, f, default_flow_style=False, sort_keys=False)

    except ImportError:
        # If yaml is not available, write manually
        lines = [
            f"default_model: {model_id}",
            f"default_provider: {provider}",
            f"default_backend: {backend}",
            f"default_gpu: {gpu_type}",
            "auto_confirm_under: 0.1",
            "auto_save: true",
            "capture_traces: true",
            "show_profiling: true",
            "show_trajectory: true",
        ]
        settings_path.write_text("\n".join(lines) + "\n")

    con.print(f"  [green]Settings saved[/green] ({settings_path})")
    con.print()

    return settings_info


# ---------------------------------------------------------------------------
# Step 4: KERNEL.md
# ---------------------------------------------------------------------------


def _step_kernel_md(con: Console, creds: dict) -> bool:
    """Optionally create a KERNEL.md in the current directory."""
    kernel_md_path = Path("KERNEL.md")

    if kernel_md_path.exists():
        con.print(f"[dim]KERNEL.md already exists -- skipping creation.[/dim]")
        con.print()
        return False

    con.print(
        "[bold]Create a KERNEL.md in this directory?[/bold] (recommended)\n"
        "This tells kernel code about your hardware and optimization constraints."
    )

    try:
        answer = con.input("[bold cyan](y/n):[/bold cyan] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        con.print()
        con.print("[dim]Skipped KERNEL.md creation.[/dim]")
        con.print()
        return False

    if answer not in ("y", "yes"):
        con.print("[dim]Skipped.[/dim]")
        con.print()
        return False

    # Determine hardware from credentials
    gpu_type, _ = _pick_gpu(creds)

    gpu_descriptions = {
        "L40S": "NVIDIA L40S (48GB, Ada Lovelace)",
        "H100": "NVIDIA H100 (80GB, Hopper)",
        "A100-80GB": "NVIDIA A100 (80GB, Ampere)",
        "A100-40GB": "NVIDIA A100 (40GB, Ampere)",
    }
    gpu_desc = gpu_descriptions.get(gpu_type, f"NVIDIA {gpu_type}")

    kernel_md_content = f"""\
---
backend: triton
hardware: {gpu_type}
target_occupancy: 0.8
must_beat_baseline: true
---

# Kernel Optimization Config

## Constraints
- Target hardware: {gpu_desc}
- Prefer Triton backend for portability
- Must pass correctness at fp32 atol=1e-4

## Hints
- For memory-bound kernels: prioritize vectorized loads and coalescing
- For compute-bound kernels: use tensor cores via tl.dot
- Accumulate in float64 for fp32 GEMM to avoid precision issues
"""

    kernel_md_path.write_text(kernel_md_content)
    con.print(f"  [green]Created KERNEL.md[/green] (edit to customize)")
    con.print()
    return True


# ---------------------------------------------------------------------------
# Step 5: Quick demo
# ---------------------------------------------------------------------------


# Demo-specific intents and speedups for a deterministic, satisfying sequence.
_DEMO_ITERATIONS = [
    {"speedup": 0.84, "status": "discard", "intent": "naive tiling"},
    {"speedup": 1.05, "status": "keep", "intent": "vectorized loads"},
    {"speedup": 1.47, "status": "keep", "intent": "online softmax"},
    {"speedup": 1.33, "status": "discard", "intent": "register blocking"},
    {"speedup": 1.72, "status": "keep", "intent": "warp-level shuffle"},
]


def _step_quick_demo(con: Console) -> None:
    """Run a quick 5-iteration mock optimization with inline output."""
    con.print("[bold]Running a quick demo optimization (mock data, free)...[/bold]")
    con.print()

    best_speedup = 0.0

    for i, it in enumerate(_DEMO_ITERATIONS, 1):
        # Simulate a brief pause for each iteration
        time.sleep(0.4)

        speedup = it["speedup"]
        status = it["status"]
        intent = it["intent"]

        if status == "keep":
            best_speedup = max(best_speedup, speedup)
            marker = "[green]keep[/green]  "
            speedup_style = f"[green]{speedup:.2f}x[/green]"
        else:
            marker = "[red]discard[/red]"
            speedup_style = f"[dim]{speedup:.2f}x[/dim]"

        con.print(f"  Iter {i}: {speedup_style}  {marker}  [dim]{intent}[/dim]")

    con.print()

    # Summary line
    summary = Text()
    summary.append("  Best: ", style="dim")
    summary.append(f"{best_speedup:.2f}x", style="bold green")
    summary.append(" in 5 iterations", style="dim")
    con.print(summary)
    con.print()


# ---------------------------------------------------------------------------
# Step 6: What's next
# ---------------------------------------------------------------------------


def _step_whats_next(con: Console) -> None:
    """Display the what's-next panel with quick commands."""
    # Build the commands table inside the panel
    lines = [
        "[bold]You're ready![/bold]",
        "",
        "[bold]Quick commands:[/bold]",
        "  [cyan]/optimize --reference FILE[/cyan]    run optimization",
        "  [cyan]/skills[/cyan]                       browse optimization skills",
        "  [cyan]/dashboard[/cyan]                    visualize results",
        "  [cyan]/help[/cyan]                         all commands",
        "",
        "[dim]Try:[/dim] [bold]/optimize --mock --iterations 10[/bold]",
    ]

    panel = Panel(
        Text.from_markup("\n".join(lines)),
        border_style="green",
        padding=(1, 3),
    )
    con.print(panel)
    con.print()
