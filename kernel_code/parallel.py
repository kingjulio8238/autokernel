"""Parallel backend exploration for kernel optimization.

Runs optimization with both Triton and CUDA backends, compares results,
and reports the winner.  For v1 the two runs execute sequentially (Triton
first, then CUDA).  True parallelism requires two Modal containers and is
planned for a future release.

Usage::

    from kernel_code.parallel import run_parallel_backends
    results = run_parallel_backends(
        reference_path="reference.py",
        config_base=config,
        iterations=10,
    )
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kernel_code.mock_data import _rand, _INTENTS, _KERNEL_SNIPPETS


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _generate_mock_backend_session(
    backend: str,
    num_iterations: int,
    session_id: str,
) -> dict:
    """Generate a mock session for a specific backend with characteristic biases.

    Triton tends to do better on memory-bound kernels.
    CUDA tends to do better on compute-bound kernels.
    Both have random variation to keep things interesting.
    """
    import math
    import random

    ref_runtime_us = _rand(80.0, 300.0)
    best_speedup = 1.0
    iterations = []

    # Backend biases -- Triton is better on memory-bound, CUDA on compute-bound
    if backend == "triton":
        bw_bonus = 0.08       # better bandwidth utilisation
        compute_bonus = -0.03
        ceiling_bias = 0.15   # slightly higher ceiling on avg
    else:  # cuda
        bw_bonus = -0.03
        compute_bonus = 0.08  # better compute utilisation
        ceiling_bias = -0.10

    target_ceiling = _rand(2.0, 3.5) + ceiling_bias

    for i in range(1, num_iterations + 1):
        # Early iterations can error
        if i <= 2 and random.random() < 0.3:
            iterations.append({
                "iteration": i,
                "speedup": 0.0,
                "status": "compile_error",
                "runtime_us": 0.0,
                "ref_runtime_us": ref_runtime_us,
                "profile": {
                    "bandwidth_util": 0.0,
                    "compute_util": 0.0,
                    "cache_efficiency": 0.0,
                    "occupancy": 0.0,
                    "bottleneck_type": "unknown",
                },
                "kernel_code_snippet": _KERNEL_SNIPPETS[0],
                "intent": _INTENTS[min(i, len(_INTENTS) - 1)],
                "decision": "error",
                "error": "CompilationError: undeclared variable",
            })
            continue

        progress = 1 - math.exp(-0.15 * (i - 2))
        noise = _rand(-0.15, 0.15)
        raw_speedup = 1.0 + (target_ceiling - 1.0) * progress + noise

        if random.random() < 0.2:
            raw_speedup *= _rand(0.65, 0.9)

        raw_speedup = max(0.5, round(raw_speedup, 2))
        runtime_us = round(ref_runtime_us / raw_speedup, 1) if raw_speedup > 0 else 0.0

        if raw_speedup > best_speedup:
            status = "keep"
            best_speedup = raw_speedup
        else:
            status = "discard"

        bw_util = min(0.95, _rand(0.3, 0.5) + 0.3 * progress + bw_bonus)
        compute_util = min(0.90, _rand(0.2, 0.4) + 0.25 * progress + compute_bonus)
        cache_eff = min(0.92, _rand(0.3, 0.5) + 0.2 * progress)
        occupancy = min(0.95, _rand(0.4, 0.6) + 0.2 * progress)

        # Bottleneck distribution is biased by backend
        if backend == "triton":
            bottleneck = random.choices(
                ["memory_bound", "compute_bound", "latency_bound"],
                weights=[0.5, 0.3, 0.2],
            )[0]
        else:
            bottleneck = random.choices(
                ["memory_bound", "compute_bound", "latency_bound"],
                weights=[0.3, 0.5, 0.2],
            )[0]

        intent_idx = min(i, len(_INTENTS) - 1)
        snippet_idx = i % len(_KERNEL_SNIPPETS)

        iterations.append({
            "iteration": i,
            "speedup": raw_speedup,
            "status": status,
            "runtime_us": runtime_us,
            "ref_runtime_us": ref_runtime_us,
            "profile": {
                "bandwidth_util": bw_util,
                "compute_util": compute_util,
                "cache_efficiency": cache_eff,
                "occupancy": occupancy,
                "bottleneck_type": bottleneck,
            },
            "kernel_code_snippet": _KERNEL_SNIPPETS[snippet_idx],
            "intent": _INTENTS[intent_idx],
            "decision": status,
            "error": None,
        })

    session = {
        "session_id": session_id,
        "problem": "L1#23 softmax",
        "hardware": "H100",
        "backend": backend,
        "model": "claude-sonnet-4-20250514",
        "ref_runtime_us": ref_runtime_us,
        "best_speedup": best_speedup,
        "num_iterations": num_iterations,
        "iterations": iterations,
    }
    return session


def _summarise_session(session: dict) -> dict:
    """Extract summary metrics from a session dict."""
    iters = session.get("iterations", [])
    kept = [it for it in iters if it["status"] == "keep"]
    errors = [it for it in iters if it["status"] in ("compile_error", "error")]

    best_speedup = session.get("best_speedup", 0.0)
    best_iter = None
    best_profile = {}
    for it in iters:
        if it["status"] == "keep" and it["speedup"] == best_speedup:
            best_iter = it
            best_profile = it.get("profile", {})
            break

    # Dominant bottleneck across all iterations
    bottleneck_counts: dict[str, int] = {}
    for it in iters:
        bn = it.get("profile", {}).get("bottleneck_type", "unknown")
        if bn and bn != "unknown":
            bottleneck_counts[bn] = bottleneck_counts.get(bn, 0) + 1
    dominant_bottleneck = max(bottleneck_counts, key=bottleneck_counts.get) if bottleneck_counts else "unknown"

    return {
        "backend": session.get("backend", "unknown"),
        "best_speedup": best_speedup,
        "iterations_kept": len(kept),
        "iterations_total": len(iters),
        "errors": len(errors),
        "dominant_bottleneck": dominant_bottleneck,
        "best_profile": best_profile,
        "best_iter": best_iter,
        "session": session,
    }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def run_parallel_backends(
    reference_path: str,
    config_base,  # OpenKernelConfig
    iterations: int = 10,
    mock: bool = True,
    console: Console | None = None,
) -> dict:
    """Run optimization with both Triton and CUDA, compare results.

    For v1: runs sequentially (Triton first, then CUDA).
    Returns dict with both results + winner.
    """
    con = console or Console()

    if mock:
        return _run_mock_parallel(iterations=iterations, console=con)
    else:
        return _run_live_parallel(
            reference_path=reference_path,
            config_base=config_base,
            iterations=iterations,
            console=con,
        )


def _run_mock_parallel(iterations: int, console: Console) -> dict:
    """Run both backends in mock mode with biased data."""
    triton_sid = uuid.uuid4().hex[:8]
    cuda_sid = uuid.uuid4().hex[:8]

    console.print("[bold cyan]--- Triton backend ---[/bold cyan]")
    console.print(f"[dim]Session: {triton_sid}[/dim]")
    triton_session = _generate_mock_backend_session("triton", iterations, triton_sid)
    console.print(f"  Completed {iterations} iterations, best speedup: [bold]{triton_session['best_speedup']:.2f}x[/bold]")
    console.print()

    console.print("[bold green]--- CUDA backend ---[/bold green]")
    console.print(f"[dim]Session: {cuda_sid}[/dim]")
    cuda_session = _generate_mock_backend_session("cuda", iterations, cuda_sid)
    console.print(f"  Completed {iterations} iterations, best speedup: [bold]{cuda_session['best_speedup']:.2f}x[/bold]")
    console.print()

    triton_result = _summarise_session(triton_session)
    cuda_result = _summarise_session(cuda_session)

    # Determine winner
    if triton_result["best_speedup"] >= cuda_result["best_speedup"]:
        winner = "triton"
    else:
        winner = "cuda"

    return {
        "triton": triton_result,
        "cuda": cuda_result,
        "winner": winner,
        "winner_session": triton_session if winner == "triton" else cuda_session,
    }


def _run_live_parallel(
    reference_path: str,
    config_base,
    iterations: int,
    console: Console,
) -> dict:
    """Run both backends in live mode (sequentially).

    Creates two OpenKernelConfig copies with Backend.TRITON and Backend.CUDA,
    runs one after the other.  True parallelism (two Modal containers) is
    future work.
    """
    from openkernel.config import Backend, OpenKernelConfig

    ref = Path(reference_path)
    if not ref.exists():
        console.print(f"[red]Error:[/red] file not found: {reference_path}")
        return {"triton": {}, "cuda": {}, "winner": "none", "winner_session": {}}

    reference_source = ref.read_text()

    # -- Triton run --
    triton_config = config_base.model_copy(update={"backend": Backend.TRITON, "max_iterations": iterations})
    triton_sid = uuid.uuid4().hex[:8]

    console.print("[bold cyan]--- Triton backend (live) ---[/bold cyan]")
    console.print(f"[dim]Session: {triton_sid}[/dim]")

    from kernel_code.integration import OpenKernelBridge

    triton_bridge = OpenKernelBridge(
        config=triton_config,
        session_id=triton_sid,
        backend="triton",
    )
    triton_opt_result = triton_bridge.run_optimization(reference_source)

    triton_session = json.loads(triton_bridge.cache_path.read_text())
    triton_result = _summarise_session(triton_session)
    console.print(f"  Best speedup: [bold]{triton_opt_result.final_speedup:.2f}x[/bold]")
    console.print()

    # -- CUDA run --
    cuda_config = config_base.model_copy(update={"backend": Backend.CUDA, "max_iterations": iterations})
    cuda_sid = uuid.uuid4().hex[:8]

    console.print("[bold green]--- CUDA backend (live) ---[/bold green]")
    console.print(f"[dim]Session: {cuda_sid}[/dim]")

    cuda_bridge = OpenKernelBridge(
        config=cuda_config,
        session_id=cuda_sid,
        backend="cuda",
    )
    cuda_opt_result = cuda_bridge.run_optimization(reference_source)

    cuda_session = json.loads(cuda_bridge.cache_path.read_text())
    cuda_result = _summarise_session(cuda_session)
    console.print(f"  Best speedup: [bold]{cuda_opt_result.final_speedup:.2f}x[/bold]")
    console.print()

    # Determine winner
    if triton_result["best_speedup"] >= cuda_result["best_speedup"]:
        winner = "triton"
    else:
        winner = "cuda"

    return {
        "triton": triton_result,
        "cuda": cuda_result,
        "winner": winner,
        "winner_session": triton_session if winner == "triton" else cuda_session,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_comparison(triton_result: dict, cuda_result: dict, console: Console) -> None:
    """Print side-by-side comparison table.

    Columns: Metric | Triton | CUDA | Winner
    Rows: Best Speedup, Iterations Kept, Errors, Dominant Bottleneck, Bandwidth, Compute
    """
    table = Table(
        title="Backend Comparison: Triton vs CUDA",
        show_header=True,
        header_style="bold",
        border_style="dim",
        pad_edge=True,
    )
    table.add_column("Metric", style="dim", width=22)
    table.add_column("Triton", width=16, justify="right")
    table.add_column("CUDA", width=16, justify="right")
    table.add_column("Winner", width=10, justify="center")

    # -- Best Speedup --
    t_spd = triton_result.get("best_speedup", 0.0)
    c_spd = cuda_result.get("best_speedup", 0.0)
    if t_spd > c_spd:
        spd_winner = "[cyan]Triton[/cyan]"
        t_spd_str = f"[bold cyan]{t_spd:.2f}x[/bold cyan]"
        c_spd_str = f"{c_spd:.2f}x"
    elif c_spd > t_spd:
        spd_winner = "[green]CUDA[/green]"
        t_spd_str = f"{t_spd:.2f}x"
        c_spd_str = f"[bold green]{c_spd:.2f}x[/bold green]"
    else:
        spd_winner = "[dim]Tie[/dim]"
        t_spd_str = f"{t_spd:.2f}x"
        c_spd_str = f"{c_spd:.2f}x"
    table.add_row("Best Speedup", t_spd_str, c_spd_str, spd_winner)

    # -- Iterations Kept --
    t_kept = triton_result.get("iterations_kept", 0)
    c_kept = cuda_result.get("iterations_kept", 0)
    t_total = triton_result.get("iterations_total", 0)
    c_total = cuda_result.get("iterations_total", 0)
    if t_kept > c_kept:
        kept_winner = "[cyan]Triton[/cyan]"
    elif c_kept > t_kept:
        kept_winner = "[green]CUDA[/green]"
    else:
        kept_winner = "[dim]Tie[/dim]"
    table.add_row(
        "Iterations Kept",
        f"{t_kept}/{t_total}",
        f"{c_kept}/{c_total}",
        kept_winner,
    )

    # -- Errors --
    t_err = triton_result.get("errors", 0)
    c_err = cuda_result.get("errors", 0)
    if t_err < c_err:
        err_winner = "[cyan]Triton[/cyan]"
    elif c_err < t_err:
        err_winner = "[green]CUDA[/green]"
    else:
        err_winner = "[dim]Tie[/dim]"
    table.add_row("Errors", str(t_err), str(c_err), err_winner)

    # -- Profile metrics from best iteration --
    t_prof = triton_result.get("best_profile", {})
    c_prof = cuda_result.get("best_profile", {})

    for key, label in [
        ("bandwidth_util", "Bandwidth Util"),
        ("compute_util", "Compute Util"),
        ("cache_efficiency", "Cache Efficiency"),
        ("occupancy", "Occupancy"),
    ]:
        tv = t_prof.get(key, 0.0)
        cv = c_prof.get(key, 0.0)
        if tv > cv:
            w = "[cyan]Triton[/cyan]"
        elif cv > tv:
            w = "[green]CUDA[/green]"
        else:
            w = "[dim]Tie[/dim]"
        table.add_row(label, f"{tv:.0%}", f"{cv:.0%}", w)

    # -- Dominant Bottleneck --
    t_bn = triton_result.get("dominant_bottleneck", "unknown")
    c_bn = cuda_result.get("dominant_bottleneck", "unknown")
    table.add_row("Bottleneck", t_bn, c_bn, "")

    console.print()
    console.print(table)

    # -- Overall winner banner --
    if t_spd > c_spd:
        winner_text = Text("Winner: Triton", style="bold cyan")
        delta = t_spd - c_spd
        winner_text.append(f" (+{delta:.2f}x over CUDA)", style="dim")
    elif c_spd > t_spd:
        winner_text = Text("Winner: CUDA", style="bold green")
        delta = c_spd - t_spd
        winner_text.append(f" (+{delta:.2f}x over Triton)", style="dim")
    else:
        winner_text = Text("Tie -- both backends achieved the same speedup", style="bold yellow")

    console.print()
    console.print(Panel(winner_text, border_style="bold", expand=False))
    console.print()
