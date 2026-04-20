"""Full session cost tracking dashboard.

Provides granular cost tracking across LLM calls, GPU evaluations, and
optimization runs.  The ``/cost`` command renders a Rich dashboard similar
to Claude Code's built-in cost view.

Usage::

    tracker = CostTracker()
    tracker.record_llm_call("groq/llama-3.3-70b", 1200, 800, 0.001)
    tracker.record_gpu_eval("L40S", 15.0)
    tracker.format_dashboard(console)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# GPU hourly rates (mirrors permissions.py / orchestrator.py)
_GPU_RATES: dict[str, float] = {
    "H100": 3.95,
    "A100-80GB": 2.50,
    "A100-40GB": 2.10,
    "L40S": 2.00,
}


def _fmt_tokens(n: int) -> str:
    """Format a token count for display: 1234 -> '1.2K', 1234567 -> '1.2M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


@dataclass
class ModelUsage:
    """Accumulated usage for a single LLM model."""

    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0


@dataclass
class GpuUsage:
    """Accumulated GPU evaluation usage."""

    gpu_type: str = "L40S"
    eval_count: int = 0
    total_seconds: float = 0.0
    cost_usd: float = 0.0


@dataclass
class RunCost:
    """Cost breakdown for a single optimization run."""

    run_id: str
    llm_cost: float = 0.0
    gpu_cost: float = 0.0
    total: float = 0.0
    iterations: int = 0


class CostTracker:
    """Full session cost tracking -- like Claude Code's /cost.

    Records LLM API calls, GPU evaluations, and per-run cost breakdowns.
    Provides both a one-line summary and a full Rich dashboard.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelUsage] = {}
        self._gpu: GpuUsage = GpuUsage()
        self._runs: list[RunCost] = []
        self._session_start: float = time.time()
        self._active_run: RunCost | None = None

        # Snapshot of provider totals at last read, used for delta tracking
        self._last_provider_tokens: int = 0
        self._last_provider_cost: float = 0.0

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Record an LLM API call."""
        if model_id not in self._models:
            self._models[model_id] = ModelUsage(model_id=model_id)

        usage = self._models[model_id]
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.calls += 1
        usage.cost_usd += cost

        # Also attribute to the active run if one exists
        if self._active_run is not None:
            self._active_run.llm_cost += cost
            self._active_run.total += cost

    def record_gpu_eval(self, gpu_type: str, seconds: float) -> None:
        """Record a Modal GPU evaluation."""
        self._gpu.gpu_type = gpu_type
        self._gpu.eval_count += 1
        self._gpu.total_seconds += seconds

        # Calculate cost from hourly rate
        rate = _GPU_RATES.get(gpu_type, 3.95)
        eval_cost = seconds * (rate / 3600)
        self._gpu.cost_usd += eval_cost

        # Attribute to active run
        if self._active_run is not None:
            self._active_run.gpu_cost += eval_cost
            self._active_run.total += eval_cost

    def record_run_cost(
        self,
        run_id: str,
        llm_cost: float,
        gpu_cost: float,
        iterations: int,
    ) -> None:
        """Record a completed optimization run's cost (bulk recording)."""
        run = RunCost(
            run_id=run_id,
            llm_cost=llm_cost,
            gpu_cost=gpu_cost,
            total=llm_cost + gpu_cost,
            iterations=iterations,
        )
        self._runs.append(run)

    def start_run(self, run_id: str) -> None:
        """Start tracking a new optimization run."""
        self._active_run = RunCost(run_id=run_id)

    def end_run(self, run_id: str, iterations: int = 0) -> None:
        """End tracking for an optimization run."""
        if self._active_run is not None and self._active_run.run_id == run_id:
            self._active_run.iterations = iterations
            self._runs.append(self._active_run)
            self._active_run = None

    def sync_from_provider(self, model_id: str, provider_tokens: int, provider_cost: float) -> None:
        """Sync cumulative totals from an LLMProvider, recording the delta.

        Call this after each agent loop turn to pick up any new LLM usage.
        The delta since the last sync is attributed as a single call.
        """
        delta_tokens = provider_tokens - self._last_provider_tokens
        delta_cost = provider_cost - self._last_provider_cost

        if delta_tokens > 0 or delta_cost > 0:
            # Rough split: assume 40% input, 60% output tokens
            input_est = int(delta_tokens * 0.4)
            output_est = delta_tokens - input_est
            self.record_llm_call(model_id, input_est, output_est, delta_cost)

        self._last_provider_tokens = provider_tokens
        self._last_provider_cost = provider_cost

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        """Total session cost (LLM + GPU)."""
        llm_total = sum(m.cost_usd for m in self._models.values())
        return llm_total + self._gpu.cost_usd

    @property
    def total_llm_cost(self) -> float:
        """Total LLM cost only."""
        return sum(m.cost_usd for m in self._models.values())

    @property
    def total_gpu_cost(self) -> float:
        """Total GPU cost only."""
        return self._gpu.cost_usd

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all models."""
        return sum(
            m.input_tokens + m.output_tokens for m in self._models.values()
        )

    @property
    def total_llm_calls(self) -> int:
        """Total number of LLM API calls."""
        return sum(m.calls for m in self._models.values())

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_summary(self) -> str:
        """One-line summary for post-optimization output.

        Example: 'LLM: $0.08 (12.5K tok) | GPU: $0.20 (45s) | Total: $0.28'
        """
        tok_str = _fmt_tokens(self.total_tokens)
        llm_part = f"LLM: ${self.total_llm_cost:.2f} ({tok_str} tok)"

        parts = [llm_part]

        if self._gpu.eval_count > 0:
            gpu_time = f"{self._gpu.total_seconds:.0f}s"
            gpu_part = f"GPU: ${self.total_gpu_cost:.2f} ({gpu_time})"
            parts.append(gpu_part)

        parts.append(f"Total: ${self.total_cost:.2f}")
        return " | ".join(parts)

    def format_dashboard(self, console: Console | None = None) -> None:
        """Full /cost dashboard with Rich tables."""
        con = console or Console()

        # Build the inner content
        inner_parts: list[Table | Text] = []

        # ---- Table 1: Per-model LLM breakdown ----
        if self._models:
            llm_table = Table(
                show_header=True,
                header_style="bold",
                border_style="dim",
                pad_edge=False,
                expand=True,
            )
            llm_table.add_column("Model", style="cyan", ratio=2)
            llm_table.add_column("Calls", justify="right", width=7)
            llm_table.add_column("In Tok", justify="right", width=9)
            llm_table.add_column("Out Tok", justify="right", width=9)
            llm_table.add_column("Cost", justify="right", width=10)

            for usage in sorted(
                self._models.values(), key=lambda u: u.cost_usd, reverse=True
            ):
                llm_table.add_row(
                    usage.model_id,
                    str(usage.calls),
                    _fmt_tokens(usage.input_tokens),
                    _fmt_tokens(usage.output_tokens),
                    f"${usage.cost_usd:.3f}",
                )

            inner_parts.append(Text("LLM Usage", style="bold"))
            inner_parts.append(llm_table)
        else:
            inner_parts.append(Text("LLM Usage: [dim]no calls yet[/dim]"))

        # ---- Table 2: GPU usage ----
        inner_parts.append(Text())  # spacer

        if self._gpu.eval_count > 0:
            gpu_table = Table(
                show_header=True,
                header_style="bold",
                border_style="dim",
                pad_edge=False,
                expand=True,
            )
            gpu_table.add_column("GPU", style="cyan", ratio=1)
            gpu_table.add_column("Evals", justify="right", width=7)
            gpu_table.add_column("Time", justify="right", width=10)
            gpu_table.add_column("Cost", justify="right", width=10)

            time_str = (
                f"{self._gpu.total_seconds:.0f}s"
                if self._gpu.total_seconds < 120
                else f"{self._gpu.total_seconds / 60:.1f}m"
            )

            gpu_table.add_row(
                self._gpu.gpu_type,
                str(self._gpu.eval_count),
                time_str,
                f"${self._gpu.cost_usd:.3f}",
            )

            inner_parts.append(Text("GPU Usage", style="bold"))
            inner_parts.append(gpu_table)
        else:
            inner_parts.append(Text("GPU Usage: [dim]no evaluations yet[/dim]"))

        # ---- Table 3: Per-run breakdown ----
        if self._runs:
            inner_parts.append(Text())  # spacer

            run_table = Table(
                show_header=True,
                header_style="bold",
                border_style="dim",
                pad_edge=False,
                expand=True,
            )
            run_table.add_column("Run", style="cyan", width=6)
            run_table.add_column("LLM", justify="right", width=9)
            run_table.add_column("GPU", justify="right", width=9)
            run_table.add_column("Total", justify="right", width=9)
            run_table.add_column("Iters", justify="right", width=7)

            for i, run in enumerate(self._runs, 1):
                run_table.add_row(
                    f"#{i}",
                    f"${run.llm_cost:.3f}",
                    f"${run.gpu_cost:.3f}",
                    f"${run.total:.3f}",
                    str(run.iterations),
                )

            inner_parts.append(Text("Per Run", style="bold"))
            inner_parts.append(run_table)

        # ---- Session total ----
        inner_parts.append(Text())  # spacer
        elapsed = time.time() - self._session_start
        if elapsed < 120:
            elapsed_str = f"{elapsed:.0f}s"
        else:
            elapsed_str = f"{elapsed / 60:.1f}m"

        total_line = Text()
        total_line.append("Session Total: ", style="bold")
        total_line.append(f"${self.total_cost:.2f}", style="bold green")
        total_line.append(f"  ({elapsed_str} elapsed)", style="dim")
        inner_parts.append(total_line)

        # Render everything inside a panel
        from rich.console import Group

        panel = Panel(
            Group(*inner_parts),
            title="[bold]Session Cost[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )

        con.print()
        con.print(panel)
        con.print()

    # ------------------------------------------------------------------
    # Serialization (for session persistence)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize tracker state for JSON persistence."""
        return {
            "models": {
                mid: {
                    "model_id": m.model_id,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "calls": m.calls,
                    "cost_usd": m.cost_usd,
                }
                for mid, m in self._models.items()
            },
            "gpu": {
                "gpu_type": self._gpu.gpu_type,
                "eval_count": self._gpu.eval_count,
                "total_seconds": self._gpu.total_seconds,
                "cost_usd": self._gpu.cost_usd,
            },
            "runs": [
                {
                    "run_id": r.run_id,
                    "llm_cost": r.llm_cost,
                    "gpu_cost": r.gpu_cost,
                    "total": r.total,
                    "iterations": r.iterations,
                }
                for r in self._runs
            ],
            "session_start": self._session_start,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CostTracker":
        """Restore tracker state from a serialized dict."""
        tracker = cls()

        tracker._session_start = data.get("session_start", time.time())

        for mid, mdata in data.get("models", {}).items():
            tracker._models[mid] = ModelUsage(
                model_id=mdata["model_id"],
                input_tokens=mdata.get("input_tokens", 0),
                output_tokens=mdata.get("output_tokens", 0),
                calls=mdata.get("calls", 0),
                cost_usd=mdata.get("cost_usd", 0.0),
            )

        gpu_data = data.get("gpu", {})
        tracker._gpu = GpuUsage(
            gpu_type=gpu_data.get("gpu_type", "L40S"),
            eval_count=gpu_data.get("eval_count", 0),
            total_seconds=gpu_data.get("total_seconds", 0.0),
            cost_usd=gpu_data.get("cost_usd", 0.0),
        )

        for rdata in data.get("runs", []):
            tracker._runs.append(
                RunCost(
                    run_id=rdata["run_id"],
                    llm_cost=rdata.get("llm_cost", 0.0),
                    gpu_cost=rdata.get("gpu_cost", 0.0),
                    total=rdata.get("total", 0.0),
                    iterations=rdata.get("iterations", 0),
                )
            )

        # Set the last-provider snapshot to current totals so first sync
        # doesn't double-count restored data.
        tracker._last_provider_tokens = tracker.total_tokens
        tracker._last_provider_cost = tracker.total_llm_cost

        return tracker
