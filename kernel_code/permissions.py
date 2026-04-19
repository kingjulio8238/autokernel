"""Cost-gated eval permissions for kernel optimization runs.

Provides cost estimation, user confirmation, and budget tracking so
optimization runs don't surprise the user with unexpected charges.
"""

from __future__ import annotations

from rich.console import Console

from openkernel.exceptions import BudgetExceededError

# GPU hourly rates (same as orchestrator.py)
_GPU_RATES: dict[str, float] = {
    "H100": 3.95,
    "A100-80GB": 2.50,
    "A100-40GB": 2.10,
    "L40S": 2.00,
}


def estimate_cost(
    iterations: int,
    gpu_type: str = "L40S",
    avg_eval_seconds: float = 15.0,
    llm_cost_per_iter: float = 0.003,
) -> float:
    """Estimate total cost for an optimization run.

    Combines GPU compute cost (iterations * eval time * hourly rate)
    with LLM inference cost (iterations * per-call cost).
    """
    gpu_rate = _GPU_RATES.get(gpu_type, 3.95)
    compute_cost = iterations * avg_eval_seconds * (gpu_rate / 3600)
    llm_cost = iterations * llm_cost_per_iter
    return compute_cost + llm_cost


def confirm_cost(
    estimated_cost: float,
    gpu_type: str,
    iterations: int,
    console: Console | None = None,
) -> bool:
    """Show cost estimate and ask for confirmation. Returns True if approved.

    Auto-approves runs estimated under $0.10.
    """
    con = console or Console()

    con.print(
        f"[bold]Estimated cost:[/bold] ~${estimated_cost:.2f} "
        f"(up to {iterations} iterations on {gpu_type}, may stop earlier)"
    )

    if estimated_cost < 0.10:
        con.print("[dim]Auto-approved (< $0.10)[/dim]")
        return True

    try:
        answer = con.input("[bold yellow]Proceed? (y/n):[/bold yellow] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        con.print()
        return False

    return answer in ("y", "yes")


class BudgetTracker:
    """Track cumulative session cost with optional limit."""

    def __init__(self, max_budget: float | None = None) -> None:
        self._total: float = 0.0
        self._max_budget: float | None = max_budget

    def record(self, cost: float) -> None:
        """Record a cost incurred."""
        self._total += cost

    def check(self, estimated_additional: float) -> bool:
        """Return False if spending *estimated_additional* would exceed budget."""
        if self._max_budget is None:
            return True
        return (self._total + estimated_additional) <= self._max_budget

    @property
    def total_spent(self) -> float:
        return self._total

    def enforce(self, estimated_additional: float = 0.0) -> None:
        """Raise BudgetExceededError if budget would be exceeded."""
        if self._max_budget is not None:
            projected = self._total_spent + estimated_additional
            if projected > self._max_budget:
                raise BudgetExceededError(
                    budget=self._max_budget,
                    spent=self._total_spent,
                    message=f"Would exceed budget: ${projected:.2f} > ${self._max_budget:.2f} limit",
                )

    @property
    def remaining(self) -> float | None:
        if self._max_budget is None:
            return None
        return max(0.0, self._max_budget - self._total)
