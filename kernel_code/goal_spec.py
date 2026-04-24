"""Goal specification for autonomous optimization.

Defines what the KE wants to achieve — the ``/auto`` command builds a
:class:`GoalSpec` from CLI flags and settings, then passes it to the
:class:`MetaOptimizer`.

Usage::

    spec = GoalSpec(
        target_speedup=2.0,
        target_sol=0.80,
        max_budget_usd=5.00,
        max_time_seconds=None,  # None = unlimited, rely on budget/rounds cap
        reference_path="reference.py",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoalSpec:
    """What the KE wants from an autonomous optimization run."""

    # Target
    target_speedup: float = 2.0  # stop when this speedup is reached
    # Fraction of speed-of-light (0.0, 1.0] — always applied (no "disabled"
    # sentinel here; ParsedGoal carries 0.0 to mean "not stated by user").
    target_sol: float = 0.80

    # Resource limits
    max_budget_usd: float = 5.00  # hard cost cap
    # Wall-clock cap. Default None = unlimited; budget + max_rounds are the real
    # stops. A time cap is a blunt instrument — it can cut off a run mid-round
    # right after a target is met (see exploratory-round branch in
    # auto_optimizer) — so callers must opt in rather than inherit a surprise
    # default.
    max_time_seconds: int | None = None
    max_rounds: int = 5  # outer loop cap (each round = one orchestrator run)

    # Problem
    reference_path: str = "reference.py"

    # Infrastructure (from settings if not set)
    hardware: str = "L40S"
    backend: str = "triton"
    model: str = ""
    provider: str = ""

    # Per-round settings
    iterations_per_round: int = 5  # inner iterations per round

    def validate(self) -> list[str]:
        """Validate the goal spec. Returns list of error messages (empty = valid)."""
        errors: list[str] = []
        if self.target_speedup <= 0:
            errors.append(f"target_speedup must be > 0, got {self.target_speedup}")
        if not (0.0 < self.target_sol <= 1.0):
            errors.append(f"target_sol must be in (0.0, 1.0], got {self.target_sol}")
        if self.max_budget_usd <= 0:
            errors.append(f"max_budget_usd must be > 0, got {self.max_budget_usd}")
        if self.max_rounds < 1:
            errors.append(f"max_rounds must be >= 1, got {self.max_rounds}")
        if self.iterations_per_round < 1:
            errors.append(f"iterations_per_round must be >= 1, got {self.iterations_per_round}")
        ref = Path(self.reference_path)
        if not ref.is_file():
            errors.append(f"reference file not found: {self.reference_path}")
        return errors

    @property
    def estimated_max_iterations(self) -> int:
        """Upper bound on total iterations across all rounds."""
        return self.max_rounds * self.iterations_per_round

    def summary(self) -> str:
        """Human-readable summary of the goal."""
        parts = [f"Target: {self.target_speedup:.1f}x @ SOL {self.target_sol:.2f}"]
        parts.append(f"Budget: ${self.max_budget_usd:.2f}")
        if self.max_time_seconds:
            mins = self.max_time_seconds // 60
            parts.append(f"Time: {mins}m")
        parts.append(f"Rounds: up to {self.max_rounds}")
        parts.append(f"Hardware: {self.hardware} | Backend: {self.backend}")
        return "  |  ".join(parts)
