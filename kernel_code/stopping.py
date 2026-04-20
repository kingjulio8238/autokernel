"""Adaptive stopping strategy for kernel optimization.

Three-layer architecture:
    Layer 0: Hard gates (budget, absolute max) — always enforced
    Layer 1: Base strategy (convergence patience, error streaks) — static defaults
    Layer 2: LLM meta-advisor (every N iters) — can extend, tighten, or stop
    Layer 3: KE profile (across sessions) — adjusts defaults from history

Override order: CLI flags > LLM meta > KE profile > base defaults

Usage::

    from kernel_code.stopping import StoppingController, StoppingStrategy

    strategy = StoppingStrategy()
    controller = StoppingController(strategy, llm=llm_provider)

    for iteration in range(max_iterations):
        # ... run optimization iteration ...
        decision = await controller.check(iteration, history)
        if decision.stop:
            print(f"Stopping: {decision.reason}")
            break
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openkernel.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class StoppingStrategy:
    """Configurable thresholds for stopping decisions."""

    # Hard cap — never exceeded regardless of other signals
    max_iterations: int = 10

    # Convergence — stop if N consecutive iters improve < threshold
    convergence_patience: int = 3
    convergence_threshold: float = 0.02  # 2% relative improvement

    # Error streak — stop if N consecutive errors/incorrect results
    max_consecutive_errors: int = 5

    # Target — stop when this speedup is reached (None = no target)
    target_speedup: float | None = None

    # Meta-advisor — call LLM every N iterations for stop/adjust decision
    meta_check_interval: int = 3

    # Whether LLM meta-advisor is enabled
    meta_enabled: bool = True


@dataclass
class StopDecision:
    """Result of a stopping check."""

    stop: bool
    reason: str = ""
    gate: str = ""  # which gate triggered: "hard_cap", "convergence", "errors", "target", "meta", "budget"
    adjust: dict | None = None  # if meta-advisor suggests parameter changes


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class StoppingController:
    """Checks all stopping gates and returns a decision.

    Call :meth:`check` after each optimization iteration.  The controller
    tracks history internally and calls the LLM meta-advisor every
    ``strategy.meta_check_interval`` iterations.
    """

    def __init__(
        self,
        strategy: StoppingStrategy,
        llm: "LLMProvider | None" = None,
        budget_remaining: float | None = None,
    ) -> None:
        self._strategy = strategy
        self._llm = llm
        self._budget_remaining = budget_remaining

        # Internal tracking
        self._history: list[dict] = []
        self._best_speedup: float = 0.0
        self._consecutive_no_improvement: int = 0
        self._consecutive_errors: int = 0
        self._meta_last_decision: str = "continue"

    @property
    def strategy(self) -> StoppingStrategy:
        return self._strategy

    def record_iteration(self, iteration_data: dict) -> None:
        """Record an iteration result for tracking."""
        self._history.append(iteration_data)

        speedup = iteration_data.get("speedup", 0.0)
        status = iteration_data.get("status", "")

        # Track errors
        if status in ("compile_error", "error", "incorrect"):
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0

        # Track convergence
        if status == "keep" and speedup > self._best_speedup:
            improvement = (speedup - self._best_speedup) / max(self._best_speedup, 0.001)
            if improvement >= self._strategy.convergence_threshold:
                self._consecutive_no_improvement = 0
            else:
                self._consecutive_no_improvement += 1
            self._best_speedup = speedup
        elif status == "discard":
            self._consecutive_no_improvement += 1

    async def check(self, iteration: int) -> StopDecision:
        """Check all stopping gates for the current iteration.

        Gates are checked in priority order. The first gate that fires
        produces the decision.

        Parameters
        ----------
        iteration : int
            Current iteration number (1-based).

        Returns
        -------
        StopDecision
            Whether to stop, and why.
        """
        s = self._strategy

        # --- Layer 0: Hard gates (always enforced) ---

        if iteration >= s.max_iterations:
            return StopDecision(
                stop=True,
                reason=f"Reached maximum iterations ({s.max_iterations})",
                gate="hard_cap",
            )

        if s.target_speedup is not None and self._best_speedup >= s.target_speedup:
            return StopDecision(
                stop=True,
                reason=f"Target speedup reached ({self._best_speedup:.2f}x >= {s.target_speedup}x)",
                gate="target",
            )

        # --- Layer 1: Base strategy (static thresholds) ---

        if self._consecutive_errors >= s.max_consecutive_errors:
            return StopDecision(
                stop=True,
                reason=f"{self._consecutive_errors} consecutive errors — likely a systematic issue",
                gate="errors",
            )

        if self._consecutive_no_improvement >= s.convergence_patience:
            # Don't stop on convergence too early — need at least a few iterations
            if iteration >= 3:
                return StopDecision(
                    stop=True,
                    reason=(
                        f"Converged — {self._consecutive_no_improvement} iterations "
                        f"with <{s.convergence_threshold:.0%} improvement "
                        f"(best: {self._best_speedup:.2f}x)"
                    ),
                    gate="convergence",
                )

        # --- Layer 2: LLM meta-advisor (every N iterations) ---

        if (
            s.meta_enabled
            and self._llm is not None
            and iteration > 0
            and iteration % s.meta_check_interval == 0
        ):
            try:
                from kernel_code.stopping_meta import meta_evaluate

                meta_decision = await meta_evaluate(
                    history=self._history,
                    best_speedup=self._best_speedup,
                    llm=self._llm,
                )

                self._meta_last_decision = meta_decision.action

                if meta_decision.action == "stop":
                    return StopDecision(
                        stop=True,
                        reason=f"LLM advisor: {meta_decision.reason}",
                        gate="meta",
                    )

                if meta_decision.action == "adjust" and meta_decision.adjustments:
                    self._apply_adjustments(meta_decision.adjustments)
                    logger.info(
                        "Meta-advisor adjusted strategy: %s — %s",
                        meta_decision.adjustments,
                        meta_decision.reason,
                    )
                    return StopDecision(
                        stop=False,
                        reason=f"Adjusted: {meta_decision.reason}",
                        gate="meta",
                        adjust=meta_decision.adjustments,
                    )

            except Exception as exc:
                # Meta-advisor is advisory — don't fail the loop
                logger.warning("Meta-advisor call failed: %s", exc)

        # --- Continue ---
        return StopDecision(stop=False)

    def _apply_adjustments(self, adjustments: dict) -> None:
        """Apply parameter adjustments from the meta-advisor."""
        s = self._strategy
        if "convergence_patience" in adjustments:
            s.convergence_patience = int(adjustments["convergence_patience"])
        if "max_iterations" in adjustments:
            # Meta can extend up to 2x the original cap, not beyond
            new_max = int(adjustments["max_iterations"])
            s.max_iterations = min(new_max, s.max_iterations * 2)
        if "convergence_threshold" in adjustments:
            s.convergence_threshold = float(adjustments["convergence_threshold"])

    def summary(self) -> dict:
        """Return a summary of the stopping state for logging/display."""
        return {
            "iterations": len(self._history),
            "best_speedup": self._best_speedup,
            "consecutive_no_improvement": self._consecutive_no_improvement,
            "consecutive_errors": self._consecutive_errors,
            "meta_last_decision": self._meta_last_decision,
            "strategy": {
                "max_iterations": self._strategy.max_iterations,
                "convergence_patience": self._strategy.convergence_patience,
                "convergence_threshold": self._strategy.convergence_threshold,
                "max_consecutive_errors": self._strategy.max_consecutive_errors,
                "target_speedup": self._strategy.target_speedup,
            },
        }
