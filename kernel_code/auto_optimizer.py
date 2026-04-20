"""MetaOptimizer — autonomous multi-round optimization loop.

Wraps the existing orchestrator/bridge in an outer reflect-adapt cycle.
The KE provides a goal (target speedup, budget, time) and walks away.
The optimizer runs rounds of optimization, reflects between rounds,
pivots strategy when stuck, and stops when the goal is met or resources
are exhausted.

Architecture::

    Outer loop (MetaOptimizer)  — reflect, pivot, accumulate
      Middle loop (Orchestrator) — world model, intent tree
        Inner loop (InnerLoop)   — generate, eval, critic, retry

Usage::

    from kernel_code.auto_optimizer import MetaOptimizer
    from kernel_code.goal_spec import GoalSpec

    spec = GoalSpec(target_speedup=2.0, max_budget_usd=5.00)
    optimizer = MetaOptimizer(spec, settings=settings, console=console)
    result = optimizer.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import warnings

# Suppress litellm's async task cleanup warnings
warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", message=".*Task was destroyed.*")
# Suppress asyncio ERROR logs (litellm pending tasks)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from kernel_code.goal_spec import GoalSpec
from kernel_code.meta_reflect import MetaReflection, reflect_on_round

if TYPE_CHECKING:
    from kernel_code.live_display import LiveOptimizationDisplay
    from kernel_code.run_log import RunLogger
    from kernel_code.settings import KernelCodeSettings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AutoResult:
    """Final result of an autonomous optimization run."""

    best_speedup: float = 0.0
    best_kernel: str = ""
    target_reached: bool = False
    rounds_completed: int = 0
    total_iterations: int = 0
    total_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    stop_reason: str = ""
    round_history: list[dict] = field(default_factory=list)


class MetaOptimizer:
    """Autonomous multi-round kernel optimization.

    Each round runs the existing OpenKernelBridge with a strategy hint.
    Between rounds, the LLM reflects on progress and decides:
    CONTINUE (refine), PIVOT (new strategy), or STOP.
    """

    def __init__(
        self,
        goal: GoalSpec,
        settings: "KernelCodeSettings",
        console: Console | None = None,
        live_display: "LiveOptimizationDisplay | None" = None,
        run_logger: "RunLogger | None" = None,
    ) -> None:
        self._goal = goal
        self._settings = settings
        self._console = console or Console()
        self._live_display = live_display
        self._run_logger = run_logger

        # State
        self._best_speedup: float = 0.0
        self._best_kernel: str = ""
        self._round_history: list[dict] = []
        self._total_cost: float = 0.0
        self._total_iterations: int = 0
        self._current_strategy: str = "general optimization"

    def run(self) -> AutoResult:
        """Run the autonomous optimization loop."""
        start_time = time.time()
        stop_reason = ""

        for round_num in range(1, self._goal.max_rounds + 1):
            # --- Check stopping gates before starting round ---
            stop_reason = self._check_gates(start_time)
            if stop_reason:
                break

            # --- Run one round ---
            if self._live_display:
                self._live_display.start_round(round_num, self._current_strategy)
            if self._run_logger:
                self._run_logger.log_round(round_num, self._current_strategy)

            # Set strategy + best kernel + round offset in env for the bridge
            os.environ["OPENKERNEL_CURRENT_STRATEGY"] = self._current_strategy
            os.environ["OPENKERNEL_ROUND_OFFSET"] = str((round_num - 1) * self._settings.num_workers)
            if self._best_kernel:
                os.environ["OPENKERNEL_BEST_KERNEL"] = self._best_kernel
                os.environ["OPENKERNEL_BEST_SPEEDUP"] = f"{self._best_speedup:.2f}"

            round_result = self._run_round(round_num)
            self._round_history.append(round_result)
            self._total_iterations += round_result.get("total", 0)

            # Print plot C between rounds
            if len(self._round_history) > 1:
                try:
                    from kernel_code.worker_plots import render_round_columns
                    if self._live_display and self._live_display._live:
                        self._live_display._live.stop()
                    self._console.print(render_round_columns(self._round_history))
                    if self._live_display:
                        self._live_display._live = None
                        self._live_display.start()
                except Exception:
                    pass

            # Update best
            round_best = round_result.get("best_speedup", 0.0)
            if round_best > self._best_speedup:
                self._best_speedup = round_best
                self._best_kernel = round_result.get("best_kernel", "")

            # --- Check if target reached ---
            if self._best_speedup >= self._goal.target_speedup:
                stop_reason = (
                    f"Target reached: {self._best_speedup:.2f}x >= "
                    f"{self._goal.target_speedup:.1f}x"
                )
                break

            # --- Plateau detection: inform but don't stop ---
            # Keep iterating toward target — only budget/time can stop us
            if len(self._round_history) >= 3:
                recent = [r.get("best_speedup", 0.0) for r in self._round_history[-3:]]
                if recent[-1] > 0 and all(abs(s - recent[-1]) / max(recent[-1], 0.001) < 0.02 for s in recent):
                    if self._live_display:
                        self._live_display.print_permanent(
                            f"  \u23bf  [#999999]Plateau at {self._best_speedup:.2f}x for 3 rounds "
                            f"— trying different strategies...[/#999999]"
                        )

            # --- Reflect and decide next action ---
            if round_num < self._goal.max_rounds:
                reflection = self._reflect(round_num)

                if reflection.action == "stop":
                    stop_reason = (
                        f"Stopped: {reflection.reason} "
                        f"(best: {self._best_speedup:.2f}x, target: {self._goal.target_speedup:.1f}x, "
                        f"budget remaining: ${self._goal.max_budget_usd - self._total_cost:.2f})"
                    )
                    break
                elif reflection.action == "pivot":
                    self._current_strategy = reflection.next_strategy
                    if self._live_display:
                        self._live_display.print_permanent(
                            f"  [#22d3ee]Pivoting:[/#22d3ee] [white]{reflection.next_strategy}[/white]"
                        )
                elif reflection.action == "continue":
                    if reflection.next_strategy:
                        self._current_strategy = reflection.next_strategy
        else:
            stop_reason = f"Completed all {self._goal.max_rounds} rounds"

        elapsed = time.time() - start_time

        return AutoResult(
            best_speedup=self._best_speedup,
            best_kernel=self._best_kernel,
            target_reached=self._best_speedup >= self._goal.target_speedup,
            rounds_completed=len(self._round_history),
            total_iterations=self._total_iterations,
            total_cost_usd=self._total_cost,
            elapsed_seconds=elapsed,
            stop_reason=stop_reason,
            round_history=self._round_history,
        )

    def _check_gates(self, start_time: float) -> str:
        """Check stopping gates. Returns reason string if should stop, empty if OK."""
        # Budget
        if self._total_cost >= self._goal.max_budget_usd:
            return f"Budget exhausted: ${self._total_cost:.2f} >= ${self._goal.max_budget_usd:.2f}"

        # Time
        if self._goal.max_time_seconds:
            elapsed = time.time() - start_time
            if elapsed >= self._goal.max_time_seconds:
                mins = int(elapsed) // 60
                return f"Time limit reached: {mins}m"

        return ""

    def _run_round(self, round_num: int) -> dict:
        """Run one optimization round via KernelAgent."""
        from kernel_code.integration.kernel_agent_bridge import KernelAgentBridge
        from kernel_code.settings import inject_api_keys
        from kernel_code.permissions import estimate_cost

        inject_api_keys(self._settings)

        reference_source = Path(self._goal.reference_path).read_text()
        model = self._goal.model or self._settings.default_model

        bridge = KernelAgentBridge(
            reference_source=reference_source,
            model_name=model,
            num_workers=self._settings.num_workers,
            max_rounds=self._goal.iterations_per_round,
            hardware=self._goal.hardware,
            live_display=self._live_display,
            run_logger=self._run_logger,
        )

        try:
            result = bridge.run()
        except Exception as exc:
            logger.error("Round %d failed: %s", round_num, exc)
            return {
                "round": round_num,
                "strategy": self._current_strategy,
                "best_speedup": 0.0,
                "best_kernel": "",
                "kept": 0,
                "total": 0,
                "errors": 1,
                "bottleneck": "error",
            }

        # Estimate cost
        rounds_used = result.get("rounds", 0)
        round_cost = estimate_cost(max(rounds_used, 1), gpu_type=self._goal.hardware)
        self._total_cost += round_cost

        speedup = result.get("speedup", 0.0)
        success = result.get("success", False)

        return {
            "round": round_num,
            "strategy": self._current_strategy,
            "best_speedup": speedup,
            "best_kernel": result.get("kernel_code", ""),
            "kept": 1 if success else 0,
            "total": rounds_used,
            "errors": 0 if success else rounds_used,
            "bottleneck": "unknown",
        }

    def _reflect(self, round_num: int) -> MetaReflection:
        """LLM reflection between rounds."""
        from openkernel.config import ModelConfig
        from openkernel.llm.provider import LLMProvider

        budget_remaining = self._goal.max_budget_usd - self._total_cost

        try:
            model_config = ModelConfig(
                provider=self._goal.provider or self._settings.default_provider,
                model_id=self._goal.model or self._settings.default_model,
            )
            llm = LLMProvider(model_config)
            loop = asyncio.new_event_loop()
            reflection = loop.run_until_complete(
                reflect_on_round(
                    rounds=self._round_history,
                    best_speedup=self._best_speedup,
                    target_speedup=self._goal.target_speedup,
                    budget_remaining=budget_remaining,
                    llm=llm,
                )
            )
            loop.close()
            return reflection
        except Exception as exc:
            logger.warning("Reflection failed: %s", exc)
            return MetaReflection(action="continue", reason=f"reflection failed: {exc}")
