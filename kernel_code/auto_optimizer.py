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
from kernel_code.optimization_log import (
    OptimizationLog,
    OptimizationRound,
    ProfileMetrics,
    RoundStatus,
)
from kernel_code.sol_metrics import compute_sol_score, compute_bandwidth_sol, compute_compute_sol
from kernel_code.problem_classifier import classify_problem
from kernel_code.evidence_tracker import extract_and_update_evidence
from kernel_code.checkpoint import CheckpointManager, CheckpointState

if TYPE_CHECKING:
    from kernel_code.live_display import LiveOptimizationDisplay
    from kernel_code.run_log import RunLogger
    from kernel_code.settings import KernelCodeSettings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Env vars the MetaOptimizer writes to hand state to the per-round bridge.
# These MUST be cleared at run() entry and exit — otherwise problem N's best
# kernel / strategy / round offset / inline profile leaks into problem N+1
# when batch_optimizer runs problems sequentially in the same process.
_PER_RUN_ENV_KEYS = (
    "OPENKERNEL_BEST_KERNEL",
    "OPENKERNEL_BEST_SPEEDUP",
    "OPENKERNEL_CURRENT_STRATEGY",
    "OPENKERNEL_ROUND_OFFSET",
    "OPENKERNEL_INLINE_PROFILE",
)


def _clear_per_run_env() -> None:
    for k in _PER_RUN_ENV_KEYS:
        os.environ.pop(k, None)


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
    evidence_added: int = 0


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
        self._exploratory_round_done: bool = False
        self._iter_counter: int = 0
        self._optimization_log = OptimizationLog(
            target_speedup=goal.target_speedup,
        )

        # Checkpointing
        self._checkpoint_mgr: CheckpointManager | None = None
        checkpoint_dir = os.environ.get("OPENKERNEL_CHECKPOINT_DIR")
        if checkpoint_dir:
            self._checkpoint_mgr = CheckpointManager(checkpoint_dir)

        # Cached problem classification (computed lazily, once per run).
        # Holds the ProblemType from problem_classifier. Used for pivot hints
        # and bottleneck labelling. Sentinel `...` means "not yet attempted".
        self._classification: object = ...

    def _get_classification(self):
        """Return cached classifier result, or None if classification failed.

        Reads the reference source once and caches the result so the classifier
        isn't invoked per-round. Failures are swallowed so the run never crashes
        on a bad reference file.
        """
        if self._classification is ...:
            try:
                reference_source = Path(self._goal.reference_path).read_text()
                self._classification = classify_problem(reference_source)
            except Exception as exc:
                logger.warning("Problem classification failed: %s", exc)
                self._classification = None
        return self._classification

    def resume_from_checkpoint(self) -> int:
        """Restore state from checkpoint if available.

        Returns the round number to resume from (0 if no checkpoint).
        """
        if not self._checkpoint_mgr:
            return 0
        state = self._checkpoint_mgr.load_latest()
        if not state:
            return 0

        self._best_speedup = state.best_speedup
        self._best_kernel = state.best_kernel
        self._total_cost = state.total_cost_usd
        self._total_iterations = state.total_iterations
        self._current_strategy = state.current_strategy
        self._round_history = state.round_history
        self._exploratory_round_done = state.exploratory_round_done
        self._iter_counter = state.total_iterations

        # Restore optimization log for trajectory context injection
        if state.optimization_log:
            self._optimization_log = OptimizationLog.from_dict_list(
                state.optimization_log,
                target_speedup=self._goal.target_speedup,
            )

        if self._console:
            self._console.print(
                f"  [#22d3ee]Resumed from checkpoint:[/#22d3ee] "
                f"round {state.round_num}, best {state.best_speedup:.2f}x, "
                f"cost ${state.total_cost_usd:.2f}"
            )
        return state.round_num

    def run(self) -> AutoResult:
        """Run the autonomous optimization loop."""
        # Clear any env state left behind by a prior MetaOptimizer.run() in the
        # same process. batch_optimizer runs problems sequentially, so without
        # this the previous problem's best kernel / strategy / round offset /
        # inline profile would leak into this problem's round 1.
        _clear_per_run_env()
        try:
            return self._run_impl()
        finally:
            # Clear again on exit so the next caller starts clean even if
            # this run raised partway through.
            _clear_per_run_env()

    def _run_impl(self) -> AutoResult:
        start_time = time.time()
        stop_reason = ""

        # Set per-run ID for dev logging (used by worker + reflection)
        from datetime import datetime
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.environ["OPENKERNEL_RUN_ID"] = run_id

        # Resume from checkpoint if available
        resume_round = self.resume_from_checkpoint()
        start_round = resume_round + 1 if resume_round > 0 else 1

        for round_num in range(start_round, self._goal.max_rounds + 1):
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

            # Plot C is rendered ONCE at session end, not between rounds

            # Update best
            round_best = round_result.get("best_speedup", 0.0)
            if round_best > self._best_speedup:
                self._best_speedup = round_best
                self._best_kernel = round_result.get("best_kernel", "")

            # Mirror per-iteration records into the run logger (end-of-round).
            # Workers are the unit of attempt exposed by the bridge.
            if self._run_logger:
                per_worker = round_result.get("per_worker") or []
                intent = round_result.get("strategy") or self._current_strategy
                if per_worker:
                    for w in per_worker:
                        self._iter_counter += 1
                        sp = float(w.get("speedup", 0.0) or 0.0)
                        if sp > 0.0:
                            status = "keep" if sp >= self._best_speedup else "discard"
                        else:
                            status = "error"
                        profile = w.get("profile") or {}
                        self._run_logger.log_iteration(
                            num=self._iter_counter,
                            speedup=sp,
                            status=status,
                            intent=intent,
                            profile=profile,
                        )
                elif round_result.get("total", 0) > 0:
                    # Bridge didn't report per-worker details — log a single round summary
                    self._iter_counter += 1
                    sp = float(round_result.get("best_speedup", 0.0) or 0.0)
                    if round_result.get("kept", 0) > 0 and sp > 0.0:
                        status = "keep" if sp >= self._best_speedup else "discard"
                    elif int(round_result.get("errors", 0)) > 0 and sp == 0.0:
                        status = "error"
                    else:
                        status = "discard"
                    self._run_logger.log_iteration(
                        num=self._iter_counter,
                        speedup=sp,
                        status=status,
                        intent=intent,
                        profile=round_result.get("profile") or {},
                    )

            # --- Save checkpoint after each round ---
            if self._checkpoint_mgr:
                ckpt = CheckpointState(
                    round_num=round_num,
                    best_speedup=self._best_speedup,
                    best_kernel=self._best_kernel,
                    total_cost_usd=self._total_cost,
                    total_iterations=self._total_iterations,
                    current_strategy=self._current_strategy,
                    round_history=self._round_history,
                    optimization_log=self._optimization_log.to_dict_list(),
                    exploratory_round_done=self._exploratory_round_done,
                )
                self._checkpoint_mgr.save_round(ckpt)

            # --- Check if target reached ---
            if self._best_speedup >= self._goal.target_speedup:
                overshoot_ratio = self._best_speedup / self._goal.target_speedup
                if (overshoot_ratio >= 1.10
                        and round_num == 1
                        and not self._exploratory_round_done
                        and self._goal.max_rounds >= 2):
                    self._exploratory_round_done = True
                    if self._live_display:
                        self._live_display.print_permanent(
                            f"  [#f5a850]Overshot target by {(overshoot_ratio - 1) * 100:.0f}% — "
                            f"running one exploratory round for extra upside[/#f5a850]"
                        )
                    # don't break; let loop continue for one more round
                else:
                    stop_reason = (
                        f"Target reached: {self._best_speedup:.2f}x >= "
                        f"{self._goal.target_speedup:.1f}x"
                    )
                    break

            # --- Early stop: zero correct kernels on round 1 ---
            # If round 1 produced no correct kernel at all, the problem is
            # likely miscompiled (dtype/atomics/API) or unreachable. Further
            # rounds will keep failing; stop immediately with an actionable
            # reason instead of grinding through the full budget.
            #
            # SKIP this early-stop if the round's eval failed at the
            # infra level (CUDA context poisoning, subprocess crash,
            # Modal timeout) — those aren't kernel-quality signals, so
            # retrying in round 2 is the right move.
            round_was_infra_failure = bool(round_result.get("infra_failed", False))
            if (round_num == 1
                    and round_result.get("kept", 0) == 0
                    and self._best_speedup <= 0.0
                    and not round_was_infra_failure):
                stop_reason = (
                    f"No correct kernel after round 1 "
                    f"({round_result.get('total', 0)} attempts, "
                    f"{round_result.get('errors', 0)} errors) — "
                    f"check reference dtypes, atomics, or backend choice"
                )
                break
            if round_num == 1 and round_was_infra_failure and self._live_display:
                self._live_display.print_permanent(
                    "  [yellow]Infra-level eval failure in round 1 "
                    "(CUDA context / subprocess crash) — retrying in round 2[/yellow]"
                )

            # --- Early stop: kernel can't beat baseline after 2+ rounds ---
            # If we've tried 2+ rounds and still can't hit 1.0x, the reference
            # is likely near-optimal for this op type. Continuing wastes budget.
            if len(self._round_history) >= 2 and self._best_speedup < 1.0:
                stop_reason = (
                    f"Sub-baseline after {len(self._round_history)} rounds "
                    f"(best {self._best_speedup:.2f}x < 1.0x) — "
                    f"reference likely near-optimal, stopping to save budget"
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
                        f"LLM budget remaining: ${self._goal.max_budget_usd - self._total_cost:.2f})"
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

        # --- Extract and persist evidence (silently) ---
        evidence_added = 0
        try:
            reference_source = Path(self._goal.reference_path).read_text()
            evidence_added = extract_and_update_evidence(
                optimization_log=self._optimization_log.to_dict_list(),
                reference_code=reference_source,
                hardware=self._goal.hardware,
            )
            if self._run_logger:
                self._run_logger.log_event(
                    f"Evidence: {evidence_added} entries added to skill library"
                )
        except Exception as exc:
            logger.warning("Evidence extraction failed: %s", exc)
            if self._run_logger:
                self._run_logger.log_event(f"Evidence extraction failed: {exc}")

        return AutoResult(
            best_speedup=self._best_speedup,
            best_kernel=self._best_kernel,
            target_reached=self._best_speedup >= self._goal.target_speedup,
            rounds_completed=len(self._round_history),
            total_iterations=self._total_iterations,
            total_cost_usd=self._total_cost,
            elapsed_seconds=elapsed,
            stop_reason=stop_reason,
            round_history=self._optimization_log.to_dict_list(),
            evidence_added=evidence_added,
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

        # Classify problem for bottleneck population
        classification = classify_problem(reference_source)
        bottleneck_hint = "memory_bound" if classification.is_memory_bound_likely else "compute_bound" if classification.is_compute_bound_likely else "unknown"

        bridge = KernelAgentBridge(
            reference_source=reference_source,
            model_name=model,
            num_workers=self._settings.num_workers,
            max_rounds=self._goal.iterations_per_round,
            hardware=self._goal.hardware,
            live_display=self._live_display,
            run_logger=self._run_logger,
            optimization_log=self._optimization_log,
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

        # Build structured optimization round
        profile_metrics = ProfileMetrics()
        if result.get("profile"):
            profile_metrics = ProfileMetrics.from_modal_profile(result["profile"])

        # Compute SOL score if we have profiling data
        if profile_metrics.runtime_us > 0 and profile_metrics.ref_runtime_us > 0:
            profile_metrics.sol_score = compute_sol_score(
                kernel_runtime_us=profile_metrics.runtime_us,
                ref_runtime_us=profile_metrics.ref_runtime_us,
                total_flops=profile_metrics.total_flops,
                total_bytes=profile_metrics.total_bytes,
                gpu_type=self._goal.hardware,
            )

        status = RoundStatus.SUCCESS if success else RoundStatus.RUNTIME_ERROR
        if not result.get("kernel_code"):
            status = RoundStatus.COMPILE_ERROR

        opt_round = OptimizationRound(
            round=round_num,
            kernel_code=result.get("kernel_code", ""),
            is_correct=success,
            speedup=speedup,
            status=status,
            strategy=self._current_strategy,
            bottleneck=bottleneck_hint,
            profile=profile_metrics,
            worker_results=result.get("per_worker", []),
            cost_usd=round_cost,
        )
        self._optimization_log.add_round(opt_round)

        # Pass inline profile context to next round via env var
        if profile_metrics.bandwidth_utilization_pct > 0:
            profile_context = (
                f"BW={profile_metrics.bandwidth_utilization_pct:.1f}%, "
                f"Compute={profile_metrics.compute_utilization_pct:.1f}%, "
                f"Occupancy={profile_metrics.occupancy:.2f}, "
                f"OI={profile_metrics.operational_intensity:.2f}"
            )
            os.environ["OPENKERNEL_INLINE_PROFILE"] = profile_context

        # Infra-level failures (CUDA context poisoning, subprocess crash,
        # eval timeout on Modal side) must not count as kernel failures —
        # otherwise round-1-zero-correct early-stop triggers on transient
        # infra problems. Surface the flag so the caller can distinguish.
        infra_failed = bool(result.get("infra_failed", False))
        return {
            "round": round_num,
            "strategy": self._current_strategy,
            "best_speedup": speedup,
            "best_kernel": result.get("kernel_code", ""),
            "kept": 1 if success else 0,
            "total": rounds_used,
            "errors": 0 if success else rounds_used,
            "bottleneck": bottleneck_hint,
            "per_worker": result.get("per_worker", []),
            "profile": result.get("profile") or {},
            "infra_failed": infra_failed,
            "infra_error": result.get("infra_error", ""),
        }

    def _reflect(self, round_num: int) -> MetaReflection:
        """LLM reflection between rounds."""
        from openkernel.config import ModelConfig
        from openkernel.llm.provider import LLMProvider

        budget_remaining = self._goal.max_budget_usd - self._total_cost

        # Build problem context + strategy hints for the reflection prompt
        problem_context = ""
        strategy_hints: list[str] | None = None
        classif = self._get_classification()
        if classif is not None:
            try:
                ctx_parts = [f"{classif.tier.value} {classif.op_type.value}"]
                if classif.is_launch_bound_likely:
                    ctx_parts.append("launch-bound")
                elif classif.is_memory_bound_likely:
                    ctx_parts.append("memory-bound")
                elif classif.is_compute_bound_likely:
                    ctx_parts.append("compute-bound")
                if classif.estimated_tensor_elements > 0:
                    ctx_parts.append(f"~{classif.estimated_tensor_elements:,} elements")
                problem_context = ", ".join(ctx_parts)
                hints = getattr(classif, "strategy_hints", None)
                if hints:
                    strategy_hints = list(hints)
            except Exception:
                pass

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
                    problem_context=problem_context,
                    strategy_hints=strategy_hints,
                )
            )
            loop.close()
            return reflection
        except Exception as exc:
            logger.warning("Reflection failed: %s", exc)
            return MetaReflection(action="continue", reason=f"reflection failed: {exc}")
