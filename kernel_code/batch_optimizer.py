"""Batch optimizer for multi-problem kernel evaluation.

Runs MetaOptimizer across a list of problem files and aggregates
statistics (geomean speedup, % beating baseline, median SOL, etc.).
Enables Cursor-style 235-problem evaluation sweeps.

Usage::

    from kernel_code.batch_optimizer import BatchOptimizer, BatchResult

    batch = BatchOptimizer(
        problem_dir=Path("problems/"),
        goal_template=GoalSpec(target_speedup=2.0, max_budget_usd=1.00),
        settings=settings,
    )
    result = batch.run_all()
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from kernel_code.auto_optimizer import AutoResult, MetaOptimizer
from kernel_code.goal_spec import GoalSpec
from kernel_code.problem_classifier import classify_problem, ProblemTier

if TYPE_CHECKING:
    from openkernel.benchmarks.problem_spec import ProblemSpec

logger = logging.getLogger(__name__)

_SUITE_SEED = 7

_VALID_HARDWARE = {"L40S", "H100", "A100-80GB", "B200"}


@dataclass
class ProblemResult:
    """Result for a single problem in a batch run."""
    problem_path: str
    problem_name: str
    tier: str                    # L1, L2, Quant, MoE
    speedup: float = 0.0
    sol_score: float = 0.0
    target_reached: bool = False
    rounds_completed: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    error: str = ""
    best_kernel: str = ""


@dataclass
class BatchResult:
    """Aggregated results from a batch optimization run."""
    problems: list[ProblemResult] = field(default_factory=list)
    total_problems: int = 0
    total_elapsed_seconds: float = 0.0
    total_cost_usd: float = 0.0

    # Aggregate metrics (computed by finalize())
    geomean_speedup: float = 0.0
    median_speedup: float = 0.0
    mean_sol_score: float = 0.0
    pct_beating_baseline: float = 0.0
    pct_exceeding_2x: float = 0.0
    per_tier_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def finalize(self) -> None:
        """Compute aggregate statistics from individual problem results."""
        self.total_problems = len(self.problems)
        if not self.problems:
            return

        speedups = [p.speedup for p in self.problems if p.speedup > 0]
        sol_scores = [p.sol_score for p in self.problems if p.sol_score > 0]

        # Geomean speedup (geometric mean, handles <1.0 values correctly)
        if speedups:
            log_sum = sum(math.log(max(s, 0.01)) for s in speedups)
            self.geomean_speedup = math.exp(log_sum / len(speedups))

        # Median speedup
        if speedups:
            sorted_s = sorted(speedups)
            mid = len(sorted_s) // 2
            self.median_speedup = (
                sorted_s[mid] if len(sorted_s) % 2 == 1
                else (sorted_s[mid - 1] + sorted_s[mid]) / 2
            )

        # Mean SOL score
        if sol_scores:
            self.mean_sol_score = sum(sol_scores) / len(sol_scores)

        # Percentage beating baseline (speedup > 1.0) — denominator is successful problems only
        beating = sum(1 for s in speedups if s > 1.0)
        self.pct_beating_baseline = (beating / len(speedups) * 100) if speedups else 0.0

        # Percentage exceeding 2x
        exceeding_2x = sum(1 for s in speedups if s >= 2.0)
        self.pct_exceeding_2x = (exceeding_2x / len(speedups) * 100) if speedups else 0.0

        # Per-tier breakdown
        tier_problems: dict[str, list[ProblemResult]] = {}
        for p in self.problems:
            tier_problems.setdefault(p.tier, []).append(p)

        for tier, probs in tier_problems.items():
            tier_speedups = [p.speedup for p in probs if p.speedup > 0]
            tier_geomean = 0.0
            if tier_speedups:
                log_sum = sum(math.log(max(s, 0.01)) for s in tier_speedups)
                tier_geomean = math.exp(log_sum / len(tier_speedups))
            successful = len(tier_speedups)
            self.per_tier_stats[tier] = {
                "count": len(probs),
                "successful": successful,
                "geomean_speedup": tier_geomean,
                "pct_beating_baseline": sum(1 for s in tier_speedups if s > 1.0) / max(successful, 1) * 100,
                "pct_exceeding_2x": sum(1 for s in tier_speedups if s >= 2.0) / max(successful, 1) * 100,
            }

        # Totals
        self.total_cost_usd = sum(p.cost_usd for p in self.problems)
        self.total_elapsed_seconds = sum(p.elapsed_seconds for p in self.problems)

    def summary(self) -> str:
        """Human-readable summary of batch results."""
        lines = [
            "=" * 60,
            "BATCH OPTIMIZATION RESULTS",
            "=" * 60,
            f"Problems: {self.total_problems}",
            f"Geomean speedup: {self.geomean_speedup:.2f}x",
            f"Median speedup: {self.median_speedup:.2f}x",
            f"Mean SOL score: {self.mean_sol_score:.2f}",
            f"Beating baseline (>1.0x): {self.pct_beating_baseline:.1f}%",
            f"Exceeding 2x: {self.pct_exceeding_2x:.1f}%",
            f"Total cost: ${self.total_cost_usd:.2f}",
            f"Total time: {self.total_elapsed_seconds / 60:.1f} min",
        ]

        if self.per_tier_stats:
            lines.append("")
            lines.append("Per-tier breakdown:")
            lines.append(f"  {'Tier':<10} {'Count':>6} {'Geomean':>8} {'>1x':>6} {'>=2x':>6}")
            lines.append(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*6} {'-'*6}")
            for tier, stats in sorted(self.per_tier_stats.items()):
                lines.append(
                    f"  {tier:<10} {stats['count']:>6.0f} "
                    f"{stats['geomean_speedup']:>7.2f}x "
                    f"{stats['pct_beating_baseline']:>5.1f}% "
                    f"{stats['pct_exceeding_2x']:>5.1f}%"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "total_problems": self.total_problems,
            "geomean_speedup": self.geomean_speedup,
            "median_speedup": self.median_speedup,
            "mean_sol_score": self.mean_sol_score,
            "pct_beating_baseline": self.pct_beating_baseline,
            "pct_exceeding_2x": self.pct_exceeding_2x,
            "total_cost_usd": self.total_cost_usd,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "per_tier_stats": self.per_tier_stats,
            "problems": [
                {
                    "name": p.problem_name,
                    "path": p.problem_path,
                    "tier": p.tier,
                    "speedup": p.speedup,
                    "sol_score": p.sol_score,
                    "target_reached": p.target_reached,
                    "rounds": p.rounds_completed,
                    "cost_usd": p.cost_usd,
                    "elapsed_s": p.elapsed_seconds,
                    "error": p.error,
                }
                for p in self.problems
            ],
        }


class BatchOptimizer:
    """Run MetaOptimizer across multiple problem files.

    Supports both directory-based and explicit file list inputs.
    Results are aggregated with geomean, per-tier breakdown, and
    SOL score statistics.
    """

    def __init__(
        self,
        problem_paths: list[Path] | None = None,
        problem_dir: Path | None = None,
        goal_template: GoalSpec | None = None,
        settings: Any = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize batch optimizer.

        Args:
            problem_paths: Explicit list of problem file paths.
            problem_dir: Directory to scan for .py problem files.
            goal_template: GoalSpec template (reference_path will be overridden per problem).
            settings: KernelCodeSettings instance.
            output_dir: Directory for batch results output.
        """
        self._settings = settings
        self._goal_template = goal_template or GoalSpec()
        self._output_dir = output_dir or Path(".kernel-code/batch_results")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Collect problem files
        if problem_paths:
            self._problems = [p for p in problem_paths if p.is_file() and p.suffix == ".py"]
        elif problem_dir and problem_dir.is_dir():
            self._problems = sorted(problem_dir.glob("*.py"))
        else:
            self._problems = []

        if not self._problems:
            logger.warning("No problem files found for batch optimization")

    def run_all(self) -> BatchResult:
        """Run optimization on all problems sequentially.

        Returns aggregated BatchResult with per-problem and
        aggregate statistics.
        """
        batch_result = BatchResult()
        total = len(self._problems)

        logger.info("Starting batch optimization: %d problems", total)

        for idx, problem_path in enumerate(self._problems, 1):
            logger.info(
                "[%d/%d] Optimizing: %s", idx, total, problem_path.name
            )

            problem_result = self._run_single(problem_path, idx, total)
            batch_result.problems.append(problem_result)

            # Log progress
            logger.info(
                "[%d/%d] %s: %.2fx speedup, SOL=%.2f, $%.2f, %.1fs",
                idx, total, problem_path.name,
                problem_result.speedup,
                problem_result.sol_score,
                problem_result.cost_usd,
                problem_result.elapsed_seconds,
            )

        # Compute aggregates
        batch_result.finalize()

        # Save results
        self._save_results(batch_result)

        logger.info(batch_result.summary())
        return batch_result

    def _run_single(
        self, problem_path: Path, idx: int, total: int
    ) -> ProblemResult:
        """Run MetaOptimizer on a single problem.

        Returns ProblemResult with speedup, SOL score, cost, etc.
        """
        problem_name = problem_path.stem
        start_time = time.time()

        # Classify problem
        try:
            code = problem_path.read_text()
            classification = classify_problem(code)
            tier = classification.tier.value
        except Exception:
            tier = "L1"

        # Build goal for this problem
        goal = GoalSpec(
            target_speedup=self._goal_template.target_speedup,
            max_budget_usd=self._goal_template.max_budget_usd,
            max_time_seconds=self._goal_template.max_time_seconds,
            max_rounds=self._goal_template.max_rounds,
            reference_path=str(problem_path),
            hardware=self._goal_template.hardware,
            backend=self._goal_template.backend,
            model=self._goal_template.model,
            provider=self._goal_template.provider,
            iterations_per_round=self._goal_template.iterations_per_round,
        )

        # Set up per-problem checkpoint dir
        import os
        ckpt_dir = self._output_dir / f"checkpoints/{problem_name}"
        os.environ["OPENKERNEL_CHECKPOINT_DIR"] = str(ckpt_dir)

        try:
            optimizer = MetaOptimizer(
                goal=goal,
                settings=self._settings,
            )
            result: AutoResult = optimizer.run()

            # Extract SOL score from last round if available
            sol_score = 0.0
            if result.round_history:
                last_round = result.round_history[-1]
                profile = last_round.get("profile", {})
                if isinstance(profile, dict):
                    sol_score = profile.get("sol_score", 0.0)

            return ProblemResult(
                problem_path=str(problem_path),
                problem_name=problem_name,
                tier=tier,
                speedup=result.best_speedup,
                sol_score=sol_score,
                target_reached=result.target_reached,
                rounds_completed=result.rounds_completed,
                cost_usd=result.total_cost_usd,
                elapsed_seconds=result.elapsed_seconds,
                best_kernel=result.best_kernel,
            )

        except Exception as exc:
            logger.error("[%d/%d] %s failed: %s", idx, total, problem_name, exc)
            return ProblemResult(
                problem_path=str(problem_path),
                problem_name=problem_name,
                tier=tier,
                elapsed_seconds=time.time() - start_time,
                error=str(exc),
            )

    def _save_results(self, batch_result: BatchResult) -> None:
        """Save batch results to output directory."""
        # Save full results JSON
        results_file = self._output_dir / "batch_results.json"
        results_file.write_text(json.dumps(batch_result.to_dict(), indent=2))

        # Save summary text
        summary_file = self._output_dir / "batch_summary.txt"
        summary_file.write_text(batch_result.summary())

        # Save per-problem best kernels
        kernels_dir = self._output_dir / "best_kernels"
        kernels_dir.mkdir(parents=True, exist_ok=True)
        for p in batch_result.problems:
            if p.best_kernel:
                kernel_file = kernels_dir / f"{p.problem_name}_best.py"
                kernel_file.write_text(p.best_kernel)

        logger.info("Batch results saved to %s", self._output_dir)


@dataclass
class SuiteResult:
    """Aggregate of a ProblemSpec-driven suite run."""

    total: int = 0
    skipped: int = 0
    succeeded: int = 0
    failed: int = 0
    errored: int = 0
    total_cost_usd: float = 0.0
    total_elapsed_s: int = 0
    records_written: list[str] = field(default_factory=list)


def _suite_config_hash(
    model: str,
    backend: str,
    target_sol: float,
    budget_per_problem: float,
    seed: int = _SUITE_SEED,
) -> str:
    """Compute the same cfg_ hash the leaderboard writer produces."""
    config = {
        "model": model,
        "backend": backend,
        "target_sol": target_sol,
        "budget_per_problem": budget_per_problem,
        "seed": seed,
    }
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"cfg_{digest}"


def _existing_keys(
    date: str, leaderboard_root: Path | None = None
) -> set[tuple[str, str, str]]:
    """Return set of (problem_id, hardware, config_hash) already on disk for ``date``."""
    from openkernel.benchmarks.leaderboard_reader import load_all

    try:
        records = load_all(root=leaderboard_root)
    except Exception as exc:
        logger.warning("run_suite: leaderboard load_all failed: %s", exc)
        return set()

    keys: set[tuple[str, str, str]] = set()
    for r in records:
        if r.get("date") != date:
            continue
        pid = r.get("problem_id") or ""
        hw = r.get("hardware") or ""
        chash = r.get("config_hash") or ""
        if pid and hw and chash:
            keys.add((pid, hw, chash))
    return keys


def _extract_profile_metrics(auto_result: AutoResult) -> dict[str, float]:
    """Pull compute_util / bandwidth_util / sol_score from the best round."""
    compute_util = 0.0
    bandwidth_util = 0.0
    sol_score = 0.0

    best_speedup = auto_result.best_speedup
    best_round = None
    for r in auto_result.round_history:
        if not isinstance(r, dict):
            continue
        if r.get("speedup", 0.0) == best_speedup and best_round is None:
            best_round = r

    # Fall back to last round if no exact speedup match
    if best_round is None and auto_result.round_history:
        last = auto_result.round_history[-1]
        if isinstance(last, dict):
            best_round = last

    if best_round is not None:
        profile = best_round.get("profile", {})
        if isinstance(profile, dict):
            # Phase 0 contract: modal_infra/app.py emits short-form keys
            # `compute_util` / `bandwidth_util` (0-100 percent), with
            # `compute_utilization` / `bandwidth_utilization` kept as legacy
            # aliases. Accept both so future renames don't silently zero out.
            compute_util = float(
                profile.get("compute_util")
                or profile.get("compute_utilization")
                or 0.0
            )
            bandwidth_util = float(
                profile.get("bandwidth_util")
                or profile.get("bandwidth_utilization")
                or 0.0
            )
            sol_score = float(profile.get("sol_score", 0.0))

    # Clamp to writer's schema ranges so valid runs never fail validation.
    compute_util = max(0.0, min(100.0, compute_util))
    bandwidth_util = max(0.0, min(100.0, bandwidth_util))
    sol_score = max(0.0, min(1.0, sol_score))

    return {
        "compute_util": compute_util,
        "bandwidth_util": bandwidth_util,
        "sol_score": sol_score,
    }


def _bottleneck_label(compute_util: float, bandwidth_util: float) -> str:
    """Map utilization pair to the writer's bottleneck_type enum."""
    if compute_util == 0.0 and bandwidth_util == 0.0:
        return "unknown"
    if compute_util >= 70.0 and bandwidth_util < 70.0:
        return "compute-bound"
    if bandwidth_util >= 70.0 and compute_util < 70.0:
        return "memory-bound"
    return "balanced"


def _run_single_spec(
    spec: "ProblemSpec",
    hardware: str,
    budget_per_problem: float,
    model: str,
    target_sol: float,
    settings: Any,
    config_hash: str,
    date: str,
    leaderboard_root: Path | None,
) -> tuple[str, AutoResult | None, str | None]:
    """Execute MetaOptimizer for one spec and write its leaderboard record.

    Returns ``(status, auto_result, record_path)`` where ``status`` is one of
    ``"succeeded"``, ``"failed"``, or ``"errored"``. ``record_path`` is ``None``
    when no record was written (failures with no kernel, or errors).
    """
    from openkernel.benchmarks.leaderboard_writer import write_record

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"{spec.id}_", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(spec.reference_source)
        ref_path = Path(tf.name)

    start = time.time()
    try:
        goal = GoalSpec(
            target_speedup=1.0 / max(1.0 - target_sol, 0.01),
            max_budget_usd=budget_per_problem,
            max_rounds=5,
            reference_path=str(ref_path),
            hardware=hardware,
            backend="triton",
            model=model,
        )
        optimizer = MetaOptimizer(goal=goal, settings=settings)
        auto_result: AutoResult = optimizer.run()
    except Exception as exc:
        logger.error("run_suite: %s errored: %s", spec.id, exc)
        ref_path.unlink(missing_ok=True)
        return ("errored", None, None)

    ref_path.unlink(missing_ok=True)

    # No correct kernel produced — skip writing a record.
    if not auto_result.best_kernel or auto_result.best_speedup <= 0.0:
        return ("failed", auto_result, None)

    profile = _extract_profile_metrics(auto_result)
    bottleneck = _bottleneck_label(profile["compute_util"], profile["bandwidth_util"])

    now = datetime.now(timezone.utc)
    record = {
        "kernel_source": auto_result.best_kernel,
        "hardware": hardware,
        "date": date,
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "speedup": float(auto_result.best_speedup),
        "sol_score": profile["sol_score"],
        "compute_util": profile["compute_util"],
        "bandwidth_util": profile["bandwidth_util"],
        "bottleneck_type": bottleneck,
        "correct": True,
        "cost_usd": float(auto_result.total_cost_usd),
        "elapsed_s": int(auto_result.elapsed_seconds or (time.time() - start)),
        "stop_reason": (auto_result.stop_reason or "completed")[:256],
        "rounds": int(max(auto_result.rounds_completed, 1)),
        "iterations": int(auto_result.total_iterations),
        "config": {
            "model": model,
            "backend": "triton",
            "target_sol": target_sol,
            "budget_per_problem": budget_per_problem,
            "seed": _SUITE_SEED,
        },
    }

    try:
        if leaderboard_root is not None:
            path = write_record(spec, record, root=leaderboard_root)
        else:
            path = write_record(spec, record)
    except Exception as exc:
        logger.error("run_suite: write_record failed for %s: %s", spec.id, exc)
        return ("errored", auto_result, None)

    return ("succeeded", auto_result, str(path))


def run_suite(
    specs: list["ProblemSpec"],
    hardware: str,
    budget_per_problem: float = 0.50,
    concurrency: int = 4,
    date: str | None = None,
    model: str = "o3-mini",
    target_sol: float = 0.80,
    settings: Any = None,
    leaderboard_root: Path | None = None,
) -> SuiteResult:
    """Run the full /optimize pipeline across a list of ``ProblemSpec``.

    Behavior:
      - Resume: specs already present in today's leaderboard (same
        ``(problem_id, hardware, config_hash)``) are skipped.
      - Concurrency: bounded by ``asyncio.Semaphore(concurrency)``.
      - Error isolation: a single-spec crash is counted and the suite
        continues.
      - Ctrl-C: cancels in-flight tasks and returns a partial
        ``SuiteResult`` without corrupting the leaderboard.
    """
    if hardware not in _VALID_HARDWARE:
        raise ValueError(
            f"invalid hardware {hardware!r}, must be one of {sorted(_VALID_HARDWARE)}"
        )

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    config_hash = _suite_config_hash(
        model=model,
        backend="triton",
        target_sol=target_sol,
        budget_per_problem=budget_per_problem,
    )
    existing = _existing_keys(date=date, leaderboard_root=leaderboard_root)

    suite = SuiteResult(total=len(specs))
    start_wall = time.time()

    pending: list["ProblemSpec"] = []
    for spec in specs:
        key = (spec.id, hardware, config_hash)
        if key in existing:
            suite.skipped += 1
            logger.info("run_suite: skipping %s (already on leaderboard)", spec.id)
            continue
        pending.append(spec)

    if not pending:
        suite.total_elapsed_s = int(time.time() - start_wall)
        return suite

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _gated(spec: "ProblemSpec") -> tuple[str, str, AutoResult | None, str | None]:
        async with sem:
            loop = asyncio.get_running_loop()
            try:
                status, auto_result, record_path = await loop.run_in_executor(
                    None,
                    _run_single_spec,
                    spec,
                    hardware,
                    budget_per_problem,
                    model,
                    target_sol,
                    settings,
                    config_hash,
                    date,
                    leaderboard_root,
                )
            except Exception as exc:
                logger.error("run_suite: unexpected error for %s: %s", spec.id, exc)
                return (spec.id, "errored", None, None)
            return (spec.id, status, auto_result, record_path)

    async def _driver() -> list[tuple[str, str, AutoResult | None, str | None]]:
        tasks = [asyncio.create_task(_gated(s)) for s in pending]
        try:
            return await asyncio.gather(*tasks, return_exceptions=False)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            # Await cancellation so we don't return while tasks are still
            # mid-write (the writer itself is atomic, but we wait to be tidy).
            results: list[tuple[str, str, AutoResult | None, str | None]] = []
            for t in tasks:
                try:
                    results.append(await t)
                except (asyncio.CancelledError, Exception):
                    continue
            return results

    try:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            raise RuntimeError("run_suite must not be called from a running loop")
        results = asyncio.run(_driver())
    except KeyboardInterrupt:
        logger.warning("run_suite: KeyboardInterrupt — cancelling outstanding work")
        results = []
    finally:
        suite.total_elapsed_s = int(time.time() - start_wall)

    for _spec_id, status, auto_result, record_path in results:
        if status == "succeeded":
            suite.succeeded += 1
            if record_path is not None:
                suite.records_written.append(record_path)
        elif status == "failed":
            suite.failed += 1
        else:
            suite.errored += 1

        if auto_result is not None:
            suite.total_cost_usd += float(auto_result.total_cost_usd or 0.0)

    return suite


__all__ = [
    "BatchOptimizer",
    "BatchResult",
    "ProblemResult",
    "SuiteResult",
    "run_suite",
]
