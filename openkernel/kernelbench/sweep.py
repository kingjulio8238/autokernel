"""KernelBench sweep orchestration.

Runs openkernel's optimize() across all (or selected) problems in a
KernelBench level, collects results, and computes aggregate metrics.

Public API:
    run_sweep(level, config, ...)  -> SweepResult
    SweepResult                    — aggregate sweep output
    ProblemResult                  — per-problem output
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openkernel.config import OpenKernelConfig
from openkernel.kernelbench.problems import get_problem_count, load_problem
from openkernel.kernelbench.scoring import (
    compute_correctness_rate,
    compute_fast_p,
    compute_geomean_speedup,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ProblemResult:
    """Result of optimizing a single KernelBench problem."""

    problem_id: int
    problem_name: str
    final_speedup: float = 0.0
    correct: bool = False
    iterations: int = 0
    cost: float = 0.0
    time: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for scoring functions."""
        return {
            "problem_id": self.problem_id,
            "problem_name": self.problem_name,
            "final_speedup": self.final_speedup,
            "correct": self.correct,
            "iterations": self.iterations,
            "cost": self.cost,
            "time": self.time,
            "error": self.error,
        }


@dataclass
class SweepResult:
    """Aggregate result of a full KernelBench level sweep."""

    level: int
    results: list[ProblemResult] = field(default_factory=list)
    fast_p_scores: dict[float, float] = field(default_factory=dict)
    geomean_speedup: float = 0.0
    correctness_rate: float = 0.0
    total_cost: float = 0.0
    total_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "level": self.level,
            "results": [r.to_dict() for r in self.results],
            "fast_p_scores": {str(k): v for k, v in self.fast_p_scores.items()},
            "geomean_speedup": self.geomean_speedup,
            "correctness_rate": self.correctness_rate,
            "total_cost": self.total_cost,
            "total_time": self.total_time,
        }


# ---------------------------------------------------------------------------
# Roofline pre-screening
# ---------------------------------------------------------------------------


def _prescreen_problems(
    problems: list[dict[str, Any]],
    config: OpenKernelConfig,
) -> list[dict[str, Any]]:
    """Sort problems by estimated optimization headroom (descending).

    Uses the analytical roofline estimator to estimate headroom without
    needing GPU time. Problems with more headroom are prioritized.
    """
    if not config.analytical_prescreen:
        return problems

    scored: list[tuple[float, dict[str, Any]]] = []

    for problem in problems:
        try:
            from openkernel.eval.profilers.roofline import estimate_roofline

            profile = estimate_roofline(
                reference_source=problem["reference_source"],
                hardware=config.modal.gpu_type.value,
            )
            headroom = profile.raw_metrics.get("estimated_headroom", 1.0)
        except Exception as exc:
            logger.debug(
                "Roofline pre-screen failed for %s: %s",
                problem.get("problem_name", "?"),
                exc,
            )
            headroom = 1.0  # neutral score on failure

        scored.append((headroom, problem))

    # Sort by headroom descending (most promising first).
    scored.sort(key=lambda x: x[0], reverse=True)

    logger.info(
        "Pre-screening: sorted %d problems by headroom (top: %.2fx, bottom: %.2fx)",
        len(scored),
        scored[0][0] if scored else 0,
        scored[-1][0] if scored else 0,
    )

    return [problem for _, problem in scored]


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


async def run_sweep(
    level: int,
    config: OpenKernelConfig | None = None,
    max_iterations_per_problem: int = 50,
    problems: list[int] | None = None,
    eval_fn: Any | None = None,
    source: str = "local",
) -> SweepResult:
    """Run openkernel on all (or selected) problems in a KernelBench level.

    Parameters
    ----------
    level : int
        KernelBench level (1, 2, or 3).
    config : OpenKernelConfig, optional
        Configuration for the optimization engine.
    max_iterations_per_problem : int
        Maximum optimization iterations per problem.
    problems : list[int], optional
        Specific problem IDs to run. ``None`` runs all problems in the level.
    eval_fn : callable, optional
        Async eval function to pass to ``optimize()``. If ``None``, the
        real Modal eval function is used.
    source : str
        Problem source: ``"local"`` loads from the ``kernelbench`` package,
        ``"mock"`` uses synthetic problems for testing.

    Returns
    -------
    SweepResult
        Aggregate results across all problems.

    Raises
    ------
    ImportError
        If the ``kernelbench`` package is not installed (when *source* is
        ``"local"``).
    """
    from openkernel import optimize

    if config is None:
        config = OpenKernelConfig()

    # Override max_iterations for per-problem budget.
    per_problem_config = config.model_copy(
        update={"max_iterations": max_iterations_per_problem}
    )

    # Select problem loader based on source.
    if source == "mock":
        from tests.mocks import MockProblemLoader

        _load = MockProblemLoader.load
    else:
        _load = load_problem

    # Load problems.
    if problems is not None:
        problem_list = [
            _load(level, pid) for pid in problems
        ]
    else:
        total = get_problem_count(level)
        problem_list = [
            _load(level, pid) for pid in range(total)
        ]

    logger.info(
        "Sweep: Level %d, %d problems, max %d iterations/problem",
        level,
        len(problem_list),
        max_iterations_per_problem,
    )

    # Pre-screen with roofline (sort by headroom).
    problem_list = _prescreen_problems(problem_list, per_problem_config)

    # Run optimization on each problem.
    sweep_results: list[ProblemResult] = []
    sweep_start = time.monotonic()

    for i, problem in enumerate(problem_list):
        pid = problem["problem_id"]
        pname = problem["problem_name"]
        ref_source = problem["reference_source"]

        logger.info(
            "Sweep [%d/%d]: Starting L%d#%d — %s",
            i + 1,
            len(problem_list),
            level,
            pid,
            pname,
        )

        problem_start = time.monotonic()

        try:
            opt_result = await optimize(
                reference_source=ref_source,
                backend=per_problem_config.backend.value,
                config=per_problem_config,
                eval_fn=eval_fn,
            )

            problem_time = time.monotonic() - problem_start
            correct = opt_result.final_speedup > 0.0 and opt_result.final_kernel != ""

            result = ProblemResult(
                problem_id=pid,
                problem_name=pname,
                final_speedup=opt_result.final_speedup,
                correct=correct,
                iterations=opt_result.iterations_total,
                cost=opt_result.total_cost_usd,
                time=problem_time,
            )

            logger.info(
                "Sweep [%d/%d]: L%d#%d — %s — speedup=%.2fx, correct=%s, iters=%d, time=%.1fs",
                i + 1,
                len(problem_list),
                level,
                pid,
                pname,
                opt_result.final_speedup,
                correct,
                opt_result.iterations_total,
                problem_time,
            )

        except Exception as exc:
            problem_time = time.monotonic() - problem_start
            logger.error(
                "Sweep [%d/%d]: L%d#%d — %s — FAILED: %s",
                i + 1,
                len(problem_list),
                level,
                pid,
                pname,
                exc,
            )
            result = ProblemResult(
                problem_id=pid,
                problem_name=pname,
                final_speedup=0.0,
                correct=False,
                iterations=0,
                cost=0.0,
                time=problem_time,
                error=str(exc),
            )

        sweep_results.append(result)

    sweep_time = time.monotonic() - sweep_start

    # Compute aggregate metrics.
    result_dicts = [r.to_dict() for r in sweep_results]

    fast_p_scores = {
        1.0: compute_fast_p(result_dicts, p=1.0),
        1.5: compute_fast_p(result_dicts, p=1.5),
        2.0: compute_fast_p(result_dicts, p=2.0),
    }

    return SweepResult(
        level=level,
        results=sweep_results,
        fast_p_scores=fast_p_scores,
        geomean_speedup=compute_geomean_speedup(result_dicts),
        correctness_rate=compute_correctness_rate(result_dicts),
        total_cost=sum(r.cost for r in sweep_results),
        total_time=sweep_time,
    )
