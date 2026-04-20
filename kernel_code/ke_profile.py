"""KE (Kernel Engineer) profile — learns stopping defaults from session history.

Tracks optimization outcomes across sessions and computes per-problem-type
defaults for the stopping strategy.  Over 10+ runs, the system learns:
"for GEMM problems, this KE's best results come by iteration 5."

Data is stored in ``.kernel-code/ke_profile.json`` and loaded at the start
of each optimization run to adjust the :class:`StoppingStrategy` defaults.

Usage::

    from kernel_code.ke_profile import KEProfile

    profile = KEProfile()
    defaults = profile.get_defaults("gemm")
    # defaults = {"max_iterations": 8, "convergence_patience": 4}

    # After a run:
    profile.record_run(
        problem_type="gemm",
        total_iterations=10,
        best_at_iteration=4,
        final_speedup=2.1,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROFILE_PATH = _PROJECT_ROOT / ".kernel-code" / "ke_profile.json"


# ---------------------------------------------------------------------------
# Problem type classification
# ---------------------------------------------------------------------------

_PROBLEM_KEYWORDS: list[tuple[str, list[str]]] = [
    # Order matters — more specific types checked first
    ("gemm", ["matmul", "gemm", "matrix_mult", "matrix mult", "linear", " mm "]),
    ("attention", ["attention", "flash", "mha", "multihead", "sdpa"]),
    ("conv", ["conv", "convolution", "conv2d", "conv1d", "depthwise"]),
    ("reduction", ["softmax", "layernorm", "batchnorm", "reduce", "norm"]),
    ("fusion", ["fused", "fusion", "epilogue", "residual_add"]),
    ("elementwise", ["elementwise", "pointwise", "relu", "gelu", "sigmoid", "tanh", "scalar"]),
]


def classify_problem(problem_name: str) -> str:
    """Classify a problem into a type based on its name/description."""
    name_lower = problem_name.lower()
    for problem_type, keywords in _PROBLEM_KEYWORDS:
        if any(kw in name_lower for kw in keywords):
            return problem_type
    return "custom"


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One optimization run outcome."""

    problem_type: str
    total_iterations: int
    best_at_iteration: int
    final_speedup: float
    approaches_tried: list[str] = field(default_factory=list)


class KEProfile:
    """Persistent KE profile that learns stopping defaults from history."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _PROFILE_PATH
        self._runs: list[RunRecord] = []
        self._load()

    def _load(self) -> None:
        """Load profile from disk."""
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text())
            for r in data.get("runs", []):
                self._runs.append(RunRecord(
                    problem_type=r.get("problem_type", "custom"),
                    total_iterations=r.get("total_iterations", 0),
                    best_at_iteration=r.get("best_at_iteration", 0),
                    final_speedup=r.get("final_speedup", 0.0),
                    approaches_tried=r.get("approaches_tried", []),
                ))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load KE profile: %s", exc)

    def _save(self) -> None:
        """Persist profile to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "runs": [
                {
                    "problem_type": r.problem_type,
                    "total_iterations": r.total_iterations,
                    "best_at_iteration": r.best_at_iteration,
                    "final_speedup": r.final_speedup,
                    "approaches_tried": r.approaches_tried,
                }
                for r in self._runs
            ],
            "learned": self._compute_learned(),
        }
        try:
            self._path.write_text(json.dumps(data, indent=2))
        except OSError as exc:
            logger.warning("Failed to save KE profile: %s", exc)

    def record_run(
        self,
        problem_type: str,
        total_iterations: int,
        best_at_iteration: int,
        final_speedup: float,
        approaches_tried: list[str] | None = None,
    ) -> None:
        """Record a completed optimization run."""
        self._runs.append(RunRecord(
            problem_type=problem_type,
            total_iterations=total_iterations,
            best_at_iteration=best_at_iteration,
            final_speedup=final_speedup,
            approaches_tried=approaches_tried or [],
        ))
        self._save()
        logger.info(
            "KE profile: recorded %s run (%d iters, best at %d, %.2fx)",
            problem_type, total_iterations, best_at_iteration, final_speedup,
        )

    def get_defaults(self, problem_type: str) -> dict:
        """Get learned stopping defaults for a problem type.

        Returns a dict of StoppingStrategy field overrides.  Only returns
        values when there are >= 3 historical runs for the problem type
        (need enough data for the averages to be meaningful).
        """
        learned = self._compute_learned()
        return learned.get(problem_type, {})

    def _compute_learned(self) -> dict[str, dict]:
        """Compute learned defaults from run history."""
        from collections import defaultdict

        by_type: dict[str, list[RunRecord]] = defaultdict(list)
        for r in self._runs:
            by_type[r.problem_type].append(r)

        learned: dict[str, dict] = {}
        for ptype, runs in by_type.items():
            if len(runs) < 3:
                continue  # not enough data

            avg_best_at = sum(r.best_at_iteration for r in runs) / len(runs)
            avg_total = sum(r.total_iterations for r in runs) / len(runs)
            avg_speedup = sum(r.final_speedup for r in runs) / len(runs)

            # Set recommended cap at 2x the average best-at iteration,
            # clamped to [5, 20]
            recommended_cap = max(5, min(20, int(avg_best_at * 2)))

            # Set patience based on how spread out the best-at values are
            best_ats = [r.best_at_iteration for r in runs]
            spread = max(best_ats) - min(best_ats) if len(best_ats) > 1 else 0
            recommended_patience = max(2, min(5, int(spread / 2) + 2))

            learned[ptype] = {
                "max_iterations": recommended_cap,
                "convergence_patience": recommended_patience,
                "runs_count": len(runs),
                "avg_best_at": round(avg_best_at, 1),
                "avg_speedup": round(avg_speedup, 2),
            }

        return learned

    @property
    def total_runs(self) -> int:
        return len(self._runs)

    def summary(self) -> str:
        """Human-readable summary of the KE profile."""
        if not self._runs:
            return "No optimization history yet."

        learned = self._compute_learned()
        parts = [f"{self.total_runs} runs recorded"]
        for ptype, defaults in learned.items():
            parts.append(
                f"  {ptype}: {defaults['runs_count']} runs, "
                f"best at iter {defaults['avg_best_at']}, "
                f"avg {defaults['avg_speedup']}x, "
                f"recommended cap: {defaults['max_iterations']}"
            )
        return "\n".join(parts)
