"""KernelBench sweep infrastructure for openkernel.

Provides problem loading, optimization sweeps, scoring metrics, and
baseline comparisons for the KernelBench benchmark suite.

Usage::

    from openkernel.kernelbench import run_sweep, SweepResult, ProblemResult
    from openkernel.kernelbench.scoring import compute_fast_p, format_scores
    from openkernel.kernelbench.compare import generate_comparison_table, BASELINES
"""

from openkernel.kernelbench.compare import (
    BASELINES,
    generate_comparison_table,
    save_comparison,
)
from openkernel.kernelbench.problems import (
    get_all_problems,
    get_problem_count,
    load_problem,
)
from openkernel.kernelbench.scoring import (
    compute_correctness_rate,
    compute_fast_p,
    compute_geomean_speedup,
    format_scores,
)
from openkernel.kernelbench.sweep import ProblemResult, SweepResult, run_sweep

__all__ = [
    # sweep
    "run_sweep",
    "SweepResult",
    "ProblemResult",
    # problems
    "load_problem",
    "get_all_problems",
    "get_problem_count",
    # scoring
    "compute_fast_p",
    "compute_geomean_speedup",
    "compute_correctness_rate",
    "format_scores",
    # compare
    "BASELINES",
    "generate_comparison_table",
    "save_comparison",
]
