"""KernelBench scoring metrics.

Implements the standard KernelBench metrics:
- fast_p: fraction of problems that are correct AND achieve speedup >= p
- geomean_speedup: geometric mean of speedup across correct problems
- correctness_rate: fraction of problems that produce correct output

Public API:
    compute_fast_p(results, p)            -> float
    compute_geomean_speedup(results)      -> float
    compute_correctness_rate(results)     -> float
    format_scores(results)                -> str
"""

from __future__ import annotations

import math


def compute_fast_p(results: list[dict], p: float = 1.0) -> float:
    """Compute fast@p — fraction of problems correct AND speedup >= p.

    Parameters
    ----------
    results : list[dict]
        Each dict must have ``correct`` (bool) and ``final_speedup`` (float).
    p : float
        Speedup threshold (e.g. 1.0, 1.5, 2.0).

    Returns
    -------
    float
        Fraction in [0.0, 1.0].
    """
    if not results:
        return 0.0

    passing = sum(
        1
        for r in results
        if r.get("correct", False) and r.get("final_speedup", 0.0) >= p
    )
    return passing / len(results)


def compute_geomean_speedup(results: list[dict]) -> float:
    """Compute geometric mean speedup across all correct problems.

    Problems that are incorrect are excluded. If no problems are correct,
    returns 0.0.

    Parameters
    ----------
    results : list[dict]
        Each dict must have ``correct`` (bool) and ``final_speedup`` (float).

    Returns
    -------
    float
        Geometric mean speedup (>= 0.0).
    """
    correct_speedups = [
        r["final_speedup"]
        for r in results
        if r.get("correct", False) and r.get("final_speedup", 0.0) > 0.0
    ]
    if not correct_speedups:
        return 0.0

    # Use log-sum-exp to avoid overflow for large product chains.
    log_sum = sum(math.log(s) for s in correct_speedups)
    return math.exp(log_sum / len(correct_speedups))


def compute_correctness_rate(results: list[dict]) -> float:
    """Compute the fraction of problems that are correct.

    Parameters
    ----------
    results : list[dict]
        Each dict must have ``correct`` (bool).

    Returns
    -------
    float
        Fraction in [0.0, 1.0].
    """
    if not results:
        return 0.0

    correct = sum(1 for r in results if r.get("correct", False))
    return correct / len(results)


def format_scores(results: list[dict]) -> str:
    """Pretty-print a summary table of KernelBench scores.

    Parameters
    ----------
    results : list[dict]
        Per-problem result dicts with ``problem_id``, ``problem_name``,
        ``correct``, ``final_speedup``, ``iterations``, ``cost``, ``time``.

    Returns
    -------
    str
        Human-readable summary table.
    """
    if not results:
        return "No results to display."

    # Compute aggregate metrics.
    n = len(results)
    correctness = compute_correctness_rate(results)
    fast_1 = compute_fast_p(results, p=1.0)
    fast_1_5 = compute_fast_p(results, p=1.5)
    fast_2 = compute_fast_p(results, p=2.0)
    geomean = compute_geomean_speedup(results)

    total_cost = sum(r.get("cost", 0.0) for r in results)
    total_time = sum(r.get("time", 0.0) for r in results)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("KernelBench Sweep Results")
    lines.append("=" * 80)
    lines.append("")

    # Aggregate scores
    lines.append("Aggregate Scores:")
    lines.append(f"  Problems:          {n}")
    lines.append(f"  Correctness:       {correctness:.1%}")
    lines.append(f"  fast@1.0:          {fast_1:.1%}")
    lines.append(f"  fast@1.5:          {fast_1_5:.1%}")
    lines.append(f"  fast@2.0:          {fast_2:.1%}")
    lines.append(f"  Geomean speedup:   {geomean:.2f}x")
    lines.append(f"  Total cost:        ${total_cost:.2f}")
    lines.append(f"  Total time:        {total_time:.1f}s")
    lines.append("")

    # Per-problem table
    lines.append("Per-Problem Results:")
    header = f"  {'ID':>4}  {'Name':<35}  {'OK?':>4}  {'Speedup':>8}  {'Iters':>5}  {'Cost':>7}  {'Time':>7}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for r in results:
        pid = r.get("problem_id", "?")
        name = r.get("problem_name", "unknown")
        if len(name) > 35:
            name = name[:32] + "..."
        ok = "yes" if r.get("correct", False) else "NO"
        speedup = r.get("final_speedup", 0.0)
        iters = r.get("iterations", 0)
        cost = r.get("cost", 0.0)
        t = r.get("time", 0.0)

        lines.append(
            f"  {pid:>4}  {name:<35}  {ok:>4}  {speedup:>7.2f}x  {iters:>5}  ${cost:>6.2f}  {t:>6.1f}s"
        )

    lines.append("=" * 80)
    return "\n".join(lines)
