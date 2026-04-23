#!/usr/bin/env python3
"""Aggregate leaderboard records into publication-ready scores.

Reads the leaderboard via ``openkernel.benchmarks.leaderboard_reader`` for a
given ``(date, hardware)`` pair and emits per-tier aggregated metrics as JSON
on stdout. The output is consumed by ``scripts/emit_report.py`` to render the
markdown report.

Usage:
    python scripts/score_suite.py --date 2026-04-22 --hardware L40S
    python scripts/score_suite.py --date 2026-04-22 --hardware H100 --tier L1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openkernel.benchmarks.leaderboard_reader import (  # noqa: E402
    filter_records,
    latest_per_problem,
    load_all,
)

_VALID_HARDWARE = ("L40S", "H100", "A100-80GB", "B200")
_TIERS = ("L1", "L2", "GPU_MODE")
_SOL_SOLVED_THRESHOLD = 0.3


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy-compatible default)."""
    if not values:
        raise ValueError("percentile requires at least one value")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _geomean(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        raise ValueError("geomean requires at least one positive value")
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def _tier_problem_count(tier: str) -> int:
    """Return the canonical problem count for a tier.

    KernelBench L1/L2 are fixed-size suites (100 each); GPU_MODE is a curated
    sample (~50). ALL is their sum. These are published denominators for the
    ``solved_rate_at_sol_0.3`` metric.
    """
    if tier == "L1":
        return 100
    if tier == "L2":
        return 100
    if tier == "GPU_MODE":
        return 50
    if tier == "ALL":
        return _tier_problem_count("L1") + _tier_problem_count("L2") + _tier_problem_count("GPU_MODE")
    raise ValueError(f"unknown tier: {tier!r}")


def _aggregate(records: list[dict], tier: str) -> dict:
    """Compute aggregated metrics for a single tier bucket.

    ``records`` is the set of deduped records (latest-per-problem) for this
    tier; it may contain both correct and incorrect records. Metrics that
    depend on kernel quality (SOL, speedup) use only the correct subset.
    """
    problem_count = _tier_problem_count(tier)
    attempted = len(records)
    correct_records = [r for r in records if r.get("correct") is True]
    correct = len(correct_records)
    correctness_rate = correct / attempted if attempted else 0.0

    solved_at_threshold = sum(
        1 for r in correct_records if r.get("sol_score", 0.0) >= _SOL_SOLVED_THRESHOLD
    )
    solved_rate = solved_at_threshold / problem_count if problem_count else 0.0

    if correct:
        sols = [float(r["sol_score"]) for r in correct_records]
        mean_sol = sum(sols) / len(sols)
        median_sol = _percentile(sols, 50.0)
        p90_sol = _percentile(sols, 90.0)
        speedups = [float(r["speedup"]) for r in correct_records if float(r.get("speedup", 0)) > 0]
        geomean_speedup = _geomean(speedups) if speedups else None
    else:
        mean_sol = None
        median_sol = None
        p90_sol = None
        geomean_speedup = None

    return {
        "problem_count": problem_count,
        "attempted": attempted,
        "correct": correct,
        "correctness_rate": correctness_rate,
        "solved_rate_at_sol_0.3": solved_rate,
        "mean_sol": mean_sol,
        "median_sol": median_sol,
        "p90_sol": p90_sol,
        "geomean_speedup": geomean_speedup,
    }


def score(
    date: str,
    hardware: str,
    tier: str = "ALL",
    root: Path | str | None = None,
) -> dict:
    """Compute the score suite for a ``(date, hardware, tier)`` slice.

    ``tier`` may be ``"L1"``, ``"L2"``, ``"GPU_MODE"``, or ``"ALL"``. When
    ``"ALL"``, per-tier buckets are populated plus a combined ``ALL`` bucket.
    When a specific tier is requested, only that bucket appears under
    ``tiers`` (no ``ALL`` synthesis).
    """
    all_records = load_all(root)
    # include incorrect records so attempted count is honest
    date_hw_records = filter_records(
        all_records,
        hardware=hardware,
        date_range=(date, date),
        correct_only=False,
    )
    deduped = latest_per_problem(date_hw_records)

    tiers_to_report: list[str] = [tier] if tier != "ALL" else list(_TIERS) + ["ALL"]
    tiers_out: dict[str, dict] = {}
    for t in tiers_to_report:
        if t == "ALL":
            bucket = deduped
        else:
            bucket = [r for r in deduped if r.get("tier") == t]
        tiers_out[t] = _aggregate(bucket, t)

    cost_total = sum(float(r.get("cost_usd", 0.0)) for r in deduped)
    elapsed_total = sum(int(r.get("elapsed_s", 0)) for r in deduped)

    result: dict = {
        "date": date,
        "hardware": hardware,
        "tiers": tiers_out,
        "cost_usd_total": cost_total,
        "elapsed_s_total": elapsed_total,
    }
    if not deduped:
        result["note"] = (
            f"no leaderboard records found for date={date} hardware={hardware}; "
            "all tier metrics are zeroed/null"
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate leaderboard records into per-tier JSON scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date", required=True, help="ISO-8601 date (YYYY-MM-DD).")
    parser.add_argument(
        "--hardware",
        required=True,
        choices=_VALID_HARDWARE,
        help="Target hardware (must match record 'hardware' field).",
    )
    parser.add_argument(
        "--tier",
        default="ALL",
        choices=list(_TIERS) + ["ALL"],
        help="Restrict aggregation to one tier (default: ALL).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Leaderboard root override (tests only). Defaults to results/leaderboard/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = score(
        date=args.date,
        hardware=args.hardware,
        tier=args.tier,
        root=args.root,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
