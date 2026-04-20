#!/usr/bin/env python3
"""Format and publish KernelBench sweep results.

Usage:
    python scripts/publish_results.py --results results/sweeps/L1_20260416_results.json
    python scripts/publish_results.py --results results/sweeps/L1_20260416_results.json --upload
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``openkernel`` can be imported
# when running the script directly (e.g. ``python scripts/publish_results.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openkernel.kernelbench.compare import generate_comparison_table
from openkernel.kernelbench.sweep import ProblemResult, SweepResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format sweep results for publication and optionally upload to HF Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to a sweep results JSON file (output of run_sweep.py).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to HF Hub (requires HF_TOKEN).",
    )
    return parser.parse_args()


def load_sweep_result(path: Path) -> SweepResult:
    """Load a SweepResult from a JSON file produced by run_sweep.py."""
    data = json.loads(path.read_text(encoding="utf-8"))

    problem_results = [
        ProblemResult(
            problem_id=r["problem_id"],
            problem_name=r["problem_name"],
            final_speedup=r.get("final_speedup", 0.0),
            correct=r.get("correct", False),
            iterations=r.get("iterations", 0),
            cost=r.get("cost", 0.0),
            time=r.get("time", 0.0),
            error=r.get("error"),
        )
        for r in data.get("results", [])
    ]

    # fast_p_scores keys are stored as strings in JSON; convert back to float.
    fast_p_scores = {
        float(k): v for k, v in data.get("fast_p_scores", {}).items()
    }

    return SweepResult(
        level=data["level"],
        results=problem_results,
        fast_p_scores=fast_p_scores,
        geomean_speedup=data.get("geomean_speedup", 0.0),
        correctness_rate=data.get("correctness_rate", 0.0),
        total_cost=data.get("total_cost", 0.0),
        total_time=data.get("total_time", 0.0),
    )


def format_blog_summary(sweep: SweepResult) -> str:
    """Return a blog-post-ready summary with headline numbers."""
    n = len(sweep.results)
    fast_1 = sweep.fast_p_scores.get(1.0, 0.0)
    geomean = sweep.geomean_speedup
    cost_per_kernel = sweep.total_cost / n if n > 0 else 0.0

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("BLOG-POST SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Level {sweep.level} | {n} problems")
    lines.append("")
    lines.append(f"  fast@1.0 (correct & faster): {fast_1:.1%}")
    lines.append(f"  Geomean speedup:             {geomean:.2f}x")
    lines.append(f"  Cost per kernel:             ${cost_per_kernel:.2f}")
    lines.append("")
    lines.append(
        f"openkernel optimized {fast_1:.0%} of KernelBench Level {sweep.level} "
        f"kernels to match or beat the reference, achieving a "
        f"{geomean:.2f}x geometric mean speedup at ${cost_per_kernel:.2f}/kernel."
    )
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    # Load results.
    sweep = load_sweep_result(results_path)
    print(f"Loaded sweep: Level {sweep.level}, {len(sweep.results)} problems\n")

    # Generate and print comparison table.
    comparison = generate_comparison_table(sweep)
    print(comparison)
    print()

    # Print blog-post summary.
    summary = format_blog_summary(sweep)
    print(summary)

    # Optionally upload to HF Hub.
    if args.upload:
        from openkernel.hub.client import HubClient
        from openkernel.hub.datasets import upload_results

        client = HubClient()
        upload_results(results_path.parent, client)
        print(f"\nResults uploaded to HF Hub from {results_path.parent}")


if __name__ == "__main__":
    main()
