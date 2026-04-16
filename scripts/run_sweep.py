#!/usr/bin/env python3
"""CLI for running KernelBench sweeps.

Usage:
    python scripts/run_sweep.py --level 1
    python scripts/run_sweep.py --level 1 --problems 0,1,2
    python scripts/run_sweep.py --level 2 --max-iterations 50 --output results/sweeps/
    python scripts/run_sweep.py --level 1 --source mock   # synthetic problems for testing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so ``openkernel`` can be imported
# when running the script directly (e.g. ``python scripts/run_sweep.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openkernel.config import OpenKernelConfig
from openkernel.kernelbench.compare import generate_comparison_table, save_comparison
from openkernel.kernelbench.scoring import format_scores
from openkernel.kernelbench.sweep import run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an openkernel KernelBench sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--level",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="KernelBench level to sweep.",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="Comma-separated problem IDs to run (e.g. 0,1,5). Default: all.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Max optimization iterations per problem (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sweeps/",
        help="Output directory for results (default: results/sweeps/).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="local",
        choices=["local", "mock"],
        help="Problem source: 'local' (kernelbench package) or 'mock' (synthetic).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        choices=["triton", "cuda"],
        help="Target backend (default: triton).",
    )
    parser.add_argument(
        "--no-prescreen",
        action="store_true",
        help="Disable analytical roofline pre-screening.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Configure logging.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse problem IDs.
    problem_ids: list[int] | None = None
    if args.problems:
        problem_ids = [int(p.strip()) for p in args.problems.split(",")]

    # Build config.
    config = OpenKernelConfig(
        backend=args.backend,
        analytical_prescreen=not args.no_prescreen,
    )

    print(f"\n{'=' * 60}")
    print(f"openkernel KernelBench Sweep")
    print(f"{'=' * 60}")
    print(f"Level:           {args.level}")
    print(f"Problems:        {problem_ids or 'all'}")
    print(f"Max iterations:  {args.max_iterations}")
    print(f"Backend:         {args.backend}")
    print(f"Source:          {args.source}")
    print(f"Pre-screen:      {'yes' if config.analytical_prescreen else 'no'}")
    print(f"Output:          {args.output}")
    print(f"{'=' * 60}\n")

    # Run the sweep.
    sweep_result = await run_sweep(
        level=args.level,
        config=config,
        max_iterations_per_problem=args.max_iterations,
        problems=problem_ids,
        source=args.source,
    )

    # Print results.
    result_dicts = [r.to_dict() for r in sweep_result.results]
    print("\n" + format_scores(result_dicts))

    # Generate comparison table.
    comparison = generate_comparison_table(sweep_result)
    print("\n" + comparison)

    # Save results.
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"L{args.level}_{timestamp}"

    # Save raw results JSON.
    results_path = output_dir / f"{base_name}_results.json"
    results_path.write_text(
        json.dumps(sweep_result.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {results_path}")

    # Save comparison table.
    comparison_path = output_dir / f"{base_name}_comparison.md"
    save_comparison(comparison, comparison_path)
    print(f"Comparison saved to: {comparison_path}")

    # Save summary scores.
    scores_path = output_dir / f"{base_name}_scores.txt"
    scores_path.write_text(format_scores(result_dicts), encoding="utf-8")
    print(f"Scores saved to: {scores_path}")


if __name__ == "__main__":
    asyncio.run(main())
