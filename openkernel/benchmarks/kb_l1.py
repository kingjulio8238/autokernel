"""KernelBench Level 1 loader.

Wraps all 100 KernelBench L1 problems as ``ProblemSpec`` instances for
consumption by the batch-runner. Problems are 1-indexed (``1..100``).
"""

from __future__ import annotations

from openkernel.benchmarks.problem_spec import ProblemSpec
from openkernel.kernelbench.problems import load_problem

# KernelBench L1 has 100 problems, 1-indexed.
_L1_COUNT = 100


def load_l1() -> list[ProblemSpec]:
    """Load all KernelBench Level 1 problems as ProblemSpec instances.

    Returns exactly 100 specs. Any problem whose source cannot be loaded
    raises (don't silently skip — Round B gate asserts len == 100).
    """
    specs: list[ProblemSpec] = []
    for pid in range(1, _L1_COUNT + 1):
        record = load_problem(1, pid)
        specs.append(
            ProblemSpec(
                id=f"kb_l1_{pid:04d}",
                name=record["problem_name"],
                tier="L1",
                source="kernelbench",
                reference_source=record["reference_source"],
                workload_spec={},
                expected_dtype="float32",
            )
        )
    return specs


if __name__ == "__main__":
    specs = load_l1()
    print(f"Loaded {len(specs)} L1 specs")
    print(f"First: {specs[0].id} — {specs[0].name}")
