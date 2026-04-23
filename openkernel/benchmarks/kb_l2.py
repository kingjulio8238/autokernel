from __future__ import annotations

from openkernel.kernelbench.problems import load_problem
from openkernel.benchmarks.problem_spec import ProblemSpec

_L2_COUNT = 100


def load_l2() -> list[ProblemSpec]:
    """Load all KernelBench Level 2 problems as ProblemSpec instances."""
    specs: list[ProblemSpec] = []
    for pid in range(1, _L2_COUNT + 1):
        record = load_problem(2, pid)
        specs.append(
            ProblemSpec(
                id=f"kb_l2_{pid:04d}",
                name=record["problem_name"],
                tier="L2",
                source="kernelbench",
                reference_source=record["reference_source"],
                workload_spec={},
                expected_dtype="float32",
            )
        )
    return specs


if __name__ == "__main__":
    specs = load_l2()
    print(f"Loaded {len(specs)} L2 specs")
    print(f"First: {specs[0].id} — {specs[0].name}")
