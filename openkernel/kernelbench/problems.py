"""KernelBench problem loader.

Loads problems from the KernelBench dataset by level and problem ID.

Public API:
    load_problem(level, problem_id)  -> dict
    get_all_problems(level)          -> list[dict]
    get_problem_count(level)         -> int
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Number of problems per level in the official KernelBench dataset.
_LEVEL_PROBLEM_COUNTS: dict[int, int] = {
    1: 100,
    2: 50,
    3: 50,
}


def load_problem(
    level: int,
    problem_id: int,
) -> dict[str, Any]:
    """Load a single KernelBench problem.

    Parameters
    ----------
    level : int
        KernelBench level (1, 2, or 3).
    problem_id : int
        Zero-indexed problem ID within the level.

    Returns
    -------
    dict
        ``{reference_source, problem_name, level, problem_id}``

    Raises
    ------
    ImportError
        If the ``kernelbench`` package is not installed.
    """
    return _load_from_kernelbench(level, problem_id)


def get_all_problems(level: int) -> list[dict[str, Any]]:
    """Load all problems for a KernelBench level.

    Parameters
    ----------
    level : int
        KernelBench level (1, 2, or 3).

    Returns
    -------
    list[dict]
        List of problem dicts, one per problem in the level.

    Raises
    ------
    ImportError
        If the ``kernelbench`` package is not installed.
    """
    count = get_problem_count(level)
    problems: list[dict[str, Any]] = []
    for pid in range(count):
        problems.append(load_problem(level, pid))
    return problems


def get_problem_count(level: int) -> int:
    """Return the number of problems in a KernelBench level."""
    return _LEVEL_PROBLEM_COUNTS.get(level, 0)


# ---------------------------------------------------------------------------
# Real loader (requires ``kernelbench`` package)
# ---------------------------------------------------------------------------


def _load_from_kernelbench(level: int, problem_id: int) -> dict[str, Any]:
    """Load a problem using the ``kernelbench`` package.

    Wraps ``kernelbench.dataset.construct_kernelbench_dataset()`` to
    extract the reference source for a single problem.
    """
    try:
        from kernelbench.dataset import construct_kernelbench_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'kernelbench' package is not installed. "
            "Install it with: pip install kernelbench"
        ) from exc

    dataset = construct_kernelbench_dataset(level=level)

    if problem_id < 0 or problem_id >= len(dataset):
        raise IndexError(
            f"Problem ID {problem_id} out of range for level {level} "
            f"(has {len(dataset)} problems)."
        )

    problem = dataset[problem_id]

    # The kernelbench dataset returns problem objects; extract the source code.
    # The exact attribute depends on the kernelbench version.
    if hasattr(problem, "source"):
        reference_source = problem.source
    elif hasattr(problem, "code"):
        reference_source = problem.code
    elif isinstance(problem, dict):
        reference_source = problem.get("source", problem.get("code", str(problem)))
    elif isinstance(problem, str):
        reference_source = problem
    else:
        # Last resort: try reading the file path if the problem is a path-like
        reference_source = str(problem)

    # Extract the problem name if available.
    if hasattr(problem, "name"):
        problem_name = problem.name
    elif isinstance(problem, dict) and "name" in problem:
        problem_name = problem["name"]
    else:
        problem_name = f"L{level}_Problem_{problem_id}"

    return {
        "reference_source": reference_source,
        "problem_name": problem_name,
        "level": level,
        "problem_id": problem_id,
    }
