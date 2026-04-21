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

    Wraps ``kernelbench.dataset.construct_kernelbench_dataset()`` and
    uses ``get_problem_by_id`` to fetch a single problem. Uses
    ``source="huggingface"`` so the dataset works from a plain pip
    install — ``source="local"`` needs the KernelBench repo cloned
    alongside the package.
    """
    try:
        from kernelbench.dataset import construct_kernelbench_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'kernelbench' package is not installed. Install it with: "
            "uv pip install 'kernelbench @ git+https://github.com/ScalingIntelligence/KernelBench.git'"
        ) from exc

    dataset = construct_kernelbench_dataset(level=level, source="huggingface")
    problem = dataset.get_problem_by_id(problem_id)

    reference_source = getattr(problem, "code", None)
    if reference_source is None:
        reference_source = getattr(problem, "source", None)
    if reference_source is None:
        reference_source = str(problem)

    problem_name = getattr(problem, "name", None) or f"L{level}_Problem_{problem_id}"

    return {
        "reference_source": reference_source,
        "problem_name": problem_name,
        "level": level,
        "problem_id": problem_id,
    }
