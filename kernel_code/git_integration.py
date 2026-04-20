"""Git integration for tracking kernel optimization experiments.

Creates per-run branches under ``openkernel/`` and commits kept variants so
the full optimization history is preserved in version control -- similar to
how autoresearch tracks experiment lineage.

Safety invariants:
- Never force-pushes.
- Never modifies history on main/master.
- Always works on an ``openkernel/*`` branch.
- ``revert_discarded`` is opt-in only (not called by default hooks).
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_git(*args: str, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command, capturing output."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def is_git_repo() -> bool:
    """Check if we're in a git repository."""
    result = _run_git("rev-parse", "--git-dir")
    return result.returncode == 0


def get_current_branch() -> str:
    """Get current branch name."""
    result = _run_git("branch", "--show-current")
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _is_protected_branch(branch: str) -> bool:
    """Return True if the branch should never be modified by automation."""
    return branch in ("main", "master")


def create_optimization_branch(problem_name: str) -> str | None:
    """Create a new branch for this optimization run.

    Branch naming: ``openkernel/<problem_name>-<monthday>``
    Appends a counter (``-2``, ``-3``, ...) if the name already exists.

    Returns the branch name, or ``None`` if git is unavailable.
    """
    if not is_git_repo():
        return None

    date = datetime.now().strftime("%b%d").lower()
    branch = f"openkernel/{problem_name}-{date}"

    # Check if branch exists; append counter if so
    result = _run_git("branch", "--list", branch)
    if result.stdout.strip():
        for i in range(2, 100):
            alt = f"{branch}-{i}"
            r = _run_git("branch", "--list", alt)
            if not r.stdout.strip():
                branch = alt
                break

    result = _run_git("checkout", "-b", branch)
    if result.returncode != 0:
        logger.warning("Failed to create branch %s: %s", branch, result.stderr.strip())
        return None

    logger.info("Created optimization branch: %s", branch)
    return branch


def commit_kept_variant(
    iteration: int,
    speedup: float,
    intent: str,
    kernel_path: str | Path,
) -> bool:
    """Commit a kept kernel variant.

    The commit message embeds the iteration number, speedup, and a truncated
    intent string so ``git log --oneline`` doubles as an optimization diary.
    """
    if not is_git_repo():
        return False

    # Safety: refuse to commit on main/master
    current = get_current_branch()
    if _is_protected_branch(current):
        logger.warning("Refusing to commit on protected branch %s", current)
        return False

    _run_git("add", str(kernel_path))
    msg = f"iter {iteration}: {speedup:.2f}x -- {intent[:60]}"
    result = _run_git("commit", "-m", msg)
    if result.returncode != 0:
        logger.warning("Git commit failed: %s", result.stderr.strip())
        return False
    return True


def revert_discarded() -> bool:
    """Revert the last commit (discarded variant).

    **Destructive** -- only call when explicitly requested by the user.
    Refuses to operate on main/master.
    """
    if not is_git_repo():
        return False

    current = get_current_branch()
    if _is_protected_branch(current):
        logger.warning("Refusing to revert on protected branch %s", current)
        return False

    result = _run_git("reset", "--hard", "HEAD~1")
    if result.returncode != 0:
        logger.warning("Git revert failed: %s", result.stderr.strip())
        return False
    return True


def get_optimization_log(max_entries: int = 20) -> str:
    """Get git log for the current branch (one-line format)."""
    result = _run_git("log", "--oneline", f"-{max_entries}")
    return result.stdout.strip() if result.returncode == 0 else "Git log unavailable"


def return_to_original_branch(original_branch: str) -> bool:
    """Switch back to the original branch after optimization."""
    result = _run_git("checkout", original_branch)
    if result.returncode != 0:
        logger.warning(
            "Failed to return to branch %s: %s",
            original_branch,
            result.stderr.strip(),
        )
        return False
    return True
