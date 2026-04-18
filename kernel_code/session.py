"""Session management and post-optimization save logic for kernel code.

Handles persistent session tracking across optimization runs, saving optimized
kernels and result summaries to disk, and resume-from-previous-session logic.

Storage layout::

    .kernel-code/
      sessions/
        {session_id}.json   # Session with embedded run history

    {reference_dir}/
      {stem}_optimized.py   # Best kernel code (written by save_optimization_result)
      {stem}_results.md     # Human-readable summary
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SESSIONS_DIR = _PROJECT_ROOT / ".kernel-code" / "sessions"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SessionRun:
    """A single optimization run within a session."""

    run_id: str
    reference_path: str
    backend: str
    config_path: str | None
    started_at: str
    completed_at: str | None
    iterations: int
    kept: int
    best_speedup: float
    cost: float
    best_kernel_path: str | None  # path to saved optimized kernel
    cache_session_id: str  # for dashboard link
    summary: str


@dataclass
class Session:
    """Persistent session that groups multiple optimization runs."""

    session_id: str
    created_at: str
    last_active: str
    runs: list[SessionRun] = field(default_factory=list)
    current_reference: str | None = None
    current_backend: str = "triton"
    conversation_history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _ensure_sessions_dir() -> Path:
    """Create the sessions directory if it does not exist and return its path."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


def save_session(session: Session) -> Path:
    """Serialize a Session to JSON on disk.

    Returns:
        Path to the written JSON file.
    """
    dest = _ensure_sessions_dir() / f"{session.session_id}.json"
    dest.write_text(json.dumps(asdict(session), indent=2))
    logger.info("Session saved: %s", dest)
    return dest


def load_session(session_id: str) -> Session:
    """Load a Session from its JSON file.

    Raises:
        FileNotFoundError: If no session file exists for *session_id*.
    """
    path = _SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {path}")

    data = json.loads(path.read_text())
    runs = [SessionRun(**r) for r in data.get("runs", [])]
    return Session(
        session_id=data["session_id"],
        created_at=data["created_at"],
        last_active=data["last_active"],
        runs=runs,
        current_reference=data.get("current_reference"),
        current_backend=data.get("current_backend", "triton"),
        conversation_history=data.get("conversation_history", []),
    )


def load_latest_session() -> Session | None:
    """Return the most recently active session, or ``None`` if none exist."""
    sessions = list_sessions()
    if not sessions:
        return None

    # Sort by last_active descending and return the first
    sessions.sort(key=lambda s: s.last_active, reverse=True)
    return sessions[0]


def list_sessions() -> list[Session]:
    """List all saved sessions, ordered by last_active (most recent first)."""
    if not _SESSIONS_DIR.exists():
        return []

    results: list[Session] = []
    for path in _SESSIONS_DIR.glob("*.json"):
        try:
            results.append(load_session(path.stem))
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Skipping malformed session file: %s", path)

    results.sort(key=lambda s: s.last_active, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Post-optimization save
# ---------------------------------------------------------------------------


def save_optimization_result(
    *,
    session: Session,
    run_id: str,
    reference_path: str | Path,
    backend: str,
    config_path: str | Path | None,
    cache_session_id: str,
    best_kernel_code: str,
    best_speedup: float,
    best_iteration: int,
    iterations_total: int,
    iterations_kept: int,
    cost: float,
    wall_time_seconds: float,
    hardware: str,
    top_iterations: list[dict],
    dashboard_port: int = 8050,
) -> SessionRun:
    """Save optimization outputs and update the session.

    Writes:
      1. ``{reference_dir}/{stem}_optimized.py`` -- best kernel source
      2. ``{reference_dir}/{stem}_results.md``   -- human-readable summary
      3. Updated session JSON with the new run appended

    Args:
        session: The active Session to append the run to.
        run_id: Unique identifier for this run.
        reference_path: Path to the original reference kernel file.
        backend: Code-generation backend used (``"triton"`` or ``"cuda"``).
        config_path: Path to the YAML config file, if any.
        cache_session_id: Session ID used by the dashboard / cache layer.
        best_kernel_code: Source code of the best optimized kernel.
        best_speedup: Peak speedup achieved.
        best_iteration: Iteration number that achieved the best speedup.
        iterations_total: Total number of iterations executed.
        iterations_kept: Number of iterations that beat the previous best.
        cost: Estimated total cost in USD.
        wall_time_seconds: Total wall-clock time.
        hardware: GPU hardware used (e.g. ``"L40S"``).
        top_iterations: List of dicts with ``iteration``, ``speedup``, and
            ``intent`` keys, sorted by speedup descending.
        dashboard_port: Port for the dashboard URL (default 8050).

    Returns:
        The newly created :class:`SessionRun`.
    """
    ref = Path(reference_path)
    ref_dir = ref.parent
    stem = ref.stem

    # -- 1. Save best kernel ------------------------------------------------
    optimized_path = ref_dir / f"{stem}_optimized.py"
    optimized_path.write_text(best_kernel_code)
    logger.info("Optimized kernel saved: %s", optimized_path)

    # -- 2. Build results markdown ------------------------------------------
    dashboard_url = get_dashboard_url(cache_session_id, port=dashboard_port)

    # Format wall time
    if wall_time_seconds >= 60:
        time_str = f"{wall_time_seconds / 60:.1f}m"
    else:
        time_str = f"{wall_time_seconds:.0f}s"

    table_rows = ""
    for entry in top_iterations:
        it_num = entry.get("iteration", "?")
        spd = entry.get("speedup", 0.0)
        intent = entry.get("intent", "")
        table_rows += f"| {it_num} | {spd:.2f}x | {intent} |\n"

    summary_md = (
        f"# Optimization Results: {ref.name}\n"
        f"\n"
        f"**Best speedup:** {best_speedup:.2f}x (iteration {best_iteration})\n"
        f"**Iterations:** {iterations_total} total, {iterations_kept} kept\n"
        f"**Cost:** ${cost:.2f}\n"
        f"**Time:** {time_str}\n"
        f"**Backend:** {backend}\n"
        f"**Hardware:** {hardware}\n"
        f"\n"
        f"## Top Optimizations\n"
        f"| # | Speedup | Intent |\n"
        f"|---|---------|--------|\n"
        f"{table_rows}"
        f"\n"
        f"## Dashboard\n"
        f"View full analysis: {dashboard_url}\n"
    )

    results_path = ref_dir / f"{stem}_results.md"
    results_path.write_text(summary_md)
    logger.info("Results summary saved: %s", results_path)

    # -- 3. Build summary string for the SessionRun -------------------------
    summary = (
        f"{best_speedup:.2f}x speedup in {iterations_total} iterations "
        f"({iterations_kept} kept), ${cost:.2f}, {time_str}"
    )

    now = datetime.now(timezone.utc).isoformat()

    run = SessionRun(
        run_id=run_id,
        reference_path=str(ref),
        backend=backend,
        config_path=str(config_path) if config_path else None,
        started_at=session.last_active,  # approximate; caller can override
        completed_at=now,
        iterations=iterations_total,
        kept=iterations_kept,
        best_speedup=best_speedup,
        cost=cost,
        best_kernel_path=str(optimized_path),
        cache_session_id=cache_session_id,
        summary=summary,
    )

    # -- 4. Update session --------------------------------------------------
    session.runs.append(run)
    session.last_active = now
    session.current_reference = str(ref)
    session.current_backend = backend
    save_session(session)

    return run


# ---------------------------------------------------------------------------
# Dashboard link
# ---------------------------------------------------------------------------


def get_dashboard_url(session_id: str, port: int = 8050) -> str:
    """Return the localhost URL for viewing a session in the dashboard."""
    return f"http://localhost:{port}/session/{session_id}"


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

_RESUME_THRESHOLD = timedelta(hours=24)


def should_resume() -> bool:
    """Check whether a recent session exists that could be resumed.

    A session is considered "recent" if its ``last_active`` timestamp is less
    than 24 hours old.  When this returns ``True`` the caller should prompt
    the user with something like::

        Resume session abc123? (y/n)

    Returns:
        ``True`` if a resumable session exists, ``False`` otherwise.
    """
    latest = load_latest_session()
    if latest is None:
        return False

    try:
        last_active = datetime.fromisoformat(latest.last_active)
    except (ValueError, TypeError):
        return False

    # Ensure timezone-aware comparison
    now = datetime.now(timezone.utc)
    if last_active.tzinfo is None:
        last_active = last_active.replace(tzinfo=timezone.utc)

    return (now - last_active) < _RESUME_THRESHOLD


def prompt_resume() -> Session | None:
    """If a recent session exists, prompt the user to resume it.

    Returns:
        The loaded :class:`Session` if the user chooses to resume,
        or ``None`` if they decline or no recent session is available.
    """
    latest = load_latest_session()
    if latest is None:
        return None

    try:
        last_active = datetime.fromisoformat(latest.last_active)
    except (ValueError, TypeError):
        return None

    now = datetime.now(timezone.utc)
    if last_active.tzinfo is None:
        last_active = last_active.replace(tzinfo=timezone.utc)

    if (now - last_active) >= _RESUME_THRESHOLD:
        return None

    num_runs = len(latest.runs)
    ref = latest.current_reference or "unknown"
    print(
        f"Resume session {latest.session_id}? "
        f"({num_runs} run{'s' if num_runs != 1 else ''}, "
        f"ref: {Path(ref).name}, "
        f"backend: {latest.current_backend})"
    )
    answer = input("(y/n) ").strip().lower()
    if answer in ("y", "yes"):
        return latest
    return None
