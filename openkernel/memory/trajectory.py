"""Within-problem trajectory memory.

Tracks every iteration attempted during a single optimization run so the
engine can avoid retrying failed approaches and summarise progress for the
LLM context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _TrajectoryEntry:
    iteration: int
    intent: str
    speedup: float
    status: str
    diagnosis: str | None = None


class TrajectoryMemory:
    """Records the history of optimization attempts for one problem.

    Usage::

        mem = TrajectoryMemory()
        mem.record(1, "tiled GEMM with autotune", 1.2, "correct", None)
        mem.record(2, "online softmax reduction", 0.0, "compile_error", "missing tl import")
        assert mem.was_tried("tiled gemm")
        print(mem.get_summary())
    """

    def __init__(self) -> None:
        self._entries: list[_TrajectoryEntry] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        iteration: int,
        intent: str,
        speedup: float,
        status: str,
        diagnosis: str | None = None,
    ) -> None:
        """Append an iteration record."""
        self._entries.append(
            _TrajectoryEntry(
                iteration=iteration,
                intent=intent,
                speedup=speedup,
                status=status,
                diagnosis=diagnosis,
            )
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 10) -> list[dict]:
        """Return the *n* most recent entries as plain dicts."""
        return [
            {
                "iteration": e.iteration,
                "intent": e.intent,
                "speedup": e.speedup,
                "status": e.status,
                "diagnosis": e.diagnosis,
            }
            for e in self._entries[-n:]
        ]

    def was_tried(self, approach: str) -> bool:
        """Fuzzy check: were most keywords of *approach* seen in a past intent?

        Returns ``True`` if any recorded intent shares at least half of the
        query keywords (case-insensitive).
        """
        query_tokens = set(_tokenize(approach))
        if not query_tokens:
            return False
        threshold = max(1, len(query_tokens) // 2)
        for entry in self._entries:
            entry_tokens = set(_tokenize(entry.intent))
            if len(query_tokens & entry_tokens) >= threshold:
                return True
        return False

    def get_summary(self) -> str:
        """Human-/LLM-readable summary of the trajectory so far."""
        if not self._entries:
            return "No iterations recorded yet."

        lines: list[str] = [
            f"Trajectory: {len(self._entries)} iterations, "
            f"best speedup {self.best_speedup:.2f}x"
        ]
        for e in self._entries:
            marker = "OK" if e.status == "correct" else e.status.upper()
            line = f"  [{marker}] iter {e.iteration}: {e.intent} -> {e.speedup:.2f}x"
            if e.diagnosis:
                line += f"  ({e.diagnosis})"
            lines.append(line)
        return "\n".join(lines)

    @property
    def best_speedup(self) -> float:
        """Highest speedup achieved so far (0.0 if nothing recorded)."""
        if not self._entries:
            return 0.0
        return max(e.speedup for e in self._entries)

    def clear(self) -> None:
        """Reset all recorded entries."""
        self._entries.clear()

    # ------------------------------------------------------------------
    # Serialization helpers (used by store.py)
    # ------------------------------------------------------------------

    def to_list(self) -> list[dict]:
        """Serialize all entries to a list of dicts."""
        return [
            {
                "iteration": e.iteration,
                "intent": e.intent,
                "speedup": e.speedup,
                "status": e.status,
                "diagnosis": e.diagnosis,
            }
            for e in self._entries
        ]

    @classmethod
    def from_list(cls, entries: list[dict]) -> TrajectoryMemory:
        """Deserialize from a list of dicts."""
        mem = cls()
        for e in entries:
            mem.record(
                iteration=e["iteration"],
                intent=e["intent"],
                speedup=e["speedup"],
                status=e["status"],
                diagnosis=e.get("diagnosis"),
            )
        return mem


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase split, dropping very short noise words."""
    return [w for w in text.lower().split() if len(w) > 2]
