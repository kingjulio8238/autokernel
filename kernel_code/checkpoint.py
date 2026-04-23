"""Round-level checkpointing for MetaOptimizer.

Saves optimization state after each round so that a crashed or
interrupted run can resume from the last checkpoint. Enables
multi-hour/multi-day autonomous kernel optimization.

Checkpoint format::

    {checkpoint_dir}/
        checkpoint_meta.json      # Latest state: round, best_speedup, strategy
        round_001.json            # Per-round snapshot
        round_002.json
        best_kernel.py            # Best kernel code (always up to date)

Usage::

    from kernel_code.checkpoint import CheckpointManager

    mgr = CheckpointManager(checkpoint_dir=".kernel-code/checkpoints/run_001")
    mgr.save_round(round_num=1, state={...})

    # On restart:
    restored = mgr.load_latest()
    if restored:
        optimizer.resume_from(restored)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Serializable snapshot of MetaOptimizer state after a round."""
    round_num: int
    best_speedup: float
    best_kernel: str
    total_cost_usd: float
    total_iterations: int
    current_strategy: str
    round_history: list[dict]
    optimization_log: list[dict]
    exploratory_round_done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_num": self.round_num,
            "best_speedup": self.best_speedup,
            "best_kernel": self.best_kernel,
            "total_cost_usd": self.total_cost_usd,
            "total_iterations": self.total_iterations,
            "current_strategy": self.current_strategy,
            "round_history": self.round_history,
            "optimization_log": self.optimization_log,
            "exploratory_round_done": self.exploratory_round_done,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointState":
        return cls(
            round_num=d.get("round_num", 0),
            best_speedup=d.get("best_speedup", 0.0),
            best_kernel=d.get("best_kernel", ""),
            total_cost_usd=d.get("total_cost_usd", 0.0),
            total_iterations=d.get("total_iterations", 0),
            current_strategy=d.get("current_strategy", "general optimization"),
            round_history=d.get("round_history", []),
            optimization_log=d.get("optimization_log", []),
            exploratory_round_done=d.get("exploratory_round_done", False),
        )


class CheckpointManager:
    """Manages round-level checkpointing for MetaOptimizer."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / "checkpoint_meta.json"
        self._kernel_path = self._dir / "best_kernel.py"

    def save_round(self, state: CheckpointState) -> None:
        """Save a checkpoint after a completed round.

        Uses atomic writes (write to temp, rename) to prevent
        corruption if the process crashes mid-write.

        Writes:
        - round_{N:03d}.json: Full round snapshot
        - checkpoint_meta.json: Latest state summary
        - best_kernel.py: Best kernel code
        """
        # Save per-round snapshot (atomic)
        round_file = self._dir / f"round_{state.round_num:03d}.json"
        self._atomic_write(round_file, json.dumps(state.to_dict(), indent=2))

        # Update meta (atomic — this is the critical file for resume)
        meta = {
            "latest_round": state.round_num,
            "best_speedup": state.best_speedup,
            "total_cost_usd": state.total_cost_usd,
            "total_iterations": state.total_iterations,
            "current_strategy": state.current_strategy,
        }
        self._atomic_write(self._meta_path, json.dumps(meta, indent=2))

        # Save best kernel code (atomic)
        if state.best_kernel:
            self._atomic_write(self._kernel_path, state.best_kernel)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write content to file atomically (write to temp, rename)."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)  # atomic on POSIX

        logger.info(
            "Checkpoint saved: round %d, best %.2fx, cost $%.2f",
            state.round_num, state.best_speedup, state.total_cost_usd,
        )

    def load_latest(self) -> CheckpointState | None:
        """Load the most recent checkpoint.

        Returns None if no checkpoint exists.
        """
        if not self._meta_path.exists():
            return None

        try:
            meta = json.loads(self._meta_path.read_text())
            latest_round = meta.get("latest_round", 0)
            if latest_round <= 0:
                return None

            # Load the full round snapshot
            round_file = self._dir / f"round_{latest_round:03d}.json"
            if not round_file.exists():
                logger.warning("Checkpoint meta exists but round file missing: %s", round_file)
                return None

            data = json.loads(round_file.read_text())
            state = CheckpointState.from_dict(data)

            # Restore best kernel from file if not in snapshot
            if not state.best_kernel and self._kernel_path.exists():
                state.best_kernel = self._kernel_path.read_text()

            logger.info(
                "Checkpoint restored: round %d, best %.2fx, cost $%.2f",
                state.round_num, state.best_speedup, state.total_cost_usd,
            )
            return state

        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load checkpoint: %s", exc)
            return None

    def has_checkpoint(self) -> bool:
        """Check if a valid checkpoint exists."""
        return self._meta_path.exists()

    @property
    def checkpoint_dir(self) -> Path:
        return self._dir
