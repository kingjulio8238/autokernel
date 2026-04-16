"""Persistence helpers for the memory layer.

Provides simple JSON-based save/load for skills and trajectory data.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from openkernel.memory.skill_library import OptimizationSkill
from openkernel.memory.trajectory import TrajectoryMemory


# ------------------------------------------------------------------
# Skills
# ------------------------------------------------------------------

def save_skills(skills: list[OptimizationSkill], path: Path) -> None:
    """Serialize a list of skills to a single JSON file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(s) for s in skills], f, indent=2)
        f.write("\n")


def load_skills(path: Path) -> list[OptimizationSkill]:
    """Deserialize skills from a JSON file produced by :func:`save_skills`."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [OptimizationSkill(**entry) for entry in data]


# ------------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------------

def save_trajectory(trajectory: TrajectoryMemory, path: Path) -> None:
    """Serialize a :class:`TrajectoryMemory` to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(trajectory.to_list(), f, indent=2)
        f.write("\n")


def load_trajectory(path: Path) -> TrajectoryMemory:
    """Deserialize a :class:`TrajectoryMemory` from JSON."""
    path = Path(path)
    if not path.exists():
        return TrajectoryMemory()
    with open(path) as f:
        data = json.load(f)
    return TrajectoryMemory.from_list(data)
