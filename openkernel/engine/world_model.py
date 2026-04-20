"""World model search types (K-Search-style intent tree).

The world model operates in strategy space, not code space:
- IntentNode: an optimization intent (what to try, not how to implement it)
- Strategy: a high-level optimization approach from the Pareto frontier
- IntentTree: the full search tree with node management, priority selection,
  stagnation detection, serialization, and non-monotonic path support.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class IntentStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class IntentNode:
    """A node in the world model search tree.

    Each node represents an optimization intent — what to try, decoupled from
    how to implement it. If code is buggy but strategy is sound, the intent
    survives for retry.
    """

    id: str
    description: str  # "Vectorize loads with float4"
    parent_id: str | None = None
    priority: float = 0.5  # LLM-estimated value (0-1)
    status: IntentStatus = IntentStatus.PENDING
    attempts: int = 0
    max_attempts: int = 5
    best_speedup: float = 0.0
    profiler_summary: str = ""  # last CriticDiagnosis summary
    children: list[str] = field(default_factory=list)  # child intent IDs
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize this node to a plain dict."""
        return {
            "id": self.id,
            "description": self.description,
            "parent_id": self.parent_id,
            "priority": self.priority,
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "best_speedup": self.best_speedup,
            "profiler_summary": self.profiler_summary,
            "children": list(self.children),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> IntentNode:
        """Deserialize a node from a plain dict."""
        return cls(
            id=data["id"],
            description=data["description"],
            parent_id=data.get("parent_id"),
            priority=data.get("priority", 0.5),
            status=IntentStatus(data.get("status", "pending")),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 5),
            best_speedup=data.get("best_speedup", 0.0),
            profiler_summary=data.get("profiler_summary", ""),
            children=list(data.get("children", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Strategy:
    """A high-level optimization strategy from the Pareto frontier.

    Strategies evolve via GEPA-style reflection and are persisted across problems.
    """

    id: str
    description: str  # "For memory-bound elementwise: vectorize, minimize shared mem, fuse"
    problem_types: list[str] = field(default_factory=list)  # ["elementwise", "reduction"]
    backend: str = "any"  # "triton" | "cuda" | "any"
    hardware_targets: list[str] = field(default_factory=list)  # ["nvidia_ampere", "nvidia_hopper"]
    success_history: list[dict] = field(default_factory=list)
    failure_history: list[dict] = field(default_factory=list)
    pareto_scores: dict = field(default_factory=dict)  # {speedup, correctness_rate, iterations}


class IntentTree:
    """Full search tree for the K-Search-style world model.

    Manages a tree of IntentNodes representing optimization intents. Supports:
    - Adding/removing nodes with parent-child relationships
    - Priority-based selection of the next node to explore
    - Stagnation detection (K consecutive non-improvements)
    - Non-monotonic paths (children can improve even if parent regressed)
    - Full serialization/deserialization for persistence and LLM context
    """

    def __init__(self, root_description: str) -> None:
        root_id = str(uuid4())
        self._root_id: str = root_id
        self._nodes: dict[str, IntentNode] = {}
        # Track the global best speedup and a history of speedup results
        # for stagnation detection. Each entry is the best_speedup achieved
        # by the most recently completed node.
        self._result_history: list[float] = []
        self._global_best_speedup: float = 0.0

        root = IntentNode(
            id=root_id,
            description=root_description,
            parent_id=None,
            priority=1.0,
            status=IntentStatus.PENDING,
        )
        self._nodes[root_id] = root

    # -- Properties ----------------------------------------------------------

    @property
    def root(self) -> IntentNode:
        """The root node of the tree."""
        return self._nodes[self._root_id]

    @property
    def all_nodes(self) -> list[IntentNode]:
        """All nodes in the tree (insertion order)."""
        return list(self._nodes.values())

    @property
    def pending_nodes(self) -> list[IntentNode]:
        """All nodes with PENDING status, sorted by priority descending."""
        return sorted(
            [n for n in self._nodes.values() if n.status == IntentStatus.PENDING],
            key=lambda n: n.priority,
            reverse=True,
        )

    @property
    def succeeded_nodes(self) -> list[IntentNode]:
        """All nodes with SUCCEEDED status."""
        return [n for n in self._nodes.values() if n.status == IntentStatus.SUCCEEDED]

    @property
    def failed_count_streak(self) -> int:
        """Number of consecutive non-improving results at the tail of history.

        A result is "non-improving" if it did not set a new global best at the
        time it was recorded. We walk backwards from the most recent result.
        """
        if not self._result_history:
            return 0

        streak = 0
        running_best = 0.0
        # Rebuild "was this a new best?" from scratch so the metric is
        # well-defined even after deserialization.
        improvements: list[bool] = []
        for speedup in self._result_history:
            if speedup > running_best:
                running_best = speedup
                improvements.append(True)
            else:
                improvements.append(False)

        # Count from the tail
        for was_improvement in reversed(improvements):
            if was_improvement:
                break
            streak += 1
        return streak

    @property
    def global_best_speedup(self) -> float:
        return self._global_best_speedup

    # -- Node operations -----------------------------------------------------

    def add_node(
        self,
        parent_id: str,
        description: str,
        priority: float = 0.5,
    ) -> IntentNode:
        """Create a new child intent under the given parent.

        Raises ValueError if parent_id does not exist in the tree.
        """
        if parent_id not in self._nodes:
            raise ValueError(f"Parent node {parent_id!r} not found in tree")
        node_id = str(uuid4())
        node = IntentNode(
            id=node_id,
            description=description,
            parent_id=parent_id,
            priority=priority,
            status=IntentStatus.PENDING,
        )
        self._nodes[node_id] = node
        self._nodes[parent_id].children.append(node_id)
        return node

    def get_node(self, node_id: str) -> IntentNode | None:
        """Return the node with the given ID, or None."""
        return self._nodes.get(node_id)

    def get_highest_priority_pending(self) -> IntentNode | None:
        """Return the pending node with the highest priority, or None."""
        pending = self.pending_nodes
        return pending[0] if pending else None

    def update_node(
        self,
        node_id: str,
        status: IntentStatus | None = None,
        best_speedup: float | None = None,
        profiler_summary: str | None = None,
    ) -> None:
        """Update a node's status and/or results.

        When best_speedup is provided the result is appended to the history
        for stagnation tracking, and the global best is updated if beaten.
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in tree")
        if status is not None:
            node.status = status
        if best_speedup is not None:
            node.best_speedup = best_speedup
            self._result_history.append(best_speedup)
            if best_speedup > self._global_best_speedup:
                self._global_best_speedup = best_speedup
        if profiler_summary is not None:
            node.profiler_summary = profiler_summary

    def update_priorities(self, updates: dict[str, float]) -> None:
        """Batch-update priorities for multiple nodes.

        ``updates`` maps node_id -> new priority (clamped to [0, 1]).
        Unknown node IDs are silently ignored.
        """
        for node_id, priority in updates.items():
            node = self._nodes.get(node_id)
            if node is not None:
                node.priority = max(0.0, min(1.0, priority))

    def prune_subtree(self, node_id: str) -> None:
        """Recursively mark a node and all its descendants as PRUNED.

        Also removes the node from its parent's children list.
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id!r} not found in tree")

        # Remove from parent's children list
        if node.parent_id and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if node_id in parent.children:
                parent.children.remove(node_id)

        # Recursively prune
        self._prune_recursive(node_id)

    def _prune_recursive(self, node_id: str) -> None:
        """Mark node and descendants as PRUNED (internal helper)."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.status = IntentStatus.PRUNED
        for child_id in list(node.children):
            self._prune_recursive(child_id)

    def get_path_to_root(self, node_id: str) -> list[IntentNode]:
        """Return the path from the given node up to the root (inclusive).

        Returns [node, parent, ..., root]. Raises ValueError if node not found.
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id!r} not found in tree")
        path: list[IntentNode] = []
        current_id: str | None = node_id
        visited: set[str] = set()
        while current_id is not None:
            if current_id in visited:
                break  # safety: prevent infinite loop on corrupted data
            visited.add(current_id)
            node = self._nodes.get(current_id)
            if node is None:
                break
            path.append(node)
            current_id = node.parent_id
        return path

    # -- Stagnation detection ------------------------------------------------

    def stagnation_detected(self, threshold: int = 7) -> bool:
        """Return True if the last ``threshold`` results were all non-improving.

        Non-improving means the result did not set a new global best speedup
        at the time it was recorded.
        """
        return self.failed_count_streak >= threshold

    # -- Serialization -------------------------------------------------------

    def serialize(self) -> dict:
        """Serialize the entire tree to a JSON-compatible dict."""
        return {
            "root_id": self._root_id,
            "global_best_speedup": self._global_best_speedup,
            "result_history": list(self._result_history),
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
        }

    @classmethod
    def deserialize(cls, data: dict) -> IntentTree:
        """Reconstruct an IntentTree from a serialized dict."""
        # We bypass __init__ to avoid creating a new root.
        tree = object.__new__(cls)
        tree._root_id = data["root_id"]
        tree._global_best_speedup = data.get("global_best_speedup", 0.0)
        tree._result_history = list(data.get("result_history", []))
        tree._nodes = {}
        for nid, node_data in data["nodes"].items():
            tree._nodes[nid] = IntentNode.from_dict(node_data)
        return tree

    # -- Display / debugging -------------------------------------------------

    def to_summary_json(self) -> dict:
        """Compact summary suitable for LLM context windows.

        Includes all nodes with their status, priority, speedup, and
        parent-child relationships, but omits bulky metadata.
        """
        nodes_summary = []
        for node in self._nodes.values():
            nodes_summary.append({
                "id": node.id,
                "description": node.description,
                "parent_id": node.parent_id,
                "priority": round(node.priority, 3),
                "status": node.status.value,
                "attempts": node.attempts,
                "best_speedup": round(node.best_speedup, 3),
                "children": node.children,
            })
        return {
            "root_id": self._root_id,
            "global_best_speedup": round(self._global_best_speedup, 3),
            "total_nodes": len(self._nodes),
            "pending_count": len(self.pending_nodes),
            "succeeded_count": len(self.succeeded_nodes),
            "stagnation_streak": self.failed_count_streak,
            "nodes": nodes_summary,
        }

    def __repr__(self) -> str:
        return (
            f"IntentTree(nodes={len(self._nodes)}, "
            f"pending={len(self.pending_nodes)}, "
            f"best_speedup={self._global_best_speedup:.3f}, "
            f"stagnation_streak={self.failed_count_streak})"
        )
