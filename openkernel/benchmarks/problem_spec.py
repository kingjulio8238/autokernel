"""Unified problem descriptor across KernelBench + GPU MODE.

``ProblemSpec`` is the shared input contract for all Phase 1 loaders
(``kb_l1``, ``kb_l2``, ``gpumode``) and for the batch-runner that consumes
them. Loaders produce instances; the batch-runner reads them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["L1", "L2", "GPU_MODE"]
Source = Literal["kernelbench", "gpumode"]

_VALID_TIERS = {"L1", "L2", "GPU_MODE"}
_VALID_SOURCES = {"kernelbench", "gpumode"}


@dataclass(frozen=True)
class ProblemSpec:
    """Unified problem descriptor across KernelBench + GPU MODE.

    Consumed by batch-runner; produced by per-source loaders.

    Attributes:
        id: Stable unique identifier, used as primary key in the
            leaderboard. Convention: ``"kb_l1_{problem_id:04d}"``,
            ``"kb_l2_{problem_id:04d}"``, or ``"gpumode_{slug}"``.
        name: Human-readable label, e.g. ``"Softmax"``.
        tier: One of ``"L1"``, ``"L2"``, or ``"GPU_MODE"``.
        source: Originating benchmark — ``"kernelbench"`` or ``"gpumode"``.
        reference_source: Full Python source of the reference kernel.
            For gpumode this defines ``ref_kernel(data)``; for kernelbench
            it defines ``Model(nn.Module)``. The batch-runner writes this
            to a temp file and passes it to Modal.
        workload_spec: Optional workload parameters mirroring the
            ``WORKLOAD_SPEC`` dict in ``reference.py`` (e.g. ``size``,
            ``seed``, ``contention``, ``m``/``n``/``k``). The Modal
            harness already reads it. Loaders populate it when the right
            workload shape is known; otherwise it stays empty.
        expected_dtype: Dominant dtype of the workload. Used as a hint
            for the classifier and for later auto-routing (e.g.
            ``"float8"`` would select a quant strategy). Defaults to
            ``"float32"``, which is reasonable for KernelBench.

    Note:
        Frozen (``frozen=True``) so instances are hashable value objects
        and cannot be mutated accidentally by loaders or the runner.
    """

    id: str
    name: str
    tier: Tier
    source: Source
    reference_source: str
    workload_spec: dict = field(default_factory=dict)
    expected_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.tier not in _VALID_TIERS:
            raise ValueError(
                f"invalid tier {self.tier!r}, must be one of {sorted(_VALID_TIERS)}"
            )
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"invalid source {self.source!r}, must be one of {sorted(_VALID_SOURCES)}"
            )
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.reference_source:
            raise ValueError(
                "reference_source cannot be empty — a loader must capture the source code"
            )
