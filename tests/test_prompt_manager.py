"""Tests for PromptManager's profile plumbing (Phase 3a, task #1).

Asserts that ``render_kernel_refinement_prompt``, ``render_kernel_optimization_prompt``
and ``render_reflexion_prompt`` accept an optional ``profile`` kwarg and thread
it into the Jinja template context. Backcompat: ``profile=None`` and ``profile={}``
must both produce identical output to the pre-change baseline (no PROFILE block).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kernel_agent.prompt_manager import PromptManager


FULL_PROFILE = {
    "compute_utilization": 0.42,
    "bandwidth_utilization": 0.75,
    "cache_efficiency": 0.55,
    "bottleneck_type": "memory_bound",
    "sol_score": 0.63,
    "gpu_type": "L40S",
}


@dataclass
class _Attempt:
    round_num: int = 2
    bottleneck_category: str = "memory"
    root_cause: str = "uncoalesced global loads"
    recommended_fix: str = "load multiple elements per iteration"
    config_changes: dict | None = None
    time_before_ms: float = 1.2345
    time_after_ms: float = 0.9876
    improvement_pct: float = 20.0
    compute_sol_pct: float = 35.0
    memory_sol_pct: float = 68.0
    passed_verification: bool = True
    error_message: str | None = None


@pytest.fixture(scope="module")
def pm() -> PromptManager:
    return PromptManager()


def _refine_kwargs() -> dict:
    return {
        "problem_description": "optimize matmul",
        "test_code": "def test(): pass",
        "kernel_code": "def kernel_function(): ...",
        "error_info": {"stdout": "", "stderr": "RuntimeError"},
    }


def _optimize_kwargs() -> dict:
    return {
        "problem_description": "optimize matmul",
        "kernel_code": "def kernel_function(): ...",
        "gpu_specs": {"name": "L40S"},
        "roofline": {},
        "category": "memory",
        "summary": "bandwidth-starved",
        "reasoning": "DRAM throughput at 40%",
        "root_cause": {"cause": "uncoalesced loads", "evidence": []},
        "recommended_fix": {"fix": "widen block", "rationale": "increase memory-level parallelism"},
        "pytorch_baseline_ms": 1.0,
        "current_best_ms": 0.5,
    }


# ---------------------------------------------------------------------------
# render_kernel_refinement_prompt
# ---------------------------------------------------------------------------


def test_refinement_profile_none_matches_baseline(pm: PromptManager) -> None:
    baseline = pm.render_kernel_refinement_prompt(**_refine_kwargs())
    explicit_none = pm.render_kernel_refinement_prompt(**_refine_kwargs(), profile=None)
    empty = pm.render_kernel_refinement_prompt(**_refine_kwargs(), profile={})
    assert baseline == explicit_none == empty
    assert "PROFILE FROM LAST ATTEMPT" not in baseline


def test_refinement_profile_populated_emits_block(pm: PromptManager) -> None:
    out = pm.render_kernel_refinement_prompt(**_refine_kwargs(), profile=FULL_PROFILE)
    assert "## PROFILE FROM LAST ATTEMPT" in out
    assert "bandwidth_utilization" not in out  # key names not leaked as-is
    assert "Bandwidth: 75% of peak" in out
    assert "Compute:   42% of peak" in out
    assert "L2 cache hit rate: 55%" in out
    assert "Bottleneck: memory_bound" in out
    assert "SOL: 0.63" in out
    assert "(L40S)" in out


# ---------------------------------------------------------------------------
# render_kernel_optimization_prompt
# ---------------------------------------------------------------------------


def test_optimization_profile_none_matches_baseline(pm: PromptManager) -> None:
    baseline = pm.render_kernel_optimization_prompt(**_optimize_kwargs())
    explicit_none = pm.render_kernel_optimization_prompt(**_optimize_kwargs(), profile=None)
    empty = pm.render_kernel_optimization_prompt(**_optimize_kwargs(), profile={})
    assert baseline == explicit_none == empty
    assert "PROFILE FROM LAST ATTEMPT" not in baseline


def test_optimization_profile_populated_emits_block(pm: PromptManager) -> None:
    out = pm.render_kernel_optimization_prompt(**_optimize_kwargs(), profile=FULL_PROFILE)
    assert "## PROFILE FROM LAST ATTEMPT" in out
    assert "Bandwidth: 75% of peak" in out
    assert "Bottleneck: memory_bound" in out


# ---------------------------------------------------------------------------
# render_reflexion_prompt
# ---------------------------------------------------------------------------


def test_reflexion_profile_none_matches_baseline(pm: PromptManager) -> None:
    attempt = _Attempt(config_changes={"BLOCK_M": 128})
    baseline = pm.render_reflexion_prompt(attempt)
    explicit_none = pm.render_reflexion_prompt(attempt, profile=None)
    empty = pm.render_reflexion_prompt(attempt, profile={})
    assert baseline == explicit_none == empty
    assert "PROFILE FROM LAST ATTEMPT" not in baseline


def test_reflexion_profile_populated_emits_block(pm: PromptManager) -> None:
    attempt = _Attempt(config_changes={"BLOCK_M": 128})
    out = pm.render_reflexion_prompt(attempt, profile=FULL_PROFILE)
    assert "## PROFILE FROM LAST ATTEMPT" in out
    assert "Bandwidth: 75% of peak" in out
    assert "Bottleneck: memory_bound" in out
