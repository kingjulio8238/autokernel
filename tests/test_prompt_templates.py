"""Tests for the PROFILE FROM LAST ATTEMPT block in Jinja prompt templates.

Covers kernel_refinement.j2, kernel_optimization.j2, and reflexion_prompt.j2.
Renders templates via a minimal Jinja environment that matches production
(trim_blocks / lstrip_blocks) so the assertions reflect real prompt output.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, FileSystemLoader


TEMPLATES_DIR = (
    Path(__file__).resolve().parents[1] / "kernel_agent" / "templates"
)


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


FULL_PROFILE = {
    "compute_utilization": 0.42,
    "bandwidth_utilization": 0.75,
    "cache_efficiency": 0.55,
    "bottleneck_type": "memory_bound",
    "sol_score": 0.63,
    "gpu_type": "L40S",
}

PARTIAL_PROFILE = {"bottleneck_type": "memory_bound"}


def _reflexion_attempt() -> SimpleNamespace:
    return SimpleNamespace(
        round_num=2,
        bottleneck_category="memory",
        root_cause="uncoalesced global loads",
        recommended_fix="load multiple elements per iteration",
        config_changes={"BLOCK_M": 128},
        time_before_ms=1.2345,
        time_after_ms=0.9876,
        improvement_pct=20.0,
        compute_sol_pct=35.0,
        memory_sol_pct=68.0,
        passed_verification=True,
        error_message=None,
    )


def _refinement_ctx(profile=None) -> dict:
    return {
        "triton_guidelines": "-- guidelines --",
        "test_code": "def test(): pass",
        "kernel_code": "def kernel_function(): ...",
        "error_info": {"stderr": "RuntimeError", "stdout": ""},
        "history_context": None,
        "no_cusolver": False,
        "profile": profile,
    }


def _optimization_ctx(profile=None) -> dict:
    return {
        "gpu_specs": None,
        "problem_description": "matmul",
        "pytorch_baseline_ms": 1.0,
        "kernel_code": "def kernel_function(): ...",
        "roofline": None,
        "bottleneck": {
            "category": "memory",
            "summary": "bandwidth-starved",
            "reasoning": "DRAM throughput at 40%",
            "root_cause": {"cause": "uncoalesced loads", "evidence": []},
            "recommended_fix": {"fix": "vectorize", "rationale": "increase memory-level parallelism"},
        },
        "rag_context": None,
        "recent_attempts": None,
        "reflexions": None,
        "error_feedback": None,
        "current_best_ms": None,
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# Full profile renders every metric line + guidance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template_name,ctx_fn",
    [
        ("reflexion_prompt.j2", lambda p: {"attempt": _reflexion_attempt(), "profile": p}),
        ("kernel_refinement.j2", _refinement_ctx),
        ("kernel_optimization.j2", _optimization_ctx),
    ],
)
def test_full_profile_renders_all_lines(env, template_name, ctx_fn):
    out = env.get_template(template_name).render(**ctx_fn(FULL_PROFILE))
    assert "## PROFILE FROM LAST ATTEMPT" in out
    assert "Bandwidth: 75% of peak" in out
    assert "(L40S)" in out
    assert "Compute:   42% of peak" in out
    assert "L2 cache hit rate: 55%" in out
    assert "Bottleneck: memory_bound" in out
    assert "SOL: 0.63" in out
    # Guidance block present
    assert "Guidance:" in out
    assert "memory_bound" in out
    assert "compute_bound" in out
    assert "latency_bound" in out


# ---------------------------------------------------------------------------
# profile=None → no PROFILE section, baseline untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template_name,ctx_fn",
    [
        ("reflexion_prompt.j2", lambda p: {"attempt": _reflexion_attempt(), "profile": p}),
        ("kernel_refinement.j2", _refinement_ctx),
        ("kernel_optimization.j2", _optimization_ctx),
    ],
)
def test_no_profile_omits_block(env, template_name, ctx_fn):
    out_none = env.get_template(template_name).render(**ctx_fn(None))
    out_empty = env.get_template(template_name).render(**ctx_fn({}))
    out_missing = env.get_template(template_name).render(
        **{k: v for k, v in ctx_fn(None).items() if k != "profile"}
    )

    for out in (out_none, out_empty, out_missing):
        assert "PROFILE FROM LAST ATTEMPT" not in out
        assert "Guidance:" not in out or "guidance" in out.lower() and "PROFILE" not in out
        assert "of peak" not in out

    # None / empty / missing all render identically (true backcompat)
    assert out_none == out_empty == out_missing


# ---------------------------------------------------------------------------
# Partial profile renders gracefully with defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template_name,ctx_fn",
    [
        ("reflexion_prompt.j2", lambda p: {"attempt": _reflexion_attempt(), "profile": p}),
        ("kernel_refinement.j2", _refinement_ctx),
        ("kernel_optimization.j2", _optimization_ctx),
    ],
)
def test_partial_profile_defaults_gracefully(env, template_name, ctx_fn):
    out = env.get_template(template_name).render(**ctx_fn(PARTIAL_PROFILE))
    assert "## PROFILE FROM LAST ATTEMPT" in out
    # Missing metrics default to 0%
    assert "Bandwidth: 0% of peak" in out
    assert "Compute:   0% of peak" in out
    assert "L2 cache hit rate: 0%" in out
    # bottleneck_type passes through
    assert "Bottleneck: memory_bound" in out
    # gpu_type missing → no "(GPU)" suffix
    assert "of peak (" not in out
    # sol_score missing → no "- SOL: 0.xx" line from the PROFILE block
    assert "- SOL: 0." not in out


# ---------------------------------------------------------------------------
# Placement: reflexion PROFILE block sits right before the Task section
# ---------------------------------------------------------------------------


def test_reflexion_profile_placement(env):
    out = env.get_template("reflexion_prompt.j2").render(
        attempt=_reflexion_attempt(), profile=FULL_PROFILE
    )
    profile_idx = out.index("## PROFILE FROM LAST ATTEMPT")
    task_idx = out.index("## Task")
    analyze_idx = out.index("Analyze whether the bottleneck diagnosis")
    assert profile_idx < task_idx < analyze_idx


def test_refinement_profile_placement(env):
    out = env.get_template("kernel_refinement.j2").render(**_refinement_ctx(FULL_PROFILE))
    profile_idx = out.index("## PROFILE FROM LAST ATTEMPT")
    fusion_idx = out.index("FUSION PRIORITY:")
    error_idx = out.index("TEST RESULTS:")
    assert error_idx < profile_idx < fusion_idx


def test_optimization_profile_placement(env):
    out = env.get_template("kernel_optimization.j2").render(**_optimization_ctx(FULL_PROFILE))
    profile_idx = out.index("## PROFILE FROM LAST ATTEMPT")
    target_idx = out.index("## PERFORMANCE TARGET")
    recommended_idx = out.index("**Recommended Fix:**")
    assert recommended_idx < profile_idx < target_idx


@pytest.mark.parametrize(
    "template_name,ctx_fn",
    [
        ("reflexion_prompt.j2", lambda p: {"attempt": _reflexion_attempt(), "profile": p}),
        ("kernel_refinement.j2", _refinement_ctx),
        ("kernel_optimization.j2", _optimization_ctx),
    ],
)
def test_gpu_type_does_not_collapse_next_line(env, template_name, ctx_fn):
    """Regression: trim_blocks was merging the gpu_type line with the next bullet."""
    out = env.get_template(template_name).render(**ctx_fn(FULL_PROFILE))
    # The '(L40S)' label must be followed by a newline then the Compute bullet, not jammed together.
    assert "(L40S)\n- Compute:" in out
    assert "(L40S)- Compute:" not in out


@pytest.mark.parametrize(
    "template_name,ctx_fn",
    [
        ("reflexion_prompt.j2", lambda p: {"attempt": _reflexion_attempt(), "profile": p}),
        ("kernel_refinement.j2", _refinement_ctx),
        ("kernel_optimization.j2", _optimization_ctx),
    ],
)
def test_guidance_is_triton_aware(env, template_name, ctx_fn):
    """Regression guard from phase3a prompt-quality review.

    float4/uint4 are CUDA vector types and will cause Haiku to hallucinate CUDA code
    in a Triton-only codebase. "increase grid size" is backwards advice for latency_bound
    (grid affects throughput, not per-kernel latency). "shared-memory staging" reads as
    manual SMEM management (CUDA style) — Triton uses implicit caching.
    """
    out = env.get_template(template_name).render(**ctx_fn(FULL_PROFILE))
    assert "float4" not in out
    assert "uint4" not in out
    assert "shared-memory staging" not in out
    assert "increase grid size" not in out
    # Affirmatively verify the Triton-aware replacements landed.
    assert "`tl.load`" in out
    assert "arithmetic intensity" in out
