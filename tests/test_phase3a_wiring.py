"""Phase 3a end-to-end wiring test (synthetic A/B, task #7).

Chains the pipeline-plumbing layer (PromptManager ``profile`` kwarg), the
Jinja PROFILE block (task #2), the Triton-aware guidance fix (task #9), and
the kernel_code-side generator fill helper (task #3 / base.py::profile_placeholders)
together in-process to prove code-level correctness without Modal.

A separate test module (``test_prompt_manager.py``) already covers backcompat
(``profile=None`` / ``{}`` matches baseline) with six focused cases. This module
intentionally stays narrower: end-to-end wiring + the Triton-aware regression
guard against CUDA jargon (``float4``).

Run: ``uv run pytest tests/test_phase3a_wiring.py -v``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from kernel_agent.prompt_manager import PromptManager
from openkernel.backends.base import profile_placeholders, safe_format


FULL_PROFILE: dict = {
    "compute_utilization": 0.42,
    "bandwidth_utilization": 0.75,
    "cache_efficiency": 0.55,
    "bottleneck_type": "memory_bound",
    "sol_score": 0.63,
    "gpu_type": "L40S",
    "hardware_peak_tflops": 183,
    "hardware_peak_gbps": 864,
}


@dataclass
class _Attempt:
    round_num: int = 2
    bottleneck_category: str = "memory"
    root_cause: str = "uncoalesced global loads"
    recommended_fix: str = "vectorize loads and widen block"
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


# ---------------------------------------------------------------------------
# 1. refinement prompt wires the profile end-to-end (heading + metrics +
#    Triton-aware guidance + float4 regression guard)
# ---------------------------------------------------------------------------


def test_refinement_prompt_includes_profile_when_available(pm: PromptManager) -> None:
    out = pm.render_kernel_refinement_prompt(**_refine_kwargs(), profile=FULL_PROFILE)

    assert "## PROFILE FROM LAST ATTEMPT" in out

    # All 4 metric bullets with the correct percentages / bottleneck.
    assert "Bandwidth: 75% of peak" in out
    assert "Compute:   42% of peak" in out
    assert "L2 cache hit rate: 55%" in out
    assert "Bottleneck: memory_bound" in out

    # GUIDANCE block — post-task-#9 Triton-aware suggestions. We anchor on
    # two independent substrings so a single wording tweak won't churn the
    # test, but a full regression to pre-#9 CUDA jargon would.
    assert "coalesced" in out, "expected Triton-aware 'coalesced' guidance"
    assert "arithmetic intensity" in out, (
        "expected Triton-aware 'arithmetic intensity' guidance"
    )

    # Regression guard: the CUDA-specific `float4` vectorization jargon
    # was removed in task #9. If it reappears, the Triton-aware fix has
    # regressed.
    assert "float4" not in out, (
        "'float4' reappeared in the refinement prompt — task #9 regression"
    )


# ---------------------------------------------------------------------------
# 2. refinement prompt omits PROFILE block on None / empty dict, and both
#    produce byte-identical output (backcompat invariant)
# ---------------------------------------------------------------------------


def test_refinement_prompt_omits_profile_block_when_none(pm: PromptManager) -> None:
    none_out = pm.render_kernel_refinement_prompt(**_refine_kwargs(), profile=None)
    empty_out = pm.render_kernel_refinement_prompt(**_refine_kwargs(), profile={})

    for out in (none_out, empty_out):
        assert "PROFILE FROM LAST ATTEMPT" not in out
        assert "Bandwidth:" not in out
        assert "Bottleneck:" not in out

    # None and {} must be byte-identical — callers without eval-time
    # profile data should not accidentally branch on which falsy they pass.
    assert none_out == empty_out


# ---------------------------------------------------------------------------
# 3. reflexion prompt (highest-leverage site per Phase 3a rationale) — the
#    PROFILE block must appear BEFORE the "Analyze whether the bottleneck
#    diagnosis was correct" pivot, so the model sees the profile while it's
#    reasoning about diagnosis accuracy.
# ---------------------------------------------------------------------------


def test_reflexion_prompt_includes_profile(pm: PromptManager) -> None:
    attempt = _Attempt(config_changes={"BLOCK_M": 128})
    out = pm.render_reflexion_prompt(attempt, profile=FULL_PROFILE)

    assert "## PROFILE FROM LAST ATTEMPT" in out
    assert "Bandwidth: 75% of peak" in out
    assert "Compute:   42% of peak" in out
    assert "L2 cache hit rate: 55%" in out
    assert "Bottleneck: memory_bound" in out

    profile_idx = out.find("## PROFILE FROM LAST ATTEMPT")
    pivot_idx = out.find("Analyze whether the bottleneck diagnosis was correct")
    assert profile_idx != -1 and pivot_idx != -1, "expected both markers present"
    assert profile_idx < pivot_idx, (
        "PROFILE block must render before the diagnosis-analysis pivot so the "
        "model has the profile in-context when it evaluates diagnosis correctness"
    )

    # Same regression guard as test #1 — the reflexion template embeds the
    # same Guidance block and must stay Triton-aware.
    assert "float4" not in out


# ---------------------------------------------------------------------------
# 4. kernel_code-side: the Triton generator template is filled via
#    base.py::profile_placeholders. Assert the 4 profile-derived fields
#    reach the rendered prompt with the canonical "of peak" framing used
#    by the critic surface (critic_v1.md).
# ---------------------------------------------------------------------------


def test_generator_prompt_includes_profile_fields() -> None:
    tpl_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "prompts"
        / "triton_generator_v1.md"
    )
    template = tpl_path.read_text()

    placeholders = profile_placeholders(FULL_PROFILE)
    # Sanity: the helper normalizes 0-1 fractions into 1-dp percent strings.
    assert placeholders == {
        "bandwidth_utilization": "75.0",
        "compute_utilization": "42.0",
        "cache_efficiency": "55.0",
        "bottleneck_type": "memory_bound",
    }

    rendered = safe_format(template, **placeholders)

    # The 4 metric values from the profile dict must all appear in the
    # rendered prompt. We check the *values* (not the surrounding prose)
    # so the assertion is robust across the task-#10 trim of the refinement
    # section while still failing loudly if the placeholder wiring breaks.
    assert "75.0" in rendered
    assert "42.0" in rendered
    assert "55.0" in rendered
    assert "memory_bound" in rendered

    # The critic surface (critic_v1.md line 24–26) pins the canonical
    # "% of peak" phrasing. Assert it independently on that template so a
    # future refactor can't silently strip the peak-normalized framing
    # from the critic's diagnosis input.
    critic_path = tpl_path.with_name("critic_v1.md")
    critic_rendered = safe_format(critic_path.read_text(), **placeholders)
    assert "Bandwidth utilization: 75.0% of peak" in critic_rendered
    assert "Compute utilization: 42.0% of peak" in critic_rendered
    assert "L2 cache hit rate: 55.0%" in critic_rendered

    # Placeholders must not leak unfilled.
    assert "{bandwidth_utilization}" not in rendered
    assert "{compute_utilization}" not in rendered
    assert "{cache_efficiency}" not in rendered
    assert "{bottleneck_type}" not in rendered
