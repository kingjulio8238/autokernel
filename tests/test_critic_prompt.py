"""Tests for profile-dict plumbing into kernel_code-side prompt templates.

Covers:
- critic_v1.md: all 4 profile placeholders fill from a stub profile.
- Generator templates (Triton + CUDA): all 4 profile placeholders render when
  a profile dict is passed to ``get_generator_prompt``.
- Missing-key resilience: a partial/empty profile dict does not crash and
  defaults to 0.0 / "unknown".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openkernel.backends.base import profile_placeholders, safe_format
from openkernel.backends.cuda_backend import CudaBackend
from openkernel.backends.triton_backend import TritonBackend

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"


def _critic_template() -> str:
    return (_PROMPTS_DIR / "critic_v1.md").read_text()


def _stub_profile() -> dict:
    return {
        "bandwidth_utilization": 0.42,
        "compute_utilization": 0.18,
        "cache_efficiency": 0.73,
        "bottleneck_type": "memory_bound",
    }


def test_critic_template_fills_profile_placeholders() -> None:
    """critic_v1.md's 3 profile INPUT placeholders render from the stub profile.

    Note: ``bottleneck_type`` in critic_v1.md is a JSON OUTPUT field in the
    diagnosis schema (not an input placeholder the harness fills), so the
    critic input carries 3 profile metrics — bandwidth, compute, cache. The
    critic agent itself *produces* the bottleneck_type in its structured
    response; the generator templates then consume it downstream.
    """
    template = _critic_template()
    placeholders = profile_placeholders(_stub_profile())
    rendered = safe_format(template, **placeholders)

    assert "{bandwidth_utilization}" not in rendered
    assert "{compute_utilization}" not in rendered
    assert "{cache_efficiency}" not in rendered
    assert "42.0" in rendered
    assert "18.0" in rendered
    assert "73.0" in rendered
    # The JSON enum referencing "memory_bound" as a valid value still survives.
    assert '"memory_bound"' in rendered


def test_profile_placeholders_missing_key_defaults_do_not_crash() -> None:
    """Empty / partial profile dicts default to 0.0 and 'unknown' without raising."""
    empty = profile_placeholders({})
    assert empty["bandwidth_utilization"] == "0.0"
    assert empty["compute_utilization"] == "0.0"
    assert empty["cache_efficiency"] == "0.0"
    assert empty["bottleneck_type"] == "unknown"

    none_profile = profile_placeholders(None)
    assert none_profile == empty

    partial = profile_placeholders({"bottleneck_type": "compute_bound"})
    assert partial["bottleneck_type"] == "compute_bound"
    assert partial["bandwidth_utilization"] == "0.0"

    # Confirm the critic template renders fine with empty values too —
    # all 3 profile input rows show 0.0%.
    rendered = safe_format(_critic_template(), **empty)
    assert "Bandwidth utilization: 0.0% of peak" in rendered
    assert "Compute utilization: 0.0% of peak" in rendered
    assert "L2 cache hit rate: 0.0%" in rendered


@pytest.mark.parametrize("backend_cls", [TritonBackend, CudaBackend])
def test_generator_template_renders_all_four_metrics(backend_cls) -> None:
    """Both generator templates surface all 4 metric values when profile is passed.

    Format differs per backend — Triton uses a compact single-line format
    (token-budget-trimmed), CUDA uses the verbose bullet list — but both
    must substitute the 4 profile values and leave no literal placeholders.
    """
    backend = backend_cls()
    prompt = backend.get_generator_prompt(
        reference="import torch\nclass Model(torch.nn.Module):\n    pass\n",
        hardware="H100",
        intent="basic tiled kernel",
        profile=_stub_profile(),
    )

    assert "42.0" in prompt
    assert "18.0" in prompt
    assert "73.0" in prompt
    assert "memory_bound" in prompt
    # Template should not leak unfilled placeholders for the profile keys.
    assert "{bandwidth_utilization}" not in prompt
    assert "{compute_utilization}" not in prompt
    assert "{cache_efficiency}" not in prompt
    assert "{bottleneck_type}" not in prompt


@pytest.mark.parametrize("backend_cls", [TritonBackend, CudaBackend])
def test_generator_template_empty_profile_defaults(backend_cls) -> None:
    """Calling a generator backend without a profile does not crash and defaults sanely."""
    backend = backend_cls()
    prompt = backend.get_generator_prompt(
        reference="import torch\n",
        hardware="H100",
        intent="basic",
    )

    # Both formats render "0.0" for the 3 numeric metrics and "unknown" for bottleneck.
    assert prompt.count("0.0") >= 3
    assert "unknown" in prompt
    assert "{bandwidth_utilization}" not in prompt
    assert "{bottleneck_type}" not in prompt


def test_profile_placeholders_accepts_percentage_values() -> None:
    """Values already in percent (>1.0) are passed through without double-scaling."""
    out = profile_placeholders(
        {
            "bandwidth_utilization": 55.0,
            "compute_utilization": 12.5,
            "cache_efficiency": 80.0,
            "bottleneck_type": "compute_bound",
        }
    )
    assert out["bandwidth_utilization"] == "55.0"
    assert out["compute_utilization"] == "12.5"
    assert out["cache_efficiency"] == "80.0"
