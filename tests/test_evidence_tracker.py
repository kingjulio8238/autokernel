"""Tests for kernel_code.evidence_tracker SOL-first promotion gate.

Runnable via:
    uv run pytest tests/test_evidence_tracker.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from kernel_code.evidence_tracker import (  # noqa: E402
    compute_skill_priority,
    extract_and_update_evidence,
)
from openkernel.benchmarks.leaderboard_reader import prior_best_sol  # noqa: E402
from openkernel.memory.skill_library import (  # noqa: E402
    OptimizationSkill,
    SkillLibrary,
)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

_REFERENCE_CODE = dedent(
    """
    import torch

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.matmul(x, x)
    """
).strip()


def _write_skill_lib(skills_dir: Path) -> None:
    """Seed a matchable skill so evidence can be attached."""
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill = OptimizationSkill(
        id="test_skill",
        name="test-skill",
        # Heavy keyword overlap with our synthetic round + the
        # L1/gemm classifier output so _match_to_skills routes
        # evidence here.
        trigger="matmul gemm l1 memory bound tile",
        approach="tile and vectorize",
        backend="any",
        tags=["gemm", "matmul", "l1"],
    )
    with open(skills_dir / f"{skill.id}.json", "w") as f:
        json.dump(
            {
                "id": skill.id,
                "name": skill.name,
                "trigger": skill.trigger,
                "approach": skill.approach,
                "backend": skill.backend,
                "evidence": [],
                "code_template": None,
                "tags": skill.tags,
                "pitfalls": [],
            },
            f,
        )


def _round(
    *,
    round_num: int = 1,
    speedup: float = 1.0,
    sol: float | None = 0.0,
    status: str = "success",
) -> dict:
    profile: dict = {}
    if sol is not None:
        profile["sol_score"] = sol
    return {
        "round": round_num,
        "speedup": speedup,
        "status": status,
        "strategy": "matmul tile l1",
        "bottleneck": "memory-bound",
        "method_applied": "matmul kernel",
        "profile": profile,
        "is_correct": True,
        "kernel_code": "",
    }


def _count_new_evidence(skills_dir: Path) -> int:
    lib = SkillLibrary(skills_dir)
    lib.load()
    return sum(len(s.evidence) for s in lib.all_skills)


# ----------------------------------------------------------------------
# Gate behavior
# ----------------------------------------------------------------------


def test_sol_above_floor_with_no_prior_best_promotes(tmp_path: Path) -> None:
    _write_skill_lib(tmp_path)
    added = extract_and_update_evidence(
        optimization_log=[_round(speedup=1.5, sol=0.60)],
        reference_code=_REFERENCE_CODE,
        hardware="L40S",
        skills_dir=tmp_path,
    )
    assert added > 0
    assert _count_new_evidence(tmp_path) > 0


def test_sol_below_floor_rejected(tmp_path: Path) -> None:
    _write_skill_lib(tmp_path)
    added = extract_and_update_evidence(
        # 0.30 SOL is sub-baseline; speedup value is irrelevant when
        # SOL is present.
        optimization_log=[_round(speedup=1.5, sol=0.30)],
        reference_code=_REFERENCE_CODE,
        hardware="L40S",
        skills_dir=tmp_path,
    )
    assert added == 0
    assert _count_new_evidence(tmp_path) == 0


def test_sol_below_prior_best_rejected(tmp_path: Path, monkeypatch) -> None:
    _write_skill_lib(tmp_path)

    # Stub prior_best_sol to return 0.70 for any problem+hw. The module
    # under test imports it inside _load_prior_best_sol via a lazy
    # import, so patch the *source* attribute.
    import openkernel.benchmarks.leaderboard_reader as lr
    monkeypatch.setattr(lr, "prior_best_sol", lambda *a, **kw: 0.70)

    added = extract_and_update_evidence(
        optimization_log=[_round(speedup=1.5, sol=0.60)],
        reference_code=_REFERENCE_CODE,
        hardware="L40S",
        skills_dir=tmp_path,
        problem_id="kb_l1_matmul",
    )
    assert added == 0
    assert _count_new_evidence(tmp_path) == 0


def test_speedup_fallback_when_sol_missing(tmp_path: Path) -> None:
    _write_skill_lib(tmp_path)
    added = extract_and_update_evidence(
        optimization_log=[_round(speedup=1.5, sol=0.0)],
        reference_code=_REFERENCE_CODE,
        hardware="L40S",
        skills_dir=tmp_path,
    )
    assert added > 0


def test_speedup_fallback_rejects_baseline_run(tmp_path: Path) -> None:
    _write_skill_lib(tmp_path)
    added = extract_and_update_evidence(
        optimization_log=[_round(speedup=1.0, sol=0.0)],
        reference_code=_REFERENCE_CODE,
        hardware="L40S",
        skills_dir=tmp_path,
    )
    assert added == 0


# ----------------------------------------------------------------------
# prior_best_sol reader helper
# ----------------------------------------------------------------------


def _write_leaderboard_record(root: Path, **overrides) -> None:
    rec = {
        "schema_version": "1.0",
        "problem_id": "kb_l1_0001",
        "problem_name": "Softmax",
        "tier": "L1",
        "hardware": "L40S",
        "date": "2026-04-22",
        "timestamp": "2026-04-22T14:17:03Z",
        "kernel_hash": "h",
        "kernel_source_path": "kernels/h.py",
        "model": "claude-sonnet-4-6",
        "speedup": 1.85,
        "sol_score": 0.42,
        "correct": True,
        "cost_usd": 0.1,
        "elapsed_s": 10,
        "stop_reason": "done",
        "config_hash": "cfg_a",
        "rounds": 1,
        "iterations": 1,
    }
    rec.update(overrides)
    (root / f"{rec['problem_id']}_{rec['config_hash']}.json").write_text(json.dumps(rec))


def test_prior_best_sol_returns_zero_when_no_records(tmp_path: Path) -> None:
    assert prior_best_sol("kb_l1_0001", "L40S", root=tmp_path) == 0.0


def test_prior_best_sol_returns_max_sol(tmp_path: Path) -> None:
    _write_leaderboard_record(tmp_path, config_hash="cfg_a", sol_score=0.40)
    _write_leaderboard_record(tmp_path, config_hash="cfg_b", sol_score=0.65)
    _write_leaderboard_record(tmp_path, config_hash="cfg_c", sol_score=0.50)
    assert prior_best_sol("kb_l1_0001", "L40S", root=tmp_path) == 0.65


def test_prior_best_sol_ignores_incorrect_runs(tmp_path: Path) -> None:
    _write_leaderboard_record(tmp_path, config_hash="cfg_a", sol_score=0.40, correct=True)
    _write_leaderboard_record(tmp_path, config_hash="cfg_b", sol_score=0.90, correct=False)
    assert prior_best_sol("kb_l1_0001", "L40S", root=tmp_path) == 0.40


def test_prior_best_sol_hardware_scoped(tmp_path: Path) -> None:
    _write_leaderboard_record(tmp_path, config_hash="cfg_a", hardware="L40S", sol_score=0.40)
    _write_leaderboard_record(tmp_path, config_hash="cfg_b", hardware="H100", sol_score=0.90)
    assert prior_best_sol("kb_l1_0001", "L40S", root=tmp_path) == 0.40


# ----------------------------------------------------------------------
# Priority scoring
# ----------------------------------------------------------------------


def test_compute_skill_priority_weights_sol() -> None:
    skill_high_sol = OptimizationSkill(
        id="a", name="a", trigger="", approach="", backend="any",
        evidence=[{"sol_score": 0.9, "speedup": 1.1, "hardware": "L40S", "problem": "L1/matmul"}],
    )
    skill_high_speedup_low_sol = OptimizationSkill(
        id="b", name="b", trigger="", approach="", backend="any",
        evidence=[{"sol_score": 0.1, "speedup": 5.0, "hardware": "L40S", "problem": "L1/matmul"}],
    )

    # SOL-driven scoring: 0.9 SOL should outrank 5x speedup at SOL 0.1.
    assert compute_skill_priority(skill_high_sol) > compute_skill_priority(skill_high_speedup_low_sol)


def test_compute_skill_priority_handles_legacy_evidence_without_sol() -> None:
    # Pre-SOL skills with only speedup should still score above empty.
    legacy = OptimizationSkill(
        id="legacy", name="legacy", trigger="", approach="", backend="any",
        evidence=[{"speedup": 2.0, "hardware": "L40S", "problem": "L1/matmul"}],
    )
    empty = OptimizationSkill(
        id="empty", name="empty", trigger="", approach="", backend="any",
    )
    assert compute_skill_priority(legacy) > compute_skill_priority(empty)
