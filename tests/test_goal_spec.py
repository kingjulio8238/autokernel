"""Tests for GoalSpec (Phase 2 — SOL target addition)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel_code.goal_spec import GoalSpec


@pytest.fixture
def ref_file(tmp_path: Path) -> Path:
    p = tmp_path / "ref.py"
    p.write_text("# reference stub\n")
    return p


def test_default_target_sol_is_080(ref_file: Path) -> None:
    spec = GoalSpec(reference_path=str(ref_file))
    assert spec.target_sol == 0.80


def test_target_speedup_still_defaults_to_2x(ref_file: Path) -> None:
    # Backcompat: existing default behavior unchanged.
    spec = GoalSpec(reference_path=str(ref_file))
    assert spec.target_speedup == 2.0


def test_construct_with_target_sol_kwarg(ref_file: Path) -> None:
    spec = GoalSpec(target_sol=0.90, reference_path=str(ref_file))
    assert spec.target_sol == 0.90
    assert spec.target_speedup == 2.0  # default preserved


def test_construct_with_both_targets(ref_file: Path) -> None:
    spec = GoalSpec(target_speedup=3.0, target_sol=0.75, reference_path=str(ref_file))
    assert spec.target_speedup == 3.0
    assert spec.target_sol == 0.75


def test_backcompat_positional_speedup_only(ref_file: Path) -> None:
    # Callers that pre-date target_sol should still work.
    spec = GoalSpec(target_speedup=2.5, reference_path=str(ref_file))
    assert spec.target_speedup == 2.5
    assert spec.target_sol == 0.80  # default fills in


def test_validate_rejects_sol_above_one(ref_file: Path) -> None:
    spec = GoalSpec(target_sol=1.5, reference_path=str(ref_file))
    errors = spec.validate()
    assert any("target_sol" in e for e in errors)


def test_validate_rejects_sol_below_zero(ref_file: Path) -> None:
    spec = GoalSpec(target_sol=-0.1, reference_path=str(ref_file))
    errors = spec.validate()
    assert any("target_sol" in e for e in errors)


def test_validate_rejects_sol_equal_zero(ref_file: Path) -> None:
    # Per spec: 0.0 < target_sol <= 1.0; zero is invalid in GoalSpec.
    spec = GoalSpec(target_sol=0.0, reference_path=str(ref_file))
    errors = spec.validate()
    assert any("target_sol" in e for e in errors)


def test_validate_accepts_sol_equal_one(ref_file: Path) -> None:
    spec = GoalSpec(target_sol=1.0, reference_path=str(ref_file))
    errors = spec.validate()
    assert not any("target_sol" in e for e in errors)


def test_validate_accepts_default_sol(ref_file: Path) -> None:
    spec = GoalSpec(reference_path=str(ref_file))
    errors = spec.validate()
    assert not any("target_sol" in e for e in errors)


def test_summary_includes_sol(ref_file: Path) -> None:
    spec = GoalSpec(target_speedup=2.0, target_sol=0.80, reference_path=str(ref_file))
    s = spec.summary()
    assert "SOL" in s
    assert "0.80" in s
    # Speedup still shown — no regression.
    assert "2.0x" in s
