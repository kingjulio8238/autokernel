"""Tests for goal_parser (Phase 2 — SOL grammar addition)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel_code.goal_parser import parse_goal, validate_goal


# --- Speedup grammar (backcompat) ------------------------------------------


def test_speedup_bare_x() -> None:
    g = parse_goal("2x")
    assert g.target_speedup == 2.0
    assert "target" in g.explicit


def test_speedup_with_suffix() -> None:
    g = parse_goal("need 2x speedup")
    assert g.target_speedup == 2.0


def test_speedup_target_2x() -> None:
    g = parse_goal("target 2x")
    assert g.target_speedup == 2.0


def test_speedup_decimal() -> None:
    g = parse_goal("2.5x faster")
    assert g.target_speedup == 2.5


def test_speedup_does_not_set_sol() -> None:
    g = parse_goal("2x speedup")
    assert g.target_sol == 0.0


# --- SOL grammar -----------------------------------------------------------


def test_sol_prefix_space() -> None:
    g = parse_goal("SOL 0.8")
    assert g.target_sol == pytest.approx(0.8)
    assert "target_sol" in g.explicit


def test_sol_lowercase() -> None:
    g = parse_goal("sol 0.8")
    assert g.target_sol == pytest.approx(0.8)


def test_sol_suffix() -> None:
    g = parse_goal("0.8 SOL")
    assert g.target_sol == pytest.approx(0.8)


def test_sol_percent() -> None:
    g = parse_goal("80% SOL")
    assert g.target_sol == pytest.approx(0.8)


def test_target_sol_phrase() -> None:
    g = parse_goal("target sol 0.8")
    assert g.target_sol == pytest.approx(0.8)


def test_sol_does_not_set_speedup() -> None:
    g = parse_goal("SOL 0.8")
    assert g.target_speedup == 0.0


# --- Both targets in one string --------------------------------------------


def test_both_targets_speedup_then_sol() -> None:
    g = parse_goal("target 2x speedup SOL 0.8")
    assert g.target_speedup == 2.0
    assert g.target_sol == pytest.approx(0.8)
    assert "target" in g.explicit
    assert "target_sol" in g.explicit


def test_both_targets_sol_then_speedup() -> None:
    g = parse_goal("SOL 0.9 and 3x speedup")
    assert g.target_speedup == 3.0
    assert g.target_sol == pytest.approx(0.9)


def test_both_targets_in_realistic_sentence() -> None:
    g = parse_goal("optimize @kernel.py for H100, need 2x speedup, SOL 0.8, budget $10")
    assert g.target_speedup == 2.0
    assert g.target_sol == pytest.approx(0.8)
    assert g.hardware == "H100"
    assert g.budget_usd == 10.0
    assert g.file == "kernel.py"


# --- Ambiguity rejection ---------------------------------------------------


def test_bare_number_rejected() -> None:
    with pytest.raises(ValueError, match="Ambiguous"):
        parse_goal("0.8")


def test_bare_number_with_percent_rejected() -> None:
    with pytest.raises(ValueError, match="Ambiguous"):
        parse_goal("80%")


def test_target_number_no_unit_rejected() -> None:
    with pytest.raises(ValueError, match="Ambiguous"):
        parse_goal("target 0.8")


def test_ambiguity_error_lists_accepted_forms() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_goal("0.8")
    msg = str(excinfo.value)
    assert "2x" in msg
    assert "SOL" in msg


def test_ambiguity_not_raised_when_target_is_well_formed() -> None:
    # Sanity: well-formed input must not trigger the guard.
    parse_goal("target 2x")
    parse_goal("target sol 0.8")
    parse_goal("target 2x speedup SOL 0.8")


def test_ambiguity_not_raised_on_unrelated_text() -> None:
    # No target at all — fine, just returns empty parse.
    g = parse_goal("optimize @kernel.py on H100")
    assert g.target_speedup == 0.0
    assert g.target_sol == 0.0


# --- validate_goal ---------------------------------------------------------


def test_validate_rejects_sol_above_one(tmp_path: Path) -> None:
    g = parse_goal("SOL 0.8")
    g.target_sol = 1.5
    errors = validate_goal(g, tmp_path)
    assert any("SOL" in e or "sol" in e for e in errors)


def test_validate_rejects_sol_below_zero(tmp_path: Path) -> None:
    g = parse_goal("SOL 0.8")
    g.target_sol = -0.1
    errors = validate_goal(g, tmp_path)
    assert any("SOL" in e or "sol" in e for e in errors)


def test_validate_accepts_valid_sol(tmp_path: Path) -> None:
    g = parse_goal("SOL 0.8")
    errors = validate_goal(g, tmp_path)
    assert not any("SOL" in e or "sol" in e for e in errors)


def test_validate_still_rejects_negative_speedup(tmp_path: Path) -> None:
    # Backcompat: existing check unchanged.
    g = parse_goal("2x")
    g.target_speedup = -1.0
    errors = validate_goal(g, tmp_path)
    assert any("target" in e.lower() for e in errors)
