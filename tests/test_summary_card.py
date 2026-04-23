"""Tests for `kernel_code.summary_card.render_optimization_summary`.

Asserts Phase 2 dual-display conventions:
- When SOL is measured: headline shows "SOL <v>" primary, speedup secondary.
- When SOL is absent (or 0.0): speedup-only with explicit "SOL unknown" tag.
- Never emits SOL in isolation — speedup is always present in the text.

Fast, hermetic: renders into an in-memory Rich Console and asserts on the
captured plain-text output. No Modal, no GPU.
"""

from __future__ import annotations

import io

from rich.console import Console

from kernel_code.summary_card import render_optimization_summary


def _render(
    iterations: list[dict],
    *,
    best_speedup: float,
    best_iteration: int = 1,
    kept_count: int = 1,
    total_count: int = 1,
    best_sol: float = 0.0,
    bottleneck_type: str = "",
) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, color_system=None)
    render_optimization_summary(
        iterations=iterations,
        best_speedup=best_speedup,
        best_iteration=best_iteration,
        kept_count=kept_count,
        total_count=total_count,
        bottleneck_type=bottleneck_type,
        best_sol=best_sol,
        console=console,
    )
    return buf.getvalue()


def test_summary_card_shows_sol_primary_and_speedup_secondary() -> None:
    iterations = [
        {"speedup": 1.0, "decision": "discard", "intent": "a", "profile": {"sol_score": 0.10}},
        {"speedup": 1.4, "decision": "discard", "intent": "b", "profile": {"sol_score": 0.25}},
        {"speedup": 1.6, "decision": "keep", "intent": "c", "profile": {"sol_score": 0.35}},
        {"speedup": 1.8, "decision": "keep", "intent": "tile", "profile": {"sol_score": 0.42}},
    ]
    out = _render(
        iterations,
        best_speedup=1.8,
        best_iteration=4,
        kept_count=2,
        total_count=4,
        best_sol=0.42,
        bottleneck_type="memory_bound",
    )
    # Primary: SOL headline with percent-of-peak and bottleneck.
    assert "Best SOL" in out
    assert "0.42" in out
    assert "42% of peak" in out
    assert "memory-bound" in out
    # Secondary: speedup is still present as the secondary row.
    assert "1.80x" in out
    # Panel subtitle should show both.
    assert "SOL 0.42" in out


def test_summary_card_autoderives_best_sol_from_iterations() -> None:
    # Caller did not pass best_sol — derived from iterations[].profile.
    # Use enough rows to avoid the preexisting ascii-chart single-row bug.
    iterations = [
        {"speedup": 1.1, "decision": "discard", "intent": "a", "profile": {"sol_score": 0.15}},
        {"speedup": 1.2, "decision": "discard", "intent": "b", "profile": {"sol_score": 0.20}},
        {"speedup": 1.5, "decision": "discard", "intent": "c", "profile": {"sol_score": 0.40}},
        {"speedup": 1.8, "decision": "keep", "intent": "d", "profile": {"sol_score": 0.55}},
    ]
    out = _render(iterations, best_speedup=1.8, kept_count=1, total_count=4)
    assert "Best SOL" in out
    assert "0.55" in out
    assert "1.80x" in out


def test_summary_card_falls_back_to_speedup_only_when_sol_missing() -> None:
    iterations = [
        {"speedup": 1.0, "decision": "discard", "intent": "a"},
        {"speedup": 1.3, "decision": "discard", "intent": "b"},
        {"speedup": 1.6, "decision": "keep", "intent": "c"},
        {"speedup": 1.8, "decision": "keep", "intent": "tile"},
    ]
    out = _render(iterations, best_speedup=1.8, kept_count=2, total_count=4, best_sol=0.0)
    # No SOL row.
    assert "Best SOL" not in out
    # Speedup is shown and explicitly tagged as SOL-unknown.
    assert "1.80x" in out
    assert "SOL unknown" in out


def test_summary_card_treats_zero_sol_score_in_profile_as_missing() -> None:
    iterations = [
        {"speedup": 1.0, "decision": "discard", "intent": "a", "profile": {"sol_score": 0.0}},
        {"speedup": 1.3, "decision": "discard", "intent": "b", "profile": {"sol_score": 0.0}},
        {"speedup": 1.6, "decision": "keep", "intent": "c", "profile": {"sol_score": 0.0}},
        {"speedup": 1.8, "decision": "keep", "intent": "x", "profile": {"sol_score": 0.0}},
    ]
    out = _render(iterations, best_speedup=1.8, kept_count=2, total_count=4)
    assert "Best SOL" not in out
    assert "1.80x" in out
    assert "SOL unknown" in out


def test_summary_card_never_emits_sol_in_isolation() -> None:
    # Even when SOL is present, speedup must appear in the output.
    iterations = [
        {"speedup": 1.0, "decision": "discard", "intent": "a", "profile": {"sol_score": 0.10}},
        {"speedup": 1.3, "decision": "discard", "intent": "b", "profile": {"sol_score": 0.20}},
        {"speedup": 1.6, "decision": "keep", "intent": "c", "profile": {"sol_score": 0.30}},
        {"speedup": 1.8, "decision": "keep", "intent": "x", "profile": {"sol_score": 0.42}},
    ]
    out = _render(iterations, best_speedup=1.8, kept_count=2, total_count=4, best_sol=0.42)
    assert "1.80x" in out, "speedup must always accompany SOL"
