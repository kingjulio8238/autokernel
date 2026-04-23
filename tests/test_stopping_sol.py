"""Tests for SOL-first stopping in MetaOptimizer.

Covers Task #2 (Phase 2 SOL UX):
  - SOL-reached path fires when latest profile has sol_score >= target_sol.
  - Speedup-fallback path fires when latest profile lacks SOL.
  - Both-criteria behavior (SOL wins when both could fire).
  - ``MetaOptimizer.run(target_sol=...)`` overrides ``GoalSpec.target_sol``.

Runnable via:
    uv run pytest tests/test_stopping_sol.py -v
    uv run python tests/test_stopping_sol.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _make_optimizer(target_speedup: float, target_sol: float, max_rounds: int = 3):
    from kernel_code.auto_optimizer import MetaOptimizer
    from kernel_code.goal_spec import GoalSpec
    from kernel_code.settings import KernelCodeSettings

    # Give the goal a real reference file so validate() passes if anyone calls it.
    # MetaOptimizer.run() itself does not call validate(), but _run_round reads
    # the reference path — we patch _run_round so the content is irrelevant.
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="stopping_sol_ref_", delete=False
    )
    tf.write("# dummy reference\nimport torch\n")
    tf.close()

    goal = GoalSpec(
        target_speedup=target_speedup,
        target_sol=target_sol,
        max_budget_usd=10.00,
        max_rounds=max_rounds,
        reference_path=tf.name,
        hardware="L40S",
        backend="triton",
    )
    settings = KernelCodeSettings()
    settings.num_workers = 1
    return MetaOptimizer(goal=goal, settings=settings), goal, Path(tf.name)


def _round_result(
    round_num: int,
    speedup: float,
    sol_score: float | None,
    strategy: str = "general optimization",
) -> dict:
    """Build a round_result dict matching MetaOptimizer._run_round's shape.

    ``sol_score=None`` omits the key entirely so from_modal_profile stores 0.0
    (the "no SOL" sentinel used by the fallback branch).
    """
    profile: dict = {
        "bandwidth_util": 30.0,
        "compute_util": 10.0,
        "runtime_us": 100.0,
        "ref_runtime_us": 200.0,
    }
    if sol_score is not None:
        profile["sol_score"] = sol_score
    return {
        "round": round_num,
        "strategy": strategy,
        "best_speedup": speedup,
        "best_kernel": f"# kernel r{round_num}\n",
        "kept": 1 if speedup > 0 else 0,
        "total": 1,
        "errors": 0,
        "bottleneck": "memory-bound",
        "per_worker": [],
        "profile": profile,
        "infra_failed": False,
        "infra_error": "",
    }


def _patch_run_round(optimizer, round_results: list[dict]):
    """Replace optimizer._run_round so it yields the given pre-built results.

    Uses the same control flow as the real method but skips the bridge call —
    we drive the stopping logic directly. Also appends to _optimization_log so
    _latest_sol() can read the round's SOL score.
    """
    from kernel_code.optimization_log import (
        OptimizationRound,
        ProfileMetrics,
        RoundStatus,
    )

    idx = {"i": 0}

    def fake_run_round(round_num: int) -> dict:
        i = idx["i"]
        idx["i"] += 1
        if i >= len(round_results):
            # Defensive: shouldn't be hit when max_rounds matches results length.
            return _round_result(round_num, speedup=0.0, sol_score=None)
        rr = round_results[i]
        profile_metrics = ProfileMetrics.from_modal_profile(rr.get("profile") or {})
        opt_round = OptimizationRound(
            round=round_num,
            kernel_code=rr.get("best_kernel", ""),
            is_correct=rr.get("kept", 0) > 0,
            speedup=rr.get("best_speedup", 0.0),
            status=RoundStatus.SUCCESS,
            strategy=rr.get("strategy", ""),
            bottleneck=rr.get("bottleneck", ""),
            profile=profile_metrics,
        )
        optimizer._optimization_log.add_round(opt_round)
        return rr

    optimizer._run_round = fake_run_round

    # Also avoid the between-round LLM reflection call.
    from kernel_code.meta_reflect import MetaReflection
    optimizer._reflect = lambda round_num: MetaReflection(  # type: ignore[assignment]
        action="continue", reason="test stub"
    )


def _cleanup(ref_path: Path) -> None:
    try:
        ref_path.unlink()
    except OSError:
        pass


def test_sol_criterion_fires_when_sol_available():
    """Latest profile has valid SOL >= target_sol → stop on SOL.

    Uses sol just above target (below the 1.10x overshoot threshold) so the
    stop fires immediately on round 1 rather than deferring for an
    exploratory round.
    """
    optimizer, goal, ref = _make_optimizer(
        target_speedup=10.0,  # intentionally unreachable so only SOL can fire
        target_sol=0.50,
        max_rounds=3,
    )
    try:
        # Round 1: sol=0.52 >= 0.50, overshoot ratio 1.04 < 1.10 → stop now.
        _patch_run_round(
            optimizer,
            [_round_result(1, speedup=1.5, sol_score=0.52)],
        )
        # Swallow evidence extraction to keep the test hermetic.
        with mock.patch(
            "kernel_code.auto_optimizer.extract_and_update_evidence",
            return_value=0,
        ):
            result = optimizer.run()

        assert result.target_reached is True, (
            f"target_reached should be True, got {result.target_reached}"
        )
        assert "SOL" in result.stop_reason, (
            f"stop_reason should mention SOL, got {result.stop_reason!r}"
        )
        assert result.rounds_completed == 1, (
            f"should stop after round 1 (overshoot 1.04 < 1.10), got "
            f"{result.rounds_completed}"
        )
    finally:
        _cleanup(ref)


def test_speedup_fallback_when_sol_missing():
    """Latest profile has no SOL → fall back to speedup criterion."""
    optimizer, goal, ref = _make_optimizer(
        target_speedup=2.0,
        target_sol=0.80,
        max_rounds=3,
    )
    try:
        # sol_score missing → speedup-fallback should fire when speedup >= target.
        # speedup 2.1 / target 2.0 = 1.05 < 1.10 overshoot threshold → stop now.
        _patch_run_round(
            optimizer,
            [_round_result(1, speedup=2.1, sol_score=None)],
        )
        with mock.patch(
            "kernel_code.auto_optimizer.extract_and_update_evidence",
            return_value=0,
        ):
            result = optimizer.run()

        assert result.target_reached is True
        assert "speedup" in result.stop_reason, (
            f"stop_reason should mention speedup, got {result.stop_reason!r}"
        )
        assert "SOL" not in result.stop_reason, (
            f"SOL should not fire when missing, got {result.stop_reason!r}"
        )
        assert result.rounds_completed == 1
    finally:
        _cleanup(ref)


def test_speedup_fallback_when_sol_zero():
    """sol_score=0.0 is the sentinel for "no SOL" → speedup fallback fires."""
    optimizer, goal, ref = _make_optimizer(
        target_speedup=2.0,
        target_sol=0.80,
        max_rounds=3,
    )
    try:
        _patch_run_round(
            optimizer,
            [_round_result(1, speedup=2.1, sol_score=0.0)],
        )
        with mock.patch(
            "kernel_code.auto_optimizer.extract_and_update_evidence",
            return_value=0,
        ):
            result = optimizer.run()

        assert result.target_reached is True
        assert "speedup" in result.stop_reason
    finally:
        _cleanup(ref)


def test_sol_wins_when_both_criteria_could_fire():
    """Both SOL and speedup reach targets → SOL is the criterion logged."""
    optimizer, goal, ref = _make_optimizer(
        target_speedup=2.0,
        target_sol=0.50,
        max_rounds=3,
    )
    try:
        # Round 1 hits both speedup=2.1 (>= 2.0) and sol=0.52 (>= 0.50).
        # Kept below 1.10 overshoot on both so we stop on round 1.
        # Both could trigger a stop; SOL must win.
        _patch_run_round(
            optimizer,
            [_round_result(1, speedup=2.1, sol_score=0.52)],
        )
        with mock.patch(
            "kernel_code.auto_optimizer.extract_and_update_evidence",
            return_value=0,
        ):
            result = optimizer.run()

        assert result.target_reached is True
        assert "SOL" in result.stop_reason, (
            f"SOL should take precedence when available, got {result.stop_reason!r}"
        )
    finally:
        _cleanup(ref)


def test_target_sol_runtime_override():
    """MetaOptimizer.run(target_sol=0.5) overrides GoalSpec.target_sol=0.3."""
    optimizer, goal, ref = _make_optimizer(
        target_speedup=10.0,  # keep speedup fallback from firing
        target_sol=0.30,
        max_rounds=3,
    )
    try:
        # Round 1: sol=0.40. Goal.target_sol=0.30 would trigger stop (met).
        # But override=0.50 > 0.40, so round 1 should NOT stop.
        # Round 2: sol=0.51 >= override (0.50), overshoot 1.02 < 1.10 → stop.
        _patch_run_round(
            optimizer,
            [
                _round_result(1, speedup=1.5, sol_score=0.40),
                _round_result(2, speedup=1.7, sol_score=0.51),
            ],
        )
        with mock.patch(
            "kernel_code.auto_optimizer.extract_and_update_evidence",
            return_value=0,
        ):
            result = optimizer.run(target_sol=0.50)

        assert result.target_reached is True
        assert result.rounds_completed == 2, (
            f"override=0.50 should defer stop to round 2, got "
            f"{result.rounds_completed} rounds"
        )
        assert "SOL" in result.stop_reason
        assert "0.51" in result.stop_reason, (
            f"stop_reason should show latest SOL 0.51, got {result.stop_reason!r}"
        )
    finally:
        _cleanup(ref)


def test_target_sol_override_none_uses_goal():
    """run(target_sol=None) leaves goal.target_sol in effect."""
    optimizer, goal, ref = _make_optimizer(
        target_speedup=10.0,
        target_sol=0.30,
        max_rounds=3,
    )
    try:
        # With no override, goal.target_sol=0.30 is the bar. Round 1 sol=0.31
        # clears it with overshoot 1.03 < 1.10 → stop immediately.
        _patch_run_round(
            optimizer,
            [_round_result(1, speedup=1.5, sol_score=0.31)],
        )
        with mock.patch(
            "kernel_code.auto_optimizer.extract_and_update_evidence",
            return_value=0,
        ):
            result = optimizer.run()  # no override

        assert result.target_reached is True
        assert result.rounds_completed == 1
    finally:
        _cleanup(ref)


def test_target_sol_override_zero_falls_back_to_goal():
    """run(target_sol=0.0) is treated as "use goal.target_sol", not "disable"."""
    optimizer, goal, ref = _make_optimizer(
        target_speedup=10.0,
        target_sol=0.30,
        max_rounds=3,
    )
    try:
        _patch_run_round(
            optimizer,
            [_round_result(1, speedup=1.5, sol_score=0.31)],
        )
        with mock.patch(
            "kernel_code.auto_optimizer.extract_and_update_evidence",
            return_value=0,
        ):
            result = optimizer.run(target_sol=0.0)

        assert result.target_reached is True, (
            "target_sol=0.0 override should fall back to goal.target_sol, "
            "not silently disable the SOL criterion"
        )
    finally:
        _cleanup(ref)


def main() -> int:
    tests = [
        test_sol_criterion_fires_when_sol_available,
        test_speedup_fallback_when_sol_missing,
        test_speedup_fallback_when_sol_zero,
        test_sol_wins_when_both_criteria_could_fire,
        test_target_sol_runtime_override,
        test_target_sol_override_none_uses_goal,
        test_target_sol_override_zero_falls_back_to_goal,
    ]
    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  [PASS] {fn.__name__}")

    total = passed + failed
    print(f"\n  RESULTS: {passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
