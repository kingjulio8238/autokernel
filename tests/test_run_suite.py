"""Tests for kernel_code.batch_optimizer.run_suite.

Runnable via:
    python -m pytest tests/test_run_suite.py -v
    python tests/test_run_suite.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _make_spec(idx: int):
    from openkernel.benchmarks import ProblemSpec

    return ProblemSpec(
        id=f"kb_l1_{idx:04d}",
        name=f"Problem{idx}",
        tier="L1",
        source="kernelbench",
        reference_source=f"# reference kernel for problem {idx}\nimport torch\n",
    )


def _fake_auto_result(speedup: float = 1.5, kernel: str = "# kernel\n"):
    from kernel_code.auto_optimizer import AutoResult

    return AutoResult(
        best_speedup=speedup,
        best_kernel=kernel,
        target_reached=speedup >= 2.0,
        rounds_completed=2,
        total_iterations=8,
        total_cost_usd=0.10,
        elapsed_seconds=42.0,
        stop_reason="target reached" if speedup >= 2.0 else "budget exhausted",
        round_history=[
            {
                "round": 1,
                "speedup": speedup * 0.8,
                "profile": {
                    "bandwidth_util": 30.0,
                    "compute_util": 10.0,
                    "sol_score": 0.2,
                },
            },
            {
                "round": 2,
                "speedup": speedup,
                "profile": {
                    "bandwidth_util": 55.0,
                    "compute_util": 25.0,
                    "sol_score": 0.45,
                },
            },
        ],
    )


def test_resume_skips_existing(tmp_path):
    """Pre-populate leaderboard with 1 record; run a 2-spec suite where one matches."""
    from kernel_code import batch_optimizer
    from kernel_code.batch_optimizer import run_suite, _suite_config_hash
    from openkernel.benchmarks.leaderboard_writer import write_record

    hardware = "L40S"
    model = "o3-mini"
    target_sol = 0.80
    budget = 0.50

    spec_a = _make_spec(1)
    spec_b = _make_spec(2)

    chash = _suite_config_hash(
        model=model,
        backend="triton",
        target_sol=target_sol,
        budget_per_problem=budget,
    )

    preexisting = {
        "kernel_source": "# already done\n",
        "hardware": hardware,
        "date": "2026-04-21",
        "timestamp": "2026-04-21T10:00:00Z",
        "model": model,
        "speedup": 1.2,
        "sol_score": 0.5,
        "compute_util": 20.0,
        "bandwidth_util": 40.0,
        "bottleneck_type": "memory-bound",
        "correct": True,
        "cost_usd": 0.08,
        "elapsed_s": 30,
        "stop_reason": "target reached",
        "rounds": 2,
        "iterations": 5,
        "config": {
            "model": model,
            "backend": "triton",
            "target_sol": target_sol,
            "budget_per_problem": budget,
            "seed": 7,
        },
    }
    write_record(spec_a, preexisting, root=tmp_path)

    # Sanity: hash we write matches what run_suite will compute.
    from openkernel.benchmarks.leaderboard_reader import load_all
    loaded = load_all(root=tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["config_hash"] == chash

    # Mock _run_single_spec so only the unmatched spec actually runs.
    called_for: list[str] = []

    def fake_run_single(spec, *args, **kwargs):
        called_for.append(spec.id)
        return ("succeeded", _fake_auto_result(speedup=1.8), "/fake/path.json")

    with mock.patch.object(batch_optimizer, "_run_single_spec", side_effect=fake_run_single):
        result = run_suite(
            specs=[spec_a, spec_b],
            hardware=hardware,
            budget_per_problem=budget,
            concurrency=2,
            date="2026-04-21",
            model=model,
            target_sol=target_sol,
            leaderboard_root=tmp_path,
        )

    assert result.total == 2, f"total should be 2, got {result.total}"
    assert result.skipped == 1, f"skipped should be 1, got {result.skipped}"
    assert result.succeeded == 1, f"succeeded should be 1, got {result.succeeded}"
    assert result.errored == 0
    assert result.failed == 0
    assert called_for == [spec_b.id], (
        f"_run_single_spec should run only for unmatched spec, called for {called_for}"
    )


def test_error_isolation(tmp_path):
    """One spec raising should not take down the whole suite."""
    from kernel_code import batch_optimizer
    from kernel_code.batch_optimizer import run_suite

    specs = [_make_spec(i) for i in range(1, 4)]

    def fake_run_single(spec, *args, **kwargs):
        if spec.id == "kb_l1_0002":
            raise RuntimeError("simulated Modal crash")
        return ("succeeded", _fake_auto_result(speedup=1.5), f"/fake/{spec.id}.json")

    with mock.patch.object(batch_optimizer, "_run_single_spec", side_effect=fake_run_single):
        result = run_suite(
            specs=specs,
            hardware="H100",
            budget_per_problem=0.25,
            concurrency=3,
            date="2026-04-21",
            model="o3-mini",
            target_sol=0.80,
            leaderboard_root=tmp_path,
        )

    assert result.total == 3
    assert result.skipped == 0
    assert result.errored == 1, f"errored should be 1, got {result.errored}"
    assert result.succeeded == 2, f"succeeded should be 2, got {result.succeeded}"
    assert result.failed == 0
    assert len(result.records_written) == 2


def test_budget_overage_graceful(tmp_path):
    """A spec returning no best_kernel (budget-blown before producing) is counted failed, not errored."""
    from kernel_code import batch_optimizer
    from kernel_code.batch_optimizer import run_suite
    from kernel_code.auto_optimizer import AutoResult

    specs = [_make_spec(i) for i in range(1, 4)]

    def fake_run_single(spec, *args, **kwargs):
        if spec.id == "kb_l1_0002":
            # Budget blown before a correct kernel was produced — no best_kernel.
            empty = AutoResult(
                best_speedup=0.0,
                best_kernel="",
                target_reached=False,
                rounds_completed=0,
                total_iterations=0,
                total_cost_usd=0.55,  # over the 0.50 budget
                elapsed_seconds=10.0,
                stop_reason="Budget exhausted",
                round_history=[],
            )
            return ("failed", empty, None)
        return ("succeeded", _fake_auto_result(speedup=1.5), f"/fake/{spec.id}.json")

    with mock.patch.object(batch_optimizer, "_run_single_spec", side_effect=fake_run_single):
        result = run_suite(
            specs=specs,
            hardware="L40S",
            budget_per_problem=0.50,
            concurrency=2,
            date="2026-04-21",
            model="o3-mini",
            target_sol=0.80,
            leaderboard_root=tmp_path,
        )

    assert result.total == 3
    assert result.failed == 1, f"over-budget spec should be failed, got {result.failed}"
    assert result.errored == 0, (
        f"over-budget is graceful, should not be errored: {result.errored}"
    )
    assert result.succeeded == 2
    # Total cost includes the over-budget attempt.
    assert result.total_cost_usd >= 0.55


def main() -> int:
    import tempfile

    tests = [
        test_resume_skips_existing,
        test_error_isolation,
        test_budget_overage_graceful,
    ]
    passed = 0
    failed = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
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
