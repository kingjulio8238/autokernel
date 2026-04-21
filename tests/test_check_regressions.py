"""Tests for scripts/check_regressions.py.

Runnable via:
    python -m pytest tests/test_check_regressions.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_check_regressions():
    path = _REPO_ROOT / "scripts" / "check_regressions.py"
    spec = importlib.util.spec_from_file_location("check_regressions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_regressions = _load_check_regressions()


def _base_record(**overrides) -> dict:
    rec = {
        "schema_version": "1.0",
        "problem_id": "kb_l1_0001",
        "problem_name": "Softmax",
        "tier": "L1",
        "hardware": "L40S",
        "date": "2026-04-22",
        "timestamp": "2026-04-22T14:17:03Z",
        "kernel_hash": "a3f2b91c4d7e6081",
        "kernel_source_path": "kernels/a3f2b91c4d7e6081.py",
        "model": "claude-sonnet-4-6",
        "speedup": 1.85,
        "sol_score": 0.42,
        "compute_util": 18.4,
        "bandwidth_util": 45.2,
        "bottleneck_type": "memory-bound",
        "correct": True,
        "cost_usd": 0.12,
        "elapsed_s": 142,
        "stop_reason": "Target reached",
        "config_hash": "cfg_f7d12e",
        "rounds": 3,
        "iterations": 12,
    }
    rec.update(overrides)
    return rec


def _write(root: Path, name: str, record: dict) -> None:
    path = root / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _run(argv: list[str]) -> tuple[int, str]:
    """Run the CLI and capture stdout + exit code."""
    rc = check_regressions.run(argv)
    return rc, ""  # stdout captured via capsys in tests


def test_clean_run_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Two problems: today's SOL is equal-or-better than prior best.
    _write(tmp_path, "prior_a", _base_record(
        problem_id="kb_l1_0001", date="2026-04-20",
        timestamp="2026-04-20T10:00:00Z", sol_score=0.40,
    ))
    _write(tmp_path, "today_a", _base_record(
        problem_id="kb_l1_0001", date="2026-04-22",
        timestamp="2026-04-22T10:00:00Z", sol_score=0.42,
    ))
    _write(tmp_path, "prior_b", _base_record(
        problem_id="kb_l1_0002", date="2026-04-21",
        timestamp="2026-04-21T10:00:00Z", sol_score=0.30,
    ))
    _write(tmp_path, "today_b", _base_record(
        problem_id="kb_l1_0002", date="2026-04-22",
        timestamp="2026-04-22T10:00:00Z", sol_score=0.31,
    ))

    rc = check_regressions.run([
        "--date", "2026-04-22",
        "--hardware", "L40S",
        "--leaderboard-root", str(tmp_path),
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert "No regressions detected" in out
    assert "2 problems" in out


def test_perf_regression_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Prior max 0.45, today 0.12 → Δ -0.33 > 0.05 threshold.
    _write(tmp_path, "prior_best", _base_record(
        problem_id="kb_l1_0042", problem_name="GEMM",
        date="2026-04-20", timestamp="2026-04-20T10:00:00Z",
        sol_score=0.45,
    ))
    _write(tmp_path, "prior_mid", _base_record(
        problem_id="kb_l1_0042", problem_name="GEMM",
        date="2026-04-21", timestamp="2026-04-21T10:00:00Z",
        sol_score=0.40, config_hash="cfg_other1",
    ))
    _write(tmp_path, "today", _base_record(
        problem_id="kb_l1_0042", problem_name="GEMM",
        date="2026-04-22", timestamp="2026-04-22T10:00:00Z",
        sol_score=0.12,
    ))

    rc = check_regressions.run([
        "--date", "2026-04-22",
        "--hardware", "L40S",
        "--leaderboard-root", str(tmp_path),
    ])
    out = capsys.readouterr().out

    assert rc == 1
    assert "kb_l1_0042" in out
    assert "Regressions detected" in out
    # Prior best date should surface.
    assert "2026-04-20" in out


def test_correctness_regression_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Prior correct=True with decent SOL; today correct=False.
    # Even though sol_score didn't drop (arbitrary value), correctness loss flags.
    _write(tmp_path, "prior", _base_record(
        problem_id="kb_l1_0007", problem_name="LayerNorm",
        date="2026-04-20", timestamp="2026-04-20T10:00:00Z",
        sol_score=0.30, correct=True,
    ))
    _write(tmp_path, "today", _base_record(
        problem_id="kb_l1_0007", problem_name="LayerNorm",
        date="2026-04-22", timestamp="2026-04-22T10:00:00Z",
        sol_score=0.50, correct=False,
    ))

    rc = check_regressions.run([
        "--date", "2026-04-22",
        "--hardware", "L40S",
        "--leaderboard-root", str(tmp_path),
    ])
    out = capsys.readouterr().out

    assert rc == 1
    assert "kb_l1_0007" in out
    assert "correctness regression" in out


def test_no_prior_data_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Only today's record exists — no baseline to regress against.
    _write(tmp_path, "today", _base_record(
        problem_id="kb_l1_0001", date="2026-04-22",
        timestamp="2026-04-22T10:00:00Z", sol_score=0.42,
    ))

    rc = check_regressions.run([
        "--date", "2026-04-22",
        "--hardware", "L40S",
        "--leaderboard-root", str(tmp_path),
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert "No prior data" in out
