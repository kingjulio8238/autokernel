"""Tests for scripts/score_suite.py.

Runnable via:
    python -m pytest tests/test_score_suite.py -v
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# score_suite lives under scripts/ (not a package); load via spec.
_SCORE_SUITE_PATH = _REPO_ROOT / "scripts" / "score_suite.py"
_spec = importlib.util.spec_from_file_location("score_suite", _SCORE_SUITE_PATH)
assert _spec is not None and _spec.loader is not None
score_suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_suite)


def _record(
    problem_id: str,
    tier: str,
    sol: float,
    speedup: float,
    correct: bool = True,
    hardware: str = "L40S",
    date: str = "2026-04-22",
    cost_usd: float = 0.5,
    elapsed_s: int = 100,
    config_hash: str = "cfg_aaa",
) -> dict:
    return {
        "schema_version": "1.0",
        "problem_id": problem_id,
        "problem_name": problem_id,
        "tier": tier,
        "hardware": hardware,
        "date": date,
        "timestamp": f"{date}T12:00:00Z",
        "kernel_hash": "deadbeefcafe0000",
        "kernel_source_path": "kernels/deadbeefcafe0000.py",
        "model": "claude-sonnet-4-6",
        "speedup": speedup,
        "sol_score": sol,
        "compute_util": 20.0,
        "bandwidth_util": 40.0,
        "bottleneck_type": "memory-bound",
        "correct": correct,
        "cost_usd": cost_usd,
        "elapsed_s": elapsed_s,
        "stop_reason": "Target reached",
        "config_hash": config_hash,
        "rounds": 2,
        "iterations": 5,
    }


def _write(root: Path, record: dict) -> None:
    date_dir = root / record["date"]
    date_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{record['problem_id']}_{record['hardware']}_{record['config_hash']}.json"
    (date_dir / fname).write_text(json.dumps(record), encoding="utf-8")


def test_aggregation_on_synthetic(tmp_path: Path) -> None:
    """Hand-computed aggregation over 10 synthetic L40S records."""
    # L1: 3 correct (sol 0.1, 0.4, 0.7, speedup 1.0, 2.0, 4.0) + 1 incorrect
    _write(tmp_path, _record("l1_a", "L1", sol=0.1, speedup=1.0, config_hash="cfg_a"))
    _write(tmp_path, _record("l1_b", "L1", sol=0.4, speedup=2.0, config_hash="cfg_b"))
    _write(tmp_path, _record("l1_c", "L1", sol=0.7, speedup=4.0, config_hash="cfg_c"))
    _write(
        tmp_path,
        _record("l1_d", "L1", sol=0.0, speedup=1.0, correct=False, config_hash="cfg_d"),
    )

    # L2: 2 correct (sol 0.2, 0.5)
    _write(tmp_path, _record("l2_a", "L2", sol=0.2, speedup=1.5, config_hash="cfg_e"))
    _write(tmp_path, _record("l2_b", "L2", sol=0.5, speedup=3.0, config_hash="cfg_f"))

    # GPU_MODE: 3 correct (sol 0.3, 0.35, 0.8) + 1 incorrect
    _write(tmp_path, _record("gm_a", "GPU_MODE", sol=0.3, speedup=2.0, config_hash="cfg_g"))
    _write(tmp_path, _record("gm_b", "GPU_MODE", sol=0.35, speedup=2.5, config_hash="cfg_h"))
    _write(tmp_path, _record("gm_c", "GPU_MODE", sol=0.8, speedup=5.0, config_hash="cfg_i"))
    _write(
        tmp_path,
        _record("gm_d", "GPU_MODE", sol=0.0, speedup=1.0, correct=False, config_hash="cfg_j"),
    )

    out = score_suite.score(
        date="2026-04-22",
        hardware="L40S",
        tier="ALL",
        root=tmp_path,
    )

    assert out["date"] == "2026-04-22"
    assert out["hardware"] == "L40S"
    assert out["cost_usd_total"] == 10 * 0.5
    assert out["elapsed_s_total"] == 10 * 100

    l1 = out["tiers"]["L1"]
    assert l1["problem_count"] == 100
    assert l1["attempted"] == 4
    assert l1["correct"] == 3
    assert math.isclose(l1["correctness_rate"], 3 / 4)
    # 2 correct L1 records have sol >= 0.3 (0.4, 0.7); denom is L1 problem_count=100
    assert math.isclose(l1["solved_rate_at_sol_0.3"], 2 / 100)
    # mean of [0.1, 0.4, 0.7] = 0.4
    assert math.isclose(l1["mean_sol"], 0.4)
    # median of [0.1, 0.4, 0.7] = 0.4
    assert math.isclose(l1["median_sol"], 0.4)
    # geomean of [1.0, 2.0, 4.0] = 2.0
    assert math.isclose(l1["geomean_speedup"], 2.0)

    l2 = out["tiers"]["L2"]
    assert l2["attempted"] == 2
    assert l2["correct"] == 2
    # 1 correct L2 record has sol >= 0.3 (0.5); denom 100
    assert math.isclose(l2["solved_rate_at_sol_0.3"], 1 / 100)
    assert math.isclose(l2["mean_sol"], (0.2 + 0.5) / 2)

    gm = out["tiers"]["GPU_MODE"]
    assert gm["problem_count"] == 50
    assert gm["attempted"] == 4
    assert gm["correct"] == 3
    # All 3 correct GPU_MODE records have sol >= 0.3 (0.3, 0.35, 0.8); denom 50
    assert math.isclose(gm["solved_rate_at_sol_0.3"], 3 / 50)

    all_t = out["tiers"]["ALL"]
    assert all_t["problem_count"] == 250  # 100 + 100 + 50
    assert all_t["attempted"] == 10
    assert all_t["correct"] == 8
    # 6 correct records across tiers have sol >= 0.3: L1(0.4,0.7)=2, L2(0.5)=1, GM(0.3,0.35,0.8)=3
    assert math.isclose(all_t["solved_rate_at_sol_0.3"], 6 / 250)


def test_empty_tier_null_metrics(tmp_path: Path) -> None:
    """Tier with zero records must emit null (not 0/NaN) for quality metrics."""
    out = score_suite.score(
        date="2026-04-22",
        hardware="L40S",
        tier="ALL",
        root=tmp_path,
    )

    assert "note" in out
    for tier_name in ("L1", "L2", "GPU_MODE", "ALL"):
        t = out["tiers"][tier_name]
        assert t["attempted"] == 0
        assert t["correct"] == 0
        assert t["correctness_rate"] == 0.0
        assert t["solved_rate_at_sol_0.3"] == 0.0
        assert t["mean_sol"] is None
        assert t["median_sol"] is None
        assert t["p90_sol"] is None
        assert t["geomean_speedup"] is None

    assert out["cost_usd_total"] == 0.0
    assert out["elapsed_s_total"] == 0
