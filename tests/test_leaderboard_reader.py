"""Tests for openkernel.benchmarks.leaderboard_reader.

Runnable via:
    python -m pytest tests/test_leaderboard_reader.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from openkernel.benchmarks.leaderboard_reader import (  # noqa: E402
    filter_records,
    latest_per_problem,
    load_all,
)


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


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_load_all_skips_underscore_files(tmp_path: Path) -> None:
    _write(tmp_path / "rec1.json", _base_record(problem_id="kb_l1_0001"))
    _write(tmp_path / "_example.json", _base_record(problem_id="kb_l1_9999"))

    records = load_all(tmp_path)

    assert len(records) == 1
    assert records[0]["problem_id"] == "kb_l1_0001"


def test_filter_by_tier() -> None:
    records = [
        _base_record(tier="L1", problem_id="a"),
        _base_record(tier="L1", problem_id="b"),
        _base_record(tier="L2", problem_id="c"),
    ]

    got = filter_records(records, tier="L1")

    assert len(got) == 2
    assert {r["problem_id"] for r in got} == {"a", "b"}


def test_filter_by_date_range() -> None:
    records = [
        _base_record(problem_id="early", date="2026-04-20"),
        _base_record(problem_id="mid", date="2026-04-25"),
        _base_record(problem_id="late", date="2026-05-02"),
    ]

    got = filter_records(records, date_range=("2026-04-21", "2026-04-30"))

    assert [r["problem_id"] for r in got] == ["mid"]


def test_filter_correct_only_default() -> None:
    records = [
        _base_record(problem_id="ok", correct=True),
        _base_record(problem_id="broken", correct=False),
    ]

    default = filter_records(records)
    explicit_off = filter_records(records, correct_only=False)

    assert [r["problem_id"] for r in default] == ["ok"]
    assert len(explicit_off) == 2


def test_latest_per_problem_dedupes() -> None:
    records = [
        _base_record(
            problem_id="kb_l1_0001",
            hardware="L40S",
            config_hash="cfg_abc123",
            timestamp="2026-04-22T10:00:00Z",
            speedup=1.5,
        ),
        _base_record(
            problem_id="kb_l1_0001",
            hardware="L40S",
            config_hash="cfg_abc123",
            timestamp="2026-04-23T10:00:00Z",
            speedup=2.1,
        ),
    ]

    got = latest_per_problem(records)

    assert len(got) == 1
    assert got[0]["speedup"] == 2.1
    assert got[0]["timestamp"] == "2026-04-23T10:00:00Z"


def test_missing_schema_version_treated_as_0_0(
    tmp_path: Path, caplog: logging.LogRecordFactory
) -> None:
    record = _base_record()
    del record["schema_version"]
    _write(tmp_path / "legacy.json", record)

    with caplog.at_level(logging.WARNING, logger="openkernel.benchmarks.leaderboard_reader"):
        records = load_all(tmp_path)

    # Phase 1 has no migration path defined for major 0, so it is skipped
    # with a warning (per reader contract). The missing-version warning
    # must still fire.
    assert records == []
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("missing schema_version" in m for m in warnings)


def test_unknown_major_skipped(
    tmp_path: Path, caplog: logging.LogRecordFactory
) -> None:
    _write(tmp_path / "future.json", _base_record(schema_version="99.0"))

    with caplog.at_level(logging.WARNING, logger="openkernel.benchmarks.leaderboard_reader"):
        records = load_all(tmp_path)

    assert records == []
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("unknown major" in m for m in warnings)
