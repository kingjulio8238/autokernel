"""Tests for the Option B table layout in LiveOptimizationDisplay.

Covers:
- default layout is table, legacy is opt-in via OPENKERNEL_LEGACY_DISPLAY=1
- rows are emitted exactly once per worker when status transitions to
  ``passed``, even under repeated polling
- per-worker runtime is derived from the baseline (baseline / speedup)
- SOL score latched from ``update_iteration`` lands on the most recent
  row that's still missing one
- workers still in-flight at ``finish`` get a terminal ``cancelled`` /
  ``stopped`` row so every worker is represented

These tests avoid starting the Rich.Live context by calling the
``update_*`` / ``finish`` methods directly on a display instance.
"""

from __future__ import annotations

import pytest

from rich.console import Console


def _mk_display(monkeypatch: pytest.MonkeyPatch, legacy: bool = False):
    from kernel_code.live_display import LiveOptimizationDisplay

    if legacy:
        monkeypatch.setenv("OPENKERNEL_LEGACY_DISPLAY", "1")
    else:
        monkeypatch.delenv("OPENKERNEL_LEGACY_DISPLAY", raising=False)

    # Silent console so tests don't emit to stdout.
    console = Console(quiet=True, record=True, width=100)
    d = LiveOptimizationDisplay(
        console=console,
        problem="reference.py",
        hardware="L40S",
        backend="triton",
    )
    # Bypass .start() — we don't want a real Live context in tests.
    import time as _t
    d._start_time = _t.time()
    return d


def test_table_layout_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    d = _mk_display(monkeypatch, legacy=False)
    assert d._use_table_layout is True


def test_legacy_layout_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    d = _mk_display(monkeypatch, legacy=True)
    assert d._use_table_layout is False


def test_passed_worker_emits_exactly_one_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated polling (typical 2 Hz) must not duplicate rows."""
    d = _mk_display(monkeypatch)
    d.set_baseline(2730.0)

    passed = [{"id": 2, "global_id": 2, "round": 3, "max_rounds": 5,
               "status": "passed", "speedup": 47.47}]

    for _ in range(5):
        d.update_workers(passed)

    assert len(d._worker_rows) == 1
    row = d._worker_rows[0]
    assert row["global_id"] == 2
    assert row["status"] == "passed"
    assert row["speedup"] == pytest.approx(47.47)


def test_runtime_derived_from_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    d = _mk_display(monkeypatch)
    d.set_baseline(2730.0)

    d.update_workers([{"id": 0, "global_id": 0, "round": 1, "max_rounds": 5,
                       "status": "passed", "speedup": 27.3}])

    row = d._worker_rows[0]
    # 2730 / 27.3 == 100
    assert row["runtime_us"] == pytest.approx(100.0)


def test_runtime_is_none_without_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    d = _mk_display(monkeypatch)
    # no set_baseline()
    d.update_workers([{"id": 0, "global_id": 0, "round": 1, "max_rounds": 5,
                       "status": "passed", "speedup": 5.0}])
    assert d._worker_rows[0]["runtime_us"] is None


def test_sol_score_latched_onto_latest_passed_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _mk_display(monkeypatch)
    d.set_baseline(1000.0)

    d.update_workers([{"id": 0, "global_id": 0, "round": 1, "max_rounds": 5,
                       "status": "passed", "speedup": 2.0}])
    d.update_workers([{"id": 1, "global_id": 1, "round": 1, "max_rounds": 5,
                       "status": "passed", "speedup": 4.0}])

    # update_iteration carries the SOL from the Modal re-eval of the round
    # winner. It should light up the most-recently-emitted passing row.
    d.update_iteration(num=1, speedup=4.0, status="keep", intent="round 1",
                       sol_score=0.72)

    assert d._worker_rows[0]["sol_score"] is None
    assert d._worker_rows[1]["sol_score"] == pytest.approx(0.72)


def test_waiting_workers_get_cancelled_row_at_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _mk_display(monkeypatch)
    d.set_baseline(2730.0)

    d.update_workers([
        {"id": 0, "global_id": 0, "round": 3, "max_rounds": 5,
         "status": "passed", "speedup": 10.0},
        {"id": 1, "global_id": 1, "round": 2, "max_rounds": 5,
         "status": "working", "speedup": 0.0},
        {"id": 2, "global_id": 2, "round": 0, "max_rounds": 5,
         "status": "waiting", "speedup": 0.0},
    ])
    d.finish()

    by_gid = {r["global_id"]: r for r in d._worker_rows}
    assert by_gid[0]["status"] == "passed"
    assert by_gid[1]["status"] == "cancelled"
    assert by_gid[2]["status"] == "cancelled"
    assert len(d._worker_rows) == 3


def test_stopped_worker_gets_stopped_row_at_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _mk_display(monkeypatch)
    d.update_workers([{"id": 0, "global_id": 0, "round": 4, "max_rounds": 5,
                       "status": "stopped", "speedup": 0.0}])
    d.finish()
    assert len(d._worker_rows) == 1
    assert d._worker_rows[0]["status"] == "stopped"


def test_build_table_view_renders_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: after pushing data, the rendered Group contains the
    expected per-row cells (worker label, runtime, SOL, speedup)."""
    d = _mk_display(monkeypatch)
    d.set_baseline(2730.0)
    d.update_workers([{"id": 2, "global_id": 2, "round": 3, "max_rounds": 5,
                       "status": "passed", "speedup": 47.47}])
    d.update_iteration(num=1, speedup=47.47, status="keep",
                       intent="general optimization", sol_score=1.00)

    rendered = d._build()
    console = Console(width=120, record=True)
    console.print(rendered)
    out = console.export_text()

    assert "W3" in out
    # 2730 / 47.47 ≈ 57.5 → rounds to "58 µs"
    assert "58" in out and "\u00b5s" in out
    assert "100%" in out
    assert "47.47" in out


def test_legacy_view_still_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opting back into the legacy layout must not blow up. We don't
    assert on exact chart contents — just that _build() produces a Group
    and no table rows get appended."""
    d = _mk_display(monkeypatch, legacy=True)
    d.set_baseline(2730.0)
    d.update_workers([{"id": 0, "global_id": 0, "round": 2, "max_rounds": 5,
                       "status": "passed", "speedup": 10.0}])
    d.update_iteration(num=1, speedup=10.0, status="keep",
                       intent="legacy", sol_score=0.5)

    # Table state must stay empty in legacy mode.
    assert d._worker_rows == []
    # And the renderer shouldn't crash.
    from rich.console import Group as _Group
    assert isinstance(d._build(), _Group)
