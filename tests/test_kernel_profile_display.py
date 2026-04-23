"""Tests for dual-display surfaces: kernel_profile, live_display, run_log,
sol_plots.

Phase 2 convention:
- SOL primary when ``profile["sol_score"] > 0``; speedup secondary (dimmed).
- Speedup-only fallback with explicit "SOL unknown" tag when SOL is absent.
- Never SOL-in-isolation — speedup is always shown alongside SOL.

Fast, hermetic: captures Rich Console output in memory. No Modal, no GPU.
"""

from __future__ import annotations

import io

from rich.console import Console

from kernel_code.kernel_profile import render_kernel_profile
from kernel_code.live_display import LiveOptimizationDisplay, _iteration_log_compact
from kernel_code.run_log import RunLogger
from kernel_code.sol_plots import render_sol_trajectory, render_strategy_timeline


def _render_profile(**kwargs: object) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, color_system=None)
    render_kernel_profile(console=console, **kwargs)  # type: ignore[arg-type]
    return buf.getvalue()


# ────────────────────────────────────────────────────────────
# kernel_profile.render_kernel_profile
# ────────────────────────────────────────────────────────────

def test_kernel_profile_sol_primary_when_present() -> None:
    out = _render_profile(
        speedup=1.8,
        ref_runtime_us=1000,
        kernel_runtime_us=556,
        profile={"sol_score": 0.42, "bottleneck_type": "memory_bound"},
    )
    assert "SOL" in out
    assert "0.42" in out
    assert "42% of hardware peak" in out
    assert "memory-bound" in out
    # Speedup is still present as secondary.
    assert "1.80x" in out


def test_kernel_profile_falls_back_to_speedup_when_sol_missing() -> None:
    out = _render_profile(
        speedup=1.8,
        ref_runtime_us=1000,
        kernel_runtime_us=556,
        profile={"bottleneck_type": "memory_bound"},
    )
    # No SOL headline row.
    assert "of hardware peak" not in out
    # Speedup is shown with explicit SOL-unknown tag.
    assert "1.80x" in out
    assert "SOL unknown" in out


def test_kernel_profile_treats_zero_sol_score_as_missing() -> None:
    out = _render_profile(
        speedup=1.8,
        ref_runtime_us=1000,
        kernel_runtime_us=556,
        profile={"sol_score": 0.0},
    )
    assert "of hardware peak" not in out
    assert "1.80x" in out
    assert "SOL unknown" in out


def test_kernel_profile_baseline_mode_not_affected() -> None:
    # Baseline renders a reference line only — no speedup/SOL headlines.
    out = _render_profile(
        speedup=0.0,
        ref_runtime_us=1234,
        profile={},
        is_baseline=True,
    )
    assert "Reference" in out
    assert "1234" in out
    # Baseline must NOT emit either headline.
    assert "Speedup" not in out
    assert "SOL unknown" not in out


def test_kernel_profile_never_emits_sol_in_isolation() -> None:
    out = _render_profile(
        speedup=1.8,
        ref_runtime_us=1000,
        kernel_runtime_us=556,
        profile={"sol_score": 0.42},
    )
    assert "1.80x" in out, "speedup must always accompany SOL"


# ────────────────────────────────────────────────────────────
# live_display._iteration_log_compact
# ────────────────────────────────────────────────────────────

def test_live_display_history_dual_format_when_sol_present() -> None:
    iterations = [
        {"num": 1, "speedup": 1.8, "status": "keep", "intent": "tile",
         "is_best": True, "sol_score": 0.42},
    ]
    lines = _iteration_log_compact(iterations, width=80)
    rendered = "\n".join(line.plain for line in lines)
    assert "SOL 0.42" in rendered
    assert "1.80x" in rendered


def test_live_display_history_speedup_only_when_sol_missing() -> None:
    iterations = [
        {"num": 1, "speedup": 1.8, "status": "keep", "intent": "tile",
         "is_best": True, "sol_score": 0.0},
    ]
    lines = _iteration_log_compact(iterations, width=80)
    rendered = "\n".join(line.plain for line in lines)
    assert "SOL" not in rendered
    assert "1.80x" in rendered


def test_live_display_band1_best_line_sol_primary() -> None:
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, color_system=None)
    disp = LiveOptimizationDisplay(console=console, problem="matmul", max_iterations=3)
    disp._start_time = 0.0
    disp._best_speedup = 1.8
    disp._current_round = 2
    disp._iterations = [
        {"num": 1, "speedup": 1.8, "status": "keep", "intent": "x",
         "is_best": True, "sol_score": 0.42},
    ]
    group = disp._build()
    console.print(group)
    out = buf.getvalue()
    assert "SOL 0.42" in out
    # Speedup rendered as dim secondary — must still appear in the output.
    assert "1.80x" in out


def test_live_display_band1_best_line_speedup_fallback_when_no_sol() -> None:
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, color_system=None)
    disp = LiveOptimizationDisplay(console=console, problem="matmul", max_iterations=3)
    disp._start_time = 0.0
    disp._best_speedup = 1.8
    disp._current_round = 2
    disp._iterations = [
        {"num": 1, "speedup": 1.8, "status": "keep", "intent": "x",
         "is_best": True, "sol_score": 0.0},
    ]
    group = disp._build()
    console.print(group)
    out = buf.getvalue()
    # No SOL in BAND 1 rollup.
    assert "SOL 0.00" not in out
    assert "1.80x" in out


# ────────────────────────────────────────────────────────────
# run_log.RunLogger.end_run
# ────────────────────────────────────────────────────────────

def test_run_log_result_block_sol_primary(tmp_path) -> None:
    logger = RunLogger()
    # Point the log writer at a tmp dir by redirecting the module's _RUNS_DIR.
    import kernel_code.run_log as rl
    orig = rl._RUNS_DIR
    rl._RUNS_DIR = tmp_path
    try:
        logger.start_run(command="/optimize test.py",
                         config={"model": "m", "hardware": "L40S", "backend": "triton"})
        logger.log_iteration(1, speedup=1.8, status="keep", intent="tile",
                             profile={"sol_score": 0.42})
        logger.end_run(best_speedup=1.8, best_sol=0.42, stop_reason="converged")
    finally:
        rl._RUNS_DIR = orig

    log_text = logger.log_path.read_text()  # type: ignore[union-attr]
    assert "Best SOL:" in log_text
    assert "0.42" in log_text
    assert "Best speedup:" in log_text
    assert "1.80x" in log_text
    # Per-iteration line keeps the SOL+speedup combined format.
    assert "SOL 0.42" in log_text


def test_run_log_result_block_falls_back_when_sol_missing(tmp_path) -> None:
    logger = RunLogger()
    import kernel_code.run_log as rl
    orig = rl._RUNS_DIR
    rl._RUNS_DIR = tmp_path
    try:
        logger.start_run(command="/optimize test.py",
                         config={"model": "m", "hardware": "L40S", "backend": "triton"})
        logger.log_iteration(1, speedup=1.8, status="keep", intent="tile",
                             profile=None)
        logger.end_run(best_speedup=1.8, stop_reason="converged")
    finally:
        rl._RUNS_DIR = orig

    log_text = logger.log_path.read_text()  # type: ignore[union-attr]
    assert "Best SOL:" not in log_text
    assert "Best speedup:" in log_text
    assert "1.80x" in log_text
    assert "SOL unknown" in log_text


def test_run_log_autoderives_best_sol_from_iterations(tmp_path) -> None:
    logger = RunLogger()
    import kernel_code.run_log as rl
    orig = rl._RUNS_DIR
    rl._RUNS_DIR = tmp_path
    try:
        logger.start_run(command="/optimize t.py",
                         config={"model": "m", "hardware": "L40S", "backend": "triton"})
        logger.log_iteration(1, speedup=1.2, status="discard", intent="a",
                             profile={"sol_score": 0.20})
        logger.log_iteration(2, speedup=1.8, status="keep", intent="b",
                             profile={"sol_score": 0.55})
        # Caller did not pass best_sol — logger derives it.
        logger.end_run(best_speedup=1.8)
    finally:
        rl._RUNS_DIR = orig

    log_text = logger.log_path.read_text()  # type: ignore[union-attr]
    assert "Best SOL:" in log_text
    assert "0.55" in log_text


# ────────────────────────────────────────────────────────────
# sol_plots — confirm no 0.5*speedup fallback
# ────────────────────────────────────────────────────────────

def test_sol_trajectory_reads_profile_sol_score_not_speedup_fallback() -> None:
    # speedup present, but no sol_score in profile → trajectory must show
    # SOL unavailable (rather than synthesising sol = 0.5 * speedup).
    rounds = [
        {"round": 1, "strategy": "initial", "speedup": 2.0, "profile": {}},
        {"round": 2, "strategy": "initial", "speedup": 3.0, "profile": {}},
    ]
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, color_system=None)
    console.print(render_sol_trajectory(rounds, width=40, height=7))
    out = buf.getvalue()
    assert "SOL unavailable" in out


def test_sol_trajectory_uses_measured_sol_when_present() -> None:
    rounds = [
        {"round": 1, "strategy": "initial", "speedup": 2.0,
         "profile": {"sol_score": 0.42}},
    ]
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, color_system=None)
    console.print(render_sol_trajectory(rounds, width=40, height=7))
    out = buf.getvalue()
    # Shows a measured-SOL header, not the "SOL unavailable" fallback.
    assert "0.42" in out
    assert "SOL unavailable" not in out


def test_strategy_timeline_sol_primary_when_any_stage_has_sol() -> None:
    rounds = [
        {"round": 1, "strategy": "initial", "speedup": 1.2,
         "profile": {"sol_score": 0.20}},
        {"round": 2, "strategy": "tiled", "speedup": 1.8,
         "profile": {"sol_score": 0.55}},
    ]
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, color_system=None)
    console.print(render_strategy_timeline(rounds, width=60))
    out = buf.getvalue()
    assert "SOL:" in out
    assert "0.55" in out
    # Secondary speedup annotation.
    assert "1.80x speedup vs reference" in out


def test_strategy_timeline_speedup_only_when_no_stage_has_sol() -> None:
    rounds = [
        {"round": 1, "strategy": "initial", "speedup": 1.2, "profile": {}},
        {"round": 2, "strategy": "tiled", "speedup": 1.8, "profile": {}},
    ]
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, color_system=None)
    console.print(render_strategy_timeline(rounds, width=60))
    out = buf.getvalue()
    assert "SOL:" not in out
    assert "1.80x" in out or "1.8x" in out
    assert "SOL unknown" in out
