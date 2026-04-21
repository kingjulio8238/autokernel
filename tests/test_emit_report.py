"""Tests for scripts/emit_report.py.

Runnable via:
    python -m pytest tests/test_emit_report.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_emit_report():
    """Load scripts/emit_report.py as a module without needing scripts/ to be a package."""
    path = _REPO_ROOT / "scripts" / "emit_report.py"
    spec = importlib.util.spec_from_file_location("emit_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_scores() -> dict:
    return {
        "date": "2026-04-22",
        "hardware": "L40S",
        "total_cost": 18.23,
        "total_wall_seconds": 7342,
        "headline": {
            "mean_sol_l1": 0.34,
            "solved_at_0_3_l1": 60,
            "attempted_l1": 42,
            "correct_l1": 38,
        },
        "tiers": {
            "L1": {
                "total": 100,
                "attempted": 42,
                "correct": 38,
                "solved_at_0_3": 60,
                "mean_sol": 0.34,
                "p90_sol": 0.72,
                "geomean_speedup": 1.42,
            },
            "L2": {
                "total": 100,
                "attempted": 20,
                "correct": 18,
                "solved_at_0_3": 25,
                "mean_sol": 0.22,
                "p90_sol": 0.55,
                "geomean_speedup": 1.15,
            },
            "GPU_MODE": {
                "total": 8,
                "attempted": 8,
                "correct": 6,
                "solved_at_0_3": 4,
                "mean_sol": 0.40,
                "p90_sol": 0.68,
                "geomean_speedup": 1.30,
            },
        },
    }


def test_emit_report_produces_expected_structure(tmp_path):
    emit_report = _load_emit_report()

    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(_synthetic_scores()), encoding="utf-8")

    output_path = tmp_path / "report.md"
    reports_dir = tmp_path / "reports"

    rc = emit_report.main(
        [
            "--date",
            "2026-04-22",
            "--hardware",
            "L40S",
            "--scores-json",
            str(scores_path),
            "--output",
            str(output_path),
            "--reports-dir",
            str(reports_dir),
        ]
    )
    assert rc == 0
    assert output_path.exists()

    text = output_path.read_text(encoding="utf-8")

    # Title
    assert "# Phase 1 Sweep — 2026-04-22 · L40S" in text

    # Headline
    assert "## Headline" in text
    assert "Mean SOL on L1" in text
    assert "0.34" in text
    assert "60/100 solved" in text
    assert "$18.23" in text
    assert "7342s" in text

    # Per-tier breakdown
    assert "## Per-tier breakdown" in text
    assert "### KernelBench L1 (100 problems)" in text
    assert "### KernelBench L2 (100 problems)" in text
    assert "### GPU MODE (8 problems)" in text
    assert "| Attempted | 42 |" in text
    assert "| P90 SOL | 0.72 |" in text
    assert "1.42×" in text

    # Baseline comparison
    assert "## Baseline comparison" in text
    assert "Cursor multi-agent (H200)" in text
    assert "0.56 (median)" in text
    assert "OpenAI o1" in text
    assert "10%" in text
    assert "DeepSeek R1" in text
    assert "12%" in text
    assert "Stanford SOTA" in text
    assert "<20%" in text
    assert "**kernel+ (this run)**" in text

    # Citations (inline links, not fabricated)
    assert "https://cursor.com/blog/multi-agent-kernels" in text
    assert "https://arxiv.org/abs/2502.10517" in text
    assert "scalingintelligence.stanford.edu/KernelBenchLeaderboard" in text

    # Trend — no prior report should yield first-sweep marker
    assert "## Day-over-day trend" in text
    assert "first sweep" in text


def test_emit_report_trend_with_prior_report(tmp_path):
    emit_report = _load_emit_report()

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    # Seed a prior report the module can parse.
    prior = reports_dir / "2026-04-20_L40S.md"
    prior.write_text(
        "# Phase 1 Sweep — 2026-04-20 · L40S\n\n"
        "## Headline\n\n"
        "- **Mean SOL on L1**: 0.29 (55/100 solved @ SOL ≥ 0.3, 40 attempted, 36 correct)\n"
        "- **Total cost**: $12.10 across 6000s wall\n",
        encoding="utf-8",
    )

    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(_synthetic_scores()), encoding="utf-8")

    output_path = reports_dir / "2026-04-22_L40S.md"

    rc = emit_report.main(
        [
            "--date",
            "2026-04-22",
            "--hardware",
            "L40S",
            "--scores-json",
            str(scores_path),
            "--output",
            str(output_path),
            "--reports-dir",
            str(reports_dir),
        ]
    )
    assert rc == 0
    text = output_path.read_text(encoding="utf-8")

    assert "Prior (2026-04-20)" in text
    assert "0.29" in text
    assert "+0.05" in text  # 0.34 - 0.29
    assert "first sweep" not in text


def test_find_prior_report_ignores_other_hardware(tmp_path):
    emit_report = _load_emit_report()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "2026-04-20_H100.md").write_text("x", encoding="utf-8")
    (reports_dir / "2026-04-21_L40S.md").write_text("x", encoding="utf-8")
    (reports_dir / "2026-04-22_L40S.md").write_text("x", encoding="utf-8")  # today — excluded

    prior = emit_report.find_prior_report(reports_dir, "L40S", "2026-04-22")
    assert prior is not None
    assert prior.name == "2026-04-21_L40S.md"

    none_prior = emit_report.find_prior_report(reports_dir, "A100", "2026-04-22")
    assert none_prior is None
