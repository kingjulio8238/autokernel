#!/usr/bin/env python3
"""Emit a publication-format markdown report from score_suite.py output.

Usage:
    python scripts/emit_report.py --date 2026-04-22 --hardware L40S
    python scripts/emit_report.py --date 2026-04-22 --hardware L40S --output results/reports/custom.md
    python scripts/emit_report.py --scores-json path/to/scores.json --date 2026-04-22 --hardware L40S

Reads scores either from a JSON file (``--scores-json``) or by shelling out to
``scripts/score_suite.py``. Writes a markdown report to
``results/reports/{date}_{hardware}.md`` unless ``--output`` is provided.

The expected score JSON schema is forgiving; missing fields default to N/A or
zero. Top-level keys read:

    {
      "date": "2026-04-22",
      "hardware": "L40S",
      "total_cost": 18.23,
      "total_wall_seconds": 7342,
      "headline": {
        "mean_sol_l1": 0.34,
        "solved_at_0_3_l1": 60,
        "attempted_l1": 42,
        "correct_l1": 38
      },
      "tiers": {
        "L1": {"total": 100, "attempted": 42, "correct": 38,
                "solved_at_0_3": 60, "mean_sol": 0.34, "p90_sol": 0.72,
                "geomean_speedup": 1.42},
        "L2": {...},
        "GPU_MODE": {...}
      }
    }
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Baseline reference points (see citations at bottom of report).
BASELINES = [
    {
        "system": "**kernel+ (this run)**",
        "l1_mean_sol": None,  # filled in at render time
        "l1_solved_at_0_3": None,
        "bold": True,
    },
    {
        "system": "Cursor multi-agent (B200)",
        "l1_mean_sol": "0.56 (median)",
        "l1_solved_at_0_3": "—",
    },
    {
        "system": "OpenAI o1",
        "l1_mean_sol": "—",
        "l1_solved_at_0_3": "10%",
    },
    {
        "system": "DeepSeek R1",
        "l1_mean_sol": "—",
        "l1_solved_at_0_3": "12%",
    },
    {
        "system": "Stanford SOTA",
        "l1_mean_sol": "—",
        "l1_solved_at_0_3": "<20%",
    },
]

CITATIONS = (
    "*Citations*: "
    "[Cursor](https://cursor.com/blog/multi-agent-kernels), "
    "[KernelBench paper](https://arxiv.org/abs/2502.10517), "
    "[Stanford leaderboard](https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/)"
)

TIER_DISPLAY = [
    ("L1", "KernelBench L1"),
    ("L2", "KernelBench L2"),
    ("GPU_MODE", "GPU MODE"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a markdown report from score_suite.py JSON output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date", required=True, help="Report date, YYYY-MM-DD.")
    parser.add_argument("--hardware", required=True, help="Hardware tag, e.g. L40S.")
    parser.add_argument(
        "--scores-json",
        type=Path,
        default=None,
        help="Path to pre-computed scores JSON. If omitted, runs score_suite.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to results/reports/{date}_{hardware}.md.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_PROJECT_ROOT / "results" / "reports",
        help="Directory for report discovery and writing (default: results/reports).",
    )
    return parser.parse_args(argv)


def load_scores(scores_json: Path | None, date: str, hardware: str) -> dict[str, Any]:
    """Load scores either from a JSON file or by invoking score_suite.py."""
    if scores_json is not None:
        return json.loads(scores_json.read_text(encoding="utf-8"))

    # Shell out to score_suite.py — loose coupling over import coupling while
    # the metrics agent is developing in parallel.
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "score_suite.py"),
        "--date",
        date,
        "--hardware",
        hardware,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def _fmt(val: Any, fmt: str = "{}", na: str = "N/A") -> str:
    if val is None:
        return na
    try:
        return fmt.format(val)
    except (ValueError, TypeError):
        return str(val)


def _pct(numer: Any, denom: Any) -> str:
    try:
        n, d = float(numer), float(denom)
        if d <= 0:
            return "N/A"
        return f"{int(round(100 * n / d))}%"
    except (ValueError, TypeError):
        return "N/A"


def render_headline(scores: dict[str, Any]) -> str:
    headline = scores.get("headline", {}) or {}
    tiers = scores.get("tiers", {}) or {}
    l1 = tiers.get("L1", {}) or {}

    mean_sol = headline.get("mean_sol_l1", l1.get("mean_sol"))
    solved = headline.get("solved_at_0_3_l1", l1.get("solved_at_0_3"))
    attempted = headline.get("attempted_l1", l1.get("attempted"))
    correct = headline.get("correct_l1", l1.get("correct"))
    total = l1.get("total", 100)

    total_cost = scores.get("total_cost")
    total_wall = scores.get("total_wall_seconds")

    lines = ["## Headline", ""]
    lines.append(
        f"- **Mean SOL on L1**: {_fmt(mean_sol, '{:.2f}')} "
        f"({_fmt(solved, '{}')}/{total} solved @ SOL ≥ 0.3, "
        f"{_fmt(attempted, '{}')} attempted, {_fmt(correct, '{}')} correct)"
    )
    cost_str = _fmt(total_cost, "${:.2f}")
    wall_str = _fmt(total_wall, "{}s")
    lines.append(f"- **Total cost**: {cost_str} across {wall_str} wall")
    return "\n".join(lines)


def render_tier_table(tier_key: str, display_name: str, tier: dict[str, Any]) -> str:
    total = tier.get("total", "N/A")
    attempted = tier.get("attempted")
    correct = tier.get("correct")
    solved_at_0_3 = tier.get("solved_at_0_3")
    mean_sol = tier.get("mean_sol")
    p90_sol = tier.get("p90_sol")
    geomean_speedup = tier.get("geomean_speedup")

    correct_pct = _pct(correct, attempted)
    solved_pct = _pct(solved_at_0_3, total)

    lines = [
        f"### {display_name} ({total} problems)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Attempted | {_fmt(attempted)} |",
        f"| Correct | {_fmt(correct)} ({correct_pct}) |",
        f"| Solved @ SOL ≥ 0.3 | {_fmt(solved_at_0_3)}/{total} ({solved_pct}) |",
        f"| Mean SOL (correct) | {_fmt(mean_sol, '{:.2f}')} |",
        f"| P90 SOL | {_fmt(p90_sol, '{:.2f}')} |",
        f"| Geomean speedup | {_fmt(geomean_speedup, '{:.2f}×')} |",
    ]
    return "\n".join(lines)


def render_per_tier(scores: dict[str, Any]) -> str:
    tiers = scores.get("tiers", {}) or {}
    parts = ["## Per-tier breakdown", ""]
    for key, display in TIER_DISPLAY:
        tier = tiers.get(key, {}) or {}
        parts.append(render_tier_table(key, display, tier))
        parts.append("")
    return "\n".join(parts).rstrip()


def render_baseline_table(scores: dict[str, Any]) -> str:
    tiers = scores.get("tiers", {}) or {}
    l1 = tiers.get("L1", {}) or {}
    headline = scores.get("headline", {}) or {}
    total = l1.get("total", 100)

    mean_sol = headline.get("mean_sol_l1", l1.get("mean_sol"))
    solved = headline.get("solved_at_0_3_l1", l1.get("solved_at_0_3"))
    solved_pct = _pct(solved, total)

    lines = [
        "## Baseline comparison",
        "",
        "| System | L1 Mean SOL | L1 Solved @ SOL ≥ 0.3 |",
        "|---|---|---|",
    ]
    for b in BASELINES:
        if b.get("bold"):
            mean_str = f"**{_fmt(mean_sol, '{:.2f}')}**"
            solved_str = f"**{solved_pct}**"
            lines.append(f"| {b['system']} | {mean_str} | {solved_str} |")
        else:
            lines.append(
                f"| {b['system']} | {b['l1_mean_sol']} | {b['l1_solved_at_0_3']} |"
            )
    lines.append("")
    lines.append(CITATIONS)
    return "\n".join(lines)


# Regex matches the headline line we emit ourselves — used for trend extraction
# from prior reports. Tolerant of formatting tweaks: requires "Mean SOL on L1"
# label followed by a number.
_HEADLINE_MEAN_SOL_RE = re.compile(
    r"Mean SOL on L1\*?\*?\s*:\s*\*?\*?\s*([0-9]+\.[0-9]+)",
)
_HEADLINE_SOLVED_RE = re.compile(
    r"Mean SOL on L1[^\n]*?\(\s*([0-9]+)\s*/\s*([0-9]+)\s+solved",
)


def find_prior_report(
    reports_dir: Path, hardware: str, today: str
) -> Path | None:
    """Return the most recent prior report for ``hardware`` before ``today``."""
    if not reports_dir.exists():
        return None
    candidates: list[tuple[str, Path]] = []
    for p in reports_dir.glob(f"*_{hardware}.md"):
        # Filename format: {date}_{hardware}.md
        stem = p.stem  # drops .md
        if not stem.endswith(f"_{hardware}"):
            continue
        date_part = stem[: -len(f"_{hardware}")]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_part):
            continue
        if date_part >= today:
            continue
        candidates.append((date_part, p))
    if not candidates:
        return None
    candidates.sort()  # ISO dates sort lexicographically
    return candidates[-1][1]


def extract_prior_metrics(prior_path: Path) -> dict[str, Any]:
    """Parse headline numbers from a prior report's markdown."""
    text = prior_path.read_text(encoding="utf-8")
    metrics: dict[str, Any] = {}
    m = _HEADLINE_MEAN_SOL_RE.search(text)
    if m:
        metrics["mean_sol_l1"] = float(m.group(1))
    m = _HEADLINE_SOLVED_RE.search(text)
    if m:
        metrics["solved_at_0_3_l1"] = int(m.group(1))
        metrics["total_l1"] = int(m.group(2))
    return metrics


def render_trend(
    scores: dict[str, Any], reports_dir: Path, hardware: str, today: str
) -> str:
    prior = find_prior_report(reports_dir, hardware, today)
    lines = ["## Day-over-day trend", ""]
    if prior is None:
        lines.append("(first sweep — no prior comparison available)")
        return "\n".join(lines)

    prior_date = prior.stem[: -len(f"_{hardware}")]
    prior_metrics = extract_prior_metrics(prior)

    tiers = scores.get("tiers", {}) or {}
    headline = scores.get("headline", {}) or {}
    l1 = tiers.get("L1", {}) or {}
    today_mean = headline.get("mean_sol_l1", l1.get("mean_sol"))
    today_solved = headline.get("solved_at_0_3_l1", l1.get("solved_at_0_3"))

    def _delta(today: Any, prior: Any, fmt: str = "{:+.2f}") -> str:
        try:
            return fmt.format(float(today) - float(prior))
        except (ValueError, TypeError):
            return "N/A"

    lines.append(f"| Metric | Today | Prior ({prior_date}) | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Mean SOL (L1) | {_fmt(today_mean, '{:.2f}')} | "
        f"{_fmt(prior_metrics.get('mean_sol_l1'), '{:.2f}')} | "
        f"{_delta(today_mean, prior_metrics.get('mean_sol_l1'))} |"
    )
    lines.append(
        f"| Solved @ 0.3 (L1) | {_fmt(today_solved)} | "
        f"{_fmt(prior_metrics.get('solved_at_0_3_l1'))} | "
        f"{_delta(today_solved, prior_metrics.get('solved_at_0_3_l1'), '{:+d}') if isinstance(today_solved, int) and isinstance(prior_metrics.get('solved_at_0_3_l1'), int) else 'N/A'} |"
    )
    return "\n".join(lines)


def render_report(
    scores: dict[str, Any],
    date: str,
    hardware: str,
    reports_dir: Path,
) -> str:
    sections = [
        f"# Phase 1 Sweep — {date} · {hardware}",
        "",
        render_headline(scores),
        "",
        render_per_tier(scores),
        "",
        render_baseline_table(scores),
        "",
        render_trend(scores, reports_dir, hardware, date),
        "",
    ]
    return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scores = load_scores(args.scores_json, args.date, args.hardware)

    reports_dir: Path = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    output: Path = args.output or (reports_dir / f"{args.date}_{args.hardware}.md")
    output.parent.mkdir(parents=True, exist_ok=True)

    report = render_report(scores, args.date, args.hardware, reports_dir)
    output.write_text(report, encoding="utf-8")
    print(str(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
