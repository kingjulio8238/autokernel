#!/usr/bin/env python3
"""Compare today's leaderboard vs prior N days and flag SOL regressions.

Exits 1 if any regressions detected (CI-hookable); 0 otherwise.

Usage:
    python scripts/check_regressions.py --date 2026-04-22 --hardware L40S
    python scripts/check_regressions.py --date 2026-04-22 --hardware L40S --lookback 7 --threshold 0.05
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openkernel.benchmarks.leaderboard_reader import filter_records, load_all


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect SOL regressions on today's leaderboard vs the last N days.",
    )
    parser.add_argument("--date", required=True, help="Today's date (YYYY-MM-DD).")
    parser.add_argument(
        "--hardware",
        required=True,
        choices=["L40S", "H100", "A100-80GB", "B200"],
        help="Hardware target to analyze.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=7,
        help="Number of prior days to compare against (default 7).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Flag if today's SOL dropped by more than this from the N-day max (default 0.05).",
    )
    parser.add_argument(
        "--leaderboard-root",
        type=str,
        default=None,
        help="Override leaderboard root directory (default: results/leaderboard/).",
    )
    return parser.parse_args(argv)


def _iso_date(s: str) -> date:
    return date.fromisoformat(s)


def _best_prior_by_problem(records: list[dict]) -> dict[str, dict]:
    """For each problem_id, pick the record with the highest sol_score.

    Only `correct == true` records are considered — incorrect priors
    cannot serve as a baseline for perf regression. Correctness-baseline
    tracking uses a separate pass.
    """
    best: dict[str, dict] = {}
    for r in records:
        pid = r.get("problem_id", "")
        if not pid:
            continue
        if r.get("correct") is not True:
            continue
        sol = r.get("sol_score")
        if not isinstance(sol, (int, float)):
            continue
        current = best.get(pid)
        if current is None or sol > current["sol_score"]:
            best[pid] = r
    return best


def _any_prior_correct_by_problem(records: list[dict]) -> dict[str, dict]:
    """For each problem_id, return one correct prior record (if any).

    Used to detect correctness regressions: today incorrect, prior correct.
    The specific record returned is arbitrary (first seen); we only need
    to know *that* a correct prior existed and its date for reporting.
    """
    out: dict[str, dict] = {}
    for r in records:
        pid = r.get("problem_id", "")
        if not pid:
            continue
        if r.get("correct") is not True:
            continue
        if pid not in out:
            out[pid] = r
    return out


def _today_by_problem(records: list[dict]) -> dict[str, dict]:
    """Pick the latest record per problem for today's slice.

    Today may contain multiple configs for the same problem; we take
    the highest sol_score among correct records, and fall back to any
    record if none are correct (so correctness regressions surface).
    """
    best_correct: dict[str, dict] = {}
    any_record: dict[str, dict] = {}
    for r in records:
        pid = r.get("problem_id", "")
        if not pid:
            continue
        any_record.setdefault(pid, r)
        if r.get("correct") is not True:
            continue
        sol = r.get("sol_score")
        if not isinstance(sol, (int, float)):
            continue
        current = best_correct.get(pid)
        if current is None or sol > current["sol_score"]:
            best_correct[pid] = r

    out: dict[str, dict] = {}
    for pid, rec in any_record.items():
        out[pid] = best_correct.get(pid, rec)
    return out


def detect_regressions(
    records: list[dict],
    today: date,
    lookback: int,
    threshold: float,
) -> tuple[list[dict], int]:
    """Return (regressions, problems_checked_today).

    Each regression dict has: problem_id, problem_name, sol_today,
    sol_prior, delta, prior_date, kernel_source_path, kind
    ("perf" or "correctness").
    """
    today_str = today.isoformat()
    prior_start = (today - timedelta(days=lookback)).isoformat()
    prior_end = (today - timedelta(days=1)).isoformat()

    # `filter_records` with correct_only=False so we can see correctness regressions.
    today_records = filter_records(
        records, date_range=(today_str, today_str), correct_only=False
    )
    prior_records = filter_records(
        records, date_range=(prior_start, prior_end), correct_only=False
    )

    today_by_pid = _today_by_problem(today_records)
    best_prior = _best_prior_by_problem(prior_records)
    any_correct_prior = _any_prior_correct_by_problem(prior_records)

    regressions: list[dict] = []
    for pid, today_rec in today_by_pid.items():
        # Correctness regression: prior was correct, today is not.
        if today_rec.get("correct") is not True:
            prior_correct = any_correct_prior.get(pid)
            if prior_correct is not None:
                regressions.append(
                    {
                        "problem_id": pid,
                        "problem_name": today_rec.get("problem_name", ""),
                        "kind": "correctness",
                        "sol_today": today_rec.get("sol_score", 0.0),
                        "sol_prior": prior_correct.get("sol_score", 0.0),
                        "delta": None,
                        "prior_date": prior_correct.get("date", ""),
                        "kernel_source_path": today_rec.get("kernel_source_path", ""),
                    }
                )
            continue

        # Perf regression: correct today, correct prior, SOL dropped > threshold.
        prior = best_prior.get(pid)
        if prior is None:
            continue
        sol_today = today_rec.get("sol_score")
        sol_prior = prior.get("sol_score")
        if not isinstance(sol_today, (int, float)) or not isinstance(sol_prior, (int, float)):
            continue
        delta = sol_today - sol_prior
        if sol_prior - sol_today > threshold:
            regressions.append(
                {
                    "problem_id": pid,
                    "problem_name": today_rec.get("problem_name", ""),
                    "kind": "perf",
                    "sol_today": sol_today,
                    "sol_prior": sol_prior,
                    "delta": delta,
                    "prior_date": prior.get("date", ""),
                    "kernel_source_path": today_rec.get("kernel_source_path", ""),
                }
            )

    return regressions, len(today_by_pid)


def format_markdown(
    regressions: list[dict],
    today: date,
    hardware: str,
) -> str:
    lines = [f"# Regressions detected ({today.isoformat()} · {hardware})", ""]
    for r in regressions:
        name_tag = f" (`{r['problem_name']}`)" if r.get("problem_name") else ""
        if r["kind"] == "correctness":
            lines.append(
                f"- **{r['problem_id']}**{name_tag}: correctness regression "
                f"(prior SOL {r['sol_prior']:.2f}, today incorrect)"
            )
        else:
            delta = r["delta"]
            lines.append(
                f"- **{r['problem_id']}**{name_tag}: "
                f"{r['sol_prior']:.2f} → {r['sol_today']:.2f} (Δ {delta:+.2f})"
            )
        lines.append(f"  - Prior best: {r['prior_date']}")
        lines.append(f"  - Today's kernel: results/leaderboard/{r['kernel_source_path']}")
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    today = _iso_date(args.date)

    root = args.leaderboard_root
    all_records = load_all(root)

    # Scope to hardware before slicing by date; keeps downstream simple.
    hw_records = [r for r in all_records if r.get("hardware") == args.hardware]

    # Early-exit messages (happy paths for CI): empty history, no today.
    prior_start = (today - timedelta(days=args.lookback)).isoformat()
    prior_end = (today - timedelta(days=1)).isoformat()
    today_str = today.isoformat()
    has_prior = any(prior_start <= r.get("date", "") <= prior_end for r in hw_records)
    has_today = any(r.get("date") == today_str for r in hw_records)

    if not has_prior:
        print("No prior data to regress against.")
        return 0
    if not has_today:
        print(f"No records for today ({today_str}) on {args.hardware}.")
        return 0

    regressions, n_checked = detect_regressions(
        hw_records, today, args.lookback, args.threshold
    )

    if not regressions:
        print(
            f"No regressions detected across {n_checked} problems "
            f"(threshold {args.threshold:.2f})"
        )
        return 0

    print(format_markdown(regressions, today, args.hardware))
    return 1


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
