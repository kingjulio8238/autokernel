#!/usr/bin/env python3
"""10-problem dry-run sweep on L40S for Phase 1 Round D.5 verification.

Runs a curated 10-problem subset end-to-end (3 L1, 3 L2, 4 GPU MODE) against
live Modal L40S using ``kernel_code.batch_optimizer.run_suite``. On completion
it shells out to ``scripts/score_suite.py`` and ``scripts/emit_report.py`` to
produce the scored JSON and markdown report.

Exit codes:
    0 — all 10 problems processed (succeeded/failed/errored all count)
    1 — suite crashed mid-run or a fatal setup error occurred

Usage:
    python scripts/run_dry_run.py
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import re  # noqa: E402

from kernel_code.batch_optimizer import run_suite  # noqa: E402
from kernel_code.settings import inject_api_keys, load_settings  # noqa: E402
from openkernel.benchmarks.gpumode_loader import load_gpumode  # noqa: E402
from openkernel.benchmarks.problem_spec import ProblemSpec  # noqa: E402
from openkernel.kernelbench.problems import load_problem  # noqa: E402


# 5-problem default for fast iteration — picks one-per-category coverage:
# compute-bound L1, elementwise L1, fused L2, trivial gpumode (likely passes),
# atomic-heavy gpumode (edge case). Override with --ids or --limit.
_TARGET_IDS: list[str] = [
    "kb_l1_0001",        # matmul (compute-bound L1)
    "kb_l1_0031",        # ELU (elementwise L1)
    "kb_l2_0001",        # Conv2D_ReLU_BiasAdd (fused L2)
    "gpumode_vectoradd", # trivial elementwise (likely passes)
    "gpumode_histogram", # atomic-heavy (stress test)
]

_HARDWARE = "L40S"
_BUDGET_PER_PROBLEM = 0.50
_CONCURRENCY = 4
_MODEL = "o3-mini"
_TARGET_SOL = 0.80


def _pick(specs_by_id: dict[str, object], target_ids: list[str]) -> tuple[list[object], list[str]]:
    picked: list[object] = []
    missing: list[str] = []
    for tid in target_ids:
        spec = specs_by_id.get(tid)
        if spec is None:
            missing.append(tid)
        else:
            picked.append(spec)
    return picked, missing


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 dry-run sweep driver.")
    p.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated spec IDs to run. Defaults to the full 10-problem set.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Take only the first N IDs after filtering (for small dry-runs).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete today's existing records for the target IDs before running so "
             "the resume filter doesn't skip them. Useful for dev iteration.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=f"Override the default concurrency ({_CONCURRENCY}). Use 1 to run "
             "problems serially — useful when diagnosing CUDA-state cross-"
             "contamination in concurrent Modal containers.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_dry_run")

    if args.ids:
        target_ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    else:
        target_ids = list(_TARGET_IDS)
    if args.limit is not None:
        target_ids = target_ids[: args.limit]
    log.info("Dry-run target: %d IDs — %s", len(target_ids), target_ids)

    concurrency = args.concurrency if args.concurrency is not None else _CONCURRENCY

    settings = load_settings()
    injected = inject_api_keys(settings)
    log.info("Injected %d API keys into environment", injected)

    # Load ONLY the target specs — avoid pulling all 208 from HF.
    # KB targets → one load_problem() per id. GPU MODE → cheap bulk load (8 items).
    log.info("Loading %d target specs (not the full 208)...", len(target_ids))

    kb_re = re.compile(r"^kb_l(\d+)_(\d+)$")
    picked: list[ProblemSpec] = []
    missing: list[str] = []
    gm_cache: dict[str, ProblemSpec] | None = None

    for tid in target_ids:
        try:
            m = kb_re.match(tid)
            if m:
                level = int(m.group(1))
                pid = int(m.group(2))
                record = load_problem(level, pid)
                picked.append(ProblemSpec(
                    id=tid,
                    name=record["problem_name"],
                    tier=f"L{level}",
                    source="kernelbench",
                    reference_source=record["reference_source"],
                    workload_spec={},
                    expected_dtype="float32",
                ))
                log.info("  loaded %s (%s)", tid, record["problem_name"][:50])
            elif tid.startswith("gpumode_"):
                if gm_cache is None:
                    gm_cache = {s.id: s for s in load_gpumode()}
                spec = gm_cache.get(tid)
                if spec is None:
                    missing.append(tid)
                    continue
                picked.append(spec)
                log.info("  loaded %s (%s)", tid, spec.name)
            else:
                missing.append(tid)
        except Exception as exc:
            log.error("  failed to load %s: %s", tid, exc)
            missing.append(tid)

    if missing:
        log.warning("%d target IDs not found and will be skipped: %s", len(missing), missing)
    log.info("Resolved %d/%d target specs", len(picked), len(target_ids))

    if not picked:
        log.error("No specs resolved — nothing to run")
        return 1

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if args.force:
        # Delete today's records for the target IDs so resume doesn't skip.
        # Also wipe companion kernel files if they're not referenced elsewhere
        # (conservative: leave kernel files in place, only drop record files).
        day_dir = _PROJECT_ROOT / "results" / "leaderboard" / date_str
        removed = 0
        if day_dir.exists():
            target_id_set = {s.id for s in picked}
            for fp in day_dir.glob("*.json"):
                # Filename convention: {problem_id}_{hardware}_{config_hash}.json
                for tid in target_id_set:
                    if fp.name.startswith(f"{tid}_{_HARDWARE}_"):
                        fp.unlink()
                        removed += 1
                        break
        log.info("--force: removed %d existing record(s) for today's targets", removed)

    log.info(
        "Starting run_suite: hardware=%s budget=$%.2f concurrency=%d model=%s target_sol=%.2f date=%s",
        _HARDWARE, _BUDGET_PER_PROBLEM, concurrency, _MODEL, _TARGET_SOL, date_str,
    )
    wall_start = time.time()
    try:
        suite = run_suite(
            specs=picked,
            hardware=_HARDWARE,
            budget_per_problem=_BUDGET_PER_PROBLEM,
            concurrency=concurrency,
            date=date_str,
            model=_MODEL,
            target_sol=_TARGET_SOL,
            settings=settings,
        )
    except Exception as exc:
        log.exception("run_suite crashed: %s", exc)
        return 1
    wall_s = time.time() - wall_start

    processed = suite.succeeded + suite.failed + suite.errored + suite.skipped
    print("\n" + "=" * 60)
    print("DRY-RUN SUITE SUMMARY")
    print("=" * 60)
    print(f"Date:         {date_str}")
    print(f"Hardware:     {_HARDWARE}")
    print(f"Model:        {_MODEL}")
    print(f"Wall time:    {wall_s:.1f}s ({wall_s / 60:.2f} min)")
    print(f"Total cost:   ${suite.total_cost_usd:.2f}")
    print(f"Specs total:  {suite.total}")
    print(f"Succeeded:    {suite.succeeded}")
    print(f"Failed:       {suite.failed}")
    print(f"Errored:      {suite.errored}")
    print(f"Skipped:      {suite.skipped}")
    print(f"Processed:    {processed}")
    print(f"Records:      {len(suite.records_written)} written")
    for rp in suite.records_written:
        print(f"  - {rp}")
    print("=" * 60)

    # Score + emit report (non-fatal if they fail).
    score_cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "score_suite.py"),
        "--date", date_str,
        "--hardware", _HARDWARE,
    ]
    log.info("Running score_suite.py: %s", " ".join(score_cmd))
    try:
        score_proc = subprocess.run(score_cmd, capture_output=True, text=True, check=False)
        if score_proc.returncode != 0:
            log.warning("score_suite.py exited %d; stderr: %s", score_proc.returncode, score_proc.stderr[:500])
        else:
            log.info("score_suite.py ok (%d bytes of JSON)", len(score_proc.stdout))
    except Exception as exc:
        log.error("score_suite.py failed: %s", exc)

    report_cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "emit_report.py"),
        "--date", date_str,
        "--hardware", _HARDWARE,
    ]
    log.info("Running emit_report.py: %s", " ".join(report_cmd))
    report_path: Path | None = None
    try:
        rep_proc = subprocess.run(report_cmd, capture_output=True, text=True, check=False)
        if rep_proc.returncode == 0 and rep_proc.stdout.strip():
            report_path = Path(rep_proc.stdout.strip().splitlines()[-1])
            log.info("Report written: %s", report_path)
        else:
            log.warning("emit_report.py rc=%d stdout=%s stderr=%s",
                        rep_proc.returncode, rep_proc.stdout[:300], rep_proc.stderr[:300])
    except Exception as exc:
        log.error("emit_report.py failed: %s", exc)

    if report_path is not None and report_path.exists():
        print(f"\nMarkdown report: {report_path}")

    # Exit rule: 0 if we made it here without crashing mid-run.
    return 0


if __name__ == "__main__":
    sys.exit(main())
