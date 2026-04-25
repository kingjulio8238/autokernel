"""KernelBench L1 sweep runner.

Runs the full /optimize pipeline on N problems from KernelBench Level 1
and captures per-problem artifacts for post-hoc diagnosis:

  - reference.py               (the loaded KernelBench source)
  - dev_log.jsonl              (every LLM prompt + response, per iteration)
  - workers/                   (KernelAgent per-worker round_*.json + logs)
  - run.log                    (the run-level log emitted by RunLogger)
  - result.json                (AutoResult: best_speedup, stop_reason, cost)
  - problem.json               (metadata)

At sweep end: a summary.json + summary.csv with aggregate stats.

Usage::

    python scripts/kb_l1_sweep.py                          # 10 problems, target 1.5x, $1 each
    python scripts/kb_l1_sweep.py --count 20 --budget 2    # 20 problems, $2 each
    python scripts/kb_l1_sweep.py --start-id 50 --count 5  # problems 50-54

Resuming a partial sweep: pass ``--resume`` to skip per-problem dirs that
already contain a complete ``result.json`` (parseable + has either
``best_speedup`` and ``target_reached`` fields OR an ``error`` field).
Partial dirs (no result.json or malformed) are wiped and re-run. Resumed
problems contribute identically to summary.json aggregation::

    python scripts/kb_l1_sweep.py --resume --out-dir .kernel-code/sweeps/strat_l1_abc1234

Prerequisites: MODAL_PROFILE=kernel+ exported, kernelbench pip-installed,
OPENAI_API_KEY set (or whatever the configured default_model needs).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _ensure_env(var: str, hint: str) -> None:
    if not os.environ.get(var):
        raise SystemExit(f"[kb_l1_sweep] ${var} not set. {hint}")


def _load_problem(problem_id: int) -> dict:
    """Wrap openkernel.kernelbench.problems.load_problem with a clear error."""
    from openkernel.kernelbench.problems import load_problem
    try:
        return load_problem(level=1, problem_id=problem_id)
    except ImportError as exc:
        raise SystemExit(
            f"[kb_l1_sweep] kernelbench package not installed. Run:\n"
            f"  uv pip install 'kernelbench @ git+https://github.com/"
            f"ScalingIntelligence/KernelBench.git'\n"
            f"  (underlying error: {exc})"
        )


def _snapshot_session_dir(run_start: float) -> Path | None:
    """Find KernelAgent's session dir (created after run_start) for snapshot.

    Returns the most recent session_* directory under triton_kernel_logs/
    that was created after ``run_start``, or None if nothing qualifies.
    """
    logs = REPO / "triton_kernel_logs"
    if not logs.exists():
        return None
    candidates = sorted(
        (d for d in logs.glob("session_*") if d.is_dir() and d.stat().st_mtime >= run_start - 2),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _install_llm_tee(llm_log_path: Path):
    """Monkey-patch KernelAgent's provider classes so every LLM call is
    teed to ``llm_log_path`` as a JSONL stream. Returns a ``restore()``
    callable that reverts all patches.

    Patches BOTH ``get_response`` AND ``get_multiple_responses`` across
    every provider class used by KernelAgent. The initial version only
    patched ``get_response`` on the OpenAI-compatible base, which
    missed:
      - ``get_multiple_responses`` (used by Kernel+'s multi-sample path
        where one prompt returns N parallel candidates)
      - Anthropic and Ollama providers that have their own methods
      - The relay provider that wraps the above

    Without these, the tee captures maybe 1-2 test-generation calls per
    problem and misses the main kernel-generation calls. Earlier
    smoke-run artifacts for L1 #6 only showed test-generation LLM
    interactions, not the actual kernel prompts where the large-K
    matmul strategy was chosen — exactly the data we wanted.
    """
    fh = open(llm_log_path, "a", buffering=1)  # line-buffered

    # Collect (class, attr, original) triples so we can restore everything.
    patches: list[tuple[type, str, object]] = []

    def _make_wrapper(original, kind: str):
        def wrapped(self, model_name, messages, *args, **kwargs):
            t0 = time.time()
            resp = original(self, model_name, messages, *args, **kwargs)
            try:
                # get_response returns a single LLMResponse; get_multiple
                # returns list[LLMResponse]. Normalize for logging.
                if isinstance(resp, list):
                    for idx, r in enumerate(resp):
                        fh.write(json.dumps({
                            "ts": time.time(),
                            "elapsed_s": time.time() - t0,
                            "kind": kind,
                            "choice_idx": idx,
                            "model": model_name,
                            "provider": getattr(self, "name", type(self).__name__),
                            "messages": messages,
                            "response": getattr(r, "content", str(r)),
                            "usage": getattr(r, "usage", None),
                            "kwargs": {k: v for k, v in kwargs.items()
                                       if isinstance(v, (str, int, float, bool))},
                        }, default=str) + "\n")
                else:
                    fh.write(json.dumps({
                        "ts": time.time(),
                        "elapsed_s": time.time() - t0,
                        "kind": kind,
                        "model": model_name,
                        "provider": getattr(self, "name", type(self).__name__),
                        "messages": messages,
                        "response": getattr(resp, "content", str(resp)),
                        "usage": getattr(resp, "usage", None),
                        "kwargs": {k: v for k, v in kwargs.items()
                                   if isinstance(v, (str, int, float, bool))},
                    }, default=str) + "\n")
            except Exception as exc:
                fh.write(json.dumps({"ts": time.time(), "log_error": str(exc)}) + "\n")
            return resp
        return wrapped

    # Gather provider classes — import failures are non-fatal; missing
    # providers just don't get patched (they weren't going to be used).
    provider_classes = []
    try:
        from kernel_agent.ka_utils.providers.openai_base import OpenAICompatibleProvider
        provider_classes.append(OpenAICompatibleProvider)
    except ImportError:
        pass
    try:
        from kernel_agent.ka_utils.providers.anthropic_provider import AnthropicProvider
        provider_classes.append(AnthropicProvider)
    except ImportError:
        pass
    try:
        from kernel_agent.ka_utils.providers.ollama_provider import OllamaProvider
        provider_classes.append(OllamaProvider)
    except ImportError:
        pass
    try:
        from kernel_agent.ka_utils.providers.relay_provider import RelayProvider
        provider_classes.append(RelayProvider)
    except ImportError:
        pass

    for cls in provider_classes:
        for attr in ("get_response", "get_multiple_responses"):
            if hasattr(cls, attr):
                original = getattr(cls, attr)
                patches.append((cls, attr, original))
                setattr(cls, attr, _make_wrapper(original, kind=attr))

    def restore():
        for cls, attr, original in patches:
            try:
                setattr(cls, attr, original)
            except Exception:
                pass
        try:
            fh.close()
        except Exception:
            pass

    return restore


def _run_single_problem(
    problem_id: int,
    out_dir: Path,
    *,
    target_speedup: float,
    budget: float,
    hardware: str,
    backend: str,
    dev_log_path: Path,
) -> dict:
    """Run /optimize equivalent on a single problem. Returns a result dict."""
    from kernel_code.auto_optimizer import MetaOptimizer
    from kernel_code.goal_spec import GoalSpec
    from kernel_code.run_log import RunLogger
    from kernel_code.settings import load_settings, inject_api_keys

    settings = load_settings()
    inject_api_keys(settings)

    # Load problem
    print(f"  loading problem {problem_id}…", flush=True)
    try:
        problem = _load_problem(problem_id)
    except SystemExit:
        raise
    except Exception as exc:
        return {
            "problem_id": problem_id,
            "error": f"load failed: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    problem_name = problem.get("problem_name", f"L1_Problem_{problem_id}")
    reference_source = problem.get("reference_source", "")
    if not reference_source:
        return {
            "problem_id": problem_id,
            "problem_name": problem_name,
            "error": "empty reference_source from kernelbench loader",
        }

    # Snapshot problem metadata + reference
    (out_dir / "problem.json").write_text(json.dumps({
        "problem_id": problem_id,
        "problem_name": problem_name,
        "level": 1,
        "reference_bytes": len(reference_source),
    }, indent=2))
    (out_dir / "reference.py").write_text(reference_source)

    # Write reference to a transient path the MetaOptimizer can point at.
    # Use a per-problem file so concurrent sweeps (if any) don't trample.
    transient_ref = REPO / f"_sweep_ref_{problem_id}.py"
    transient_ref.write_text(reference_source)

    # Env setup — each problem gets its own dev-log run-id so
    # OPENKERNEL_DEV_LOG captures the LLM thought trace separately.
    run_id = f"kb_l1_{problem_id}_{uuid.uuid4().hex[:8]}"
    os.environ["OPENKERNEL_DEV_LOG"] = "1"
    os.environ["OPENKERNEL_RUN_ID"] = run_id

    spec = GoalSpec(
        target_speedup=target_speedup,
        max_budget_usd=budget,
        max_rounds=settings.max_rounds,
        reference_path=str(transient_ref),
        hardware=hardware,
        backend=backend,
        model=settings.default_model,
        provider=settings.default_provider,
    )

    run_logger = RunLogger()
    run_logger.start_run(
        command=f"kb_l1_sweep: problem_id={problem_id}",
        config={
            "model": settings.default_model,
            "hardware": hardware,
            "backend": backend,
            "target_speedup": target_speedup,
            "budget": budget,
            "file": transient_ref.name,
            "sweep": True,
        },
    )

    run_start = time.time()
    # Tee every LLM call to a per-problem jsonl so we can diagnose what the
    # model thought and what it proposed on each iteration.
    llm_tee_restore = _install_llm_tee(out_dir / "llm_calls.jsonl")
    try:
        optimizer = MetaOptimizer(
            goal=spec,
            settings=settings,
            console=None,          # no Rich stdout during sweep
            live_display=None,     # no live table — headless
            run_logger=run_logger,
        )
        result = optimizer.run()
        result_dict = {
            "best_speedup": float(result.best_speedup),
            "target_reached": bool(result.target_reached),
            "rounds_completed": int(result.rounds_completed),
            "total_iterations": int(result.total_iterations),
            "total_cost_usd": float(result.total_cost_usd),
            "elapsed_seconds": float(result.elapsed_seconds),
            "stop_reason": str(result.stop_reason),
            "evidence_added": int(result.evidence_added),
            "best_kernel_bytes": len(result.best_kernel or ""),
        }
        # Save best kernel if non-empty
        if result.best_kernel:
            (out_dir / "best_kernel.py").write_text(result.best_kernel)
    except BaseException as exc:
        result_dict = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - run_start,
        }
    finally:
        # Restore the patched provider even if the optimizer raised so
        # subsequent problems don't keep accumulating into a dangling file.
        try:
            llm_tee_restore()
        except Exception:
            pass

    # Snapshot dev-log (LLM prompts + responses)
    src_dev_log = REPO / ".kernel-code" / "dev_logs" / f"run_{run_id}.jsonl"
    if src_dev_log.exists():
        shutil.copy2(src_dev_log, out_dir / "dev_log.jsonl")

    # Snapshot KernelAgent session workers dir
    session_dir = _snapshot_session_dir(run_start)
    if session_dir is not None:
        workers_src = session_dir / "workers"
        if workers_src.exists():
            shutil.copytree(workers_src, out_dir / "workers", dirs_exist_ok=True)
        # Also grab the top-level agent log if present
        for log_file in session_dir.glob("agent_*.log"):
            shutil.copy2(log_file, out_dir / log_file.name)

    # Snapshot run-log
    runs_dir = REPO / ".kernel-code" / "runs"
    if runs_dir.exists():
        candidate_runs = sorted(
            (p for p in runs_dir.glob("*_smart.log") if p.stat().st_mtime >= run_start - 2),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidate_runs:
            shutil.copy2(candidate_runs[0], out_dir / "run.log")

    # Cleanup transient ref
    try:
        transient_ref.unlink()
    except OSError:
        pass

    # Persist result.json last so a crash during snapshotting doesn't leave
    # the reader without any status indicator.
    final = {
        "problem_id": problem_id,
        "problem_name": problem_name,
        "run_id": run_id,
        **result_dict,
    }
    (out_dir / "result.json").write_text(json.dumps(final, indent=2))
    return final


def _load_complete_result(problem_dir: Path) -> dict | None:
    """Return a parsed result.json iff it indicates a fully-finished run.

    A result is considered complete when it parses as JSON and either:
      * carries an ``error`` field (the prior run errored — re-running
        unlikely to help and would burn budget), OR
      * carries both ``best_speedup`` and ``target_reached`` fields (the
        prior run finished cleanly).

    Anything else (missing file, malformed JSON, missing fields) returns
    None so the caller can wipe + re-run.
    """
    rj = problem_dir / "result.json"
    if not rj.exists():
        return None
    try:
        data = json.loads(rj.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if "error" in data:
        return data
    if "best_speedup" in data and "target_reached" in data:
        return data
    return None


def _write_summary(out_root: Path, per_problem: list[dict]) -> None:
    """Write summary.json + summary.csv with aggregate stats."""
    total = len(per_problem)
    errored = sum(1 for r in per_problem if "error" in r)
    succeeded = [r for r in per_problem if "error" not in r]
    target_hit = sum(1 for r in succeeded if r.get("target_reached"))
    beat_baseline = sum(1 for r in succeeded if float(r.get("best_speedup") or 0) > 1.02)
    speedups = sorted([float(r.get("best_speedup") or 0) for r in succeeded])

    def _pct(numer: int, denom: int) -> float:
        return 100.0 * numer / denom if denom else 0.0

    summary = {
        "total_problems": total,
        "errored": errored,
        "target_reached": target_hit,
        "beat_baseline": beat_baseline,
        "target_reached_pct": _pct(target_hit, total),
        "beat_baseline_pct": _pct(beat_baseline, total),
        "median_speedup": speedups[len(speedups) // 2] if speedups else 0.0,
        "max_speedup": max(speedups) if speedups else 0.0,
        "total_cost_usd": sum(float(r.get("total_cost_usd") or 0) for r in succeeded),
        "total_elapsed_seconds": sum(float(r.get("elapsed_seconds") or 0) for r in per_problem),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "problem_id", "problem_name", "best_speedup", "target_reached",
            "rounds", "iterations", "cost_usd", "elapsed_s", "stop_reason",
            "error",
        ])
        for r in per_problem:
            writer.writerow([
                r.get("problem_id"),
                r.get("problem_name", ""),
                f"{float(r.get('best_speedup') or 0):.3f}",
                r.get("target_reached", ""),
                r.get("rounds_completed", ""),
                r.get("total_iterations", ""),
                f"{float(r.get('total_cost_usd') or 0):.3f}",
                f"{float(r.get('elapsed_seconds') or 0):.1f}",
                (r.get("stop_reason") or "")[:120],
                r.get("error", ""),
            ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=10, help="Number of problems (default 10)")
    ap.add_argument("--start-id", type=int, default=1, help="First problem ID (default 1)")
    ap.add_argument("--target", type=float, default=1.5, help="Target speedup (default 1.5)")
    ap.add_argument("--budget", type=float, default=1.0, help="USD budget per problem (default $1)")
    ap.add_argument("--hardware", default="L40S", help="Hardware (default L40S)")
    ap.add_argument("--backend", default="triton", help="Backend (default triton)")
    ap.add_argument("--out-dir", default=None, help="Output dir (default .kernel-code/sweeps/l1_<ts>)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip per-problem dirs that already contain a complete result.json. "
            "Partial/malformed dirs are wiped and re-run. Resumed problems are "
            "aggregated into summary.json identically to fresh runs."
        ),
    )
    args = ap.parse_args()

    _ensure_env("MODAL_PROFILE", "Export with: export MODAL_PROFILE=kernel+")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir or REPO / ".kernel-code" / "sweeps" / f"l1_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[kb_l1_sweep] writing artifacts to {out_root}")
    print(f"[kb_l1_sweep] {args.count} problems, target {args.target}x, "
          f"${args.budget:.2f} budget each, hardware={args.hardware}")
    print()

    per_problem: list[dict] = []
    sweep_start = time.time()
    for i in range(args.count):
        problem_id = args.start_id + i
        problem_dir = out_root / f"problem_{problem_id:03d}"

        resumed_result: dict | None = None
        if args.resume and problem_dir.exists():
            resumed_result = _load_complete_result(problem_dir)
            if resumed_result is None:
                # Partial/malformed dir — wipe before re-running so stale
                # artifacts don't bleed into the new attempt.
                shutil.rmtree(problem_dir)

        problem_dir.mkdir(parents=True, exist_ok=True)

        if resumed_result is not None:
            per_problem.append(resumed_result)
            sp = float(resumed_result.get("best_speedup") or 0)
            if "error" in resumed_result:
                print(
                    f"[{i + 1}/{args.count}] problem {problem_id}… (resumed) "
                    f"ERROR: {resumed_result['error']}",
                    flush=True,
                )
            else:
                hit = "✓" if resumed_result.get("target_reached") else "·"
                print(
                    f"[{i + 1}/{args.count}] problem {problem_id}… (resumed) "
                    f"{hit} best={sp:.2f}x "
                    f"cost=${float(resumed_result.get('total_cost_usd') or 0):.3f} "
                    f"elapsed={float(resumed_result.get('elapsed_seconds') or 0):.0f}s "
                    f"({(resumed_result.get('stop_reason') or '')[:60]})",
                    flush=True,
                )
            continue

        print(f"[{i + 1}/{args.count}] problem {problem_id}…", flush=True)
        t0 = time.time()
        try:
            result = _run_single_problem(
                problem_id,
                problem_dir,
                target_speedup=args.target,
                budget=args.budget,
                hardware=args.hardware,
                backend=args.backend,
                dev_log_path=problem_dir / "dev_log.jsonl",
            )
        except BaseException as exc:
            result = {
                "problem_id": problem_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - t0,
            }
            (problem_dir / "result.json").write_text(json.dumps(result, indent=2))

        per_problem.append(result)

        # One-line progress
        if "error" in result:
            print(f"  [{i + 1}/{args.count}] ERROR: {result['error']}")
        else:
            sp = float(result.get("best_speedup") or 0)
            hit = "✓" if result.get("target_reached") else "·"
            print(
                f"  [{i + 1}/{args.count}] {hit} best={sp:.2f}x "
                f"cost=${float(result.get('total_cost_usd') or 0):.3f} "
                f"elapsed={float(result.get('elapsed_seconds') or 0):.0f}s "
                f"({result.get('stop_reason', '')[:60]})"
            )

    _write_summary(out_root, per_problem)
    total_s = time.time() - sweep_start

    # Final report
    print()
    print(f"[kb_l1_sweep] done in {total_s:.0f}s")
    summary = json.loads((out_root / "summary.json").read_text())
    print(
        f"  target-reached: {summary['target_reached']}/{summary['total_problems']}"
        f" ({summary['target_reached_pct']:.0f}%)"
    )
    print(
        f"  beat-baseline:  {summary['beat_baseline']}/{summary['total_problems']}"
        f" ({summary['beat_baseline_pct']:.0f}%)"
    )
    print(f"  median speedup: {summary['median_speedup']:.2f}x")
    print(f"  max speedup:    {summary['max_speedup']:.2f}x")
    print(f"  total cost:     ${summary['total_cost_usd']:.2f}")
    print(f"  errored:        {summary['errored']}")
    print()
    print(f"  artifacts: {out_root}")
    print(f"  summary:   {out_root / 'summary.json'}")
    print(f"             {out_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
