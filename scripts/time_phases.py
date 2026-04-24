"""Phase-timing diagnostic for the /optimize pipeline.

Times each step of a full optimize cycle *without* running the LLM loop,
so you can see which phase is eating the wallclock (Modal cold boot,
serial baseline trials, classification, etc.) before spending credits on
an actual run.

Usage:
    python scripts/time_phases.py                           # times with reference.py
    python scripts/time_phases.py reference_relu.py         # pick a reference
    python scripts/time_phases.py reference_relu.py --llm   # also time one LLM call

Output: a Rich table with wallclock per phase and a short note.
No LLM calls are made unless --llm is passed. Modal calls always happen
(that's the whole point) and will burn ~$0.05 on L40S time.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _fmt_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)}m{secs:04.1f}s"


class PhaseTimer:
    def __init__(self) -> None:
        self.rows: list[tuple[str, float, str]] = []

    def time(self, label: str, note: str = ""):
        timer = self
        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()
                print(f"  → {label}…", flush=True)
                return self_inner
            def __exit__(self_inner, exc_type, exc, tb):
                dt = time.perf_counter() - self_inner.t0
                tag = "FAIL" if exc_type else "OK"
                print(f"    {tag} · {_fmt_time(dt)}" + (f" · {note}" if note else ""))
                timer.rows.append((label, dt, note or ""))
                return False
        return _Ctx()

    def add(self, label: str, seconds: float, note: str = "") -> None:
        self.rows.append((label, seconds, note))

    def render(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
        except ImportError:
            # Fallback: plain text
            print("\n" + "=" * 70)
            print(f"{'Phase':<36}{'Time':>12}  Notes")
            print("-" * 70)
            for label, dt, note in self.rows:
                print(f"{label:<36}{_fmt_time(dt):>12}  {note}")
            print("=" * 70)
            total = sum(dt for _, dt, _ in self.rows)
            print(f"{'Total':<36}{_fmt_time(total):>12}")
            return

        console = Console()
        table = Table(
            box=box.ROUNDED,
            header_style="bold white",
            border_style="white",
            pad_edge=False,
            padding=(0, 2),
            expand=False,
        )
        table.add_column("phase", no_wrap=True)
        table.add_column("time", justify="right", no_wrap=True)
        table.add_column("notes")
        for label, dt, note in self.rows:
            style = "bold #ef4444" if dt > 30 else ("#ffc107" if dt > 5 else "white")
            table.add_row(label, f"[{style}]{_fmt_time(dt)}[/{style}]", note or "")
        total = sum(dt for _, dt, _ in self.rows)
        table.add_section()
        table.add_row("[bold]total[/bold]", f"[bold]{_fmt_time(total)}[/bold]", "")
        console.print()
        console.print(table)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "reference",
        nargs="?",
        default="reference.py",
        help="Reference file to profile (default: reference.py)",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of Modal eval trials to time (default: 5, matches /optimize)",
    )
    ap.add_argument(
        "--llm",
        action="store_true",
        help="Also time one LLM generation call using the configured default model",
    )
    args = ap.parse_args()

    ref_path = REPO / args.reference
    if not ref_path.exists():
        print(f"[ERROR] Reference file not found: {ref_path}", file=sys.stderr)
        return 1

    print(f"Timing /optimize phases for: {ref_path.name}")
    print(f"  Trials: {args.trials}  ·  LLM: {'yes' if args.llm else 'no'}")
    print()

    timer = PhaseTimer()

    # --- 1. Local setup: load + sanitize + classify --------------------
    from kernel_code.problem import detect_format, make_self_contained, Problem
    from kernel_code.problem_classifier import classify_problem
    from kernel_code.settings import load_settings

    with timer.time("load reference + detect format"):
        raw_src = ref_path.read_text()
        fmt = detect_format(raw_src)
    note = f"format={fmt}, {len(raw_src)} bytes"
    timer.rows[-1] = (timer.rows[-1][0], timer.rows[-1][1], note)

    with timer.time("sanitize for Modal"):
        if fmt == "gpumode":
            p = Problem(reference_code=raw_src, format="gpumode")
            ref_src = make_self_contained(p)
        else:
            ref_src = raw_src

    with timer.time("classify problem"):
        classif = classify_problem(ref_src)
    timer.rows[-1] = (
        timer.rows[-1][0], timer.rows[-1][1],
        f"{classif.op_type.value}, "
        f"{classif.estimated_tensor_elements:,} elements, "
        f"{'mem-bound' if classif.is_memory_bound_likely else 'compute-bound' if classif.is_compute_bound_likely else 'unclassified'}"
    )

    with timer.time("HBM ceiling check"):
        from kernel_code.shell import _hbm_ceiling_check
        settings = load_settings()
        warning = _hbm_ceiling_check(
            ref_us=3380.0,  # placeholder; real value comes from profile
            ref_code=ref_src,
            classif=classif,
            hardware=settings.default_gpu,
            target_speedup=1.3,
        )
    timer.rows[-1] = (
        timer.rows[-1][0], timer.rows[-1][1],
        "at-ceiling" if warning else "below-ceiling"
    )

    # --- 2. Modal: cold boot + warm evals -------------------------------
    import modal
    eval_fn = modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")

    # KernelBench-style: alias ModelNew = Model so the ref itself passes.
    if fmt == "kernelbench":
        kernel_src = ref_src + "\n\nModelNew = Model\n"
    else:
        kernel_src = ref_src.replace("def ref_kernel(", "def kernel_function(")
    pf = fmt if fmt in ("kernelbench", "gpumode") else "kernelbench"

    def _one_eval() -> dict:
        return eval_fn.remote(
            kernel_source=kernel_src,
            reference_source=ref_src,
            eval_mode="fast",
            problem_format=pf,
        )

    with timer.time("Modal eval — trial 1 (cold boot risk)"):
        r1 = _one_eval()
    timer.rows[-1] = (
        timer.rows[-1][0], timer.rows[-1][1],
        f"ref={r1.get('ref_runtime_us', 0):.0f}µs"
    )
    # Fold the child-side phase breakdown for trial 1 into the table as
    # indented sub-rows so we can see where the 40s went.
    for label, dt in (r1.get("phases") or []):
        timer.add(f"    └ {label}", dt, "")

    warm_times: list[float] = []
    last_warm_phases: list[tuple[str, float]] = []
    for i in range(2, args.trials + 1):
        with timer.time(f"Modal eval — trial {i} (warm)"):
            r = _one_eval()
        warm_times.append(timer.rows[-1][1])
        timer.rows[-1] = (
            timer.rows[-1][0], timer.rows[-1][1],
            f"ref={r.get('ref_runtime_us', 0):.0f}µs"
        )
        last_warm_phases = r.get("phases") or []
    # Show a single warm-trial breakdown (the most recent) so we can
    # compare cold vs warm per-phase without exploding the table.
    if last_warm_phases:
        timer.add("  warm trial phase breakdown (last):", 0.0, "")
        for label, dt in last_warm_phases:
            timer.add(f"    └ {label}", dt, "")

    if warm_times:
        timer.add(
            "  warm eval — min / median / max",
            statistics.median(warm_times),
            f"min={_fmt_time(min(warm_times))}, max={_fmt_time(max(warm_times))}",
        )

    # --- 3. Optional: one LLM generation --------------------------------
    if args.llm:
        import os
        try:
            with timer.time(f"LLM generation — {settings.default_model}"):
                from openkernel.llm.provider import LLMProvider
                from openkernel.config import OpenKernelConfig
                cfg = OpenKernelConfig(
                    provider=settings.default_provider,
                    model_id=settings.default_model,
                )
                # Forward project-scoped keys into the environment so the
                # provider wrapper can find them.
                for attr, env in (
                    ("openai_api_key", "OPENAI_API_KEY"),
                    ("nvidia_api_key", "NVIDIA_API_KEY"),
                    ("anthropic_api_key", "ANTHROPIC_API_KEY"),
                    ("groq_api_key", "GROQ_API_KEY"),
                    ("minimax_api_key", "MINIMAX_API_KEY"),
                ):
                    val = getattr(settings, attr, "") or ""
                    if val and not os.environ.get(env):
                        os.environ[env] = val
                llm = LLMProvider(cfg)
                prompt = (
                    "Write a minimal Triton kernel for y = max(x, 0) on a 1D tensor. "
                    "Just the kernel, no commentary."
                )
                out = llm.generate(prompt, max_tokens=400)
            timer.rows[-1] = (
                timer.rows[-1][0], timer.rows[-1][1],
                f"{len(out)} chars returned"
            )
        except Exception as exc:
            timer.add(
                f"LLM generation — {settings.default_model}",
                0.0,
                f"[FAILED] {type(exc).__name__}: {str(exc)[:60]}",
            )

    # --- 4. Report ------------------------------------------------------
    timer.render()

    # Quick verdict
    total = sum(dt for _, dt, _ in timer.rows)
    print()
    print("Verdict:")
    cold = timer.rows[4][1] if len(timer.rows) > 4 else 0
    if cold > 60:
        print(f"  · Modal cold boot dominates ({_fmt_time(cold)}). Expected on new containers.")
        print(f"    Container keeps warm ~60s idle, so running eval again immediately re-uses it.")
    if warm_times and statistics.median(warm_times) > 10:
        print(f"  · Warm eval is slow ({_fmt_time(statistics.median(warm_times))}). L40S eval for this problem should be <10s.")
        print(f"    Check Modal dashboard for queue delays or container resource issues.")
    if args.trials >= 5 and len(warm_times) >= 4:
        parallelizable = sum(warm_times[1:])  # if all warm trials ran in parallel instead of serial
        print(f"  · Running {args.trials} trials serially wastes ~{_fmt_time(parallelizable)}.")
        print(f"    Parallelizing via concurrent.futures would cut profile wallclock by this much.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
