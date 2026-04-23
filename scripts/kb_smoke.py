"""Phase 1 smoke test: verify KernelBench integration end-to-end.

Loads one L1 KernelBench problem, builds a passthrough ModelNew that
delegates to the reference Model (expected speedup ~1.0x), submits to the
existing Modal `eval_kernel_on_gpu` function with problem_format="kernelbench",
and asserts that status/correctness/sol_score populate as expected.

Exit code 0 on pass, 1 on fail.

Usage:
    python scripts/kb_smoke.py
"""

from __future__ import annotations

import sys
from typing import Any

import modal


def _load_problem_l1_id1() -> tuple[str, str]:
    """Load L1 problem id=1 from KernelBench via HuggingFace.

    LocalKernelBenchDataset requires the KernelBench repo on disk; the HF
    source ships the problem files inside the dataset so it works from a
    plain pip install. KernelBench problem IDs are 1-indexed — id=1 is
    square matmul, the lightest-weight L1 problem.
    """
    try:
        from kernelbench.dataset import construct_kernelbench_dataset
    except ImportError as exc:
        print(
            "ERROR: `kernelbench` not installed locally. Install with:\n"
            "  uv pip install 'kernelbench @ git+https://github.com/ScalingIntelligence/KernelBench.git'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    dataset = construct_kernelbench_dataset(level=1, source="huggingface")
    problem = dataset.get_problem_by_id(1)
    return problem.name, problem.code


def _build_passthrough_kernel(reference_source: str) -> str:
    """Build a ModelNew that delegates to Model — produces ~1.0x speedup.

    Strategy: copy the reference source verbatim (so Model, get_inputs, and
    any module-level constants are defined), then alias ModelNew = Model.
    Works for every KernelBench problem regardless of forward signature.
    """
    return reference_source + "\n\nModelNew = Model\n"


def _run_and_check() -> int:
    name, ref_src = _load_problem_l1_id1()
    print(f"Loaded KernelBench L1 problem: {name}")

    kernel_src = _build_passthrough_kernel(ref_src)

    print("Calling Modal eval_kernel_on_gpu (L40S, kernelbench format)...")
    eval_fn = modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")
    result: dict[str, Any] = eval_fn.remote(
        kernel_source=kernel_src,
        reference_source=ref_src,
        eval_mode="fast",
        problem_format="kernelbench",
    )

    print("\n=== RESULT ===")
    for k, v in result.items():
        if k == "profile":
            continue
        print(f"  {k}: {v}")

    profile = result.get("profile") or {}
    print("\n=== PROFILE ===")
    for k, v in profile.items():
        if k in ("raw_metrics", "top_stalls"):
            print(f"  {k}: <{type(v).__name__} with {len(v) if hasattr(v, '__len__') else '?'} entries>")
        else:
            print(f"  {k}: {v}")

    print("\n=== ACCEPTANCE CHECKS ===")

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        if not cond:
            failures.append(name)

    status = result.get("status")
    correct = result.get("correct")
    speedup = result.get("speedup", 0.0)
    sol = profile.get("sol_score", 0.0)

    check(
        'status == "correct"',
        status == "correct",
        f"got {status!r}; error={result.get('error')!r}",
    )
    check("correct == True", correct is True, f"got {correct!r}")
    # profile dict must not just exist, but carry a meaningful sol_score
    # (the headline metric). Zero sol_score means something upstream broke.
    check("profile has non-zero sol_score", profile.get("sol_score", 0.0) > 0.0,
          f"sol_score={profile.get('sol_score')}")
    check("sol_score > 0", sol > 0, f"sol_score={sol}")
    check(
        "speedup in [0.7, 1.3] (passthrough baseline)",
        0.7 <= speedup <= 1.3,
        f"speedup={speedup}",
    )

    if failures:
        print(f"\nSMOKE TEST: FAIL ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nSMOKE TEST: PASS")
    return 0


def main() -> None:
    raise SystemExit(_run_and_check())


if __name__ == "__main__":
    main()
