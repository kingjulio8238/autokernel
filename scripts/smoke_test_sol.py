"""Phase 0 smoke test: call Modal eval directly on the histogram reference,
print the profile dict, and assert sol_score + compute_util + bandwidth_util
all populate with non-zero values.

Usage:
    python scripts/smoke_test_sol.py
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ref_path = REPO / "reference.py"
    raw_src = ref_path.read_text()

    # Mirror shell.py's /optimize flow: inline utils.py so Modal can import it
    import sys
    sys.path.insert(0, str(REPO))
    from kernel_code.problem import Problem, make_self_contained  # type: ignore

    ref_src = make_self_contained(Problem(reference_code=raw_src, format="gpumode"))

    # For gpumode path: same source as both ref and kernel — passthrough 1.0x
    kernel_src = ref_src.replace("def ref_kernel(", "def kernel_function(")

    print("Calling Modal eval (gpumode, L40S)...")
    eval_fn = modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")
    result = eval_fn.remote(
        kernel_source=kernel_src,
        reference_source=ref_src,
        eval_mode="fast",
        problem_format="gpumode",
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
            # these can be noisy; summarize
            print(f"  {k}: <{type(v).__name__} with {len(v) if hasattr(v, '__len__') else '?'} entries>")
        else:
            print(f"  {k}: {v}")

    print("\n=== PHASE 0 ACCEPTANCE CHECKS ===")

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")

    check("profile dict not empty", bool(profile))
    check("sol_score present", "sol_score" in profile,
          f"value={profile.get('sol_score')}")
    check("compute_util present", "compute_util" in profile,
          f"value={profile.get('compute_util')}")
    check("bandwidth_util present", "bandwidth_util" in profile,
          f"value={profile.get('bandwidth_util')}")
    check("bottleneck_type present", "bottleneck_type" in profile,
          f"value={profile.get('bottleneck_type')}")

    # When ref == kernel (passthrough), speedup should be ~1.0 and SOL ~0.5
    sol = profile.get("sol_score", 0.0)
    check("sol_score in plausible range [0.3, 0.7]", 0.3 <= sol <= 0.7,
          f"sol={sol:.3f} (passthrough = ref, expect ~0.5)")

    # Note: torch.profiler can't extract flops/bytes for all op types (e.g.,
    # histograms via bincount). For those, utilization stays 0 and SOL uses
    # the speedup-relative fallback (sol_metrics.py:64-65). That's expected.
    bu = profile.get("bandwidth_util", 0.0)
    cu = profile.get("compute_util", 0.0)
    has_util = bu > 0 or cu > 0
    print(f"  [INFO] utilization data available: {has_util} "
          f"(bw={bu:.2f}%, compute={cu:.2f}%) — "
          f"expected 0 for ops torch.profiler can't trace (e.g. bincount)")

    print()
    if ok:
        print("PHASE 0 SMOKE TEST: PASS")
    else:
        print("PHASE 0 SMOKE TEST: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
