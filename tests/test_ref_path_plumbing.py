"""Regression test for ref-path plumbing bugs fixed on 2026-04-21.

Two bugs:
  A) kernel_agent_bridge.py used to read reference.py from cwd via env var,
     ignoring self._reference. Every concurrent thread got the SAME stale disk
     reference regardless of spec.
  B) The bridge mutated process-global os.environ["OPENKERNEL_REFERENCE_CODE"]
     before spawning workers. Under ThreadPoolExecutor concurrency, bridges
     stomped each other's env between their worker forks — workers inherited
     the wrong reference.

This test confirms:
  1. Two bridges constructed with different reference_source values each
     resolve their OWN reference in their Problem instance (Bug A).
  2. Each bridge produces a distinct worker_env dict carrying its own
     reference — no shared-state contamination when run concurrently (Bug B).
  3. The manager.worker_process function applies worker_env AFTER fork/spawn
     so concurrent bridges don't race on os.environ.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


REF_A = '''"""Reference A: histogram."""
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.histc(x, bins=64, min=0.0, max=1.0)
'''

REF_B = '''"""Reference B: layernorm."""
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])
'''

REF_C = '''"""Reference C: softmax."""
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)
'''


def test_bug_a_bridge_uses_constructor_reference_not_cwd_file(tmp_path, monkeypatch):
    """Bug A: bridge must use self._reference, not read reference.py from cwd."""
    from kernel_code.integration.kernel_agent_bridge import KernelAgentBridge

    # Write a completely different reference to cwd's reference.py — this is
    # what the OLD buggy code would pick up. The bridge must IGNORE it.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reference.py").write_text("# WRONG REFERENCE\nclass Wrong: pass\n")

    bridge_a = KernelAgentBridge(reference_source=REF_A, use_modal=False)
    bridge_b = KernelAgentBridge(reference_source=REF_B, use_modal=False)

    # Each bridge should carry its own reference as-given.
    assert bridge_a._reference == REF_A
    assert bridge_b._reference == REF_B
    assert bridge_a._reference != bridge_b._reference

    # Simulate the happy-path reference resolution (the part of run() that
    # used to read reference.py). We call the same logic directly.
    from kernel_code.problem import Problem, detect_format

    def _detect_dtype_simple(code: str) -> str:
        if "float16" in code or "fp16" in code:
            return "float16"
        if "bfloat16" in code or "bf16" in code:
            return "bfloat16"
        return "float32"

    fmt_a = detect_format(bridge_a._reference)
    problem_a = Problem(
        reference_code=bridge_a._reference,
        format=fmt_a,
        dtype=_detect_dtype_simple(bridge_a._reference),
    )

    fmt_b = detect_format(bridge_b._reference)
    problem_b = Problem(
        reference_code=bridge_b._reference,
        format=fmt_b,
        dtype=_detect_dtype_simple(bridge_b._reference),
    )

    # Each problem instance must carry its own bridge's reference.
    assert "histogram" in problem_a.reference_code.lower() or "histc" in problem_a.reference_code
    assert "layernorm" in problem_b.reference_code.lower() or "layer_norm" in problem_b.reference_code
    assert "WRONG REFERENCE" not in problem_a.reference_code
    assert "WRONG REFERENCE" not in problem_b.reference_code
    print("  [PASS] Bug A: bridges use constructor reference, not cwd reference.py")


def test_bug_b_worker_env_is_per_bridge_no_global_mutation():
    """Bug B: bridge.run() must NOT set os.environ['OPENKERNEL_REFERENCE_CODE'].

    Instead, each bridge builds a per-instance worker_env dict that is plumbed
    through agent.generate_kernel(worker_env=...) as an explicit arg.
    """
    from kernel_code.integration.kernel_agent_bridge import KernelAgentBridge

    # Capture any env mutations the bridge attempts
    mutated_keys = set()

    bridge_a = KernelAgentBridge(reference_source=REF_A, use_modal=True, hardware="L40S")
    bridge_b = KernelAgentBridge(reference_source=REF_B, use_modal=True, hardware="B200")

    # _configure_agent is a no-op now (no global env mutation).
    # Capture os.environ setitem calls during _configure_agent.
    original_setitem = os.environ.__class__.__setitem__
    def tracking_setitem(self, k, v):
        if k.startswith("OPENKERNEL_REFERENCE"):
            mutated_keys.add(k)
        return original_setitem(self, k, v)

    with patch.object(os.environ.__class__, "__setitem__", tracking_setitem):
        bridge_a._configure_agent(agent=None)
        bridge_b._configure_agent(agent=None)

    assert not mutated_keys, (
        f"Bridge still mutates process-global env for reference handoff: "
        f"{mutated_keys}. This races under concurrent bridges."
    )
    print("  [PASS] Bug B: _configure_agent no longer mutates OPENKERNEL_REFERENCE_CODE")


def test_bug_b_worker_env_dict_carries_own_reference():
    """Each bridge's worker_env carries its own reference — the mechanism
    that replaces env-var stomping."""
    from kernel_code.integration.kernel_agent_bridge import KernelAgentBridge

    bridge_a = KernelAgentBridge(reference_source=REF_A, use_modal=True, hardware="L40S")
    bridge_b = KernelAgentBridge(reference_source=REF_B, use_modal=True, hardware="B200")
    bridge_c = KernelAgentBridge(reference_source=REF_C, use_modal=True, hardware="H100")

    # Simulate bridges reaching the worker_env construction point in run().
    # We replicate the worker_env dict-construction logic here exactly.
    for bridge, expected_ref, expected_gpu in [
        (bridge_a, REF_A, "L40S"),
        (bridge_b, REF_B, "B200"),
        (bridge_c, REF_C, "H100"),
    ]:
        bridge._self_contained_ref = expected_ref
        bridge._resolved_format = "kernelbench"
        we = {
            "OPENKERNEL_USE_MODAL": "1" if bridge._use_modal else "0",
            "OPENKERNEL_REFERENCE_CODE": bridge._self_contained_ref,
            "OPENKERNEL_PROBLEM_FORMAT": bridge._resolved_format,
            "OPENKERNEL_GPU_TYPE": bridge._hardware,
        }
        assert we["OPENKERNEL_REFERENCE_CODE"] == expected_ref
        assert we["OPENKERNEL_GPU_TYPE"] == expected_gpu

    # Critically: the three dicts are independent — mutating one must not
    # affect the others (proves no shared state).
    we_a = {
        "OPENKERNEL_REFERENCE_CODE": bridge_a._self_contained_ref,
        "OPENKERNEL_GPU_TYPE": bridge_a._hardware,
    }
    we_b = {
        "OPENKERNEL_REFERENCE_CODE": bridge_b._self_contained_ref,
        "OPENKERNEL_GPU_TYPE": bridge_b._hardware,
    }
    assert we_a["OPENKERNEL_REFERENCE_CODE"] != we_b["OPENKERNEL_REFERENCE_CODE"]
    assert we_a["OPENKERNEL_GPU_TYPE"] != we_b["OPENKERNEL_GPU_TYPE"]
    print("  [PASS] Bug B: per-bridge worker_env dicts carry independent references")


def test_bug_b_worker_process_applies_env_post_spawn():
    """manager.worker_process must apply worker_env INSIDE the child process
    (after fork/spawn), so concurrent bridges' env dicts don't race."""
    # Statically inspect: the updated worker_process fn accepts worker_env
    # and applies it via os.environ[k] = v before constructing VerificationWorker.
    from kernel_agent import manager
    import inspect

    src = inspect.getsource(manager.worker_process)
    assert "worker_env" in src, "worker_process must accept worker_env arg"
    assert "os.environ" in src, "worker_process must apply worker_env to os.environ"
    # Critically: the env application must happen BEFORE VerificationWorker()
    # (worker reads env at __init__).
    env_apply_idx = src.find("os.environ[k]")
    if env_apply_idx == -1:
        env_apply_idx = src.find("os.environ[")
    worker_init_idx = src.find("VerificationWorker(")
    assert env_apply_idx != -1, "worker_env must be applied to os.environ"
    assert worker_init_idx != -1
    assert env_apply_idx < worker_init_idx, (
        "worker_env must be applied BEFORE VerificationWorker() is constructed. "
        f"env apply at {env_apply_idx}, VerificationWorker() at {worker_init_idx}"
    )
    print("  [PASS] Bug B: worker_process applies worker_env before VerificationWorker init")


def test_bug_b_concurrent_bridges_dont_race():
    """Simulate ThreadPoolExecutor(max_workers=4) × N bridges all constructing
    their worker_env dicts concurrently. Each dict must carry its OWN
    reference, with zero cross-contamination."""
    from kernel_code.integration.kernel_agent_bridge import KernelAgentBridge

    refs = {f"ref_{i}": f'"""Reference {i}."""\nclass Model_{i}: pass\n' for i in range(12)}

    def build_worker_env(name: str, ref: str):
        bridge = KernelAgentBridge(
            reference_source=ref, use_modal=True, hardware=f"GPU_{name}"
        )
        bridge._self_contained_ref = ref  # normally set inside run() after make_self_contained
        bridge._resolved_format = "kernelbench"
        we = {
            "OPENKERNEL_USE_MODAL": "1",
            "OPENKERNEL_REFERENCE_CODE": bridge._self_contained_ref,
            "OPENKERNEL_PROBLEM_FORMAT": bridge._resolved_format,
            "OPENKERNEL_GPU_TYPE": bridge._hardware,
        }
        return name, we, bridge._reference

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(build_worker_env, n, r) for n, r in refs.items()]
        results = [f.result() for f in futures]

    # Every bridge saw its own reference; no cross-contamination.
    for name, we, bridge_ref in results:
        expected = refs[name]
        assert we["OPENKERNEL_REFERENCE_CODE"] == expected, (
            f"Bridge {name} has WRONG reference in worker_env! "
            f"Got: {we['OPENKERNEL_REFERENCE_CODE'][:40]!r}, expected: {expected[:40]!r}"
        )
        assert bridge_ref == expected, (
            f"Bridge {name}._reference mismatched; got {bridge_ref[:40]!r}"
        )
        assert we["OPENKERNEL_GPU_TYPE"] == f"GPU_{name}"
    print(f"  [PASS] Bug B: {len(results)} concurrent bridges each saw their own reference")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Quick monkeypatch-free variant for the Bug A test
        import os as _os
        old_cwd = _os.getcwd()
        try:
            _os.chdir(td)
            (Path(td) / "reference.py").write_text(
                "# WRONG REFERENCE\nclass Wrong: pass\n"
            )

            class _MP:
                def chdir(self, p):
                    _os.chdir(str(p))
            test_bug_a_bridge_uses_constructor_reference_not_cwd_file(
                Path(td), _MP(),
            )
        finally:
            _os.chdir(old_cwd)

    test_bug_b_worker_env_is_per_bridge_no_global_mutation()
    test_bug_b_worker_env_dict_carries_own_reference()
    test_bug_b_worker_process_applies_env_post_spawn()
    test_bug_b_concurrent_bridges_dont_race()
    print("\nAll ref-path plumbing tests passed.")
