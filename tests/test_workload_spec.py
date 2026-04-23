"""Tests for reference.WORKLOAD_SPEC shape and signature compatibility.

Runnable via:
    python -m pytest tests/test_workload_spec.py
    python tests/test_workload_spec.py

These tests deliberately avoid invoking reference.generate_input — that
would require CUDA and a real input_t dataclass from the gpumode harness.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _import_reference():
    """Import reference.py, stubbing out harness-only modules.

    reference.py does `from utils import ...` and `from task import ...`,
    which resolve against the gpumode benchmark dir at runtime. For a
    pure import-time test we stub them with empty modules.
    """
    import types

    if "utils" not in sys.modules:
        utils_stub = types.ModuleType("utils")
        utils_stub.make_match_reference = lambda *a, **kw: None
        utils_stub.DeterministicContext = type(
            "DeterministicContext",
            (),
            {"__enter__": lambda self: self, "__exit__": lambda *a: False},
        )
        sys.modules["utils"] = utils_stub

    if "task" not in sys.modules:
        task_stub = types.ModuleType("task")
        task_stub.input_t = object
        task_stub.output_t = object
        sys.modules["task"] = task_stub

    import importlib
    if "reference" in sys.modules:
        return importlib.reload(sys.modules["reference"])
    return importlib.import_module("reference")


def test_workload_spec_present_and_well_formed():
    reference = _import_reference()

    assert hasattr(reference, "WORKLOAD_SPEC"), "reference.WORKLOAD_SPEC missing"
    spec = reference.WORKLOAD_SPEC
    assert isinstance(spec, dict), f"WORKLOAD_SPEC must be a dict, got {type(spec).__name__}"

    for key in ("size", "seed", "contention"):
        assert key in spec, f"WORKLOAD_SPEC missing key {key!r}"

    for key, value in spec.items():
        assert isinstance(value, int), (
            f"WORKLOAD_SPEC[{key!r}] must be int, got {type(value).__name__}"
        )


def test_generate_input_accepts_workload_spec_kwargs():
    reference = _import_reference()

    sig = inspect.signature(reference.generate_input)
    params = set(sig.parameters.keys())

    for key in reference.WORKLOAD_SPEC:
        assert key in params, (
            f"generate_input signature is missing kwarg {key!r}; "
            f"has {sorted(params)}"
        )


def main() -> int:
    tests = [
        test_workload_spec_present_and_well_formed,
        test_generate_input_accepts_workload_spec_kwargs,
    ]
    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  [PASS] {fn.__name__}")

    total = passed + failed
    print(f"\n  RESULTS: {passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
