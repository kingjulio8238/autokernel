"""Tests for openkernel.benchmarks.ProblemSpec.

Runnable via:
    python -m pytest tests/test_problem_spec.py
    python tests/test_problem_spec.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def test_defaults_when_only_required_fields_provided():
    from openkernel.benchmarks import ProblemSpec

    spec = ProblemSpec(
        id="kb_l1_0042",
        name="Softmax",
        tier="L1",
        source="kernelbench",
        reference_source="class Model(nn.Module):\n    pass\n",
    )

    assert spec.id == "kb_l1_0042"
    assert spec.name == "Softmax"
    assert spec.tier == "L1"
    assert spec.source == "kernelbench"
    assert spec.reference_source == "class Model(nn.Module):\n    pass\n"
    assert spec.workload_spec == {}, (
        f"workload_spec should default to empty dict, got {spec.workload_spec!r}"
    )
    assert spec.expected_dtype == "float32", (
        f"expected_dtype should default to 'float32', got {spec.expected_dtype!r}"
    )


def test_all_fields_preserved():
    from openkernel.benchmarks import ProblemSpec

    workload = {"size": 1024, "seed": 7, "contention": 1}
    spec = ProblemSpec(
        id="gpumode_grayscale",
        name="Grayscale",
        tier="GPU_MODE",
        source="gpumode",
        reference_source="def ref_kernel(data):\n    return data\n",
        workload_spec=workload,
        expected_dtype="float16",
    )

    assert spec.id == "gpumode_grayscale"
    assert spec.name == "Grayscale"
    assert spec.tier == "GPU_MODE"
    assert spec.source == "gpumode"
    assert spec.reference_source == "def ref_kernel(data):\n    return data\n"
    assert spec.workload_spec == workload
    assert spec.expected_dtype == "float16"


def test_frozen_mutation_raises():
    from openkernel.benchmarks import ProblemSpec

    spec = ProblemSpec(
        id="kb_l2_0001",
        name="Conv2D",
        tier="L2",
        source="kernelbench",
        reference_source="# ref\n",
    )

    try:
        spec.name = "Mutated"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError(
            "Expected FrozenInstanceError when mutating ProblemSpec.name"
        )


def test_invalid_tier_raises():
    from openkernel.benchmarks import ProblemSpec

    try:
        ProblemSpec(
            id="kb_l1_0001",
            name="Softmax",
            tier="XXL",  # type: ignore[arg-type]
            source="kernelbench",
            reference_source="# ref\n",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid tier 'XXL'")


def test_invalid_source_raises():
    from openkernel.benchmarks import ProblemSpec

    try:
        ProblemSpec(
            id="kb_l1_0001",
            name="Softmax",
            tier="L1",
            source="not-a-source",  # type: ignore[arg-type]
            reference_source="# ref\n",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid source 'not-a-source'")


def test_empty_id_raises():
    from openkernel.benchmarks import ProblemSpec

    try:
        ProblemSpec(
            id="",
            name="Softmax",
            tier="L1",
            source="kernelbench",
            reference_source="# ref\n",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty id")


def test_empty_reference_source_raises():
    from openkernel.benchmarks import ProblemSpec

    try:
        ProblemSpec(
            id="kb_l1_0001",
            name="Softmax",
            tier="L1",
            source="kernelbench",
            reference_source="",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty reference_source")


def main() -> int:
    tests = [
        test_defaults_when_only_required_fields_provided,
        test_all_fields_preserved,
        test_frozen_mutation_raises,
        test_invalid_tier_raises,
        test_invalid_source_raises,
        test_empty_id_raises,
        test_empty_reference_source_raises,
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
