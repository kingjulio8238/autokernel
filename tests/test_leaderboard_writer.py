"""Tests for openkernel.benchmarks.leaderboard_writer.

Runnable via:
    python -m pytest tests/test_leaderboard_writer.py
    python tests/test_leaderboard_writer.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _make_spec():
    from openkernel.benchmarks import ProblemSpec

    return ProblemSpec(
        id="kb_l1_0042",
        name="Softmax",
        tier="L1",
        source="kernelbench",
        reference_source="class Model(nn.Module):\n    pass\n",
    )


def _make_result(**overrides):
    base = {
        "kernel_source": "import triton\n# winning kernel source\n",
        "hardware": "L40S",
        "date": "2026-04-21",
        "timestamp": "2026-04-21T12:00:00Z",
        "model": "claude-opus-4-7",
        "speedup": 1.85,
        "sol_score": 0.42,
        "compute_util": 18.4,
        "bandwidth_util": 45.2,
        "bottleneck_type": "memory-bound",
        "correct": True,
        "cost_usd": 0.12,
        "elapsed_s": 142,
        "stop_reason": "Target reached: 0.42 >= 0.30",
        "rounds": 3,
        "iterations": 12,
        "config": {
            "model": "claude-opus-4-7",
            "backend": "triton",
            "target_sol": 0.3,
            "budget": 600,
            "seed": 7,
        },
    }
    base.update(overrides)
    return base


def test_write_then_read_back():
    from openkernel.benchmarks.leaderboard_writer import write_record

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = _make_spec()
        result = _make_result()

        path = write_record(spec, result, root=root)

        assert path.exists(), f"record file should exist at {path}"
        assert path.parent == root / "2026-04-21"
        assert path.name.startswith("kb_l1_0042_L40S_cfg_")
        assert path.name.endswith(".json")

        with open(path, "r") as f:
            loaded = json.load(f)

        assert loaded["schema_version"] == "1.0"
        assert loaded["problem_id"] == "kb_l1_0042"
        assert loaded["problem_name"] == "Softmax"
        assert loaded["tier"] == "L1"
        assert loaded["hardware"] == "L40S"
        assert loaded["date"] == "2026-04-21"
        assert loaded["timestamp"] == "2026-04-21T12:00:00Z"
        assert loaded["model"] == "claude-opus-4-7"
        assert loaded["speedup"] == 1.85
        assert loaded["sol_score"] == 0.42
        assert loaded["compute_util"] == 18.4
        assert loaded["bandwidth_util"] == 45.2
        assert loaded["bottleneck_type"] == "memory-bound"
        assert loaded["correct"] is True
        assert loaded["cost_usd"] == 0.12
        assert loaded["elapsed_s"] == 142
        assert loaded["stop_reason"] == "Target reached: 0.42 >= 0.30"
        assert loaded["rounds"] == 3
        assert loaded["iterations"] == 12

        # derived fields
        assert loaded["kernel_hash"].startswith("") and len(loaded["kernel_hash"]) == 16
        assert all(c in "0123456789abcdef" for c in loaded["kernel_hash"])
        assert loaded["kernel_source_path"] == f"kernels/{loaded['kernel_hash']}.py"
        assert loaded["config_hash"].startswith("cfg_")
        assert 10 <= len(loaded["config_hash"]) <= 20

        # all 22 fields present
        expected_fields = {
            "schema_version", "problem_id", "problem_name", "tier", "hardware",
            "date", "timestamp", "kernel_hash", "kernel_source_path", "model",
            "speedup", "sol_score", "compute_util", "bandwidth_util",
            "bottleneck_type", "correct", "cost_usd", "elapsed_s", "stop_reason",
            "config_hash", "rounds", "iterations",
        }
        assert set(loaded.keys()) == expected_fields, (
            f"field mismatch. missing={expected_fields - set(loaded.keys())}, "
            f"extra={set(loaded.keys()) - expected_fields}"
        )


def test_atomic_on_crash():
    """Simulate a crash between temp-write and rename — no partial file is visible."""
    from openkernel.benchmarks import leaderboard_writer

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = _make_spec()
        result = _make_result()

        # Make os.rename raise to simulate a crash AFTER the temp file was written.
        original_rename = os.rename
        rename_calls = {"n": 0}

        def exploding_rename(src, dst):
            rename_calls["n"] += 1
            # let the kernel companion rename happen; blow up on the record rename
            if str(dst).endswith(".json"):
                raise RuntimeError("simulated crash mid-write")
            return original_rename(src, dst)

        with mock.patch.object(leaderboard_writer.os, "rename", side_effect=exploding_rename):
            try:
                leaderboard_writer.write_record(spec, result, root=root)
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected RuntimeError from simulated crash")

        # The date dir may have a temp file, but NO finalized record.
        date_dir = root / "2026-04-21"
        if date_dir.exists():
            finalized = [
                p for p in date_dir.iterdir()
                if p.suffix == ".json" and ".tmp." not in p.name
            ]
            assert finalized == [], (
                f"no finalized record should exist after crash, found: {finalized}"
            )


def test_kernel_source_companion():
    """The kernel companion file is written and contains the exact source bytes."""
    from openkernel.benchmarks.leaderboard_writer import write_record

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = _make_spec()
        source = "import triton\n# a very specific kernel body for hashing\n"
        result = _make_result(kernel_source=source)

        path = write_record(spec, result, root=root)
        with open(path, "r") as f:
            loaded = json.load(f)

        kernel_path = root / loaded["kernel_source_path"]
        assert kernel_path.exists(), f"companion kernel should exist at {kernel_path}"
        assert kernel_path.parent == root / "kernels"
        assert kernel_path.suffix == ".py"

        with open(kernel_path, "r") as f:
            assert f.read() == source


def test_validation_rejects_missing_field():
    """Omitting a required field raises ValueError and writes nothing."""
    from openkernel.benchmarks.leaderboard_writer import write_record

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = _make_spec()
        result = _make_result()
        del result["speedup"]

        try:
            write_record(spec, result, root=root)
        except ValueError as exc:
            assert "speedup" in str(exc), f"error should mention missing field: {exc}"
        else:
            raise AssertionError("expected ValueError for missing 'speedup' field")

        # nothing should have been written
        assert list(root.rglob("*.json")) == []
        assert not (root / "kernels").exists() or list((root / "kernels").iterdir()) == []


def main() -> int:
    tests = [
        test_write_then_read_back,
        test_atomic_on_crash,
        test_kernel_source_companion,
        test_validation_rejects_missing_field,
    ]

    passed = 0
    failed = 0
    print(f"\n  Running {len(tests)} tests...\n")
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
