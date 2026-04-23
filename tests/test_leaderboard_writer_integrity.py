"""Tests for kernel-identity integrity guards in leaderboard_writer.

Runnable via:
    uv run python -m pytest tests/test_leaderboard_writer_integrity.py -v
    python -m pytest tests/test_leaderboard_writer_integrity.py -v
    python tests/test_leaderboard_writer_integrity.py

The guards defend against cross-problem kernel leakage (e.g. a histogram
kernel being written against a prefixsum record, which happened on
2026-04-22 via OPENKERNEL_BEST_KERNEL env-var leak).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


_PREFIXSUM_REF = """\
from utils import match_reference, DeterministicContext
import torch
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    \"\"\"Reference implementation of inclusive prefix sum using PyTorch.\"\"\"
    with DeterministicContext():
        data, output = data
        output = torch.cumsum(data.to(torch.float64), dim=0).to(torch.float64)
        return output
"""

# First ~60 chars of the contaminated histogram kernel that slipped past
# the writer on 2026-04-22 (results/leaderboard/kernels/1a4e9268e6226103.py).
_HISTOGRAM_KERNEL_SRC = """\
#!/usr/bin/env python3
\"\"\"
Kernel implementation for a 256-bin histogram over a uint8 tensor
using Triton. This kernel is designed to be fused into a single pass.
\"\"\"

import triton
import triton.language as tl
import torch


@triton.jit
def _histogram_kernel(data_ptr, hist_ptr):
    pass


def kernel_function(args):
    return args
"""

_PREFIXSUM_KERNEL_SRC = """\
\"\"\"Triton inclusive prefix sum / cumsum kernel.\"\"\"

import triton
import triton.language as tl
import torch


@triton.jit
def _prefixsum_kernel(x_ptr, y_ptr, N):
    pass


def kernel_function(args):
    data, out = args
    return out
"""


def _make_prefixsum_spec():
    from openkernel.benchmarks import ProblemSpec

    return ProblemSpec(
        id="gpumode_prefixsum",
        name="Prefixsum",
        tier="GPU_MODE",
        source="gpumode",
        reference_source=_PREFIXSUM_REF,
    )


def _make_other_spec():
    """A second, unrelated spec — for cross-problem reuse tests."""
    from openkernel.benchmarks import ProblemSpec

    return ProblemSpec(
        id="gpumode_histogram",
        name="Histogram",
        tier="GPU_MODE",
        source="gpumode",
        reference_source=(
            "def ref_kernel(data):\n"
            "    # 256-bin histogram reference using torch.histc\n"
            "    return torch.histc(data.float(), bins=256)\n"
        ),
    )


def _make_result(kernel_source: str, **overrides):
    base = {
        "kernel_source": kernel_source,
        "hardware": "L40S",
        "date": "2026-04-23",
        "timestamp": "2026-04-23T12:00:00Z",
        "model": "o3-mini",
        "speedup": 1.50,
        "sol_score": 0.10,
        "compute_util": 5.0,
        "bandwidth_util": 20.0,
        "bottleneck_type": "memory-bound",
        "correct": True,
        "cost_usd": 0.02,
        "elapsed_s": 100,
        "stop_reason": "target reached",
        "rounds": 2,
        "iterations": 5,
        "config": {"model": "o3-mini", "backend": "triton", "seed": 7},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Case 1: prefixsum spec + histogram kernel source → raises
# ---------------------------------------------------------------------------
def test_prefixsum_spec_with_histogram_kernel_raises():
    from openkernel.benchmarks.leaderboard_writer import write_record

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = _make_prefixsum_spec()
        result = _make_result(_HISTOGRAM_KERNEL_SRC)

        with pytest.raises(ValueError) as excinfo:
            write_record(spec, result, root=root)

        msg = str(excinfo.value)
        assert "histogram" in msg.lower(), (
            f"error should name the mismatched token: {msg}"
        )
        assert "gpumode_prefixsum" in msg, f"error should name the spec: {msg}"

        # no partial files were written
        date_dir = root / "2026-04-23"
        kernels_dir = root / "kernels"
        assert not date_dir.exists() or list(date_dir.iterdir()) == []
        assert not kernels_dir.exists() or list(kernels_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# Case 2: prefixsum spec + prefixsum kernel source → passes
# ---------------------------------------------------------------------------
def test_prefixsum_spec_with_prefixsum_kernel_passes():
    from openkernel.benchmarks.leaderboard_writer import write_record

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = _make_prefixsum_spec()
        result = _make_result(_PREFIXSUM_KERNEL_SRC)

        path = write_record(spec, result, root=root)
        assert path.exists(), f"record should be written at {path}"
        assert path.name.startswith("gpumode_prefixsum_L40S_cfg_")


# ---------------------------------------------------------------------------
# Case 3: same kernel hash reused across problems on same date → raises
# on 2nd write
# ---------------------------------------------------------------------------
def test_cross_problem_kernel_hash_reuse_raises_on_second_write():
    from openkernel.benchmarks.leaderboard_writer import write_record

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Shared neutral kernel body — no op tokens so the heuristic does
        # NOT fire; only the cross-problem-reuse guard should trigger.
        shared_src = (
            '"""A generic kernel body used to probe hash-reuse guard."""\n'
            "import triton\n\n"
            "def kernel_function(x):\n    return x\n"
        )

        # 1st write: prefixsum spec — must succeed (reference_source has
        # "cumsum"/"prefix sum" but the shared kernel mentions neither,
        # so the op-token heuristic stays silent).
        spec_a = _make_prefixsum_spec()
        result_a = _make_result(shared_src)
        path_a = write_record(spec_a, result_a, root=root)
        assert path_a.exists()

        # 2nd write: a DIFFERENT problem_id, same date, same kernel bytes
        # — must raise.
        spec_b = _make_other_spec()
        result_b = _make_result(shared_src)

        with pytest.raises(ValueError) as excinfo:
            write_record(spec_b, result_b, root=root)

        msg = str(excinfo.value)
        assert "kernel_hash" in msg or "kernel-identity" in msg, (
            f"error should mention hash-reuse cause: {msg}"
        )
        assert "gpumode_prefixsum" in msg, (
            f"error should name the prior problem: {msg}"
        )


# ---------------------------------------------------------------------------
# Case 4: placeholder source with correct=True → raises
# ---------------------------------------------------------------------------
def test_placeholder_source_with_correct_true_raises():
    from openkernel.benchmarks.leaderboard_writer import write_record

    placeholder = "# No correct kernel produced for this attempt.\n"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = _make_prefixsum_spec()
        result = _make_result(placeholder, correct=True, speedup=1.5)

        with pytest.raises(ValueError) as excinfo:
            write_record(spec, result, root=root)

        msg = str(excinfo.value)
        assert "placeholder" in msg.lower(), (
            f"error should explain the placeholder problem: {msg}"
        )


# ---------------------------------------------------------------------------
# Case 5: placeholder source with correct=False → passes (failure record)
# ---------------------------------------------------------------------------
def test_placeholder_source_with_correct_false_passes():
    from openkernel.benchmarks.leaderboard_writer import write_record

    placeholder = "# No correct kernel produced for this attempt.\n"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = _make_prefixsum_spec()
        # correct=False failure record: speedup=0.0 sentinel, rounds=0 ok
        result = _make_result(
            placeholder,
            correct=False,
            speedup=0.0,
            sol_score=0.0,
            compute_util=0.0,
            bandwidth_util=0.0,
            bottleneck_type="unknown",
            rounds=0,
            stop_reason="no correct kernel",
        )

        path = write_record(spec, result, root=root)
        assert path.exists(), f"failure record should be written at {path}"


# ---------------------------------------------------------------------------
# Case 6: failure placeholder reused across problems on same date → passes
# (guard #3 must NOT fire on the failure placeholder — by design all
# failures collapse to one companion file at
# ``kernels/27c7dcf3ca247272.py``, so collision is contract, not leak)
# ---------------------------------------------------------------------------
def test_placeholder_cross_problem_reuse_passes():
    from openkernel.benchmarks.leaderboard_writer import write_record

    placeholder = "# No correct kernel produced for this attempt.\n"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # 1st failure record: prefixsum.
        spec_a = _make_prefixsum_spec()
        result_a = _make_result(
            placeholder,
            correct=False,
            speedup=0.0,
            sol_score=0.0,
            compute_util=0.0,
            bandwidth_util=0.0,
            bottleneck_type="unknown",
            rounds=0,
            stop_reason="no correct kernel",
        )
        path_a = write_record(spec_a, result_a, root=root)
        assert path_a.exists()

        # 2nd failure record: histogram, same date. SAME placeholder hash.
        # Must succeed — this is exactly the blind spot that dropped
        # gpumode_histogram's failure record on 2026-04-23 pre-patch.
        spec_b = _make_other_spec()
        result_b = _make_result(
            placeholder,
            correct=False,
            speedup=0.0,
            sol_score=0.0,
            compute_util=0.0,
            bandwidth_util=0.0,
            bottleneck_type="unknown",
            rounds=0,
            stop_reason="no correct kernel",
        )
        path_b = write_record(spec_b, result_b, root=root)
        assert path_b.exists(), (
            f"second failure record with same placeholder must write at {path_b}"
        )
        # Both records persisted despite shared kernel_hash.
        assert path_a != path_b


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
