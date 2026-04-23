"""Regression test for intra-container CUDA contamination (phase 1.D.6-unblocker).

Bug: When a kernel triggered an illegal memory access on the Modal eval
container, subsequent evals in the same container saw a sticky CUDA error
because the driver-level device state was wedged. Container reuse amortized
cold-starts but poisoned the CUDA context across unrelated calls.

Fix: modal_infra/app.py._maybe_schedule_container_death() schedules
os._exit(1) ~2s after returning a cuda_error, forcing Modal to rotate to a
fresh container for the next call.

This test locks the fix in by:
  1) Submitting a Triton kernel that performs an unmasked out-of-bounds
     load — reliably produces an illegal memory access → status=cuda_error.
  2) Immediately submitting a trivially-correct identity-copy Triton kernel
     to the SAME Modal function. Without the fix this would inherit the
     poisoned context and fail with a sticky cuda_error. With the fix, the
     second call lands on a fresh container and returns status=correct.

Skipped automatically when Modal is unreachable so CI without Modal creds
doesn't hang.
"""

from __future__ import annotations

import pytest

# Skip the whole module if modal isn't importable (CI without deps).
modal = pytest.importorskip("modal")


# ----- GPU Mode problem format (simplest working shape for Triton kernels) -----
# _run_eval_gpumode expects:
#   reference module: ref_kernel(data), generate_input(size, seed), optional WORKLOAD_SPEC
#   kernel module:    kernel_function(data) or kernel_function(*data)

# Tiny fixed workload so Modal cost is minimal.
_WORKLOAD = '''
WORKLOAD_SPEC = {"size": 1024}

import torch

def generate_input(size, seed=0):
    torch.manual_seed(seed)
    return torch.randn(size, device="cuda", dtype=torch.float32)
'''

# Reference: identity copy. Used by BOTH good and bad tests so the same
# reference semantics apply — only the candidate kernel changes.
REFERENCE_SOURCE = _WORKLOAD + '''
def ref_kernel(x):
    return x.clone()
'''

# BAD kernel: Triton gather with deliberately-huge indices so the address
# computation `x_ptr + indices[i]` lands FAR outside the CUDA allocator's
# arena — guaranteed "CUDA illegal memory access was encountered".
#
# Important note on why a "simple unmasked OOB load past tensor end" is NOT
# enough: PyTorch's caching allocator hands out chunks from a larger pool,
# so reading a few thousand floats past the tensor end often stays inside
# mapped memory and silently returns garbage (no fault). We need to jump
# GB into unmapped address space. Using an indices tensor with a huge
# stride multiplier reliably does this and is deterministic.
BAD_KERNEL_SOURCE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _bad_gather_kernel(x_ptr, idx_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load the (deliberately bogus) indices and gather through them.
    # indices point GB past the tensor base -> unmapped address -> fault.
    idx = tl.load(idx_ptr + offsets)
    vals = tl.load(x_ptr + idx)
    tl.store(out_ptr + offsets, vals)

def kernel_function(x):
    N = x.numel()
    out = torch.empty_like(x)
    # Indices deliberately in the billions -> address is far outside any
    # CUDA allocator arena -> illegal memory access.
    bogus = torch.full((N,), 2_000_000_000, dtype=torch.int64, device=x.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _bad_gather_kernel[grid](x, bogus, out, BLOCK_SIZE=BLOCK_SIZE)
    return out
'''

# GOOD kernel: correctly-masked Triton identity copy. Must pass correctness
# against the identity-copy reference.
GOOD_KERNEL_SOURCE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _safe_copy_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    vals = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, vals, mask=mask)

def kernel_function(x):
    N = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _safe_copy_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out
'''


def _modal_available() -> bool:
    """True iff the deployed openkernel-eval Modal app is reachable."""
    try:
        # from_name is a cheap handle creation — does not make a network call.
        # We treat an ImportError / lookup failure as "Modal unavailable".
        modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _modal_available(),
    reason="Modal app 'openkernel-eval' not reachable — skipping live GPU test",
)


def test_cuda_contamination_does_not_leak_across_calls():
    """Bad kernel poisons the context; the next call must still succeed.

    Sequencing (same Modal function handle, back-to-back):
      call #1: BAD kernel  -> expect status == "cuda_error"
      call #2: GOOD kernel -> expect status != "cuda_error" AND correct is True

    Without the container-death fix, call #2 historically returned
    status=cuda_error because the prior call's illegal memory access wedged
    the CUDA driver state inside the reused container.
    """
    eval_fn = modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")

    r1 = eval_fn.remote(
        kernel_source=BAD_KERNEL_SOURCE,
        reference_source=REFERENCE_SOURCE,
        gpu_type="L40S",
    )
    assert isinstance(r1, dict), f"expected dict result, got {type(r1)}: {r1!r}"
    assert r1.get("status") == "cuda_error", (
        f"BAD kernel should surface as cuda_error; got status="
        f"{r1.get('status')!r}, error={r1.get('error')!r}"
    )

    r2 = eval_fn.remote(
        kernel_source=GOOD_KERNEL_SOURCE,
        reference_source=REFERENCE_SOURCE,
        gpu_type="L40S",
    )
    assert isinstance(r2, dict), f"expected dict result, got {type(r2)}: {r2!r}"
    assert r2.get("status") != "cuda_error", (
        f"sticky cuda_error after container rotation: status="
        f"{r2.get('status')!r}, error={r2.get('error')!r}"
    )
    assert r2.get("correct") is True, (
        f"GOOD identity-copy kernel should pass correctness: status="
        f"{r2.get('status')!r}, correct={r2.get('correct')!r}, "
        f"error={r2.get('error')!r}"
    )
