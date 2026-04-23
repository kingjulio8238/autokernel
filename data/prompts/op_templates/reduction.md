# Reduction Triton Template

Canonical Triton skeleton for 1-axis reductions — sum, mean, max, argmax, softmax along
a single dimension. One program per row (or per reduction group); reduction axis fits in
one block where possible. Adapt the combine op for your reduction.

```python
import triton
import triton.language as tl
import torch

# Reductions are bandwidth-bound for small N and latency-bound for large N.
# BLOCK_N must be >= N in the single-block pattern; otherwise use a two-pass variant.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4),
        triton.Config({'BLOCK_N': 2048}, num_warps=8),
        triton.Config({'BLOCK_N': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def reduction_kernel(
    x_ptr, out_ptr,
    M, N,                    # reduce axis N, batch axis M
    stride_xm, stride_xn,    # pass strides — do NOT assume contiguous on reduce axis
    BLOCK_N: tl.constexpr,
):
    # One program == one row == one independent reduction.
    # This pattern assumes N <= BLOCK_N (single-block reduction).
    # For larger N, tile over N and use atomic_add or two-pass.
    row = tl.program_id(axis=0)

    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    # `other=-inf` for max, `other=0.0` for sum — identity element of the op.
    x = tl.load(
        x_ptr + row * stride_xm + offs_n * stride_xn,
        mask=mask,
        other=0.0,
    )

    # --- REDUCTION BODY: pick one; mix in epilogue if fused. ---
    # Sum / mean:
    #   out = tl.sum(x, axis=0)        # mean: out / N after
    # Max / argmax:
    #   out = tl.max(x, axis=0)
    # Softmax (numerically stable online form, recommended):
    x_max = tl.max(tl.where(mask, x, -float('inf')), axis=0)
    x_exp = tl.exp(x - x_max)
    x_exp = tl.where(mask, x_exp, 0.0)
    denom = tl.sum(x_exp, axis=0)
    out = x_exp / denom
    # -----------------------------------------------------------

    # Softmax writes a vector; scalar reductions (sum/max) write one element.
    tl.store(out_ptr + row * N + offs_n, out, mask=mask)


def reduction_launch(x: torch.Tensor) -> torch.Tensor:
    # Canonicalize: reduce axis should be the LAST (innermost) axis for
    # coalesced loads. Transpose + contiguous at the wrapper if not.
    assert x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (M,)
    reduction_kernel[grid](
        x, out, M, N, x.stride(0), x.stride(1),
    )
    return out
```

## Caveats

- **Reduction axis must be innermost for coalescing.** If reducing dim 0 of a 2D tensor,
  transpose and make contiguous in the wrapper. `tl.load` along a strided axis is fine
  but slow; along contiguous it's peak bandwidth.
- **Compute in fp32.** For fp16/bf16 inputs, promote with `x.to(tl.float32)` before the
  reduction. Accumulating many fp16 values loses precision fast — softmax especially
  diverges without fp32 accumulation.
- **Single-block vs multi-block.** If `N <= 8192` (fits in one block × warps), use the
  one-program-per-row pattern above. For larger N, either (a) two-pass: kernel1 emits
  partial sums/maxes per tile, kernel2 combines; or (b) atomic_add into global output
  (sum only — not max/softmax).
- **Online softmax for attention.** For softmax inside a fused attention kernel, use the
  streaming Flash-style update (`m_new = max(m, x)`, rescale running sum) — see
  `attention.md`. The standalone form above is fine for a plain softmax op.
- **Autotune key is `N`.** Different reduction lengths want different block sizes and
  warp counts. Don't key on M (doesn't affect per-program work).
- **`tl.reduce` / `tl.sum` / `tl.max` require the reduce axis to be a constexpr-shaped
  dim.** `axis=0` reduces the `BLOCK_N` dimension here; don't try to reduce over a
  runtime-shaped axis.
