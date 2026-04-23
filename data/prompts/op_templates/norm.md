# Normalization Triton Template

Canonical Triton skeleton for normalization kernels — LayerNorm, RMSNorm, GroupNorm,
BatchNorm (inference). One program per normalization group (per row for LN/RMS,
per group for GN). Adapt for the specific variant and its affine params.

```python
import triton
import triton.language as tl
import torch

# Norms are memory-bound and require a reduction + a rescale.
# BLOCK_N must cover the norm axis in one block (standard pattern).
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4),
        triton.Config({'BLOCK_N': 2048}, num_warps=8),
        triton.Config({'BLOCK_N': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, out_ptr,
    weight_ptr, bias_ptr,     # affine params; remove for RMSNorm (no bias, no mean)
    M, N,
    stride_xm, stride_xn,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_AFFINE: tl.constexpr, # compile-time switch: skip weight/bias loads when False
):
    # One program per row. Row == independent normalization group.
    row = tl.program_id(axis=0)
    x_row = x_ptr + row * stride_xm
    out_row = out_ptr + row * N

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    # Promote to fp32 for stable variance. fp16 variance loses precision badly.
    x = tl.load(x_row + offs * stride_xn, mask=mask, other=0.0).to(tl.float32)

    # --- Two-pass mean + var (simplest, numerically fine for BLOCK_N <= 8k). ---
    # For very long rows, prefer Welford single-pass to halve memory traffic.
    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # LayerNorm: (x - mean) * rstd. RMSNorm: x * rsqrt(mean(x^2) + eps) — skip mean sub.
    x_hat = x_centered * rstd

    # Affine (per-feature weight/bias). Broadcast along row.
    if HAS_AFFINE:
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x_hat = x_hat * w + b

    tl.store(out_row + offs, x_hat.to(out_ptr.dtype.element_ty), mask=mask)


def layernorm_launch(x, weight, bias, eps=1e-5):
    # Flatten leading dims: LayerNorm acts on the last dim, treat (..., N) as (M, N).
    assert x.is_cuda
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    has_affine = weight is not None
    grid = (M,)
    layernorm_kernel[grid](
        x_2d, out, weight, bias, M, N,
        x_2d.stride(0), x_2d.stride(1),
        eps, HAS_AFFINE=has_affine,
    )
    return out.reshape_as(x)
```

## Caveats

- **Always accumulate in fp32.** Variance of fp16 values near each other cancels badly;
  a bf16/fp16 kernel that computes var in low precision will silently give NaNs or
  drifted outputs. Promote at load, cast back at store.
- **Norm axis must fit in one block** for the single-program-per-row pattern. If
  `N > 8192`, switch to a two-kernel design (kernel1: per-tile partial stats;
  kernel2: finalize + rescale) or Welford online reduction.
- **RMSNorm = LayerNorm minus the mean.** `rstd = rsqrt(mean(x^2) + eps)`, no bias.
  Same skeleton; drop the mean-subtract and the bias load.
- **GroupNorm / InstanceNorm.** One program per (batch, group). Compute mean/var
  over the channel subgroup × spatial dims. Flatten spatial × (C/G) into the BLOCK_N
  axis; same reduction pattern.
- **BatchNorm (inference).** Not really a "reduction kernel" — mean/var are frozen
  parameters. Degenerates into an elementwise affine — use `elementwise.md` instead.
- **Fused residual / dropout.** If the norm follows a residual add, load both inputs
  and sum before reducing — one kernel, one pass. Same for dropout mask if applied
  post-norm.
- **Autotune key is `N`.** Block size and warp count depend on norm-axis length.
