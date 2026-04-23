# Elementwise Triton Template

Canonical Triton skeleton for elementwise (pointwise) kernels — activations (relu, gelu, silu),
binary ops (add, mul, sub), and any unary/binary transform with no reduction axis.
Adapt the op body for your specific function and fold nearby elementwise ops into one kernel.

```python
import triton
import triton.language as tl
import torch

# Elementwise kernels are memory-bound: the right BLOCK_SIZE is a
# tradeoff between occupancy and coalesced loads. Autotune over a few.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],  # re-tune when total size changes category
)
@triton.jit
def elementwise_kernel(
    x_ptr,          # *in_dtype — input
    y_ptr,          # *in_dtype — optional second operand (remove if unary)
    out_ptr,        # *out_dtype — output
    n_elements,     # i32/i64 — total flat length (treat tensor as 1D)
    BLOCK_SIZE: tl.constexpr,
):
    # 1-D grid: each program handles BLOCK_SIZE contiguous elements.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Tail mask: final block may be partial. Required for correctness.
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # --- OP BODY: replace with your elementwise expression. ---
    # Fuse as many pointwise ops as possible here — each one is free
    # once the data is in registers. e.g. silu(x) * y, gelu(x + bias), ...
    out = tl.where(x > 0, x, 0.0) * y   # example: relu(x) * y
    # ------------------------------------------------------------

    tl.store(out_ptr + offsets, out, mask=mask)


def elementwise_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Flatten: elementwise ops don't care about shape, only total length.
    # This lets one kernel serve 1D/2D/ND tensors uniformly.
    assert x.is_cuda and x.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    elementwise_kernel[grid](x, y, out, n)
    return out
```

## Caveats

- **Flatten before launch.** Treat inputs as 1-D of `numel()`. Strided/non-contiguous inputs
  should be `.contiguous()`-ed at the wrapper boundary; supporting arbitrary strides inside
  the kernel kills coalescing and rarely pays off.
- **Fuse aggressively.** Elementwise ops are memory-bound — the cost is loading x once.
  Doing `gelu(x) * y + bias` in one kernel is ~3x faster than three separate kernels.
  If the graph has a chain of pointwise ops, pull them all into one `@triton.jit`.
- **Autotune key is `n_elements`.** Block size that's optimal for 1M elements is different
  from 1K (occupancy concerns). Don't key on shape — shape doesn't matter, size does.
- **Dtype: compute in fp32 for fp16/bf16 inputs** when the op has nonlinearity
  (`exp`, `tanh`, division). Use `x.to(tl.float32)` before the math, cast back at store.
  Pure `add`/`mul` can stay in native dtype.
- **When NOT to use this template.** If you need a reduction (softmax, sum), use
  `reduction.md`. If the op is followed by a GEMM, fuse it as a GEMM epilogue instead
  (`gemm.md` / `fused.md`).
