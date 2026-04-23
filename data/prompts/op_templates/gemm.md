# GEMM Triton Template

Canonical Triton skeleton for matrix multiplication kernels — `matmul`, `bmm`, `linear`,
and any `C = A @ B (+ bias) (+ epilogue)`. Tiled accumulation over the K axis with
`tl.dot`. This single skeleton covers 2D matmul; for batched (`bmm`), add a leading
program-id over the batch axis.

```python
import triton
import triton.language as tl
import torch

# GEMM is compute-bound on modern GPUs when tiles are large enough. The goal is
# maximum `tl.dot` throughput — pick tiles that fit in shared memory and keep
# the tensor cores fed. Autotune is essential: optimal tile depends heavily on
# M, N, K aspect ratio.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # L2-cache-friendly program ordering: group programs along M so adjacent
    # programs reuse the same B tiles. This is ~10-20% faster than plain
    # row-major launch for large matrices.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # fp32 accumulator even for fp16/bf16 inputs — tensor cores do this internally,
    # but we still need the register accumulator in fp32 to avoid precision loss
    # across many K iterations.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask.T, other=0.0)
        acc += tl.dot(a, b)      # maps to mma/wgmma tensor cores
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- EPILOGUE: fuse bias / activation / scale here while acc is in registers. ---
    # Examples:
    #   acc = acc + tl.load(bias_ptr + offs_bn, mask=offs_bn < N)[None, :]
    #   acc = tl.where(acc > 0, acc, 0.0)   # relu
    #   acc = acc * scale                    # quant dequant
    # ---------------------------------------------------------------------------

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def gemm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

## Caveats

- **Always accumulate in fp32.** Even for fp16/bf16 inputs. K can be thousands; low-
  precision accumulation drifts. `tl.dot` returns fp32 by default — keep `acc` fp32
  and only cast at store.
- **BLOCK_K=32 is a safe default.** Too large wastes shared memory; too small starves
  tensor cores. For fp8 / int8 inputs, BLOCK_K=64 is often better.
- **GROUP_M swizzling matters.** The group-row launch order above is the standard trick
  for L2 reuse. Setting `GROUP_M=1` (simple row-major) costs ~10-20% on large shapes.
- **Autotune key = `[M, N, K]`.** Shape-dependent optimal tiles: skinny matmuls
  (M << N) want different tiles than square. Re-tuning per shape bucket is critical.
- **Transposition via strides, not data movement.** `a @ b.T` = pass `b.stride(1),
  b.stride(0)` as `stride_bk, stride_bn` — no physical transpose needed. Same kernel.
- **bmm (batched matmul).** Add `pid_batch = tl.program_id(axis=1)`, offset a_ptr/b_ptr/
  c_ptr by `pid_batch * stride_batch`, launch 3D grid. Same tile/accum loop.
- **`linear(x, W)` = `x @ W.T` + bias.** Fuse the bias add into the epilogue above.
  Don't call a separate elementwise kernel afterwards.
- **When NOT to use.** Tiny M (M < 16, e.g. decode step of a transformer) is bandwidth-
  bound and this compute-oriented template under-utilizes memory. Use a GEMV-flavored
  kernel (persistent, split-K) for those shapes.
