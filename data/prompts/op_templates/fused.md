# Fused-Op Triton Template

Canonical Triton skeleton for **fused multi-op kernels** — where two or more ops from
different categories are collapsed into one kernel to avoid round-tripping through HBM.
Typical wins: `matmul + bias + activation`, `layernorm + residual + dropout`,
`elementwise chain`, `attention + output projection`. The pattern: pick the **dominant
op** as the main body (the one that controls the memory/compute pattern), then add
**prologue** (input transforms) and **epilogue** (output transforms) in registers.

```python
import triton
import triton.language as tl
import torch

# Fused GEMM + bias + activation — the most common fused pattern.
# Main body: GEMM (compute-bound, controls layout). Prologue: nothing.
# Epilogue: bias broadcast + activation + optional residual add.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_bias_act_kernel(
    a_ptr, b_ptr, c_ptr,
    bias_ptr, residual_ptr,           # epilogue inputs; pass 0-ptr if unused
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,           # compile-time switches — no runtime cost
    HAS_RESIDUAL: tl.constexpr,
    ACT: tl.constexpr,                # 0=none, 1=relu, 2=gelu, 3=silu
):
    # L2-swizzled program ID (same as gemm.md — reproduced here for self-containment).
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
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # --- MAIN BODY: GEMM accumulation (unchanged from gemm.md). ---
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask.T, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- EPILOGUE: everything "for free" while acc is hot in registers. ---
    # Order matters: (1) scale/dequant, (2) bias, (3) residual, (4) activation.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if HAS_BIAS:
        # Bias is per-output-column — broadcast along M.
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        acc += bias[None, :].to(tl.float32)

    if HAS_RESIDUAL:
        # Residual connection — load the corresponding C-shaped tile and add.
        r_ptrs = residual_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        r_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        r = tl.load(r_ptrs, mask=r_mask, other=0.0).to(tl.float32)
        acc += r

    # Activation — compile-time switched to zero branch cost on the hot path.
    if ACT == 1:   # relu
        acc = tl.where(acc > 0, acc, 0.0)
    elif ACT == 2: # gelu (tanh approx — cheap and close enough for inference)
        acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
    elif ACT == 3: # silu / swish
        acc = acc * (1.0 / (1.0 + tl.exp(-acc)))

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def fused_launch(a, b, bias=None, residual=None, act='none'):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    act_id = {'none': 0, 'relu': 1, 'gelu': 2, 'silu': 3}[act]
    grid = lambda m: (triton.cdiv(M, m['BLOCK_M']) * triton.cdiv(N, m['BLOCK_N']),)
    fused_gemm_bias_act_kernel[grid](
        a, b, c, bias, residual,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=(bias is not None),
        HAS_RESIDUAL=(residual is not None),
        ACT=act_id,
    )
    return c
```

## Caveats

- **Identify the dominant op first.** Fusion works best when one op controls the
  kernel's memory pattern and the others piggyback on registers. GEMM + epilogue,
  LayerNorm + residual, Attention + output-proj-bias — all have one clear dominant.
  If no op is dominant, you probably shouldn't fuse.
- **Epilogue only uses data already in registers.** Anything that requires a new
  reduction axis or a different tile shape can't go in the epilogue — it becomes a
  separate kernel. Bias (broadcast) is fine; a second matmul is not.
- **`constexpr` switches for optional ops.** `HAS_BIAS`, `HAS_RESIDUAL`, `ACT` should
  all be `tl.constexpr`. Triton specializes the kernel per combo, so unused branches
  disappear from the compiled SASS — zero runtime cost for the "none" path.
- **Epilogue order matters for numerical equivalence.** The convention is
  `out = act(matmul + bias + residual)`. Changing order (e.g. adding residual
  before bias) gives different results. Match the reference op's order exactly.
- **Don't over-fuse.** Fusing too much inflates register pressure → spills → slower
  than two smaller kernels. If occupancy drops below 25% after adding an epilogue,
  split it out.
- **Multi-output fusion** (e.g. attention + sum-of-squares statistic for logging)
  requires two `tl.store` calls to different tensors — fine, but mind the extra
  stride arguments.
- **When NOT to use this template.** If fusion is "two elementwise ops", just use
  `elementwise.md` — chain them in the op body. This fused template is for
  **cross-category** fusion where the dominant op is compute-heavy (GEMM, conv,
  attention) and the tail ops are pointwise/broadcast.
- **Autotune key** should match the dominant op's key — GEMM tiles are sensitive to
  M/N/K, not to whether bias is on.
