# Attention Triton Template

Canonical Triton skeleton for attention kernels — scaled-dot-product, multi-head (MHA),
grouped-query (GQA), and causal variants. Uses the **Flash-Attention online-softmax**
pattern: Q tile held in SRAM, K/V streamed in tiles, softmax statistics updated
incrementally. This is the standard forward skeleton; adapt for GQA by widening K/V
heads and for causal by masking future keys.

```python
import triton
import triton.language as tl
import torch

# Attention forward. One program per (batch, head, Q-tile). Q tile is fixed;
# we stream K/V tiles and update (m_i, l_i, acc) using the online softmax trick.
# This avoids materializing the full (S_q x S_k) attention matrix — O(N) memory.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ],
    key=['S_q', 'S_k', 'D'],
)
@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    sm_scale,
    B, H, S_q, S_k, D,
    sqb, sqh, sqs, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    sob, soh, sos, sod,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)                  # Q tile index
    pid_bh = tl.program_id(axis=1)                 # fused batch*head
    batch = pid_bh // H
    head  = pid_bh %  H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query positions
    offs_d = tl.arange(0, D_HEAD)                      # head dim

    # Load Q tile once — it lives in SRAM for the entire K/V sweep.
    q_ptrs = (q_ptr + batch * sqb + head * sqh
              + offs_m[:, None] * sqs + offs_d[None, :] * sqd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S_q, other=0.0)

    # Running softmax statistics (per query row).
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                # running sumexp
    acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)         # weighted V accum

    # Causal: only keys up to the max query index in this tile matter.
    hi = (pid_m + 1) * BLOCK_M if CAUSAL else S_k

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = (k_ptr + batch * skb + head * skh
                  + offs_n[:, None] * sks + offs_d[None, :] * skd)
        v_ptrs = (v_ptr + batch * svb + head * svh
                  + offs_n[:, None] * svs + offs_d[None, :] * svd)
        k = tl.load(k_ptrs, mask=offs_n[:, None] < S_k, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < S_k, other=0.0)

        # Scores: Q @ K^T, scaled.
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float('inf'))
        qk = tl.where(offs_n[None, :] < S_k, qk, -float('inf'))

        # Online softmax update — the heart of Flash.
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)                   # new running max
        alpha = tl.exp(m_i - m_new)                     # rescale old acc
        p = tl.exp(qk - m_new[:, None])                 # new probs against m_new
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Final normalization: divide accumulated (P @ V) by running sum-exp.
    acc = acc / l_i[:, None]

    out_ptrs = (out_ptr + batch * sob + head * soh
                + offs_m[:, None] * sos + offs_d[None, :] * sod)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty),
             mask=offs_m[:, None] < S_q)


def attention_launch(q, k, v, causal=False):
    # q, k, v: [B, H, S, D]. For GQA, broadcast K/V heads to match Q heads
    # in the wrapper (or pass a separate H_kv and index accordingly).
    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    out = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)
    grid = lambda m: (triton.cdiv(S_q, m['BLOCK_M']), B * H)
    attention_kernel[grid](
        q, k, v, out, sm_scale, B, H, S_q, S_k, D,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        D_HEAD=D, CAUSAL=causal,
    )
    return out
```

## Caveats

- **Online softmax is non-negotiable.** Materializing `QK^T` in global memory is O(S^2)
  and defeats the entire point. The three-variable update `(m_i, l_i, acc)` is the
  correct form — do not skip the `alpha` rescale of `acc` and `l_i`.
- **fp32 for softmax stats.** `m_i`, `l_i`, and `acc` must be fp32 even for fp16/bf16
  Q/K/V. `exp(qk - m)` in fp16 overflows/underflows almost immediately.
- **Head dim D must be a power of 2** (16, 32, 64, 128) for `tl.dot` efficiency on
  tensor cores. `D_HEAD` is constexpr — kernel is specialized per head-dim.
- **Causal masking.** Two-part: (a) upper-bound the K loop at `(pid_m+1)*BLOCK_M`
  (skips useless work), (b) within-tile `qk = where(q_idx >= k_idx, qk, -inf)`.
  Both are needed; skipping either is slow or wrong.
- **GQA.** `H_q != H_kv`. Each Q head maps to `H_kv * (H_q // H_kv) = head // groups`.
  Pass `head_kv = head // (H / H_kv)` when loading K/V. K/V strides use H_kv.
- **Decoder (S_q == 1).** This template is forward-training-sized. For one-token
  decode, the `BLOCK_M` dimension is wasted; use a GEMV-shaped attention kernel
  (paged or chunked KV cache) — different template.
- **Backward is a separate kernel.** This is forward only; the backward pass
  requires `(m_i, l_i)` saved at forward, then two more kernels (dQ, dK/dV).
- **Autotune key = `[S_q, S_k, D]`.** Sequence-length-dependent. Don't key on batch
  or head — they just scale the grid.
