# Triton Backend Reference

Quick reference for LLM kernel generation. Triton 3.x API.

## Core API
```python
import triton
import triton.language as tl

@triton.jit
def kernel(ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)              # block index
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # element offsets
    mask = offs < N                      # bounds check
    x = tl.load(ptr + offs, mask=mask)   # load from global
    tl.store(ptr + offs, x, mask=mask)   # store to global
```

## Key Operations
- `tl.load(ptr, mask, other=0.0)` — masked global load
- `tl.store(ptr, val, mask)` — masked global store
- `tl.dot(a, b)` — tensor core matrix multiply (a: MxK, b: KxN → MxN)
- `tl.sum(x, axis)` — reduction
- `tl.max(x, axis)` — reduction
- `tl.where(cond, true_val, false_val)` — select
- `tl.exp(x)`, `tl.log(x)`, `tl.sqrt(x)` — elementwise math
- `tl.zeros((M, N), dtype)` — create zero tensor
- `tl.arange(0, N)` — range (N must be constexpr power of 2)
- `tl.program_id(axis)` — block index (0, 1, 2)
- `tl.num_programs(axis)` — grid size

## Autotune
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],  # re-tune when these change
)
@triton.jit
def kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

## Grid Launch
```python
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), ...)
```

## Common Patterns

### Tiled GEMM
```
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=...)
    b = tl.load(b_ptrs, mask=...)
    acc += tl.dot(a, b)      # tensor core MMA
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

### Online Softmax (single pass)
```
m = tl.full([BLOCK], float('-inf'), dtype=tl.float32)
d = tl.zeros([BLOCK], dtype=tl.float32)
for block in range(0, N, BLOCK_N):
    x = tl.load(...)
    m_new = tl.maximum(m, tl.max(x, axis=1))
    d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new[:, None]), axis=1)
    m = m_new
```

### Fused Elementwise
```
x = tl.load(input_ptr + offs, mask=mask)
y = tl.load(other_ptr + offs, mask=mask)
out = some_op(x, y)  # fuse multiple ops, one global read/write
tl.store(output_ptr + offs, out, mask=mask)
```

## Critical Rules
1. BLOCK sizes must be powers of 2
2. tl.arange size must be constexpr
3. tl.dot requires both inputs to be 2D, inner dims must match
4. Accumulate GEMM in float32 even for fp16 inputs (precision)
5. Use `mask` on all loads/stores near boundaries
6. Strides must be passed as kernel args (not computed inside)
