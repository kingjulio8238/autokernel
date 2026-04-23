# Histogram / Scatter-Add Triton Template

Canonical Triton skeleton for histogram-style kernels — `bincount`, `scatter_add`,
`index_add`, `torch.histc`, embedding-bag reductions. Pattern: unordered writes to
shared output bins. Two-level strategy — per-block shared/register histogram, then
atomic merge to global — avoids global-atomic contention when bins are few.

```python
import triton
import triton.language as tl
import torch

# Histogram-class kernels are write-contention bound. The trick is to reduce
# the number of global atomics: aggregate per-block in on-chip memory first,
# then each block performs `NUM_BINS` atomics instead of one-per-input-element.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements', 'NUM_BINS'],
)
@triton.jit
def histogram_kernel(
    src_ptr,            # *i32/i64 — indices (or floats bucketed outside)
    val_ptr,            # *f — values to accumulate (1.0 for bincount, scores for scatter_add)
    out_ptr,            # *f — output histogram, length NUM_BINS, pre-zeroed
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,      # constexpr: local-hist size is known at compile
    HAS_VALUES: tl.constexpr,    # bincount (False) vs scatter_add (True)
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    idx = tl.load(src_ptr + offs, mask=mask, other=0)
    if HAS_VALUES:
        val = tl.load(val_ptr + offs, mask=mask, other=0.0)
    else:
        val = 1.0   # scalar broadcast; bincount

    # Clamp index into range — out-of-range indices are silently dropped.
    # Drop this if the caller guarantees validity.
    idx = tl.where((idx >= 0) & (idx < NUM_BINS) & mask, idx, NUM_BINS)

    # --- Strategy: bin-loop (scales when NUM_BINS is small, e.g. <= 256). ---
    # For each bin, mask elements that landed in it and sum their values.
    # Result is ONE atomic_add per bin per block — O(NUM_BINS) atomics per block
    # instead of O(BLOCK_SIZE).
    for b in tl.static_range(NUM_BINS):
        contrib = tl.sum(tl.where(idx == b, val, 0.0), axis=0)
        tl.atomic_add(out_ptr + b, contrib)

    # --- Alternative: direct atomic (scales when NUM_BINS is very large, e.g. vocab
    # size ~50k — per-bin loop above blows up). Uncomment and swap in:
    #   tl.atomic_add(out_ptr + idx, val, mask=mask & (idx < NUM_BINS))
    # Direct atomics are fine when the index distribution is near-uniform;
    # with a skewed (power-law) distribution, contention dominates.


def histogram_launch(indices: torch.Tensor,
                     values: torch.Tensor | None,
                     num_bins: int) -> torch.Tensor:
    assert indices.is_cuda and indices.dtype in (torch.int32, torch.int64)
    n = indices.numel()
    # Output must be pre-zeroed — atomics accumulate onto existing memory.
    out = torch.zeros(num_bins, device=indices.device,
                      dtype=values.dtype if values is not None else torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    histogram_kernel[grid](
        indices.contiguous().view(-1),
        values.contiguous().view(-1) if values is not None else indices,  # ignored if HAS_VALUES=False
        out, n,
        NUM_BINS=num_bins,
        HAS_VALUES=values is not None,
    )
    return out
```

## Caveats

- **Pre-zero the output.** Atomics accumulate onto whatever is there. `torch.zeros`
  the output tensor in the wrapper — do not rely on uninitialized memory being zero.
- **Two regimes, pick by NUM_BINS.**
  - **Small (<= 256 bins):** use the per-bin loop above. `NUM_BINS` atomics/block,
    near-zero contention.
  - **Large (thousands of bins, e.g. vocab-size embedding-bag):** the per-bin loop
    is too slow — switch to direct `tl.atomic_add(out_ptr + idx, val, mask=...)`.
    The hardware atomic unit handles modest contention fine.
- **Index bounds.** Out-of-range indices are a silent correctness bug. Either clamp in
  the kernel (as shown, redirect to `NUM_BINS` which is past end and masked) or assert
  in the wrapper. Do not rely on undefined behavior.
- **Dtype of `out`.** fp32 is almost always right. fp16 atomic_add exists on sm_70+ but
  loses precision fast when contention is high; bf16 atomic_add is not universally
  supported. Accumulate in fp32, cast back if needed.
- **Scatter-add with multi-dim indices.** If scattering into a (M, K) output via row
  indices, treat it as M independent bins of width K — each program handles a K-wide
  tile and does `atomic_add` along the K dim (no loop). Don't flatten naively.
- **Determinism.** Atomic-add ordering is nondeterministic in fp arithmetic. If
  bit-exact reproducibility is required, use a sort+segmented-reduce variant instead
  (different template — not this one).
- **Autotune key = `[n_elements, NUM_BINS]`.** The optimal BLOCK_SIZE depends on both
  input size and bin count (per-bin loop cost scales with NUM_BINS).
