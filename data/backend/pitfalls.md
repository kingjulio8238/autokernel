# Common Pitfalls in Kernel Optimization

Accumulated learnings from KernelBench optimization runs.

## KernelBench Contract
- Reference defines `Model` with `forward()`, `get_inputs()`, `get_init_inputs()`
- Your kernel must define `ModelNew` (not `Model`) with identical `forward()` signature
- Correctness: `torch.allclose(ref, kernel, rtol=1e-2, atol=1e-2)` for fp32
- Speedup = ref_median_time / kernel_median_time (higher is better)
- Inputs are generated fresh each trial — don't assume fixed values

## Correctness Failures
- **FP32 accumulation order**: GEMM accumulated in fp32 can differ from torch.matmul (which uses cuBLAS with higher internal precision). Fix: accumulate in float64, cast result back to float32.
- **Reduction order**: parallel reductions change summation order. For softmax/layernorm, compute running stats in float64.
- **Boundary masking**: forgetting `mask=` on tl.load near tensor edges causes garbage values. Always mask.
- **Stride assumptions**: don't assume contiguous — always pass and use `tensor.stride()`.

## Performance Traps
- **torch.matmul for large GEMM**: For 2048+ square matrices, cuBLAS is near-optimal. A naive Triton GEMM will be slower. Need autotune with large tile configs to compete.
- **Launch overhead**: Triton kernels have higher launch overhead than cuBLAS. For small tensors (<256 elements), the launch cost dominates — fuse with adjacent ops.
- **Autotune cold start**: First call triggers autotuning across all configs. This is not measured in benchmarks (warmup runs first), but be aware during interactive testing.
- **Memory layout**: Row-major (C-contiguous) loads coalesce naturally. Column-major requires transposition or stride-aware loading.

## Triton-Specific Issues
- `tl.dot` requires both operands to be 2D with matching inner dimension
- `tl.arange(0, N)` — N must be a constexpr power of 2
- Shared memory is implicit in Triton (managed by compiler) — you control it via BLOCK sizes and num_stages
- `num_stages > 1` enables software pipelining (async loads) — critical for GDDR6X latency hiding on L40S
- `num_warps` affects occupancy: 4 warps = 128 threads, 8 warps = 256 threads per block

## When NOT to Write a Custom Kernel
- Large dense GEMM (>4096): cuBLAS is highly optimized, hard to beat
- Standard ops with known fusions: torch.compile often finds them
- The win is in: custom fusions, non-standard reductions, sparse patterns, operator-specific tricks
