---
backend: triton
hardware: L40S
target_occupancy: 0.8
must_beat_baseline: true
---

# Kernel Optimization Config

## Constraints
- Target hardware: NVIDIA L40S (48GB, Ada Lovelace)
- Prefer Triton backend for portability
- Must pass correctness at fp32 atol=1e-4

## Hints
- For memory-bound kernels: prioritize vectorized loads and coalescing
- For compute-bound kernels: use tensor cores via tl.dot
- Accumulate in float64 for fp32 GEMM to avoid precision issues
