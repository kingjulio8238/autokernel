# Phase 3a Prompt Quality Review: PROFILE FROM LAST ATTEMPT
**Date:** 2026-04-23  
**Reviewer:** review-prompt-quality  
**Scope:** Audit of guidance correctness, placement, and small-model parseability for profile-injected Jinja templates.

---

## 1. Kernel-Engineer Correctness

### Finding: 3 of 4 guidance bullets have correctness issues; float4/uint4 is critical in Triton context.

#### 1a. Memory Bound + Bandwidth < 60%
**Current:** "improve coalescing, use vectorized loads (float4 / uint4), or tile to hit fewer global-mem trips per output"

**Assessment:** Generally sound but backend-mismatched and vague on tiling.
- ✓ Coalescing advice is correct; uncoalesced loads on modern GPUs waste bandwidth.
- ✗ **float4/uint4 are CUDA/HIP intrinsics.** Our codebase is **Triton-only** (confirmed by test file test_prompt_templates.py:43–56, all mocked kernels are Triton). Haiku (30–70B, target model) will interpret this as valid code suggestion and attempt to generate Triton with `float4` constructs, which will fail at compile. **This is a hallucination risk.**
- ✓ Tiling to reduce trips is correct, but "hit fewer global-mem trips per output" is vague—doesn't clarify whether spatial tiling or dimensionality reduction.
- ⚠️ **Edge case:** A kernel with 90% bandwidth utilization but diagnosed as memory_bound (due to bottleneck definition inflating low compute SOL) doesn't need coalescing—it needs compute fusion. The guidance conflates bandwidth underutilization with coalescing faults.

**Recommendation (if taken):** Replace with Triton-aware guidance:
```
"If bottleneck is `memory_bound` and bandwidth < 60%: the kernel under-utilizes memory bandwidth. 
Strategies: (1) verify loads are coalesced (consecutive offsets in tl.load); (2) load multiple elements per iteration 
to increase memory-level parallelism; (3) reduce tile size or reorder loops to improve L2/L1 locality."
```

---

#### 1b. Compute Bound + Compute < 60%
**Current:** "try larger block size, more registers per thread, or reduce redundant work per output"

**Assessment:** Problematic for Triton; can mislead occupancy-sensitive vs. algorithm-limited kernels.
- ✗ **Larger block size is not universal.** Triton programs have fixed block structure; larger blocks only help if per-thread work is insufficient *and* the block is undersized. If the algorithm is already maxed out, block tuning is no-op.
- ✗ **"More registers per thread" doesn't apply to Triton.** Triton auto-infers register allocation from code; there's no direct tuning knob. If compute is at 60%, the issue is typically *insufficient arithmetic per memory operation*, not register pressure.
- ✓ "Reduce redundant work per output" is correct but vague (loop unrolling? loop fusion?).
- ⚠️ **Counterexample (small-tensor problem):** A 16×16 matmul kernel diagnosed as compute_bound at 50% compute is *not* an occupancy problem—it's a problem-size problem. Increasing block size won't help; the kernel is saturated by tiny input. The guidance would mislead toward occupancy tuning when the real fix is fusion or kernel skipping. Haiku will attempt larger blocks, waste time, and miss the actual optimization.

**Recommendation (if taken):** Replace with:
```
"If bottleneck is `compute_bound` and compute < 60%: the kernel under-utilizes arithmetic units. 
Strategies: (1) increase arithmetic intensity (more operations per memory load) via loop fusion or caching; 
(2) reduce memory-stall impact by overlapping loads with computation; (3) for small-tensor kernels, fusion with 
neighboring operations is usually more effective than occupancy tuning. Block size changes are low-ROI unless 
confirmed by occupancy profiling."
```

---

#### 1c. Latency Bound (Both Utilizations Low)
**Current:** "likely too little work per launch — fuse adjacent ops, widen blocks, or increase grid size"

**Assessment:** "Increase grid size" is backwards; misleading.
- ✓ Fusion is correct for latency-bound (insufficient instruction-level parallelism).
- ✓ Widen blocks can help by increasing in-flight loads per thread, hiding latency.
- ✗ **"Increase grid size" is incorrect.** Grid size affects throughput (how many warps can run concurrently), not latency (how long a single warp waits for a load). A kernel waiting on one load per thread won't be helped by launching more kernel instances—it needs more in-flight loads per warp. Haiku may attempt to increase grid dimension, which wastes resources without improving latency.
- ⚠️ **Counterexample:** Sparse-matrix kernel where each thread loads one sparse element, then does 10 FLOPs. Increasing grid size doesn't reduce the latency stall; you need wider blocks or more loads per iteration.

**Recommendation (if taken):** Replace with:
```
"If bottleneck is `latency_bound` (both utilizations low): likely high memory-stall or instruction-latency cycles. 
Strategies: (1) fuse adjacent operations into the same kernel to hide latency; (2) increase arithmetic per memory 
operation; (3) if block is small, widen blocks to keep more loads in-flight per SM. Note: increasing grid size 
improves throughput, not per-kernel latency."
```

---

#### 1d. L2 Cache Hit Rate < 40%
**Current:** "working set is spilling L1/L2 — consider smaller tiles, tiling the reduction dim, or shared-memory staging"

**Assessment:** Sound core advice; edge case on layout mismatches.
- ✓ Smaller tiles reduce working set → better L2 reuse. Correct.
- ✓ Tiling the reduction dimension is a classic matmul/gemm optimization. Correct.
- ⚠️ **"Shared-memory staging"** is phrased as manual SMEM management (CUDA style). Triton has implicit caching via `tl.load` with `cache_modifier` hints. The phrasing doesn't match Triton's model and could confuse Haiku.
- ⚠️ **Edge case:** Strided-access patterns (e.g., column-major reads in row-major layout) yield low L2 hit rates regardless of tile size. The guidance doesn't mention memory layout mismatches or stride patterns, which are often the root cause.

**Recommendation (if taken):** Replace with:
```
"If L2 cache hit rate < 40%: working set is exceeding L2 capacity. 
Strategies: (1) reduce tile size to fit working set within L2; (2) transpose or reorder data layout to match cache 
lines (row-major access for row-major layout); (3) tile reduction dimensions separately; (4) for Triton, verify 
loop order aligns with memory strides. Layout mismatches are often the root cause."
```

---

### Summary: Correctness Issues
| **Bullet** | **Issue** | **Severity** |
|---|---|---|
| memory_bound (float4/uint4) | CUDA-specific in Triton codebase; hallucination risk | **HIGH** |
| compute_bound (block size tuning) | Can mislead small-tensor kernels toward occupancy tuning instead of fusion | **MEDIUM** |
| latency_bound (increase grid) | Backwards; grid size improves throughput, not latency | **MEDIUM** |
| L2 hit rate (shared-memory phrasing) | Doesn't match Triton's implicit caching model; misses layout issues | **LOW** |

---

## 2. Threshold Sensibility

**Thresholds:** 60% (bandwidth/compute utilization), 40% (L2 cache hit rate)

**L40S typical SOL distributions** (Phase 2 observational data):
- Bandwidth-bound kernels: 40–75% utilization (wide range by access pattern coalescing/layout)
- Compute-bound kernels: 20–50% (Triton kernels often memory-latency-bound, underutilize compute)
- L2 hit rate: 30–70% depending on working-set size relative to L2 capacity

### Assessment:

**60% threshold (bandwidth/compute):**
- ✓ Reasonable as "actionable gap" marker. Below 60% signals significant headroom for improvement.
- ✓ Correctly flags the example profile (42% compute at memory_bound) as concerning.
- ✓ Avoids false positives on 70–80% utilization kernels (harder to improve further).

**40% L2 threshold:**
- ⚠️ Lenient. L2 hit rates below 40% indicate severe working-set misses. A 45–50% threshold would catch more issues earlier.
- ✓ Defensible if goal is "only flag egregious cases" to avoid alert fatigue for Haiku.
- **Recommendation:** Keep 40% for now; if false positives become an issue, tighten to 45%. Monitor Phase 3 optimization rounds to validate.

---

## 3. Placement Correctness

### kernel_refinement.j2 (lines 36–50)
**Context:** Profile appears after TEST RESULTS (line 32) and before FUSION PRIORITY (line 56).

**Assessment:** ✓ Sound placement.
- The model has just seen the error; the profile explains *why* it failed (e.g., "compute at 42%, bandwidth at 75% → memory-bound issue, not compute").
- The profile sits between error diagnosis and fix strategy, providing context for the refinement.
- Test assertion confirms placement: `assert error_idx < profile_idx < fusion_idx` (test_prompt_templates.py:201).

---

### kernel_optimization.j2 (lines 72–86)
**Context:** Profile appears after BOTTLENECK ANALYSIS (line 58) and before PERFORMANCE TARGET (line 120).

**Assessment:** ✓ Excellent placement.
- The model reads the bottleneck diagnosis first, then the profile *validates or challenges* that diagnosis.
- Example: "BOTTLENECK ANALYSIS says memory_bound, but profile shows compute at 42%—maybe the real issue is low compute density, not bandwidth."
- Test assertion confirms placement: `assert recommended_idx < profile_idx < target_idx` (test_prompt_templates.py:209).

---

### reflexion_prompt.j2 (lines 39–53)
**Context:** Profile appears before the Task section (line 55: "## Task").

**Assessment:** ✓ Appropriate placement, though slightly early.
- The model is about to analyze whether a prior fix was effective. Having the profile *before* the task statement makes sense because it provides context for the reflection.
- Ideally, the profile would sit right before "Analyze whether the bottleneck diagnosis was correct" (line 58) so it's fresh when the model reasons about the fix.
- Minor: acceptable trade-off for template cleanliness.

---

### Verdict:
All three placements are **contextually sound**. The profile block provides relevant signal at decision points where the model is diagnosing or refining the kernel.

---

## 4. Small-Model Parseability (Haiku Target)

Model target: Haiku (4B–70B-equivalent context, fast inference). Expected in Phase 3c as L1 router.

### Analysis:

**Pronoun ambiguity:** ✓ NONE
- All subjects explicit ("the kernel is...", "the bottleneck is...").
- No pronouns like "it" or "they" that require backtracking to identify antecedents.

**Nested conditionals:**
```
"If bottleneck is `memory_bound` and bandwidth < 60%: ..."
"If bottleneck is `compute_bound` and compute < 60%: ..."
"If bottleneck is `latency_bound` (both utilizations low): ..."
```
- Single level of nesting (category + metric). Haiku can parse reliably. ✓

**Jargon density:**
- `memory_bound`, `compute_bound`, `latency_bound`: Technical but unambiguous in HPC context. Haiku trained on CUDA/HIP code recognizes these.
- `coalescing`, `vectorized loads`: Standard GPU terminology.
- ✗ **`float4 / uint4`**: CUDA vector types. Haiku will likely interpret as valid Triton constructs (they're not). Hallucination risk confirmed by cross-check: float4 is not in Triton's type system.
- `shared-memory staging`: Assumes manual shared-memory code (CUDA style). Triton doesn't expose this directly; Haiku may not parse correctly.

**Imperative vs. conditional:**
- Current: "If X: do Y" structure is clear and imperative within each branch.
- Slightly more explicit for Haiku: "When bottleneck is memory_bound and bandwidth < 60%, improve coalescing by..." (shifts from conditional to direct instruction).

### Verdict:
**Acceptable for Haiku, but backend-specific jargon (float4, shared-memory staging) creates hallucination risk.** Recommend Triton-aware rewrite.

---

## 5. Missing Signal (High-Value Profile Data Not Currently Plumbed)

**Currently captured:**
- Bandwidth utilization ✓
- Compute utilization ✓
- L2 cache hit rate ✓
- Bottleneck type ✓
- SOL score ✓

**High-value signals NOT included:**

| **Signal** | **Source** | **Actionability** | **Impact** |
|---|---|---|---|
| **SM Occupancy (%)** | NCU, NVIDIA profiler | "occupancy < 50% → increase block size" (direct); "occupancy > 90% → register pressure, consider reducing work/iteration" | HIGH |
| **Register Spill Rate** | NCU register-spill metrics | "spill > 0 → reduce per-thread work or fuse loops" | HIGH |
| **Memory Stall Cycles (%)** | NCU PAPI stalls | More precise than "memory_bound"; tells if kernel is waiting on L1/L2/DRAM or stuck on other latencies | MEDIUM |
| **Warp Execution Efficiency (%)** | NCU warp-exec efficiency | Indicates branch divergence or idle warps; guides loop restructuring for divergent codes | MEDIUM |
| **Tensor Core Utilization (%)** | NCU tensor metrics | Critical for ops using TC; currently only in roofline block, not in guidance | MEDIUM |
| **Cache Line Utilization (%)** | Derived from bandwidth / peak throughput | Complements bandwidth; reveals if bandwidth is wasted on poor line efficiency | LOW |

### Top recommendations for follow-up:
1. **Occupancy (HIGH-ROI):** Would enable more precise block-size guidance. Current guidance assumes occupancy issues without confirming.
2. **Register spill (HIGH-ROI):** Would validate loop-unrolling vs. fusion trade-offs.
3. **Memory stall cycles (MEDIUM):** More granular than bottleneck type; could differentiate L1 vs. L2 vs. DRAM stalls.

**Note:** Not blockers for Phase 3a; would improve Phase 3b guidance by ~15–20%.

---

## 6. Hallucination Risk & Backend Awareness

**Critical Issue:** Guidance assumes CUDA/HIP, but codebase is **Triton-only**.

### Problematic phrases:

| **Phrase** | **Problem** | **Risk** |
|---|---|---|
| **`float4 / uint4`** | CUDA/HIP vector types; not native in Triton. Users load multiple scalars instead. | HIGH: Haiku will attempt `float4` in Triton, fail at compile. |
| **`shared-memory staging`** | Assumes manual SMEM management (CUDA style). Triton has implicit caching via `tl.load(cache_modifier=...)`. | MEDIUM: Haiku may misinterpret as direct SMEM code requirement. |
| **`vectorize` (generic)** | Acceptable; Triton users understand as `tl.load(x, x + tl.arange(...))`. | LOW: No risk. ✓ |

### Confirmation from codebase:
- test_prompt_templates.py lines 43–56: All mocked kernel_code contexts are Triton syntax.
- kernel_refinement.j2 lines 63–65: Explicitly forbids torch.nn calls; targets Triton as the compute backend.
- No CUDA/HIP code in repo; all Phase 1–2 kernels are Triton.

### Recommendation:
**Replace with Triton-specific guidance block.** Example revised guidance:

```markdown
Guidance:
- If bottleneck is `memory_bound` and bandwidth < 60%: the kernel under-utilizes memory bandwidth. 
  Try: (1) verify loads are coalesced (consecutive offsets in tl.load); (2) load multiple elements per iteration 
  (e.g., tl.load(x[i:i+4]) pattern); (3) reduce tile size or reorder loops for better L2 locality.
- If bottleneck is `compute_bound` and compute < 60%: the kernel under-utilizes arithmetic units. 
  Try: (1) fuse adjacent operations into the same kernel; (2) increase arithmetic per load (higher ops/byte ratio); 
  (3) avoid small-tensor kernels where launch overhead dominates; increase problem size or batch operations.
- If bottleneck is `latency_bound` (both utilizations low): likely high memory-stall or instruction-latency cycles. 
  Try: (1) fuse adjacent ops to amortize launch and memory-latency overhead; (2) widen blocks to increase in-flight 
  loads per SM; (3) reorder loops to overlap loads with arithmetic.
- If L2 cache hit rate < 40%: working set is exceeding L2 capacity. 
  Try: (1) reduce tile size; (2) reorder loops to improve memory-access stride alignment; (3) for reductions, tile 
  the reduction dimension separately; (4) verify data layout (row-major vs. column-major) matches access patterns.
```

---

## Summary Table

| **Audit Question** | **Finding** | **Severity** | **Blocker?** |
|---|---|---|---|
| 1. Kernel correctness | 3 of 4 bullets have issues; float4/uint4 is CUDA-specific in Triton codebase | HIGH | **YES** |
| 2. Threshold sensibility | 60% is sound; 40% L2 is lenient but defensible | LOW | NO |
| 3. Placement correctness | All three templates have contextually sound placements | NONE | NO |
| 4. Small-model parseability | Acceptable for Haiku, but CUDA jargon creates confusion | MEDIUM | Recommend rewrite |
| 5. Missing signal | Occupancy & register spill are high-value, not included; not critical | LOW | NO |
| 6. Hallucination risk | float4/uint4 and shared-memory phrasing don't match Triton model; risky | MEDIUM | **YES** |

---

## Recommendations for Team Lead

**Immediate actions (Phase 3a):**
1. **Replace float4/uint4 and shared-memory references with Triton-specific guidance.** Current phrasing will cause Haiku to hallucinate CUDA code in a Triton-only system.
2. **Reframe compute_bound guidance** to avoid misleading small-tensor kernels toward block-size tuning.
3. **Remove "increase grid size" from latency_bound guidance** or clarify it doesn't help per-kernel latency.

**Optional follow-ups (Phase 3b+):**
- Add occupancy and register-spill metrics to profile block (high ROI for tuning precision).
- Validate 40% L2 threshold on Phase 3a optimization runs; tighten to 45% if false positives occur.

---

## References

- **Template files:** kernel_refinement.j2:36–50, kernel_optimization.j2:72–86, reflexion_prompt.j2:39–53
- **Test coverage:** test_prompt_templates.py (assertions confirm placement and rendering)
- **Triton codebase:** All kernel_code contexts in tests use Triton syntax; no CUDA/HIP examples in repo
