# openkernel: System Design

*April 2026*

---

## One Sentence

openkernel takes a PyTorch reference operation + a target GPU, and produces an optimized kernel (CUDA or Triton) through a 3-level hybrid search with profiler-guided feedback, running entirely on cloud GPUs.

---

## Design Principles

1. **Backend-flexible**: Kernel engineer chooses CUDA or Triton. Backend-specific optimization strategies for each.
2. **Model-agnostic (BYOM)**: Any LLM via OpenAI-compatible API. Recommended models list from internal benchmarking. No lock-in until kernelgen-1.
3. **Cloud-native**: Entire eval pipeline runs on Modal. No local GPU requirement. Develop anywhere.
4. **Better on all axes**: Faster, cheaper, and easier to use than every competitor.
5. **Compounding knowledge**: Every problem solved makes the next one easier via the skill library.
6. **Trace everything**: Full optimization traces captured for kernelgen-1 training. Follow Cursor/Composer 2 best practices.

---

## Architecture: 5 Layers

### Layer 1 — Evaluation Engine

The foundation. Everything depends on reliable, fast kernel evaluation.

**Modal function signature:**
```
kernel_code + reference_code + hardware_target → EvalResult
```

**EvalResult:**
```python
@dataclass
class EvalResult:
    status: str              # correct | compile_error | incorrect | error
    correct: bool
    speedup: float           # ref_runtime / kernel_runtime
    runtime_us: float
    ref_runtime_us: float
    profile: ProfileData     # bottleneck classification + metrics
    error: str | None        # error message if failed
```

**ProfileData** (from Proton/torch.profiler/analytical):
```python
@dataclass
class ProfileData:
    bottleneck_type: str     # compute_bound | memory_bound | latency_bound
    roofline_position: float # 0-1, how close to theoretical ceiling
    cache_efficiency: float  # L2 hit rate estimate
    occupancy: float         # achieved / theoretical
    bandwidth_utilization: float  # achieved / peak memory bandwidth
    compute_utilization: float    # achieved / peak compute
    top_stalls: list[str]    # top reasons for warp stalls (if available)
    raw_metrics: dict        # full profiler output for reference
```

**Two eval modes:**
- `fast`: 5 correctness trials, 10 perf trials. ~5 seconds. For rapid iteration.
- `thorough`: 5 correctness trials, 100 perf trials. ~15 seconds. For final validation.

**Profiling stack (production-ready on Modal):**
- Triton kernels: **Proton** (built into Triton, no admin permissions needed, instruction-level metrics)
- CUDA kernels: **torch.profiler** + CUDA events (no admin needed, kernel-level metrics)
- All kernels: **simple-torchroofline** for analytical pre-screening (no GPU needed)
- Deep profiling (optional): **NCU on RunPod** bare-metal instances when SYS_ADMIN is available

### Layer 2 — Agent Pair (Generator + Critic)

Two LLM roles. CudaForge proved role separation yields 1.5x better results than self-refine.

**Generator:**
- Input: reference code, hardware specs, optimization intent, critic feedback, relevant skills from memory
- Output: kernel code (Triton or CUDA, determined by KE's backend choice)
- Backend-specific prompting:
  - Triton: includes `@triton.autotune` config space proposal
  - CUDA: includes launch config, compilation flags, includes/headers
- BYOM: unified interface via OpenAI-compatible API (litellm or direct)

**Critic:**
- Input: kernel code, EvalResult (including ProfileData)
- Output: structured CriticDiagnosis

```python
@dataclass
class CriticDiagnosis:
    bottleneck_type: str        # compute_bound | memory_bound | latency_bound
    roofline_position: float    # 0-1
    specific_issue: str         # "L2 hit rate 45% — strided access pattern with 256-byte gaps"
    recommendation: str         # "Restructure to coalesced access with BLOCK_K=64 tiles"
    estimated_headroom: float   # "~1.4x improvement possible"
    confidence: float           # 0-1, how confident in the diagnosis
```

### Layer 3 — Search Engine (3-Level Hybrid)

**Inner Loop — Refinement (Caesar-style)**

For a given optimization intent:
1. Generator produces kernel code
2. Eval on Modal (fast mode)
3. If compile error → feed error to Generator, retry (up to 3 times)
4. If incorrect → feed max_diff to Generator, retry (up to 3 times)
5. If correct → Critic diagnoses from profile data
6. Generator produces improved version targeting the diagnosed bottleneck
7. Repeat up to K attempts per intent (default K=5)
8. Return best result to middle loop

**Middle Loop — World Model Search (K-Search-style)**

```python
@dataclass
class IntentNode:
    id: str
    description: str          # "Vectorize loads with float4"
    parent_id: str | None
    priority: float           # LLM-estimated value (0-1)
    status: str               # pending | active | succeeded | failed | pruned
    attempts: int
    best_speedup: float
    profiler_summary: str     # last CriticDiagnosis summary
    children: list[str]
```

LLM operates on the tree via 3 operations:
- **Insert**: Propose new optimization intents as children of existing nodes
- **Update**: Re-estimate priorities after seeing results
- **Prune**: Remove subtrees that are dead ends

Key properties:
- Decouples WHAT to optimize from HOW to implement it
- If good strategy produces buggy code, the strategy survives for retry
- Non-monotonic paths allowed (temporary regression OK if world model predicts payoff)
- Stagnation detection: K consecutive failures triggers strategy switch

**Outer Loop — Strategy Evolution (GEPA-style)**

Maintains a Pareto frontier of optimization strategies:

```python
@dataclass
class Strategy:
    id: str
    description: str           # "For memory-bound elementwise: vectorize, minimize shared mem, fuse"
    problem_types: list[str]   # ["elementwise", "reduction", "memory_bound"]
    backend: str               # "triton" | "cuda" | "any"
    hardware_targets: list[str]
    success_history: list[dict]
    failure_history: list[dict]
    pareto_scores: dict        # {speedup, correctness_rate, iterations_to_converge}
```

Loop:
1. For each problem: retrieve relevant strategies from frontier, seed intent tree
2. Run middle loop with those strategies
3. After solving: LLM reflects on what worked and why
4. Generate new strategies via mutation/crossover
5. Add non-dominated strategies to frontier, prune dominated ones
6. Persist frontier to skill library

### Layer 4 — Memory System

**Skill Library** (long-term, cross-problem):
```python
@dataclass
class OptimizationSkill:
    id: str
    name: str                  # "online_softmax_reduction"
    trigger: str               # "softmax on large tensors, memory-bound"
    approach: str              # "Online algorithm: compute max+sum in single pass..."
    backend: str               # "triton" | "cuda" | "any"
    evidence: list[dict]       # [{problem: "L1#23", speedup: 2.4, hardware: "H100"}, ...]
    code_template: str | None  # optional starter code
```

KernelSkill showed this achieves 3.6x per-round efficiency. Skills are retrieved by problem similarity before optimization begins.

**Trajectory Memory** (short-term, within-problem):
- Full history of what was tried for the current kernel
- Prevents oscillation (don't retry what already failed)
- Feeds into world model tree updates

**Trace Store** (for kernelgen-1):
- Every optimization run stores: all prompts, all LLM responses, all kernel variants, all eval results, all profile data
- Follows Cursor/Composer 2 best practices for data collection
- Stored in structured format (JSON/Parquet) for post-training pipeline
- Privacy-aware: opt-in for users of kernel code

### Layer 5 — Backend-Specific Strategies

**Triton strategies:**
- Tile size search via `@triton.autotune` (parametric phase)
- Proton profiling for instruction-level feedback
- Shared memory tiling patterns
- AMD portability testing (same Triton code, different autotuning configs)

**CUDA strategies:**
- NCU profiling when available (RunPod deep profiling tier)
- CUTLASS CuTe templates for GEMM
- Warp-level primitives (shuffle, reduction)
- Tensor core MMA instructions
- Inline PTX for critical paths

**Shared strategies (both backends):**
- Fusion discovery (L2 problems: identify which ops to fuse)
- Algorithmic changes (online softmax, Welford's algorithm, Kahan summation)
- Memory access pattern optimization (coalescing, vectorized loads)
- Analytical pre-screening (roofline bounds before GPU time)

---

## KernelBench Integration

openkernel is designed to hill-climb KernelBench.

**Problem loader:** Load any problem by level + ID via `kernelbench.dataset.construct_kernelbench_dataset()`

**Evaluation:** Uses KernelBench's `eval_kernel_against_ref()` inside Modal containers. Wrapped with our profiling layer.

**Sweep orchestration:**
1. Pre-screen all problems in a level with analytical roofline
2. Sort by estimated headroom (highest first)
3. Allocate iteration budget proportional to headroom
4. Run openkernel on each problem
5. Track results: per-problem speedup, correctness, iterations

**Metrics:**
- `fast_p` at p={1.0, 1.5, 2.0} — the KernelBench standard
- Geomean speedup across correct kernels
- Correctness rate per level
- Cost per kernel (Modal compute + LLM tokens)
- Iterations to convergence (efficiency)

**Results publishing:** We run sweeps and publish comparison tables vs CudaForge, Kernel-Smith, CUDA Agent, KernelSkill, etc.

---

## Profiling Decision Tree

```
Is kernel Triton?
├── YES → Use Proton (built-in, no permissions needed)
│         Provides: instruction-level metrics, custom metrics, hardware counters
└── NO (CUDA) → Use torch.profiler + CUDA events
                 Provides: kernel timing, memory bandwidth, SM utilization

For ALL kernels:
├── Pre-screen with simple-torchroofline (analytical, no GPU)
└── Optional: deep NCU profiling on RunPod (root access, $2-4/hr)
              Use for: detailed warp stall analysis, cache behavior, occupancy limiters
              Only for promising kernels (correct + speedup > 1.0x)
```
