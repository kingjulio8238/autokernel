# Architecture Review: Autokernel vs KernelMem vs HF Skills vs Arxiv Paper

## Autokernel's Current Architecture (Summary)

Our system is a **3-layer hierarchical optimization engine**:

```
MetaOptimizer (outer) → reflect, pivot, accumulate strategy
  └─ KernelAgentBridge (middle) → 4 parallel workers, Modal GPU eval
       └─ VerificationWorker (inner) → 10-round LLM refinement loop
```

**Key strengths**: Reflection-based strategy pivoting, persistent skill library with evidence tracking, format-agnostic problem interface (KernelBench + GPU Mode), multi-provider LLM system, rich live visualization. Recent commits show the core pipeline is stable — work has shifted to UX polish and plot reliability.

---

## KernelMem Comparison

### Where KernelMem is stronger

| Area | KernelMem | Autokernel |
|------|-----------|------------|
| **Deterministic gating** | Rule-based SMEM budget checks, COALESCING-FIRST rules, REMOVABLE kernel detection | Relies on LLM judgment |
| **One-method discipline** | Enforces exactly one optimization per round | No such constraint — LLM may scatter-shot |
| **Inline profiling** | NCU metrics directly feed next LLM prompt (fresh data) | Profiling decoupled (can be stale) |
| **Modification plans** | Explicit numbered code actions ("replace 4 scalar loads with 2 float2") | Less structured prescriptions |

### Where autokernel is stronger

| Area | Autokernel | KernelMem |
|------|------------|-----------|
| **Semantic retrieval** | RAG with OpenAI embeddings for pattern matching | Timestamp-based FIFO (last 10 kernels) |
| **Cross-run learning** | Persistent skill library with evidence | Static YAML memorybank, never updated |
| **Parallelism** | 4 workers racing in parallel | Single-threaded sequential |
| **Search strategy** | Beam search / greedy with strategy pivoting | Single-path refinement only |
| **Architecture** | Modular components with clear protocols | Monolithic 1,831-line main file |

---

## HF Kernels Skill Comparison

HF takes a **fixed-curriculum approach** (~550 tokens + reference docs) — no iterative optimization loop. Key patterns to adopt:

1. **Hardware-specific optimization guides** — explicit block sizing, shared memory, grid params per GPU (H100: 132 SMs, 192KB SMEM; A100: 108 SMs, 164KB)
2. **Integration pattern templates** — closure-based module patching for diffusers/transformers
3. **Pitfall documentation** — RMSNorm weight may be None, use `type()` not `isinstance()`, inject before CPU offload
4. **Comparative benchmarking context** — "2.67x kernel speedup on RMSNorm = only 6% end-to-end because RMSNorm is 5% of total compute"

---

## Arxiv Paper 2509.07506 Comparison

The paper proposes a **4-agent decomposition** (Testing, Profiling, Planning, Coding) achieving 1.32x average speedup on SGLang kernels with zero-shot prompting. Key takeaway: explicit agent specialization outperforms monolithic (1.32x vs 1.08x for single-agent baseline).

---

## Priority Recommendations

### HIGH PRIORITY — Adopt from KernelMem

1. **Deterministic Gating Module** — Port `machine_check_ver2.py` rules into a gating layer before LLM prescriptions. Validates SMEM budget, flags removable kernels, enforces coalescing-first for memory-bound ops. Prevents ~25% of invalid optimization suggestions.

2. **One-Method Discipline** — Schema change: `primary_optimisation_method: str` (not list). Forces focused optimization per round, easier to debug what helped/hurt.

3. **Inline Profiling Coupling** — Profile immediately after kernel verification, inject fresh NCU metrics into next optimization prompt rather than retrieving stale data.

### HIGH PRIORITY — Adopt from HF Skills

4. **Hardware-Specific Skill Templates** — Add GPU-specific entries to skill library with concrete parameters (block_size, shared_mem_kb, grid sizing per SM count).

5. **Integration Pattern Library** — Add skills tagged with `integration_target: ["diffusers", "transformers"]` containing closure-based forward-patching templates.

6. **Pitfall Field on Skills** — `pitfalls: list[str]` injected as warnings when a skill is matched.

### MEDIUM PRIORITY — Adopt from Arxiv

7. **Formalized Optimization Log** — Structure round history as `OptimizationRound(round, kernel_code, is_correct, speedup, profile, bottleneck)` and inject into generator context.

8. **Explicit Planning Step** — Consider decomposing the implicit critic into a distinct PlanningAgent that outputs structured diagnostics.

---

## Cursor/NVIDIA Multi-Agent Alignment (Apr 2026)

Cursor + NVIDIA optimized 235 CUDA kernels on Blackwell 200 GPUs for 3 weeks, achieving 38% geomean speedup. Key patterns to adopt:

### Architecture Gaps

| Area | Cursor | Autokernel | Gap |
|------|--------|------------|-----|
| **Scale** | 235 problems, 27 GPUs, 3 weeks | 1 problem, 1-4 GPUs, ~30 min sessions | No batch mode, no checkpointing |
| **SOL Score** | Logarithmic 0-1 metric (0.5=baseline, 1.0=theoretical limit) | Raw speedup only | No ceiling-relative metric |
| **Planner rebalancing** | Distributes + rebalances workers dynamically | Homogeneous workers, fixed allocation | No load rebalancing |
| **Problem taxonomy** | L1 (simple), L2 (complex), Quant, FlashInfer tiers | KernelBench vs GPU Mode format only | No difficulty classification |
| **Strategy specialization** | Different strategies per problem type (GQA, MoE, GEMM) | Same prompt template for all problems | No per-type strategy templates |
| **Coordination** | Single markdown file (rules, output format, tests) | Env vars + JSON + Jinja2 templates | More structured but less transparent |
| **Language targets** | CUDA C + CuTe DSL (opposite ends of abstraction) | Triton-focused with partial CUDA | Single language |

### Where We're Already Aligned

- Multi-worker parallelism (4 workers racing)
- Multi-round optimization with reflection (CONTINUE/PIVOT/STOP)
- Remote GPU evaluation (Modal) decoupled from LLM generation
- Format-agnostic problem interface (KernelBench + GPU Mode)
- Few-shot retrieval (4-level fallback: exact → KernelBook → skills → generic)
- Persistent skill library with evidence tracking (Cursor has no cross-run learning)

### Visualization Alignment — What Cursor Showed

**1. SOL Score Trajectory** (highest priority)
- Y-axis: SOL score (0-1, logarithmic), not raw speedup
- X-axis: Iteration number
- Horizontal lines at 0.5 (baseline) and 1.0 (theoretical limit)
- Annotated strategy inflection points ("Improve kernel scheduling", "Specialize for shapes")
- **Gap**: Our Plot A shows raw speedup over time with no ceiling reference or strategy annotations

**2. Speedup Distribution Histogram**
- Log-scale X-axis (0.25x to 256x)
- Problem count on Y-axis
- Gray bars below baseline, orange bars above
- **Gap**: We have no batch-level aggregation or distribution visualization

**3. Sequential Optimization Progress (Horizontal Timeline)**
- Dots on a horizontal bar showing strategy stages
- Each dot labeled with strategy name + % of theoretical ceiling
- Example: "General GEMM 4.5% → Blackwell instructions 15.8% → Read/write and compute 43.5% → Optimize overlap 62.4% → M dimension specialization 85.7%"
- **Gap**: We have no strategy progression timeline visualization

**4. Geomean Bar Chart**
- Three bars: Naive PyTorch (0.17x) | Optimized PyTorch (1x) | Agent (1.38x)
- **Gap**: We show single best_speedup, not geomean across problems

**5. Problem Taxonomy Table**
- Problems segmented by tier (L1/L2/Quant/FlashInfer) with counts
- **Gap**: No tier classification system

### HIGH PRIORITY — Adopt from Cursor

9. **SOL Score Computation** — Create `kernel_code/sol_metrics.py` with `compute_sol_score(kernel_runtime_us, ref_runtime_us, hardware_peak_tflops, hardware_bandwidth, total_flops, total_bytes)`. Returns logarithmic 0-1 score where 0.5 = optimized baseline, 1.0 = theoretical hardware limit. Add `sol_score` field to `ProfileMetrics`.

10. **Annotated SOL Trajectory** — Extend Plot A to show SOL score on Y-axis with horizontal baseline (0.5) and ceiling (1.0) lines. Annotate MetaOptimizer strategy pivots as inflection point labels on the trajectory.

11. **Problem Classifier + Strategy Templates** — Create `kernel_code/problem_classifier.py` that detects problem types (BF16 GQA, NVFP4 MoE, GEMM, Reduction, Fused ops, Attention) from reference code. Create per-type strategy templates in `data/strategies/` that inject domain-specific optimization guidance.

12. **Round-Level Checkpointing** — Save `best_kernel`, `best_speedup`, `round_history` to `OPENKERNEL_CHECKPOINT_DIR/round_{N}.json` after each round. Resume from last checkpoint on restart. Enables multi-hour/multi-day autonomous runs.

### MEDIUM PRIORITY — Adopt from Cursor

13. **Batch Optimizer** — Create `kernel_code/batch_optimizer.py` that runs MetaOptimizer across a list of problem files and aggregates: geomean speedup, % beating baseline, % exceeding 2x, median SOL score.

14. **Strategy Progression Timeline** — Render horizontal dot timeline showing optimization strategy stages with SOL score at each stage (like Cursor's GEMM progression: 4.5% → 15.8% → 43.5% → 62.4% → 85.7%).

15. **Speedup Distribution Histogram** — For batch runs, render log-scale histogram of per-problem speedups with gray (regression) / orange (improvement) bars and geomean line.

16. **Worker Load Rebalancing** — Extend MetaOptimizer to monitor per-worker throughput and dynamically reassign rounds to faster workers. Detect timeouts and re-spawn stuck workers.

---

## Cursor Kernels Blog Alignment (Blackwell/MXFP8)

Cursor also published a deep technical post on hand-writing MXFP8 MoE kernels for Blackwell B200 GPUs from scratch (pure CUDA/PTX, zero library dependencies). They achieved **3.5x MoE layer speedup** → **1.5x end-to-end training speedup**, outperforming all open-source alternatives. To outperform Cursor's kernel system, we need to understand and encode these patterns.

### Key Technical Insights

**1. Blackwell TMEM Changes Everything**
- Hopper: tensor core results accumulate in registers → easy CUDA core access
- Blackwell: results accumulate in TMEM → must transfer TMEM→registers→TMEM for custom arithmetic
- This makes microscaling dequantization 1.76x slower than matmul on Blackwell (vs 1.03x on Hopper)
- Solution: use hardware block-scaled MMA (tcgen05.mma...block_scale) which handles dequantization in tensor cores

**2. Quantization Overhead Dominates**
- For MXFP8 matmul, quantization can be 76% of matmul time if not fused
- Cursor built fastest-ever MXFP8 quantization kernel (6.2+ TB/s vs 4.5 TB/s for TransformerEngine/TorchAO)
- Key: fuse quantization into kernel prologues/epilogues, produce scale factors in hardware-compatible layout

**3. Warp Specialization Pipeline**
- 3 warpgroups (384 threads), each specialized: 2 warpgroups for TMEM→register→SMEM→HBM, 1 warpgroup for warp-specialized loads + MMA
- 5-slot circular buffers in TMEM and SMEM
- Persistent grid: one threadblock per SM (148 on Blackwell)

**4. Expert-Wise L2 Supergrouping**
- For grouped GEMMs (MoE), L2 cache optimization is critical — bad patterns cause 50% performance loss
- Apply supergrouping per expert submatrix, not whole output matrix

### HIGH PRIORITY — Adopt from Cursor Kernels

17. **Blackwell B200 GPU Support** — Add B200 to GPU_SPECS (148 SMs, 227KB SMEM, 8 TB/s HBM3e, 4500 TFLOPS FP8). Create `hw_b200_optimization.json` skill covering TMEM, tcgen05.mma, TMA, 2-CTA clustering, warp specialization. Target ceilings: 2750 TFLOPS MXFP8, 1550 TFLOPS BF16.

18. **MXFP8 Quantization-Aware Gating** — Add quantization cost analysis to optimization gate. Flag when quantization overhead exceeds matmul time. Add "fuse quantization" as allowed method. Encode Cursor's insight: CUDA cores are 1/56th tensor core speed on Blackwell, so dequantization in CUDA cores is unacceptable.

19. **CUDA/PTX Generation Path** — Add CUDA C backend alongside Triton. Cursor showed that Triton cannot reach hardware limits on Blackwell (TMEM/TMA/tcgen05 are PTX-level). Enable A/B comparison: generate in both Triton and CUDA C, keep the faster kernel.

20. **MoE Strategy Templates** — Add MoE-specific optimization strategies: grouped GEMMs (Fprop/Dgrad/Wgrad), expert-wise L2 supergrouping, persistent grid patterns, quantization-fused SwiGLU/GEGLU epilogues. MoE layers are 53% of forward-pass time — this is where the biggest gains are.

---

## Cursor Warp Decode Alignment (MoE Inference)

Cursor's "Warp Decode" blog (Apr 2026) demonstrates a fundamental insight: **the right parallelism axis depends on workload**. For small-batch MoE decode, organizing by outputs (not experts) delivers 1.84x throughput and 1.4x better accuracy.

### Key Insight: Parallelism Axis as Strategy Decision

Traditional MoE has 8 stages (5 are bookkeeping). Warp decode compresses to 3 by flipping the parallelism axis:
- **Traditional (expert-centric):** Route → Gather → Pad → Quantize → GEMM → Scatter → Reduce → Write
- **Warp decode (output-centric):** Route → Warp compute (stream weights + fused gate/up/down + fold routing) → Write

The core technique: each warp owns one output neuron, streams all weight rows, accumulates all top-k experts in private FP32 registers, reduces via `__shfl_xor_sync` butterfly (no SMEM, no barriers, no buffers).

### Results
- 1.84x decode throughput (66→122 tokens/s on B200)
- 1.4x closer to FP32 ground truth (by eliminating intermediate MXFP8 quantization)
- 58% of peak HBM bandwidth (3.95/6.8 TB/s) at batch=32
- Flat gain across all context lengths (5k-50k tokens)

### What This Means for Autokernel

21. **Workload-Aware Parallelism Selection** — The problem classifier must detect workload type (small-batch decode vs prefill/large-batch) and recommend the right parallelism axis. For decode: warp-per-output. For prefill: expert-centric grouped GEMM. This is a strategy-level decision that our MetaOptimizer's reflection should make, not something hardcoded.

22. **Warp Independence as Optimization Pattern** — Add "embarrassingly parallel warp" as a skill pattern. Key properties: no shared mutable state, all accumulation in private registers, warp-level shuffle reduction instead of SMEM. This pattern applies beyond MoE to any small-batch matrix-vector product.

23. **Stage Elimination Analysis** — Add diagnostic capability: count pipeline stages in reference implementation, identify which are bookkeeping vs compute. Flag when >50% of stages are non-compute (Cursor's traditional path had 5/8 stages as bookkeeping). This is a high-value optimization signal.

### The Broader Pattern

Across all three Cursor blog posts, one meta-pattern emerges:

| Blog | Key Move | Speedup |
|------|----------|---------|
| Multi-Agent (235 kernels) | Autonomous exploration via specialized agents | 1.38x geomean |
| Kernels (MXFP8 MoE training) | Eliminate dequant overhead via hardware block-scale MMA | 3.5x MoE layer |
| Warp Decode (MoE inference) | Flip parallelism axis from experts to outputs | 1.84x decode |

**Common thread:** All three gains came from questioning the conventional decomposition — not from micro-optimizing within it. Cursor's multi-agent system explored unconventional strategies. Their kernel team questioned whether dequantization needed to happen at all (use hardware MMA instead). Their warp decode team questioned whether expert-centric was the right axis.

**For autokernel:** Our MetaOptimizer's PIVOT mechanism is the right architecture for this. When the agent detects plateau, reflection should consider "is the parallelism axis wrong?" and "are there stages that can be eliminated entirely?" — not just "try a different tiling strategy." The skill library should encode these meta-level pivots as first-class optimization patterns.

### Strategy to Outperform Cursor

Cursor's advantage is **human expertise encoded in hand-written kernels**. Their multi-agent system (38% geomean on 235 problems) and their internal kernel team (3.5x MoE layer) represent two different levels:

1. **Multi-agent level (achievable now):** Match their 38% geomean by implementing SOL scoring, problem classification, strategy specialization, and batch evaluation. Our reflection-based pivoting + persistent skill learning + deterministic gating already exceeds their single-markdown coordination protocol.

2. **Expert kernel level (stretch goal):** Match their 3.5x MoE speedup by adding CUDA/PTX generation, Blackwell-specific skills, MXFP8 quantization awareness, and warp specialization templates. This requires the agent to reason about PTX-level instructions (tcgen05.mma, cp.async.bulk.tensor, mbarrier), which is at the frontier of what LLMs can do with sufficient context.

**The path to outperformance:** Our architectural advantage (persistent cross-run learning via skill library + evidence tracking) compounds over time. Cursor's multi-agent system starts fresh each run. If we can encode the patterns from their kernels blog as skills (TMEM pipeline, supergrouping, quantization fusion), our agent will have that expertise permanently available — effectively turning their weeks of kernel engineering into retrievable optimization patterns.
