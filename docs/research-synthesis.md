# Autokernel v2 + Kernel Code: Research Synthesis

*April 16, 2026 — Compiled from 7 parallel research agents*

---

## Executive Summary

The GPU kernel optimization space has exploded from a research curiosity to a funded competitive market in 6 months. 60+ papers/systems exist. KernelBench L1/L2 is effectively solved (multiple systems at 100% correctness). $28.5M+ in startup funding has entered (Standard Kernels $20M, Makora $8.5M). Meta's KernelEvolve achieves 100% on all 250 KernelBench problems in production.

**The opportunity is real but the window is narrowing.** Autokernel v2 needs a sharp, differentiated thesis — it cannot be "yet another kernel agent."

---

## Part 1: The Market

### Size
- AI infrastructure CapEx: ~$660-690B combined hyperscaler spend (2026)
- Kernel optimization value at stake: 10-20% of compute costs = $60-120B in potential savings
- Kernel engineering talent spend: ~$500M-$1.2B/year globally (1,000-2,000 engineers at $400-600K TC)

### Competition

| Tier | Players | Status |
|------|---------|--------|
| **Funded startups** | Standard Kernels ($20M), Makora/MAKO ($8.5M, Jeff Dean angel), Gimlet Lab | Direct competitors |
| **Big tech internal** | KernelEvolve (Meta, production), AlphaEvolve (DeepMind), KernelLLM (Meta 8B model) | Set the performance bar |
| **Research/open-source** | CudaForge, Kernel-Smith (SOTA Triton), CUDA Agent (SOTA overall), KernelSkill (5.44x L1), STARK, GEAK (AMD) | Competition for mindshare |
| **Name collision** | RightNow AI released "AutoKernel" on April 6, 2026 | Branding concern |

### Key Insight
Standard Kernels goes low (CUDA + PTX, 105% of cuBLAS). Makora goes evolutionary. Gimlet goes multi-hardware (Apple Metal + CUDA). Meta goes production-scale. **No one owns the developer experience layer.**

### Business Model (Recommended)
Phase 1: Open-core framework + design partnerships ($1-5M ARR)
Phase 2: kernel code product with usage-based pricing ($5-20M ARR)
Phase 3: Enterprise site licenses tied to compute savings ($50-100M ARR)

### Highest-Value Buyer Segments
1. **Custom silicon teams** (Google TPU, Amazon Trainium, Meta MTIA) — existential need, immature kernel stacks
2. **Frontier labs** (Anthropic, OpenAI, xAI) — GPU costs are #1 expense
3. **AI infra companies** (CoreWeave, Lambda, Together) — compete on price/performance

---

## Part 2: The Technology Landscape

### Five Paradigms for Kernel Optimization

| Paradigm | Best System | Peak Result | Sample Efficiency | Cross-Problem Transfer |
|----------|------------|-------------|-------------------|----------------------|
| Hill-climbing | Caesar | 72% L2 | High | None |
| Evolutionary | Kernel-Smith | SOTA (Triton) | Medium | Via post-training |
| World model | K-Search | 14.3x on MoE, 2.10x over evo | **Highest** | None (in-context) |
| Multi-agent | KernelSkill | 5.44x L1 | Medium-High | Dual-level memory |
| RL | CUDA Agent | SOTA (overall) | Low (training) | Via model weights |

### Critical Findings

1. **Role separation > self-refine.** CudaForge's Coder+Judge (2 agents) achieves 1.68x. Same model self-refining achieves 1.1x. Separating generation from evaluation prevents self-evaluation bias.

2. **Curated profiler metrics > raw dumps.** CudaForge selected 24 Nsight Compute metrics (from ~600) via correlation analysis. Focused metrics outperformed exhaustive output (1.32x vs 1.28x median).

3. **Strategy persistence > code persistence.** K-Search's key innovation: if a good optimization strategy produces buggy code, the strategy survives for retry. Evolutionary methods would discard both.

4. **Sequential depth > parallel breadth.** Kevin showed 16 trajectories x 8 turns beats 128 single-turn attempts under same compute budget.

5. **KernelBench L1/L2 is solved.** Multiple systems at 100%. Differentiation frontier: L3/L4, production deployment, multi-hardware, cost efficiency, developer experience.

### Backend Strategy
- **Primary: Triton** — Python-native, AMD portability, dominant LLM target, 78-82% of hand-tuned CUDA automatically
- **Secondary: CUDA C++** — escape hatch for maximum performance, warp-level primitives
- **Optional: CUTLASS CuTe DSL** — GEMM-specific when Triton's 5-10% gap matters
- **Avoid: PTX** — Standard Kernels' territory, extremely specialized

### Two-Phase Optimization (Consensus Pattern)
All top systems converge on: LLM explores structural/algorithmic space (code transformations) → traditional autotuner handles parametric space (block sizes, warp counts). Triton's `@triton.autotune` handles phase 2 natively.

---

## Part 3: Autokernel v2 Architecture

### Recommended: 3-Level Hybrid Architecture

```
OUTER LOOP: Strategy Evolution (GEPA-style)
  Maintains Pareto frontier of optimization strategies
  Evolves strategies via LLM reflection on execution traces
  Preserves diverse approaches (MAP-Elites / Pareto dominance)
  Cross-session, cross-problem knowledge transfer

  MIDDLE LOOP: World Model Search (K-Search-style)
    For each strategy, maintains search tree of optimization intents
    LLM estimates priorities, inserts/updates/prunes tree nodes
    Decouples "what to try" from "how to implement it"
    Non-monotonic path tolerance (intermediate regressions OK)

    INNER LOOP: Implementation Refinement (Caesar-style)
      Generate + compile + benchmark + profile concrete kernels
      Feed 24 curated NCU metrics back to LLM
      Retry up to K times before declaring node stagnation
      Report results back to world model for tree update
```

### Why This Specific Hybrid

- **Outer loop** solves K-Search's weakness (no cross-session learning)
- **Middle loop** solves evolutionary methods' weakness (can't reason about WHY strategies work)
- **Inner loop** solves the implementation noise problem (good strategy shouldn't die from syntax error)

### Agent Roles (Start with 2)

- **Generator**: World model reasoning + code generation (middle + inner loops)
- **Critic**: Profiling analysis + strategy evaluation (bottleneck identification, optimization suggestions)

CudaForge proved 2 roles at $0.30/kernel. KernelSkill's 7 agents get higher speedup but at much higher complexity. Start minimal, add roles when you hit specific failure modes.

### Strategy Library (Dual-Level Memory)

- **Level 1 — Optimization Skills** (long-term, cross-problem): Decision policies mapping (bottleneck type, hardware) → optimization technique. KernelSkill showed 3.6x per-round efficiency with this.
- **Level 2 — Trajectory Memory** (short-term, within-problem): What was tried, what worked, prevents oscillation. Feeds into world model tree.
- **Pareto frontier**: Keep non-dominated strategies across kernel types, hardware targets, optimization objectives.

### Profiling Integration (Non-Negotiable)

```bash
ncu --set full --csv --target-processes all <kernel> > profile.csv
```
Parse → extract roofline classification + top warp stall reasons + memory utilization → structured text → LLM prompt. CudaForge's validated 24-metric subset:
- Occupancy limits (registers, shared memory, blocks)
- Instruction execution rates
- Memory throughput (DRAM, L1, L2 hit rates)
- Warp stall types (memory dependency, scoreboard, barrier, branch)

Also integrate **simple-torchroofline** for analytical pre-screening without GPU.

### Resources to Leverage
- **OpenEvolve** — infrastructure base for evolutionary search
- **KernelBook** (18K pairs) — few-shot examples in prompts
- **KernelLLM** (Meta 8B) — cheap first-pass generator, refine with frontier model
- **FlashInfer-Bench** — real-world inference workload validation
- **Triton autotuner** — parametric optimization after LLM structural optimization

### Where autokernel v2 Wins (Differentiation)

| Angle | Why It's Defensible |
|-------|-------------------|
| **Self-recursive improvement** | Use optimized kernels to improve the optimization loop itself. No one else does this |
| **Continuous optimization** | Monitor production → detect regression → re-optimize → hot-swap. FlashInfer's dynamic replacement API enables this |
| **Cost efficiency** | Hierarchical model routing: KernelLLM (8B) for easy kernels, frontier model for hard ones. Beat CudaForge's $0.30/kernel |
| **Developer experience** | Every existing tool is a black box. Explain WHY each optimization works. Teach the developer |
| **Production integration** | Not just benchmarks — real PyTorch/torch.compile ecosystem integration |

### What NOT to Compete On
- KernelBench L1/L2 scores (solved by 5+ systems)
- One-shot kernel generation (commodity)
- "Another LLM agent that writes kernels" (15+ exist)
- PTX-level optimization (Standard Kernels' territory)

---

## Part 4: Kernel Code Product

### Build Decision: From Scratch in Python with Textual

**Don't fork Claude Code** — proprietary license, 512K lines TypeScript, wrong language for kernel ecosystem, wrong agent loop.

**Don't fork Aider** — right language but no rich TUI, no profiling integration.

**Build fresh in Python with Textual TUI** because:
- Python-native (matches PyTorch/Triton/CUDA ecosystem)
- Textual has richest widget library for data viz
- Design agent loop specifically for kernel optimization
- MCP for extensibility (GPU profiling, KernelBench, hardware monitoring as MCP servers)
- Model-agnostic from day 1 (LiteLLM)

### Borrow Patterns From
- **Claude Code** (source at `/Users/juliansaks/Downloads/src/`): Agent loop structure, sub-agent coordination (Fork/Teammate/Worktree), session persistence, MCP model
- **Aider**: Tree-sitter repo mapping, git-native workflow, model-agnostic LLM integration
- **Cursor**: Local embeddings for kernel pattern matching

### Visualization Layer (Plotly Dash + Browser Dashboard)

| Panel | What It Shows | Inspiration |
|-------|--------------|-------------|
| Optimization trajectory | Live speedup over iterations, color-coded keep/discard | PufferLib Constellation Fig 2 |
| Roofline model | Arithmetic intensity vs GFLOP/s with ceiling lines | Nsight Compute |
| Resource utilization | Occupancy, registers, shared mem, cache hit rates | Nsight "Speed of Light" |
| Experiment log | Scrollable colored table of all variants | results.tsv live |
| Code diff | Syntax-highlighted side-by-side with perf annotations | GitHub diff |
| Optimization landscape | 3D scatter / t-SNE of kernel variants | Constellation Fig 1/3 |
| Memory access heatmap | Per-warp memory access patterns (on-demand) | Nsight memory workload |

**Integration pattern**: Terminal (rich progress bar, current best) + Browser dashboard (full visualization, opt-in with `--dashboard`). Data flow via JSON cache files (same pattern as PufferLib Constellation and KernelTuner).

### Kernel-Specific Agent Loop (Different from General Coding)
1. **Analyze** reference kernel + target hardware specs + roofline bounds
2. **Generate** optimized kernel variant (Triton or CUDA)
3. **Compile** with error recovery
4. **Benchmark** against reference
5. **Profile** with Nsight Compute (24 curated metrics)
6. **Score** — update trajectory and strategy library
7. **Decide** — Critic analyzes profiling, identifies bottleneck, prescribes next optimization
8. **Branch** — optionally explore multiple strategies in parallel (sub-agent pattern)

---

## Part 5: Roadmap

### Phase 1: autokernel v2 Core (Months 1-3)
- Implement 3-level hybrid loop (strategy evolution × world model × refinement)
- 2-agent roles (Generator + Critic)
- Triton primary backend, NCU profiling integration
- Dual-level strategy library
- Hill-climb KernelBench L1-L3, target SOTA
- Open-source release

### Phase 2: kernel code MVP (Months 3-6)
- Python/Textual TUI with kernel-specific agent loop
- Plotly Dash browser dashboard (opt-in)
- MCP tool servers (nsight, KernelBench, GPU monitor)
- Model-agnostic LLM integration
- 2-3 design partnerships with frontier labs / custom silicon teams

### Phase 3: Production & Scale (Months 6-12)
- Continuous optimization pipeline (monitor → detect → re-optimize → hot-swap)
- Multi-hardware (NVIDIA + AMD, custom silicon partnerships)
- Enterprise features (team dashboards, kernel registries, CI integration)
- Usage-based pricing launch

### Phase 4: Moat Building (Months 12-18)
- Self-recursive improvement (optimize kernels used in optimization loop)
- Cross-problem knowledge transfer at scale
- Kernel marketplace / community contributions
- L4 (HuggingFace models) and production workload optimization

---

## Appendix: Key References

### Systems to Study Deeply
- **K-Search** — world model approach, most sample-efficient
- **CudaForge** — multi-agent + profiling, $0.30/kernel baseline
- **KernelEvolve** (Meta) — production-scale, 100% KernelBench
- **KernelSkill** — dual-level memory, 5.44x L1
- **CUDA Agent** — RL-trained, SOTA overall
- **AlphaEvolve/OpenEvolve** — evolutionary framework
- **GEPA** — prompt evolution, 35x more efficient than RL

### Infrastructure to Leverage
- OpenEvolve (evolutionary search framework)
- KernelBook (18K training pairs)
- KernelLLM (Meta 8B model)
- simple-torchroofline (analytical bounds)
- FlashInfer-Bench (production workload testing)
- kernelbench-tinker (RLVR integration)
- Caesar (multi-turn inference engine)
- ThunderKittens 2.0 (potential third target DSL — 1.3x faster than cuDNN on Blackwell, adopted by Cursor/Together AI)
- tritonBLAS analytical model (94.7% of exhaustive autotuning with zero overhead — first-principles can replace brute-force)

### Competitors to Track
- Standard Kernels Co. ($20M, CUDA+PTX)
- Makora/MAKO ($8.5M, evolutionary, AMD focus)
- Gimlet Lab (multi-hardware)
- Modular/Mojo ($1.6B valuation, adjacent)
- Cursor + NVIDIA multi-agent (38% geomean speedup, 235 kernels, zero human intervention)

### Design Foundations (from Autoresearch)
Karpathy's autoresearch proved 3 primitives are sufficient for recursive improvement:
1. **Single editable artifact** per optimization target (the kernel file)
2. **Fixed evaluation budget** per iteration (standardized benchmarking)
3. **Scalar metric** (correctness x speedup x profiler efficiency)

These should be the foundation. The 3-level hybrid architecture layers sophistication in search strategy on top of this simple core. Autoresearch ran ~910 experiments in 8 hours on 16 GPUs and achieved 11% speedup on already-optimized nanochat. All 20 improvements were additive and transferred across model scales.
