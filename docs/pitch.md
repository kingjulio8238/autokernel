# The Pitch

---

## The Problem

Every AI company runs on GPU kernels. Every matrix multiplication, every attention head, every activation function — it all executes as GPU kernel code. The faster those kernels run, the less you spend on chips, power, and data centers. The slower they run, the more you burn.

NVIDIA provides cuBLAS and cuDNN — hand-tuned libraries for standard operations. These are fast and free. But modern AI models don't run on standard operations. They run on custom compositions: fused attention patterns, novel activations, model-specific layer combinations. For these, there's no library. You either hire kernel engineers at $500K+ each to hand-optimize your code — or you accept 2-5x slower execution and pay for it in GPU hours.

There are roughly 1,500 kernel engineers in the world. Every frontier lab, every hyperscaler, every custom silicon team is competing for the same talent. The demand is 10x the supply. And each engineer can optimize maybe 30-50 kernels per year.

The result: **billions of dollars in GPU compute are wasted annually on unoptimized kernels that no one has time to fix.**

---

## The Insight

Kernel optimization is now solvable with AI. Over the past 12 months, 60+ research systems have demonstrated that LLMs can generate GPU kernels that match or beat hand-tuned code — when given the right feedback loop.

The breakthrough isn't one-shot code generation. It's **iterative optimization with hardware feedback**: generate a kernel, compile it, benchmark it, profile it, diagnose the bottleneck, generate a better one. Repeat. The best systems achieve 100% correctness on standard benchmarks and 2-5x speedups over PyTorch — at a cost of $0.30 per kernel.

But every one of these systems is either a research prototype, locked inside a big tech company, or a black box that gives you a kernel without explaining why.

No one has built the developer tool.

---

## openkernel

openkernel is a self-recursive kernel optimization engine. Give it a PyTorch operation and a target GPU — it produces an optimized CUDA or Triton kernel through structured search with hardware profiler feedback.

**What makes it different:**

**It searches in strategy space, not just code space.** Most systems generate code and hope it's faster. openkernel reasons about what optimization to apply before writing any code — tiling strategy, memory access pattern, fusion boundary — then implements and validates. If the code is buggy but the strategy is sound, the strategy survives. This is why it finds optimizations that brute-force approaches miss.

**It gets smarter with every kernel.** Every optimization run teaches the system something. Successful strategies are extracted into a skill library — "online softmax reduction works for memory-bound reductions" — and retrieved for similar problems in the future. Problem 50 optimizes faster than problem 1. Problem 500 is faster still.

**It explains what it does.** Two LLM roles — a Generator that writes code and a Critic that analyzes hardware profiler data — produce structured, inspectable reasoning. The Critic says: "This kernel is memory-bound. L2 cache hit rate is 45%. Recommendation: restructure to coalesced access with BLOCK_K=64 tiles." The kernel engineer sees why, learns from it, stays in control.

**It runs on cloud GPUs.** No local NVIDIA hardware required. Evaluation runs on Modal — compile, benchmark, and profile in isolated cloud containers. Develop from a laptop. Test on H100s.

**It supports both CUDA and Triton.** The kernel engineer chooses the backend. Backend-specific optimization strategies for each — Triton's autotuner for parametric search, CUDA's warp-level primitives for peak performance.

**It's model-agnostic.** Bring your own LLM. Claude, GPT, Gemini, open-source models — openkernel works with any provider. When our post-trained kernel model (kernelgen-1) ships, it'll be faster and cheaper for kernel tasks than any frontier model. Until then, you're not locked in.

---

## kernel code

kernel code is how developers use openkernel. A terminal-native tool built for kernel engineers — with visualizations that no general-purpose coding assistant has.

**What you see:**

A Textual TUI in your terminal showing the optimization running live — speedup trajectory, profiling gauges, experiment log, the Critic's analysis. Press `d` to open a full browser dashboard with roofline plots, 3D optimization landscapes, strategy trees, and side-by-side kernel diffs with performance annotations.

**Why it matters:**

Claude Code writes kernels. Cursor writes kernels. Codex writes kernels. None of them optimize kernels. They don't know what a roofline looks like. They can't read Nsight Compute output. They don't have a feedback loop between hardware profiling and code generation.

kernel code does. It's built for the workflow kernel engineers actually have: write → compile → benchmark → profile → diagnose → optimize → repeat. Every panel, every visualization, every keyboard shortcut is designed for that loop.

**The trace flywheel:**

Every optimization run in kernel code captures the full trace — every prompt, every generated kernel, every benchmark result, every profiler diagnosis. With user consent, these traces feed the training pipeline for kernelgen-1. More developers using kernel code = more traces = better model = better results = more developers. This is how Cursor built their models. We're applying the same flywheel to kernel optimization.

---

## Why Now

Three things converged:

1. **LLMs can write kernel code.** As of 2025-2026, frontier models generate correct CUDA and Triton kernels at 80-100% rates when given iterative feedback. This wasn't possible 18 months ago.

2. **The benchmark exists.** KernelBench (Stanford, ICML 2025) standardized how kernel generation is evaluated — 250 problems, 4 difficulty levels, clear metrics. For the first time, systems can be objectively compared. openkernel is designed to hill-climb this benchmark.

3. **The market is massive and underserved.** AI infrastructure CapEx exceeds $600B/year. Kernel optimization is the highest-leverage investment in the stack — multiplicative across energy, chips, infrastructure, models, and applications. A 20% kernel speedup saves a frontier lab $220M/year at 50K GPU scale. Yet the tooling barely exists. Standard Kernel raised $20M. Makora raised $8.5M. The space is early and large.

---

## The Economics

Kernel optimization sits at the narrowest waist of the AI compute stack:

```
 5. Applications     $20-200/mo subscriptions, per-token API pricing
 4. Models           $100M-$2B per training run, growing inference OpEx
 3. Infrastructure   $10-15B per hyperscale data center
 2. Chips            $25K-$40K per H100, $250-400M per 10K cluster
 1. Energy           $55M/yr electricity at 100K GPU scale
```

Every other optimization touches one layer. Cheaper electricity? Layer 1 only. Bulk GPU discount? Layer 2 only. Model distillation? Layer 4 only.

**A 20% faster kernel reduces costs across all five layers simultaneously.** Fewer GPUs needed = less power, less cooling, fewer racks, less networking, faster training, cheaper inference. For a 50K GPU deployment: ~$220M/year in savings from kernel optimization alone.

And a kernel engineer team of 15 at $500K each = $7.5M/year. openkernel does the same work for pennies per kernel.

---

## The Business

**Open-core model:**

openkernel is open source (Apache 2.0). Developers can inspect, modify, and contribute. This builds the community that feeds the trace flywheel that trains kernelgen-1 that makes the product better.

kernel code is the commercial product. Free tier for individual developers. Premium tier for teams (dashboards, collaboration, more GPU credits). Enterprise tier for frontier labs (custom hardware profiling, on-prem deployment, SLA, private model fine-tuning).

**Pricing tied to value:** Enterprise contracts priced as a fraction of compute savings delivered. If we save you $50M/year, we charge $2-5M. The ROI is obvious.

**Go-to-market:** Publish competitive KernelBench results → open-source openkernel → developers adopt kernel code → bottom-up adoption within companies → enterprise conversion. The Vercel playbook applied to GPU infrastructure.

---

## The Team

Four engineers building with coding agents. We move fast.

---

## What We're Asking For

Belief that the best kernel optimization system should be a developer tool, not a black box. That kernel engineers deserve the same quality of tooling that web developers have. And that the company that builds the trace flywheel — more users, more data, better model, better results — wins this market.

openkernel is the engine. kernel code is the product. kernelgen-1 is the moat.
