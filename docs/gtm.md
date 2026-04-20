# Go-to-Market: openkernel + kernel code

*April 2026*

---

## The Framing: Kernel Optimization as the Narrowest Waist of the AI Stack

Every GTM conversation starts with the five-layer cake (see `docs/five-layer-cake.md`):

```
 5. Applications     ← Revenue ($20-200/month subscriptions, per-token API pricing)
 4. Models           ← Training ($100M-$2B/run) + Inference (ongoing OpEx)
 3. Infrastructure   ← Data centers, networking, cooling ($10-15B hyperscale)
 2. Chips            ← GPUs, TPUs, ASICs ($25K-$40K per H100)
 1. Energy           ← Electricity + cooling ($55M/yr at 100K GPU scale)
```

**The pitch**: Kernel optimization is the only investment that's multiplicative across all five layers simultaneously. A 20% kernel speedup at 50K GPU scale = **~$220M/year in savings**. Every other optimization (cheaper power, bulk GPU discounts, model distillation) only touches one layer.

**openkernel automates this** — replacing $5-10M/year kernel engineering teams with a system that optimizes hundreds of kernels overnight, getting better with every run.

---

## Market Segmentation by Layer

### Who buys, why, and how to reach them

| Layer | Buyer Segment | Pain Point | Why openkernel | Entry Point | Deal Size |
|-------|--------------|------------|---------------|-------------|-----------|
| **Layer 2** (Chips) | Custom silicon teams (Google TPU, Amazon Trainium, Meta MTIA, OpenAI/Broadcom) | Kernel quality IS chip performance. No cuBLAS. $10B+ chip programs succeed or fail on kernel software. | Only tool targeting multi-hardware kernel optimization. Their kernel engineers are the most expensive and hardest to hire. | Design partnership — "we'll optimize your kernel stack" | $1-10M/year |
| **Layer 3** (Infra) | AI infrastructure cos (CoreWeave, Lambda, Together, Modal) | Compete on price/performance. Every % of GPU efficiency = margin. | Optimize the kernels their customers' models run on. Direct margin impact. | Integration partnership — embed openkernel in their platform | $500K-5M/year |
| **Layer 4** (Models) | Frontier labs (Anthropic, OpenAI, xAI, DeepSeek) | GPU costs are #1 expense. Training runs cost $100M-$2B. Inference OpEx growing. | 10-100x productivity multiplier for their kernel teams. Faster training = competitive advantage. | kernel code licenses for their kernel eng team | $500K-5M/year |
| **Layer 4** (Models) | Inference providers (Fireworks, Baseten, Anyscale) | Compete on inference cost/latency. Margins thin. | Automated kernel optimization for customer models. | API integration — optimize on deploy | $200K-2M/year |
| **Layer 5** (Apps) | Enterprise AI teams (Fortune 500 deploying models) | Inference costs growing as AI adoption scales. Need cost optimization without ML expertise. | Self-serve kernel optimization. No kernel engineering required. | kernel code self-serve product | $5K-50K/year |

---

## GTM Motion: Open-Source Flywheel → Product-Led Growth → Enterprise

Following the Vercel playbook (open-source tool → community → PLG → enterprise):

### Phase 1: Credibility (KernelBench Results)

**Goal**: Prove openkernel works by publishing competitive KernelBench results.

**Actions**:
- Run openkernel sweeps on KernelBench L1-L3
- Publish results publicly (HF dataset + blog post + leaderboard submission)
- Open-source the openkernel framework (Apache 2.0)
- Key metric: fast_1 score competitive with CudaForge/Kernel-Smith/CUDA Agent

**Framing**: "openkernel achieves [X]% fast_1 on KernelBench L1 — faster, cheaper, and easier to use than any existing system. Here's the data."

**Why this matters**: Every buyer segment above will Google "kernel optimization benchmark" before talking to us. KernelBench results are the credibility gate. Standard Kernel doesn't publish KernelBench numbers. We do.

**Distribution channels**:
- HF Hub (dataset + model cards)
- GPU Mode Discord (10K+ kernel engineers)
- KernelBench leaderboard (Stanford)
- Twitter/X (ML community)
- Blog post with five-layer-cake framing (why this matters economically, not just technically)

### Phase 2: Developer Adoption (Bottom-Up)

**Goal**: Get kernel engineers using openkernel and kernel code daily.

**Actions**:
- Ship kernel code (TUI + dashboard) as the primary UX
- Free tier: unlimited optimization on personal projects, community GPU credits
- Premium tier: more Modal GPU credits, priority models, team features
- Publish tutorials: "Optimize your first kernel in 5 minutes"
- GPU Mode competition integration (optimize competition kernels with openkernel)
- Hackathons and kernel optimization challenges

**The flywheel**:
```
Developer tries openkernel (free, open source)
    → Gets faster kernel (immediate value)
    → Shares result on Twitter/Discord (organic growth)
    → Teammates see it (bottom-up adoption within company)
    → Team wants dashboards + collaboration (kernel code premium)
    → Company wants enterprise features (enterprise sales conversation)
```

**Key metric**: Monthly active kernel optimizations (not just downloads — actual optimization runs)

**Trace flywheel starts here**: Every optimization run (with opt-in consent) feeds the trace store. More users → more traces → better kernelgen-1 → better results → more users.

### Phase 3: Enterprise Conversion

**Goal**: Convert bottom-up adoption into enterprise contracts.

**Signal**: Multiple engineers at the same company using kernel code = enterprise sales trigger.

**Enterprise features** (paid tier):
- Team dashboards and shared optimization history
- SSO/SAML authentication
- On-prem Modal alternative (run on customer's GPUs)
- Custom hardware profiling (NCU on customer's bare-metal)
- SLA and support
- kernelgen model fine-tuned on customer's kernel patterns (private, not shared)
- CI/CD integration (optimize kernels on every deploy)
- Production monitoring (continuous optimization pipeline)

**Pricing model**: Usage-based, tied to value delivered.
- Base: per-kernel-optimization pricing (beat CudaForge's $0.30 baseline)
- Enterprise: percentage of compute savings (if we save you $10M/year, we charge $500K-$1M)
- This naturally scales with customer size — bigger customers = more compute = more savings = higher contract value

### Phase 4: Platform (Long-Term)

**Goal**: Become the infrastructure layer for kernel optimization industry-wide.

**Actions**:
- Kernel marketplace (community-contributed optimized kernels, rated by speedup)
- kernelgen-1 → kernelgen-2 → ... (improving model from growing trace data)
- Multi-hardware optimization as a service (NVIDIA + AMD + custom silicon)
- Integration with inference engines (SGLang, vLLM, TensorRT-LLM)
- Integration with training frameworks (Megatron, DeepSpeed, FSDP)
- "Optimize on deploy" API — automatically optimize kernels when deploying a model

---

## Positioning: How We Frame Against Competitors

### vs Standard Kernel ($20M, CUDA+PTX)

**Their positioning**: "AI-generated CUDA kernels that outperform cuBLAS. 80%-4x performance gains."

**Our differentiation**:
- **Accessibility**: They go low (PTX). We go where developers are (Triton + CUDA). Lower barrier to entry.
- **Developer experience**: They're a black box. kernel code gives full visibility — roofline plots, profiling dashboards, optimization trajectories. Kernel engineers learn and stay in control.
- **Open source**: They're closed. We're open-core. Developers can inspect, modify, contribute.
- **Cost transparency**: We publish our cost per kernel. They don't.

**Framing**: "Standard Kernel optimizes kernels for you. openkernel optimizes kernels with you — and teaches your team in the process."

### vs Modular/Mojo ($1.6B, new language)

**Their positioning**: "A new programming language for AI infrastructure. Python-like but 100x faster."

**Our differentiation**:
- **No new language required**: We optimize existing CUDA and Triton code. No migration needed.
- **Works today**: Mojo 1.0 not yet released. openkernel works with your existing stack.
- **Different problem**: Mojo replaces your language. openkernel optimizes your kernels. Complementary, not competitive.

**Framing**: "Mojo asks you to rewrite your stack. openkernel makes your existing stack faster."

### vs CudaForge / Academic Systems

**Their positioning**: Research papers with benchmark numbers.

**Our differentiation**:
- **Product, not paper**: kernel code is a real developer tool with UX, not a research prototype.
- **Persistent learning**: Our skill library compounds across problems. Their systems start fresh every time.
- **Production path**: We integrate with HF Hub, Modal, inference engines. They're standalone scripts.

**Framing**: "CudaForge proved the approach works. openkernel makes it a product."

### vs Claude Code / Codex / Cursor (general coding tools)

**Their positioning**: "AI coding assistant for any programming task."

**Our differentiation**:
- **Kernel-specific**: Hardware profiling integration, roofline visualization, backend-specific optimization strategies. General tools can write kernel code but can't optimize it.
- **Benchmark-driven**: We publish KernelBench results. They don't even try.
- **The right feedback loop**: General tools have write→test→iterate. We have write→compile→benchmark→profile→diagnose→iterate. Fundamentally different.

**Framing**: "Claude Code writes kernels. kernel code optimizes them."

---

## The Five-Layer Cake in Sales Conversations

### For Layer 2 buyers (Custom Silicon)

> "Your $10B chip program is only as good as your kernel software. Without cuBLAS, your kernel team is writing everything from scratch. openkernel gives your 20 kernel engineers the productivity of 200 — and the skill library gets smarter with every kernel they optimize. We're not replacing your team; we're making them 10x more productive on your custom hardware."

### For Layer 3 buyers (Infrastructure)

> "Your customers choose you based on price/performance. A 20% kernel speedup on your GPU fleet means you can either cut prices 20% or pocket the margin. openkernel runs on Modal — no integration needed. Point it at your customers' most common workloads and watch the speedups compound."

### For Layer 4 buyers (Frontier Labs)

> "You're spending $X00M/year on GPU compute. Your kernel team of 15 engineers optimizes maybe 50 kernels per year at $500K/engineer. openkernel optimizes hundreds of kernels overnight for pennies each. The math is simple: $7.5M/year in kernel engineering talent vs $50K/year in openkernel compute. And the skill library means it gets better every month."

### For Layer 5 buyers (Enterprise)

> "You're deploying models for inference and the GPU bill is growing 3x year over year. You don't have kernel engineers and you don't need them. kernel code analyzes your model's performance bottlenecks and optimizes the kernels that matter most — automatically. Start with one model, see the speedup, then roll out across your fleet."

---

## Metrics That Matter

### Phase 1 (Credibility)
- KernelBench fast_1 score (L1, L2, L3)
- Cost per kernel (target: <$0.30, beating CudaForge)
- Time per kernel (target: <15 min average)
- GitHub stars / HF dataset downloads

### Phase 2 (Adoption)
- Monthly active optimization runs (the core metric)
- Unique developers per month
- Traces collected (opt-in)
- Community contributions (kernels, skills, bug reports)
- Net Promoter Score among kernel engineers

### Phase 3 (Revenue)
- ARR from kernel code premium + enterprise
- Number of enterprise contracts
- Average contract value
- Net revenue retention (expansion within accounts)
- Compute savings delivered to customers (our value metric)

### Phase 4 (Platform)
- kernelgen model quality (fast_1 improvement per version)
- Marketplace kernel count + download count
- Integration partner count (inference engines, training frameworks)
- Hardware platforms supported

---

## Launch Plan

### Pre-Launch (Build Phase)
- Build openkernel + kernel code in parallel
- Run KernelBench sweeps internally
- Iterate until results are competitive
- Write launch blog post with five-layer-cake framing

### Launch Day
- Open-source openkernel on GitHub (Apache 2.0)
- Ship kernel code on PyPI (`pip install kernel-code`)
- Publish KernelBench results on HF Hub + blog
- Submit to KernelBench leaderboard
- Post on Twitter/X, HN, Reddit, GPU Mode Discord
- Announce on ProductHunt

### Post-Launch (First 30 Days)
- Monitor adoption metrics
- Engage with GPU Mode community
- Publish tutorials and optimization case studies
- Identify bottom-up enterprise adoption signals
- Begin enterprise outreach to Layer 2/4 targets

### Post-Launch (First 90 Days)
- Ship kernelgen-1 if sufficient traces collected
- First enterprise design partnership signed
- Expand KernelBench coverage (L4, multi-hardware)
- Integration with one inference engine (SGLang or vLLM)

---

## Budget Allocation (Approximate)

| Category | % of Resources | Focus |
|----------|---------------|-------|
| Engineering (openkernel) | 40% | Core engine, 3-level hybrid, Modal eval |
| Engineering (kernel code) | 25% | TUI, dashboard, visualizations, trace capture |
| KernelBench + benchmarking | 15% | Sweeps, results, credibility proof |
| Community + content | 10% | Blog posts, tutorials, GPU Mode engagement |
| Enterprise sales prep | 10% | Sales materials, five-layer-cake deck, demo environments |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| KernelBench results aren't competitive | Start with simpler CudaForge-style loop first, prove it works, then add world model/strategy layers |
| Standard Kernel ships faster | Our differentiator is open-source + developer experience, not raw benchmark numbers |
| Modal costs too high for free tier | Negotiate volume pricing; offer CPU-only mode for local development |
| Trace collection insufficient for kernelgen-1 | Seed with KernelBook (18K pairs) + our own sweep traces; user traces are bonus |
| Market consolidates around NVIDIA tools | NVIDIA optimizes standard ops; we optimize composition/fusion — complementary, not competitive |
| Name confusion with RightNow AI's AutoKernel | "openkernel" is distinct; if needed, we can rebrand before launch |
