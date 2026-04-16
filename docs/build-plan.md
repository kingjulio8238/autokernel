# Build Plan: openkernel + kernel code

*April 2026*

---

## Build Phases

### Phase A — Eval Engine on Modal

**Goal**: Reliable kernel evaluation on cloud GPUs.

**Deliverable**: Modal app that takes kernel code + reference code → returns correctness, speedup, profile data.

**Work**:
- Modal app with GPU container (CUDA toolkit + Triton + KernelBench)
- Wrapper around `eval_kernel_against_ref()` with fast/thorough modes
- Profiling integration: Proton (Triton), torch.profiler (CUDA), analytical roofline
- Structured EvalResult and ProfileData output
- Test on 5 KernelBench L1 problems

**Depends on**: Nothing (start immediately)

### Phase B — Generator + Critic + Inner Loop

**Goal**: Working inner optimization loop that produces correct, faster kernels.

**Deliverable**: Given a KernelBench problem, produce an optimized kernel through iterative generation + evaluation + diagnosis.

**Work**:
- BYOM LLM interface (OpenAI-compatible API via litellm)
- Generator agent with backend-specific prompts (Triton + CUDA)
- Critic agent with structured CriticDiagnosis output
- Inner refinement loop: generate → eval → diagnose → retry
- Compile error recovery, correctness error recovery
- Backend-specific templates and strategies
- Recommended models list from internal testing

**Depends on**: Phase A (eval engine)

### Phase C — World Model Search (Middle Loop)

**Goal**: Structured search in strategy space, not just code space.

**Deliverable**: K-Search-style intent tree that guides optimization through structured reasoning.

**Work**:
- IntentNode data structure and tree management
- LLM operations: insert, update, prune
- Priority scoring and selection
- Non-monotonic path tolerance
- Stagnation detection (K consecutive failures → escalate)
- Integration with inner loop (each intent runs a refinement cycle)

**Depends on**: Phase B (inner loop)

### Phase D — Strategy Evolution + Memory (Outer Loop)

**Goal**: Cross-problem knowledge transfer and compounding intelligence.

**Deliverable**: Skill library that makes problem N+1 faster than problem N.

**Work**:
- Pareto frontier of strategies (GEPA-style)
- Skill library with trigger conditions, approaches, evidence
- Skill retrieval by problem similarity
- Trajectory memory (within-problem history)
- Strategy mutation/crossover via LLM reflection
- Persistence (JSON/SQLite for skills, Parquet for traces)

**Depends on**: Phase C (world model)

### Phase E — KernelBench Sweep + Results

**Goal**: Publishable KernelBench results proving openkernel is competitive.

**Deliverable**: fast_p scores across L1-L3, comparison tables vs baselines.

**Work**:
- Sweep orchestration: run openkernel across all problems in a level
- Analytical pre-screening: sort problems by headroom
- Budget allocation: more iterations for higher-headroom problems
- Results aggregation: fast_p at p={1.0, 1.5, 2.0}, geomean speedup
- Comparison vs CudaForge, Kernel-Smith, CUDA Agent, KernelSkill
- Cost accounting: tokens + Modal compute per kernel

**Depends on**: Phases A-D (full engine)

### Phase F — kernel code (Parallel with B-E)

**Goal**: Terminal-native product that wraps openkernel with kernel-specific visualizations.

**Deliverable**: Textual TUI + personalized Plotly Dash dashboards.

**Work**:
- Textual TUI with panels: chat/agent, trajectory, profiling, experiment log
- Plotly Dash dashboard served on localhost (7 panels)
- JSON cache file data flow (openkernel → cache → TUI/dashboard reads)
- Trace capture to Parquet for kernelgen-1
- Keyboard shortcuts, backend switching, model selection
- Integration with openkernel Python API

**Depends on**: Phase A (eval engine) for basic functionality. Layers on as B-D become available.

---

## Work Distribution (4 Engineers + Agents)

| Engineer | Primary Focus | Phases |
|----------|--------------|--------|
| **E1** | Eval engine + Modal infra + profiling | A, E (sweep infra) |
| **E2** | Search engine (3-level hybrid) | C, D |
| **E3** | Agent pair + LLM integration + memory | B, D (memory/traces) |
| **E4** | kernel code product + visualizations | F |

### Parallelism

All 4 engineers start immediately — see PLAN.md for the full parallel execution map and dependency graph. All engineers use coding agents (Claude Code) to parallelize within their focus area.

---

## Success Criteria

### openkernel
- [ ] Eval engine works reliably on Modal (>99% uptime, <30s per eval)
- [ ] Inner loop produces correct kernels on L1 problems (>90% correctness)
- [ ] World model search improves over pure hill-climbing (measurable on held-out problems)
- [ ] Skill library shows compounding returns (problem N+1 converges faster)
- [ ] KernelBench fast_1 competitive with published baselines
- [ ] Cost per kernel competitive with CudaForge ($0.30 baseline)

### kernel code
- [ ] TUI renders all panels correctly with real-time updates
- [ ] Dashboard accessible via link with all 7 visualization panels
- [ ] Trace capture stores full optimization traces in Parquet
- [ ] End-to-end: kernel engineer can optimize a kernel from `kernel-code optimize` command

### KernelBench results (for publication)
- [ ] L1: fast_1 score published with comparison table
- [ ] L2: fast_1 score published
- [ ] Cost and iteration efficiency metrics published
- [ ] Results reproducible by third parties
