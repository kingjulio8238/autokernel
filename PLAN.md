# PLAN.md — Build Execution Plan

## What We're Building

**openkernel**: Self-recursive GPU kernel optimization engine (Python library + CLI)
**kernel code**: Terminal-native developer tool wrapping openkernel (Textual TUI + Plotly Dash dashboards)

Both ship simultaneously. openkernel is the critical path — kernel code depends on it.

## Team

4 engineers (E1-E4) + coding agents. Each engineer owns a vertical and uses agent teams to parallelize within it.

---

## Dependency Graph

```
Phase A: Modal Eval Engine ──────────────────────────┐
    │                                                 │
    ▼                                                 ▼
Phase B: Generator + Critic + Inner Loop    Phase F: kernel code TUI + Dashboard
    │                                           │ (scaffolding, can start w/ mock data)
    ▼                                           │
Phase C: World Model Search                     │
    │                                           │
    ▼                                           │
Phase D: Strategy Evolution + Memory            │
    │                                           │
    ▼                                           ▼
Phase E: KernelBench Sweep + Results      Phase F: Integration (connects to live engine)
    │                                           │
    ▼                                           ▼
Phase G: HF Hub + Trace Pipeline          Phase H: Benchmark Visualizations
    │                                           │
    ▼                                           ▼
              LAUNCH (both products simultaneously)
```

**Critical path**: A → B → C → D → E → Launch
**Parallel path**: F runs alongside B-E, integrating as engine features land

---

## Engineer Assignments

| Engineer | Ownership | Primary Phases |
|----------|-----------|---------------|
| **E1** | Eval infrastructure + Modal + profiling + HF Hub | A, G |
| **E2** | Search engine (3-level hybrid loop) | C, D |
| **E3** | LLM layer + agents + memory persistence + traces | B, D (memory), G (traces) |
| **E4** | kernel code product (TUI + dashboard + benchmarks) | F, H |

---

## Phase A — Modal Eval Engine

**Owner**: E1
**Blocked by**: Nothing (start immediately)
**Blocks**: Phase B, Phase F integration

### Deliverables

- [ ] **A1**: Modal account setup + GPU access verified (H100, A100, L40S)
- [ ] **A2**: `modal_infra/Dockerfile` — custom container with CUDA toolkit, Triton, KernelBench, Proton, torch
- [ ] **A3**: `modal_infra/app.py` — Modal function: accepts kernel source + reference source → compiles, checks correctness, benchmarks
- [ ] **A4**: `openkernel/eval/types.py` — `EvalResult` and `ProfileData` dataclasses (shared contract with all other phases)
- [ ] **A5**: `openkernel/eval/harness.py` — wraps `eval_kernel_against_ref()`, supports `fast` (10 trials) and `thorough` (100 trials) modes
- [ ] **A6**: `openkernel/eval/profilers/proton_profiler.py` — Triton Proton integration (bottleneck classification, bandwidth/compute utilization)
- [ ] **A7**: `openkernel/eval/profilers/torch_profiler.py` — torch.profiler + CUDA events for CUDA kernels
- [ ] **A8**: `openkernel/eval/profilers/roofline.py` — analytical roofline via simple-torchroofline (no GPU needed)
- [ ] **A9**: `openkernel/eval/profiler.py` — orchestrator: dispatches to correct profiler based on backend (Triton→Proton, CUDA→torch.profiler)
- [ ] **A10**: Integration test — evaluate 5 KernelBench L1 problems end-to-end on Modal, verify correctness and timing match local results

### Acceptance Criteria
- `EvalResult` returned for all 5 test problems
- Correctness matches KernelBench reference within tolerance
- Speedup values are reproducible (±5% across runs)
- Fast mode completes in <10s, thorough in <30s (excluding cold start)
- ProfileData includes bottleneck_type and at least 4 utilization metrics

### Agent Team Strategy (E1)
Spawn 3 agents in parallel:
- Agent 1: Dockerfile + Modal app (A2, A3)
- Agent 2: EvalResult types + harness wrapper (A4, A5)
- Agent 3: Profiler integrations (A6, A7, A8, A9)
Then integration test (A10) after all converge.

---

## Phase B — Generator + Critic + Inner Loop

**Owner**: E3
**Blocked by**: Phase A (needs eval engine to test)
**Blocks**: Phase C

### Deliverables

- [ ] **B1**: `openkernel/llm/provider.py` — BYOM interface via litellm (supports Claude, GPT, Gemini, local models)
- [ ] **B2**: `openkernel/llm/models.py` — recommended models registry (JSON config, loaded at runtime)
- [ ] **B3**: `openkernel/llm/structured.py` — structured output parsing for CriticDiagnosis, kernel code extraction
- [ ] **B4**: `openkernel/backends/base.py` — abstract backend interface
- [ ] **B5**: `openkernel/backends/triton_backend.py` — Triton code gen templates, `@triton.autotune` config proposal, compilation helpers
- [ ] **B6**: `openkernel/backends/cuda_backend.py` — CUDA code gen templates, nvcc compilation, launch config helpers
- [ ] **B7**: `openkernel/agents/prompts/triton_generator.py` — Triton-specific generation prompts (system prompt, few-shot examples, profile-guided refinement prompt)
- [ ] **B8**: `openkernel/agents/prompts/cuda_generator.py` — CUDA-specific generation prompts
- [ ] **B9**: `openkernel/agents/prompts/critic_prompts.py` — Critic analysis prompts (bottleneck diagnosis, recommendation generation)
- [ ] **B10**: `openkernel/agents/generator.py` — Generator agent: takes (reference, intent, critic_feedback, skills) → kernel code
- [ ] **B11**: `openkernel/agents/critic.py` — Critic agent: takes (kernel, EvalResult, ProfileData) → CriticDiagnosis
- [ ] **B12**: `openkernel/engine/inner_loop.py` — refinement loop: generate → eval → diagnose → retry (up to K attempts per intent)
- [ ] **B13**: `openkernel/config.py` — configuration: eval modes, model defaults, backend selection, retry limits
- [ ] **B14**: End-to-end test — run inner loop on 5 KernelBench L1 problems, verify it produces correct kernels with speedup >1.0x on at least 3/5

### Acceptance Criteria
- Inner loop produces correct kernel on >80% of attempts (after retries)
- At least 3/5 test problems achieve speedup >1.0x
- Compile errors are recovered from (error fed back to Generator, retry succeeds)
- Critic produces structured CriticDiagnosis with bottleneck_type and recommendation
- Works with at least 2 different LLM providers (test Claude + GPT)

### Agent Team Strategy (E3)
Phase B has two independent tracks that can run in parallel:
- Track 1 (B1-B3): LLM interface layer (can build and test with mock eval)
- Track 2 (B4-B9): Backends + prompts (can write and test prompts without eval engine)
- Then converge on B10-B12 (agents + inner loop) once Phase A delivers and both tracks complete

### E3 Pre-Work (Start During Phase A)
B1-B9 don't need the eval engine. E3 should start these immediately:
- Build LLM provider interface (B1-B3)
- Write backend abstractions and prompt templates (B4-B9)
- Test prompts in isolation (generate kernel code, check it looks reasonable)
- When Phase A delivers, wire up B10-B12 and run end-to-end

---

## Phase C — World Model Search

**Owner**: E2
**Blocked by**: Phase B (needs inner loop)
**Blocks**: Phase D

### Deliverables

- [ ] **C1**: `openkernel/engine/world_model.py` — IntentNode dataclass, tree structure, priority management
- [ ] **C2**: LLM tree operations — insert (propose new intents), update (re-estimate priorities), prune (remove dead ends)
- [ ] **C3**: Node selection — pick highest-priority unvisited node, handle stagnation detection (K failures → escalate)
- [ ] **C4**: Non-monotonic path support — allow temporary regression if world model predicts long-term payoff
- [ ] **C5**: Integration with inner loop — each selected intent triggers a refinement cycle (Phase B's inner_loop)
- [ ] **C6**: `openkernel/engine/orchestrator.py` — orchestrates world model + inner loop together
- [ ] **C7**: A/B test — compare world model search vs pure hill-climbing (inner loop only) on 10 KernelBench L1 problems. World model should win on average speedup.

### Acceptance Criteria
- World model produces a structured intent tree for each problem
- LLM can insert, update, and prune nodes based on results
- Stagnation detection fires after K consecutive failures
- World model search achieves higher average speedup than pure inner loop on test set
- Intent tree is inspectable (can be serialized to JSON for visualization)

### E2 Pre-Work (Start During Phase A-B)
The world model is pure Python logic + LLM prompting — no GPU dependency. E2 can:
- Design and implement IntentNode/tree data structures (C1)
- Write LLM tree operation prompts and test with mock eval results (C2-C4)
- Build the orchestrator shell (C6) that will plug in the real inner loop once Phase B delivers

---

## Phase D — Strategy Evolution + Memory

**Owner**: E2 (search) + E3 (memory/traces)
**Blocked by**: Phase C
**Blocks**: Phase E (need full engine for sweeps)

### Deliverables (E2: Strategy Evolution)

- [ ] **D1**: `openkernel/engine/strategy_evolution.py` — GEPA-style Pareto frontier of strategies
- [ ] **D2**: Strategy dataclass — description, problem_types, backend, success/failure history, Pareto scores
- [ ] **D3**: Strategy mutation/crossover — LLM reflects on results, proposes new strategies, evolves frontier
- [ ] **D4**: Strategy retrieval — given a new problem, retrieve relevant strategies from frontier to seed intent tree
- [ ] **D5**: Integration with orchestrator — outer loop wraps world model, seeds it with strategies per problem

### Deliverables (E3: Memory + Traces)

- [ ] **D6**: `openkernel/memory/skill_library.py` — OptimizationSkill dataclass, CRUD operations, similarity-based retrieval
- [ ] **D7**: `openkernel/memory/trajectory.py` — within-problem history tracking, prevents oscillation
- [ ] **D8**: `openkernel/memory/pareto.py` — Pareto frontier management (dominance checking, pruning)
- [ ] **D9**: `openkernel/memory/store.py` — persistence: JSON for skills, SQLite for trajectory/metadata
- [ ] **D10**: `openkernel/traces/types.py` — OptimizationTrace, IterationTrace dataclasses
- [ ] **D11**: `openkernel/traces/capture.py` — captures full trace during optimization (every prompt, response, eval result)
- [ ] **D12**: `openkernel/traces/storage.py` — writes traces to Parquet files (monthly partitioned)
- [ ] **D13**: `data/skills/*.json` — pre-seeded optimization skills (5-10 skills for common kernel patterns)

### Acceptance Criteria
- Skill library retrieval returns relevant skills for a given problem
- Strategy evolution produces new strategies after reflecting on results
- Trajectory memory prevents the engine from retrying failed approaches
- Traces capture full optimization history in Parquet format
- A/B test: engine with memory converges faster on problem N+1 after solving problems 1..N

---

## Phase E — KernelBench Sweep + Results

**Owner**: E1 (sweep infra) + E2 (analysis)
**Blocked by**: Phase D (full engine)
**Blocks**: Launch (need results for credibility)

### Deliverables

- [ ] **E1**: `openkernel/kernelbench/problems.py` — load any problem by level + ID
- [ ] **E2**: `openkernel/kernelbench/sweep.py` — sweep orchestration: pre-screen with roofline, sort by headroom, allocate budget, run
- [ ] **E3**: `openkernel/kernelbench/scoring.py` — fast_p computation at p={1.0, 1.5, 2.0}, geomean speedup, correctness rate
- [ ] **E4**: `openkernel/kernelbench/compare.py` — comparison tables vs CudaForge, Kernel-Smith, CUDA Agent, KernelSkill baselines
- [ ] **E5**: `scripts/run_sweep.py` — CLI to run sweeps
- [ ] **E6**: `scripts/publish_results.py` — format results for HF Hub upload + blog post
- [ ] **E7**: Run L1 sweep (100 problems) — publish fast_1 score
- [ ] **E8**: Run L2 sweep (100 problems) — publish fast_1 score
- [ ] **E9**: Cost + efficiency analysis — cost per kernel, iterations to convergence, comparison vs baselines

### Acceptance Criteria
- L1 fast_1 score is competitive with published baselines
- Cost per kernel < $0.50 (stretch: < $0.30)
- Results are reproducible (re-run produces same scores ±5%)
- Comparison tables generated automatically
- Results uploaded to HF Hub dataset

---

## Phase F — kernel code Product

**Owner**: E4
**Blocked by**: Phase A for live data (but scaffolding can start immediately with mock data)
**Blocks**: Nothing (integrates progressively)

### Deliverables

#### F-Stage 1: Scaffolding (start immediately, use mock data)

- [ ] **F1**: `kernel_code/cli.py` — CLI entry point (`kernel-code optimize`, `kernel-code dashboard`, `kernel-code config`)
- [ ] **F2**: `kernel_code/tui/app.py` — Textual App shell with panel layout
- [ ] **F3**: `kernel_code/tui/panels/status_bar.py` — GPU, backend, model, iteration, cost display
- [ ] **F4**: `kernel_code/tui/panels/experiment_log.py` — scrollable colored results table
- [ ] **F5**: `kernel_code/tui/widgets/sparkline.py` — terminal sparkline chart widget
- [ ] **F6**: `kernel_code/tui/widgets/gauge.py` — horizontal utilization gauge widget
- [ ] **F7**: `kernel_code/tui/widgets/colored_table.py` — color-coded table widget
- [ ] **F8**: Mock data generator — produces realistic JSON cache files for testing TUI without live engine

#### F-Stage 2: Dashboard (parallel with F-Stage 1)

- [ ] **F9**: `kernel_code/dashboard/server.py` — Dash app on localhost, launched via `d` key from TUI
- [ ] **F10**: `kernel_code/dashboard/data.py` — data layer reading JSON cache files
- [ ] **F11**: `kernel_code/dashboard/layouts/trajectory.py` — Panel 1: speedup over time
- [ ] **F12**: `kernel_code/dashboard/layouts/roofline.py` — Panel 2: roofline model
- [ ] **F13**: `kernel_code/dashboard/layouts/utilization.py` — Panel 3: resource gauges
- [ ] **F14**: `kernel_code/dashboard/layouts/experiment_table.py` — Panel 4: filterable table
- [ ] **F15**: `kernel_code/dashboard/layouts/code_diff.py` — Panel 5: syntax-highlighted diff
- [ ] **F16**: `kernel_code/dashboard/layouts/landscape.py` — Panel 6: 3D scatter (Constellation-style)
- [ ] **F17**: `kernel_code/dashboard/layouts/strategy_tree.py` — Panel 7: world model intent tree

#### F-Stage 3: Live Integration (after Phase A delivers)

- [ ] **F18**: `kernel_code/tui/panels/chat.py` — chat/agent panel with LLM streaming
- [ ] **F19**: `kernel_code/tui/panels/trajectory.py` — live optimization trajectory
- [ ] **F20**: `kernel_code/tui/panels/profiling.py` — live profiling summary
- [ ] **F21**: `kernel_code/tui/keybindings.py` — keyboard shortcuts (d, k, r, s, p, b, q)
- [ ] **F22**: `kernel_code/integration/openkernel_bridge.py` — wraps `openkernel.optimize()`, streams results to TUI + dashboard
- [ ] **F23**: `kernel_code/integration/trace_bridge.py` — connects trace capture to kernel code sessions

#### F-Stage 4: Post-Hoc + Benchmark Charts (after Phase E delivers results)

- [ ] **F24**: `kernel_code/dashboard/layouts/convergence.py` — Panel 8
- [ ] **F25**: `kernel_code/dashboard/layouts/cost_efficiency.py` — Panel 9
- [ ] **F26**: `kernel_code/dashboard/layouts/strategy_stats.py` — Panel 10
- [ ] **F27**: `kernel_code/benchmarks/*.py` — 6 KernelBench result chart generators
- [ ] **F28**: `kernel_code/benchmarks/export.py` — PNG/SVG export

### Acceptance Criteria
- TUI renders all panels with real-time updates from live engine
- Dashboard opens in browser from TUI with all panels functional
- Constellation-style 3D landscape with configurable axes works
- Trace capture stores full Parquet traces
- End-to-end: `kernel-code optimize --level 1 --problem 23` works from cold start to optimized kernel

---

## Phase G — HF Hub + Trace Pipeline

**Owner**: E1 (hub) + E3 (trace export)
**Blocked by**: Phase D (needs traces to upload)
**Blocks**: kernelgen-1 (future)

### Deliverables

- [ ] **G1**: `openkernel/hub/client.py` — HF Hub upload/download orchestrator
- [ ] **G2**: `openkernel/hub/datasets.py` — upload traces, download skill library, upload KernelBench results
- [ ] **G3**: `openkernel/hub/kernels.py` — upload/download optimized kernels per problem
- [ ] **G4**: `openkernel/hub/models.py` — download kernelgen-1+ from HF (future-proofing)
- [ ] **G5**: `openkernel/traces/export.py` — filter best traces, compute rewards, format as training pairs
- [ ] **G6**: HF Hub repos created: `openkernel/optimization-traces`, `openkernel/kernelbench-results`, `openkernel/optimized-kernels`, `openkernel/skill-library`

---

## Parallel Execution Map

```
         E1 (Infra)          E2 (Search)           E3 (Agents)          E4 (Product)
         ──────────          ───────────           ───────────          ────────────
Start    A1-A3: Modal        C1: IntentNode        B1-B3: LLM          F1-F8: TUI
         setup + Docker      data structures       provider             scaffolding
         + app               (no GPU needed)       (no GPU needed)      + mock data
         │                   │                     │                    │
         A4-A9: Profilers    C2-C4: LLM tree       B4-B9: Backends     F9-F17: Dash
         + harness           ops + prompts         + prompts            panels
         │                   (mock eval)           (mock eval)          │
         │                   │                     │                    │
         A10: Integration    │                     │                    │
         test ◄──────────────┤─────────────────────┤                    │
         ═══ PHASE A DONE ═══                      │                    │
         │                   │                     ▼                    │
         │                   │                     B10-B12: Agents      │
         │                   │                     + inner loop         │
         │                   │                     │                    │
         │                   │                     B13-B14: E2E test    │
         │                   │                     ═══ PHASE B DONE ═══ │
         │                   ▼                     │                    ▼
         │                   C5-C6: Wire to        │                    F18-F23: Live
         │                   real inner loop       │                    integration
         │                   │                     │                    │
         │                   C7: A/B test          │                    │
         │                   ═══ PHASE C DONE ═══  │                    │
         │                   │                     ▼                    │
         │                   D1-D5: Strategy       D6-D13: Memory       │
         │                   evolution             + traces + skills    │
         │                   ═══ PHASE D DONE ════════════════════════  │
         ▼                   ▼                     ▼                    │
         E1-E6: Sweep        E7-E9: Run            G1-G6: HF Hub       │
         infra               sweeps + analyze      + trace pipeline     │
         ═══ PHASE E DONE ════════════════════════════════════════════  │
                                                                        ▼
                                                                        F24-F28: Post-hoc
                                                                        panels + bench charts
                                                                        ═══ PHASE F DONE ═══
                                                                        │
                                                                        ▼
                                                                     LAUNCH
```

### Key Parallelism Opportunities

1. **E1 + E3 + E4 all start day 1** — Modal setup, LLM interface, and TUI scaffolding have zero dependencies on each other
2. **E2 starts day 1 too** — world model data structures and LLM prompts can be designed and tested with mock eval data before Phase A delivers
3. **Phase F runs the entire build** — E4 is never blocked (uses mock data initially, integrates live data as it becomes available)
4. **Phase B pre-work** — E3 builds the full LLM + backends + prompts layer before Phase A finishes. Only the agent wiring + inner loop (B10-B12) needs the eval engine
5. **Phase D splits cleanly** — E2 owns strategy evolution, E3 owns memory/traces. No conflict.

### Integration Points (Where Pieces Connect)

| Integration | When | What Connects | Who Coordinates |
|------------|------|---------------|-----------------|
| **Eval → Inner Loop** | After Phase A | `EvalResult` flows from eval engine to inner loop | E1 + E3 test together |
| **Inner Loop → World Model** | After Phase B | Refinement results feed into intent tree updates | E2 + E3 test together |
| **World Model → Strategy Evo** | After Phase C | World model results inform strategy Pareto updates | E2 handles both |
| **Engine → kernel code** | After Phase A+ | `openkernel.optimize()` streams results to TUI via bridge | E3 + E4 test together |
| **Engine → Traces** | After Phase D | Full optimization traces captured in Parquet | E3 handles both |
| **Traces → HF Hub** | After Phase G | Parquet traces uploaded to HF dataset | E1 + E3 test together |
| **Sweep → Benchmarks** | After Phase E | Sweep results feed benchmark chart generators | E1 + E4 test together |

---

## Testing Strategy

### Unit Tests (each engineer owns their modules)
- E1: `tests/test_eval/` — harness, Modal app, profilers
- E2: `tests/test_engine/` — inner loop, world model, strategy evolution
- E3: `tests/test_agents/` + `tests/test_memory/` — generator, critic, skill library, trajectory
- E4: `tests/test_kernel_code/` — TUI, dashboard

### Integration Tests (at integration points)
- **Eval integration**: run real kernel through Modal → verify EvalResult
- **Engine integration**: run full 3-level loop on 1 KernelBench problem → verify optimization trace
- **Product integration**: `kernel-code optimize` end-to-end → verify TUI updates + trace captured

### Benchmark Tests (Phase E)
- L1 sweep: 100 problems, measure fast_p
- L2 sweep: 100 problems, measure fast_p
- Cost accounting: tokens + Modal compute per kernel
- Comparison vs published baselines

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Modal doesn't support Proton/NCU profiling | Medium | High | Fall back to torch.profiler + CUDA events. Test profiling on Modal ASAP (Phase A). |
| World model doesn't outperform pure hill-climbing | Low | Medium | We still have a competitive inner loop. World model is additive, not load-bearing. |
| KernelBench results aren't competitive | Medium | High | Ship inner loop first (CudaForge-competitive), layer world model + strategy evo for improvement. |
| Modal cold starts are too slow | Low | Medium | Use `modal.Cls()` with keep-warm, or switch to persistent containers. |
| LLM costs too high per kernel | Medium | Medium | Tiered routing: cheap model for classification + easy problems, frontier for hard. |
| Textual TUI limitations for complex widgets | Low | Low | Fall back to rich library for rendering, or custom widgets. |

---

## Definition of Done (Launch)

### Must Have
- [ ] `openkernel optimize` works end-to-end on any KernelBench problem (both Triton and CUDA backends)
- [ ] `kernel-code optimize` launches TUI with live panels + dashboard link
- [ ] 3-level hybrid loop (inner + world model + strategy evolution) operational
- [ ] Skill library persists across sessions and improves convergence
- [ ] Traces captured in Parquet, uploadable to HF Hub
- [ ] KernelBench L1 fast_1 score published with comparison table
- [ ] Cost per kernel documented and competitive
- [ ] README.md with installation, quickstart, and 5-minute tutorial

### Should Have
- [ ] KernelBench L2 fast_1 score published
- [ ] All 10 dashboard panels functional
- [ ] Constellation-style 3D landscape with configurable axes
- [ ] Pre-seeded skill library with 10+ optimization patterns
- [ ] Benchmark charts exported as PNG/SVG for blog post

### Nice to Have
- [ ] KernelBench L3 results
- [ ] Multi-hardware results (H100 + A100 + L40S)
- [ ] kernelgen-1 training pipeline (even if model not yet trained)
- [ ] GPU Mode community integration
