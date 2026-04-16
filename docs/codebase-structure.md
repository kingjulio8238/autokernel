# Codebase Structure: openkernel + kernel code

*April 2026*

---

## Monorepo Layout

```
openkernel/
├── README.md
├── pyproject.toml                  # Workspace root — manages both packages
├── .python-version                 # 3.11+
├── uv.lock
├── .gitignore
│
├── docs/
│   ├── five-layer-cake.md          # Economic analysis of kernel optimization
│   ├── research-synthesis.md       # Full research synthesis (60+ systems surveyed)
│   ├── openkernel-design.md        # System design document
│   ├── kernel-code-design.md       # Product design document
│   ├── build-plan.md               # Build phases and work distribution
│   ├── data-and-integrations.md    # HF Hub, traces, storage, kernelgen pipeline
│   ├── visualization-design.md     # All visualization specs (TUI, dashboard, KernelBench)
│   └── codebase-structure.md       # This file
│
│
│ ═══════════════════════════════════════════════════════════
│  PACKAGE 1: openkernel (the engine)
│ ═══════════════════════════════════════════════════════════
│
├── openkernel/
│   ├── __init__.py                 # Public API: optimize(), sweep(), evaluate()
│   ├── py.typed                    # PEP 561 type marker
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # Main entry — orchestrates the 3-level hybrid loop
│   │   ├── inner_loop.py           # Refinement: generate → eval → diagnose → retry
│   │   ├── world_model.py          # K-Search-style intent tree + priority management
│   │   └── strategy_evolution.py   # GEPA-style Pareto frontier of strategies
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── generator.py            # Code generation agent (Triton + CUDA prompts)
│   │   ├── critic.py               # Profiling analysis + bottleneck diagnosis
│   │   └── prompts/
│   │       ├── __init__.py
│   │       ├── triton_generator.py # Backend-specific Triton generation prompts
│   │       ├── cuda_generator.py   # Backend-specific CUDA generation prompts
│   │       └── critic_prompts.py   # Critic analysis prompts
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── harness.py              # Wraps KernelBench eval_kernel_against_ref()
│   │   ├── modal_app.py            # Modal GPU function: compile + correctness + benchmark
│   │   ├── profiler.py             # Profiling orchestrator (dispatches to backend-specific)
│   │   ├── profilers/
│   │   │   ├── __init__.py
│   │   │   ├── proton_profiler.py  # Triton Proton integration
│   │   │   ├── torch_profiler.py   # torch.profiler + CUDA events integration
│   │   │   ├── ncu_profiler.py     # NCU integration (RunPod deep profiling tier)
│   │   │   └── roofline.py         # Analytical roofline (simple-torchroofline)
│   │   └── types.py                # EvalResult, ProfileData dataclasses
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── skill_library.py        # Long-term cross-problem optimization skills
│   │   ├── trajectory.py           # Short-term within-problem history
│   │   ├── pareto.py               # Pareto frontier management for strategies
│   │   └── store.py                # Persistence backend (JSON/SQLite)
│   │
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                 # Backend interface (abstract)
│   │   ├── triton_backend.py       # Triton-specific: code gen templates, autotuner integration
│   │   └── cuda_backend.py         # CUDA-specific: code gen templates, compilation, launch config
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py             # BYOM interface (OpenAI-compatible via litellm)
│   │   ├── models.py               # Recommended models registry + config
│   │   └── structured.py           # Structured output parsing (CriticDiagnosis, IntentNode, etc.)
│   │
│   ├── kernelbench/
│   │   ├── __init__.py
│   │   ├── problems.py             # Load KernelBench problems by level + ID
│   │   ├── sweep.py                # Sweep orchestration across a full level
│   │   ├── scoring.py              # fast_p computation, geomean speedup
│   │   └── compare.py              # Comparison tables vs published baselines
│   │
│   ├── traces/
│   │   ├── __init__.py
│   │   ├── capture.py              # Trace capture during optimization runs
│   │   ├── storage.py              # Parquet storage for traces
│   │   ├── export.py               # Export traces for training (filter, format, reward computation)
│   │   └── types.py                # OptimizationTrace, IterationTrace dataclasses
│   │
│   ├── hub/
│   │   ├── __init__.py
│   │   ├── client.py               # Hugging Face Hub integration (upload/download)
│   │   ├── models.py               # Download/load models from HF (kernelgen-1+)
│   │   ├── datasets.py             # Upload traces, download skill library, KernelBench results
│   │   └── kernels.py              # Upload/download optimized kernels per problem
│   │
│   └── config.py                   # Configuration: eval modes, model defaults, Modal config, HF config
│
│
│ ═══════════════════════════════════════════════════════════
│  PACKAGE 2: kernel-code (the product)
│ ═══════════════════════════════════════════════════════════
│
├── kernel_code/
│   ├── __init__.py
│   ├── py.typed
│   │
│   ├── cli.py                      # CLI entry point: kernel-code optimize, kernel-code dashboard
│   │
│   ├── tui/
│   │   ├── __init__.py
│   │   ├── app.py                  # Textual App — main TUI application
│   │   ├── panels/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py             # Chat/Agent panel (LLM conversation + progress)
│   │   │   ├── trajectory.py       # Optimization trajectory sparkline chart
│   │   │   ├── profiling.py        # Profiling summary panel (bottleneck, utilization gauges)
│   │   │   ├── experiment_log.py   # Scrollable experiment table with status colors
│   │   │   ├── kernel_diff.py      # Side-by-side kernel code diff
│   │   │   └── status_bar.py       # GPU, backend, model, iteration status
│   │   ├── widgets/
│   │   │   ├── __init__.py
│   │   │   ├── sparkline.py        # Terminal sparkline chart widget
│   │   │   ├── gauge.py            # Utilization gauge widget
│   │   │   └── colored_table.py    # Color-coded results table widget
│   │   └── keybindings.py          # Keyboard shortcut handlers
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── server.py               # Plotly Dash app served on localhost
│   │   ├── layouts/
│   │   │   ├── __init__.py
│   │   │   ├── trajectory.py       # Panel 1: Interactive speedup-over-time chart
│   │   │   ├── roofline.py         # Panel 2: Roofline model (log-log scatter)
│   │   │   ├── utilization.py      # Panel 3: Resource utilization gauges
│   │   │   ├── experiment_table.py # Panel 4: Filterable/sortable experiment table
│   │   │   ├── code_diff.py        # Panel 5: Syntax-highlighted code diff
│   │   │   ├── landscape.py        # Panel 6: 3D scatter optimization landscape (Constellation-style)
│   │   │   ├── strategy_tree.py    # Panel 7: World model intent tree visualization
│   │   │   ├── convergence.py      # Panel 8: Post-hoc convergence analysis
│   │   │   ├── cost_efficiency.py  # Panel 9: Post-hoc cost-performance frontier
│   │   │   └── strategy_stats.py   # Panel 10: Post-hoc strategy effectiveness
│   │   └── data.py                 # Dashboard data layer (reads JSON cache files)
│   │
│   ├── benchmarks/                  # KernelBench results visualization (static charts)
│   │   ├── __init__.py
│   │   ├── fast_p_chart.py         # Grouped bar: fast_p scores vs competitors
│   │   ├── speedup_distribution.py # Violin/box plot: speedup distribution
│   │   ├── scaling_curve.py        # Line chart: fast_p vs iteration budget
│   │   ├── cost_frontier.py        # Scatter: cost vs performance Pareto
│   │   ├── problem_heatmap.py      # Heatmap: problems x systems
│   │   ├── hardware_comparison.py  # Grouped bar: GPU types x backends
│   │   └── export.py               # Export charts as PNG/SVG for docs/website
│   │
│   └── integration/
│       ├── __init__.py
│       ├── openkernel_bridge.py    # Wraps openkernel API, streams results to TUI/dashboard
│       └── trace_bridge.py         # Connects trace capture to kernel code sessions
│
│
│ ═══════════════════════════════════════════════════════════
│  SHARED / INFRA
│ ═══════════════════════════════════════════════════════════
│
├── modal_infra/
│   ├── app.py                      # Modal app definition (GPU functions)
│   ├── Dockerfile                  # Custom container: CUDA toolkit + Triton + KernelBench + profilers
│   ├── deploy.py                   # Deploy script: modal deploy modal_infra/app.py
│   └── config.py                   # Modal-specific config (GPU types, timeouts, concurrency)
│
├── scripts/
│   ├── setup_problem.py            # Load a KernelBench problem for manual testing
│   ├── run_sweep.py                # CLI for running KernelBench sweeps
│   ├── publish_results.py          # Format results for publication
│   └── benchmark_models.py         # Benchmark different LLMs to update recommended models list
│
├── tests/
│   ├── test_eval/
│   │   ├── test_harness.py         # Eval harness unit tests
│   │   ├── test_modal_app.py       # Modal function integration tests
│   │   └── test_profilers.py       # Profiler integration tests
│   ├── test_engine/
│   │   ├── test_inner_loop.py
│   │   ├── test_world_model.py
│   │   └── test_strategy_evolution.py
│   ├── test_agents/
│   │   ├── test_generator.py
│   │   └── test_critic.py
│   ├── test_memory/
│   │   ├── test_skill_library.py
│   │   └── test_trajectory.py
│   └── test_kernel_code/
│       ├── test_tui.py
│       └── test_dashboard.py
│
├── data/
│   ├── skills/                     # Pre-seeded optimization skills (JSON)
│   │   ├── triton_elementwise.json
│   │   ├── triton_reduction.json
│   │   ├── triton_gemm.json
│   │   ├── cuda_gemm.json
│   │   └── fusion_patterns.json
│   ├── models/                     # Recommended models config
│   │   └── recommended.json        # {model_id, provider, strengths, cost_tier}
│   └── prompts/                    # Prompt templates (versioned)
│       ├── triton_generator_v1.md
│       ├── cuda_generator_v1.md
│       └── critic_v1.md
│
├── traces/                         # Optimization traces (gitignored, local + HF Hub)
│   ├── .gitkeep
│   ├── raw/                        # Raw traces per session (Parquet)
│   │   └── YYYY-MM/
│   │       └── session_<id>.parquet
│   ├── processed/                  # Filtered + formatted for training
│   │   ├── training_pairs_v1.parquet
│   │   ├── strategy_rewards_v1.parquet
│   │   └── critic_accuracy_v1.parquet
│   └── metadata/
│       ├── schema.json             # Trace schema version
│       └── stats.json              # Aggregate statistics
│
├── results/                        # KernelBench sweep results (gitignored)
│   ├── .gitkeep
│   ├── sweeps/                     # Raw sweep data (Parquet)
│   ├── comparisons/                # vs competitor results (JSON)
│   ├── charts/                     # Generated visualization exports (PNG/SVG)
│   └── README.md
│
└── cache/                          # Live optimization JSON cache (gitignored)
    ├── .gitkeep
    └── sessions/                   # One JSON file per active/completed session
        └── session_<id>.json       # Append-only, read by TUI + dashboard
```

---

## Package Configuration

### pyproject.toml (workspace root)

```toml
[project]
name = "openkernel-workspace"
version = "0.1.0"
description = "Autonomous GPU kernel optimization engine + developer tools"
requires-python = ">=3.11"

[tool.uv.workspace]
members = ["openkernel", "kernel_code"]
```

### openkernel/pyproject.toml

```toml
[project]
name = "openkernel"
version = "0.1.0"
description = "Self-recursive GPU kernel optimization engine"
requires-python = ">=3.11"
dependencies = [
    "kernelbench",
    "modal",
    "litellm",
    "pydantic>=2.0",
    "torch>=2.9",
    "triton",
    "simple-torchroofline",
    "huggingface-hub>=0.25",
    "pyarrow>=15.0",
    "datasets>=3.0",
]

[project.scripts]
openkernel = "openkernel.cli:main"
```

### kernel_code/pyproject.toml

```toml
[project]
name = "kernel-code"
version = "0.1.0"
description = "Terminal-native kernel optimization developer tool"
requires-python = ">=3.11"
dependencies = [
    "openkernel",
    "textual>=1.0",
    "plotly>=6.0",
    "dash>=3.0",
    "rich>=13.0",
    "pyarrow>=15.0",
    "pygments>=2.18",
]

[project.scripts]
kernel-code = "kernel_code.cli:main"
```

---

## Key Files by Phase

### Phase A (Eval Engine)
```
modal_infra/app.py
modal_infra/Dockerfile
openkernel/eval/harness.py
openkernel/eval/modal_app.py
openkernel/eval/types.py
openkernel/eval/profilers/proton_profiler.py
openkernel/eval/profilers/torch_profiler.py
openkernel/eval/profilers/roofline.py
```

### Phase B (Generator + Critic + Inner Loop)
```
openkernel/agents/generator.py
openkernel/agents/critic.py
openkernel/agents/prompts/*
openkernel/llm/provider.py
openkernel/llm/models.py
openkernel/llm/structured.py
openkernel/engine/inner_loop.py
openkernel/backends/triton_backend.py
openkernel/backends/cuda_backend.py
```

### Phase C (World Model)
```
openkernel/engine/world_model.py
openkernel/engine/orchestrator.py
```

### Phase D (Strategy Evolution + Memory)
```
openkernel/engine/strategy_evolution.py
openkernel/memory/skill_library.py
openkernel/memory/trajectory.py
openkernel/memory/pareto.py
openkernel/memory/store.py
openkernel/traces/capture.py
openkernel/traces/storage.py
data/skills/*.json
```

### Phase E (KernelBench Sweep)
```
openkernel/kernelbench/problems.py
openkernel/kernelbench/sweep.py
openkernel/kernelbench/scoring.py
openkernel/kernelbench/compare.py
scripts/run_sweep.py
scripts/publish_results.py
```

### Phase F (kernel code)
```
kernel_code/cli.py
kernel_code/tui/app.py
kernel_code/tui/panels/*
kernel_code/tui/widgets/*
kernel_code/dashboard/server.py
kernel_code/dashboard/layouts/*
kernel_code/integration/openkernel_bridge.py
kernel_code/integration/trace_bridge.py
```

---

## CLI Interface

### openkernel (library/engine)

```bash
# Optimize a single kernel
openkernel optimize --reference problem.py --backend triton --model claude-sonnet-4 --mode fast

# Optimize a KernelBench problem
openkernel optimize --level 1 --problem 23 --backend triton --iterations 50

# Run a sweep (internal use, for KernelBench results)
openkernel sweep --level 1 --iterations 100 --output results/l1_sweep.tsv
```

### kernel code (product)

```bash
# Interactive optimization (launches TUI)
kernel-code optimize --reference my_kernel.py --backend triton

# Open dashboard for a previous run
kernel-code dashboard --session <session-id>

# Configure model
kernel-code config --model claude-sonnet-4 --api-key <key>
```

---

## Data Flow

```
User via kernel code TUI
        │
        ▼
kernel_code/integration/openkernel_bridge.py
        │
        ▼
openkernel/engine/orchestrator.py  (3-level hybrid loop)
        │
        ├── openkernel/engine/strategy_evolution.py  (outer loop)
        ├── openkernel/engine/world_model.py         (middle loop)
        └── openkernel/engine/inner_loop.py          (inner loop)
                │
                ├── openkernel/agents/generator.py → LLM API (BYOM)
                ├── openkernel/eval/modal_app.py   → Modal GPU (compile + bench + profile)
                └── openkernel/agents/critic.py    → LLM API (BYOM)
                        │
                        ▼
                openkernel/memory/ (skills, trajectory, pareto)
                openkernel/traces/ (capture for kernelgen-1)
                        │
                        ▼
                JSON cache files → kernel code TUI + Dash dashboard
```
