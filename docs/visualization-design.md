# Visualization Design: openkernel + kernel code + KernelBench

*April 2026*

---

## Three Visualization Contexts

openkernel needs visualizations for three distinct contexts:

1. **Live optimization** — real-time feedback during a kernel optimization run (kernel code TUI + dashboard)
2. **Post-hoc analysis** — reviewing completed optimization runs, comparing approaches
3. **KernelBench results** — benchmark leaderboard, comparison charts, credibility proof

Each context has different requirements for interactivity, update frequency, and data density.

---

## Design Principles (from PufferLib Constellation)

Constellation's architecture provides key lessons:

| Constellation Pattern | Our Application |
|----------------------|-----------------|
| **Static memory, 100K points at 60fps** | Pre-allocate visualization buffers; don't reallocate per update |
| **1,200 lines of C, single-file** | Keep visualization code lean. Dash layouts = one file per panel |
| **Configurable axes (x, y, z, color)** | Let users map ANY metric to ANY axis in the experiment landscape view |
| **Linear/log/logit scales** | Essential for speedup data (log scale reveals small improvements) |
| **Two filters on any key** | Filter experiments by backend, model, problem type, speedup range |
| **Right-click tooltips** | Hover on any data point → see full experiment details |
| **Shared settings across figures** | Filter in one panel → all panels update (linked views) |
| **JSON cache files for data flow** | openkernel writes JSON → TUI/dashboard reads. Decoupled, simple |
| **zip + scp for remote data** | Download traces from Modal/cloud → local visualization. Offline-first |

---

## Context 1: Live Optimization (kernel code)

### Terminal (Textual TUI) — Always Visible

Optimized for 80-column terminal. Refreshes every iteration (~5-30s).

```
┌─ Chat ──────────────┬─ Trajectory ─────────────────────────┐
│ ▸ Analyzing L1#23   │ 2.0x ┤                          ●    │
│   softmax, 4096     │      │                    ●          │
│                     │ 1.5x ┤              ●                │
│ ▸ Intent: online    │      │         ×                     │
│   reduction algo    │ 1.0x ┤───●─×───────────────────────  │
│                     │      │ ×                              │
│ ▸ Generated kernel  │ 0.5x ┤                               │
│   with single-pass  │      └─┬──┬──┬──┬──┬──┬──┬──┬──┬──  │
│   max+exp+sum       │        1  2  3  4  5  6  7  8  9    │
│                     │  ● keep  × discard  ○ error          │
│ ✓ Correct, 1.8x    ├──────────────────────────────────────┤
│                     │ Profiling                             │
│ Critic: "Bandwidth  │ ▓▓▓▓▓▓▓▓░░ BW   72% peak           │
│ at 72% peak. Try    │ ▓▓▓▓░░░░░░ Comp  41% peak           │
│ float4 vectorized   │ ▓▓▓▓▓▓▓░░░ L2    68% hit            │
│ loads to improve    │ ▓▓▓▓▓▓░░░░ Occ   0.62               │
│ bandwidth util."    │ Bottleneck: MEMORY BOUND              │
│                     │ Headroom: ~1.4x remaining             │
├─────────────────────┼──────────────────────────────────────┤
│ Experiment Log      │ Skills Matched                        │
│ #1  1.0x  keep  bas │ ▸ online_softmax_reduction           │
│ #2  0.7x  disc  bad │   "single-pass max+exp+sum"          │
│ #3  1.3x  keep  sha │ ▸ vectorized_elementwise             │
│ #4  1.3x  disc  no  │   "float4 loads for bandwidth"       │
│ #5  1.8x  keep  onl │                                      │
│ #6  ...   ···   run │                                      │
├─────────────────────┴──────────────────────────────────────┤
│ H100 │ Triton │ claude-sonnet-4 │ L1#23 │ Iter 6 │ $0.12  │
│ [d]ashboard [k]diff [r]oofline [s]kills [p]ause [q]uit    │
└────────────────────────────────────────────────────────────┘
```

**TUI Panels:**

| Panel | Widget | Update Frequency |
|-------|--------|-----------------|
| Chat/Agent | Rich text with streaming | Real-time (LLM streaming) |
| Trajectory | Sparkline with colored markers | Per iteration |
| Profiling | Horizontal bar gauges | Per iteration |
| Experiment Log | Colored table (scrollable) | Per iteration |
| Skills Matched | Collapsible list | Per problem |
| Status Bar | Text labels | Per iteration |

### Browser Dashboard (Plotly Dash) — On-Demand via Link

Launched with `d` key. Full interactive visualization suite. Served on `localhost:8050` with unique session URL.

**7 Dashboard Panels:**

#### Panel 1: Optimization Trajectory (Primary)
- **Type**: Interactive line chart (Plotly `go.Scatter`)
- **X-axis**: Iteration number (secondary: wall-clock time)
- **Y-axis**: Speedup over reference
- **Data points**: Each iteration, color-coded:
  - Green filled circle = keep (new best)
  - Red X = discard (slower)
  - Yellow triangle = error/incorrect
  - Blue diamond = baseline
- **Overlay**: Running best as bold step line
- **Baseline**: Dashed line at 1.0x (PyTorch reference)
- **Hover**: Full details (intent, speedup, profiler summary, kernel snippet)
- **Inspiration**: PufferLib Constellation Fig 2 + our analysis.ipynb

#### Panel 2: Roofline Model
- **Type**: Log-log scatter plot (Plotly `go.Scatter`)
- **X-axis**: Arithmetic intensity (FLOP/byte)
- **Y-axis**: Performance (GFLOP/s or GB/s)
- **Ceiling lines**: Peak compute throughput, peak memory bandwidth
- **Ridge point**: Where ceilings intersect (labeled)
- **Data points**: Each kernel variant plotted by measured intensity + performance
- **Arrow**: From current kernel to theoretical optimal position
- **Inspiration**: NVIDIA Nsight Compute roofline view

#### Panel 3: Resource Utilization
- **Type**: Horizontal bar charts / gauge meters (Plotly `go.Indicator`)
- **Metrics**: Occupancy, register usage, shared memory, L1/L2 cache hit rates, bandwidth utilization, compute utilization
- **Color**: Green (>80%), yellow (50-80%), red (<50%)
- **Comparison**: Current best vs previous iteration (delta arrows)
- **Inspiration**: Nsight Compute "Speed of Light" section

#### Panel 4: Experiment Table
- **Type**: DataTable (Dash `dash_table.DataTable`)
- **Columns**: Iteration, speedup, status, backend, intent, latency, cost
- **Features**: Sortable, filterable, searchable, color-coded rows
- **Click**: Selecting a row updates all other panels to show that iteration's data
- **Inspiration**: Enhanced results.tsv

#### Panel 5: Code Diff
- **Type**: Side-by-side syntax-highlighted diff (custom HTML/CSS component)
- **Left**: Previous best kernel
- **Right**: Current iteration kernel
- **Highlighting**: Green = added, red = removed, yellow = changed
- **Performance annotations**: Lines contributing most to speedup/regression marked
- **Inspiration**: GitHub code review

#### Panel 6: Optimization Landscape (Constellation-Inspired)
- **Type**: 3D scatter plot (Plotly `go.Scatter3d`)
- **Axes**: User-configurable — map any metric to X, Y, Z, color
  - Suggested defaults: X=iteration, Y=speedup, Z=cost, color=status
- **Controls**: Dropdown selectors for each axis (Constellation pattern)
- **Scale**: Linear/log toggle (Constellation pattern)
- **Filters**: Range sliders on any metric (Constellation pattern)
- **Hover**: Full experiment details on right-click/hover
- **Purpose**: Discover patterns — do certain backends cluster? Do expensive iterations correlate with breakthroughs?
- **Inspiration**: PufferLib Constellation Fig 1 (3D scatter with configurable axes)

#### Panel 7: Strategy Tree (World Model Visualization)
- **Type**: Tree diagram (Plotly `go.Treemap` or custom D3)
- **Nodes**: Optimization intents from the world model search tree
- **Color**: By status (green=succeeded, red=pruned, gray=pending, blue=active)
- **Size**: By speedup achieved (larger = more successful)
- **Hover**: Intent description, attempts, best speedup, critic diagnosis
- **Purpose**: Understand the search strategy — what was tried, what worked

---

## Context 2: Post-Hoc Analysis

For reviewing completed optimization runs. Accessed via:
```bash
kernel-code dashboard --session <session-id>
```

Same 7 panels as live dashboard, but with complete data. Additional post-hoc panels:

#### Panel 8: Convergence Analysis
- **Type**: Line chart
- **X-axis**: Iteration
- **Y-axis**: Cumulative best speedup + iteration delta
- **Purpose**: Where did diminishing returns kick in? At what iteration was 90% of final speedup achieved?

#### Panel 9: Cost Efficiency
- **Type**: Scatter plot
- **X-axis**: Cumulative cost (USD)
- **Y-axis**: Speedup achieved
- **Purpose**: Cost-performance frontier. How much did each unit of speedup cost?

#### Panel 10: Strategy Effectiveness
- **Type**: Bar chart grouped by strategy
- **Bars**: Average speedup per strategy, colored by problem type
- **Purpose**: Which strategies work best for which problem types? Informs skill library curation.

---

## Context 3: KernelBench Results (Public Credibility)

Visualizations for published benchmark results. These go on our website/README/HF dataset page.

### Chart 1: fast_p Scores vs Competitors
- **Type**: Grouped bar chart
- **X-axis**: System (openkernel, CudaForge, Kernel-Smith, CUDA Agent, KernelSkill, ...)
- **Y-axis**: fast_p score
- **Groups**: fast_1.0, fast_1.5, fast_2.0
- **Facets**: L1, L2, L3
- **Purpose**: THE headline chart. "openkernel achieves X% fast_1 on L1, competitive with Y"

### Chart 2: Speedup Distribution
- **Type**: Violin plot or box plot
- **X-axis**: System
- **Y-axis**: Speedup (log scale)
- **Purpose**: Show distribution, not just mean. openkernel might have higher median even if some systems have higher max.

### Chart 3: Scaling Curve (fast_p vs Iterations)
- **Type**: Line chart
- **X-axis**: Iteration budget {5, 10, 20, 50, 100}
- **Y-axis**: fast_1 score
- **Lines**: One per system (openkernel vs others at each budget)
- **Purpose**: Show that more iterations = better results, and openkernel scales better

### Chart 4: Cost-Performance Frontier
- **Type**: Scatter plot
- **X-axis**: Cost per kernel (USD)
- **Y-axis**: Average speedup
- **Points**: Each system
- **Purpose**: openkernel should be Pareto-optimal (better AND cheaper)

### Chart 5: Per-Problem Heatmap
- **Type**: Heatmap (problems x systems)
- **Color**: Speedup achieved (green gradient)
- **Purpose**: Show which problems each system solves well. openkernel should have broad coverage.
- **Inspiration**: KernelBench "Kernelsseum" leaderboard but more visual

### Chart 6: Hardware Comparison
- **Type**: Grouped bar chart
- **X-axis**: GPU type (H100, A100, L40S)
- **Y-axis**: Average speedup
- **Groups**: By backend (Triton, CUDA)
- **Purpose**: Show openkernel works across hardware

### Implementation
Static charts generated with Plotly + exported as PNG/SVG for docs. Interactive versions hosted as Dash app or HF Space.

---

## Data Flow Architecture

```
openkernel engine
    │
    ├── writes → JSON cache files (append-only, one per session)
    │            Format: {iteration, speedup, status, profile, kernel_code, ...}
    │
    ├── writes → Parquet traces (for kernelgen-1 training)
    │            Stored locally, uploaded to HF Hub periodically
    │
    └── writes → results TSV (for KernelBench tracking)

kernel code TUI
    │
    ├── reads ← JSON cache files (polls every 1s)
    │            Renders: trajectory sparkline, profiling gauges, experiment log
    │
    └── serves → Dash dashboard on localhost:8050
                 Reads same JSON cache files
                 Full interactive panels (7 live + 3 post-hoc)

KernelBench results
    │
    ├── reads ← Parquet results from HF Hub
    └── renders → Static Plotly charts (PNG/SVG) + interactive HF Space
```

---

## Framework Choices

| Component | Framework | Why |
|-----------|-----------|-----|
| TUI | Textual (Python) | Richest Python TUI widgets, terminal-native |
| TUI charts | Textual sparklines + custom widgets | Lightweight, 80-column compatible |
| Dashboard | Plotly Dash (Python) | Scientific viz, real-time updates, Python-native |
| 3D scatter | Plotly `go.Scatter3d` | Interactive, configurable axes (Constellation pattern) |
| Roofline | Plotly `go.Scatter` (log-log) | Standard roofline rendering |
| Code diff | Custom Dash component (Pygments) | Syntax highlighting + annotations |
| KernelBench charts | Plotly (static export) | Publication-quality PNG/SVG |
| Data format | JSON (cache) + Parquet (traces) | JSON for real-time, Parquet for ML training |
| Storage | Local + Hugging Face Hub | Local for speed, HF for persistence/sharing |

---

## Constellation-Inspired Features (Specific Adoptions)

| Constellation Feature | Our Implementation | Where |
|----------------------|-------------------|-------|
| Configurable X/Y/Z/color axes | Dropdown selectors on Optimization Landscape panel | Dashboard Panel 6 |
| Linear/log/logit scale toggles | Scale selector on all numeric axes | All dashboard panels |
| Two-filter system on any key | Range sliders + dropdown filters | Dashboard Panels 4, 6 |
| Right-click tooltips | Hover tooltips on all data points | All interactive charts |
| Shared settings across figures | Selecting a row in experiment table highlights across all panels | Dashboard (linked views) |
| JSON cache file data flow | openkernel writes JSON → dashboard reads | Architecture |
| Static memory / 100K points at 60fps | Plotly handles this natively with WebGL mode | Dashboard Panel 6 (3D) |
| zip + scp for remote data | Download traces from HF Hub for offline analysis | Post-hoc analysis |
