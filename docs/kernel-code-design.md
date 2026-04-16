# kernel code: Product Design

*April 2026*

---

## One Sentence

kernel code is a terminal-native developer tool (Textual TUI) purpose-built for kernel engineers, wrapping openkernel with visualizations that differentiate from Codex, Claude Code, and Cursor.

---

## Product Vision

kernel code is how most developers access openkernel. It provides:
1. **Interactive kernel optimization** — write/modify kernels with AI assistance, optimized for the kernel engineer workflow
2. **Real-time visualizations** — roofline plots, optimization trajectories, profiling panels that no general-purpose coding tool has
3. **Personalized dashboards** — web-accessible via links for deeper analysis
4. **Trace capture** — every optimization session feeds the kernelgen-1 training flywheel

---

## Interface: Textual TUI (Terminal-Native)

Built with Python Textual framework. Terminal-native because kernel engineers live in terminals.

### TUI Layout

```
┌─────────────────────────────────────────────────────────┐
│  kernel code v0.1          [H100]  [Triton]  [L1#23]   │
├──────────────────────┬──────────────────────────────────┤
│                      │  Optimization Trajectory          │
│   Chat / Agent       │  ████████████▓▓░░░░  1.8x       │
│   Panel              │  ↑keep  ↑keep  ×disc  ×err      │
│                      ├──────────────────────────────────┤
│   > Analyzing        │  Profiling Summary                │
│     reference...     │  Bottleneck: memory_bound         │
│                      │  Bandwidth:  72% of peak          │
│   Generated kernel   │  L2 hit:     45% (poor)           │
│   with vectorized    │  Occupancy:  0.78                 │
│   float4 loads...    │  Headroom:   ~1.4x                │
│                      ├──────────────────────────────────┤
│   ✓ Correct          │  Experiment Log                   │
│   ↑ 1.3x → 1.8x     │  #1  1.0x  keep   baseline       │
│                      │  #2  0.7x  disc   bad tiling      │
│   Critic: "L2 hit    │  #3  1.3x  keep   shared mem     │
│   rate improved to   │  #4  1.3x  disc   no gain        │
│   78%. Next: try     │  #5  1.8x  keep   vectorized     │
│   register blocking" │  #6  ...   ···    running         │
│                      │                                   │
├──────────────────────┴──────────────────────────────────┤
│  [d]ashboard  [k]ernel diff  [r]oofline  [q]uit        │
└─────────────────────────────────────────────────────────┘
```

### Key TUI Panels

| Panel | What It Shows |
|-------|--------------|
| **Chat/Agent** | LLM conversation, optimization progress, Generator/Critic output |
| **Optimization Trajectory** | Sparkline chart of speedup over iterations, color-coded keep/discard |
| **Profiling Summary** | Current bottleneck, bandwidth/compute utilization, cache efficiency, headroom |
| **Experiment Log** | Scrollable table of all iterations with status colors |
| **Status Bar** | Current GPU, backend, problem, model, iteration count |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `d` | Open personalized web dashboard (launches browser) |
| `k` | Show kernel diff (side-by-side current vs previous best) |
| `r` | Show roofline model overlay |
| `s` | Show/hide skill library matches |
| `p` | Pause/resume optimization |
| `b` | Switch backend (Triton ↔ CUDA) |
| `q` | Quit |

---

## Personalized Web Dashboards (Plotly Dash)

Accessible via link from TUI (`d` key). Opens in browser. Full visualization suite:

### Dashboard Panels

| Panel | What It Shows | Inspiration |
|-------|--------------|-------------|
| **Optimization Trajectory** | Full interactive speedup chart with hover details per iteration | PufferLib Constellation |
| **Roofline Model** | Arithmetic intensity vs GFLOP/s with ceiling lines, kernel positions | Nsight Compute |
| **Resource Utilization** | Gauges for occupancy, registers, shared mem, cache rates, bandwidth | Nsight "Speed of Light" |
| **Experiment Table** | Full results with filtering, sorting, search | Enhanced results.tsv |
| **Code Diff** | Syntax-highlighted side-by-side with performance annotations | GitHub diff |
| **Optimization Landscape** | 3D scatter / t-SNE of kernel variants colored by performance | Constellation Fig 1/3 |
| **Memory Access Heatmap** | Per-warp memory patterns (on-demand, from deep profiling) | Nsight memory workload |

### Data Flow

```
openkernel engine → JSON cache files (append-only) → Dashboard reads periodically
                  → TUI progress (Textual)          → terminal
                  → WebSocket (optional)             → real-time dashboard updates
```

---

## Trace Capture (for kernelgen-1)

Every kernel code session captures:

```python
@dataclass
class OptimizationTrace:
    session_id: str
    problem: str                # KernelBench level + ID or custom
    hardware: str               # GPU type
    backend: str                # triton | cuda
    model: str                  # which LLM was used
    iterations: list[IterationTrace]
    final_speedup: float
    final_kernel: str
    total_tokens: int
    total_time_seconds: float
    
@dataclass  
class IterationTrace:
    iteration: int
    intent: str                 # what optimization was attempted
    prompt: str                 # full LLM prompt
    response: str               # full LLM response
    kernel_code: str            # generated kernel
    eval_result: EvalResult     # correctness + speedup + profile
    critic_diagnosis: CriticDiagnosis | None
    decision: str               # keep | discard | error
    tokens_used: int
    latency_seconds: float
```

Storage: Parquet files (columnar, efficient for ML training pipelines). Opt-in for users.

Training pipeline (future): traces → filter for best (top 10% speedup) → format as instruction-following pairs → SFT + GRPO → kernelgen-1.

---

## Relationship to openkernel

```
kernel code (TUI + dashboards + trace capture)
    │
    ├── wraps openkernel engine via Python API
    │   openkernel.optimize(reference, backend, model, config) → stream of results
    │
    ├── renders results in TUI panels
    │
    ├── serves personalized dashboard on localhost
    │
    └── captures traces to Parquet store
```

kernel code IS the primary way developers interact with openkernel. The openkernel library can also be used directly (Python API, scripts) for automation/sweeps, but kernel code is the human-facing product.
