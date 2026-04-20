# Testing Guide: KernelBench with Dummy Data

How to test the entire openkernel + kernel code pipeline without spending money or needing real GPU results.

---

## Quick Start (2 minutes)

```bash
cd /Users/juliansaks/Desktop/code/autokernel

# 1. Run local tests (free, no credentials)
PYTHONPATH=. python scripts/test_infra.py --local-only

# 2. Launch the interactive shell with mock data
PYTHONPATH=. python -m kernel_code.cli

# In the shell:
kernel-code > /optimize --mock --iterations 20
# [TUI launches with mock optimization data]
# [Press q to exit TUI]

kernel-code > /show best
kernel-code > /show results
kernel-code > /skills
kernel-code > /skill:triton_reduction
kernel-code > /history
kernel-code > /compare
kernel-code > /config
kernel-code > /optimize --mock --parallel --iterations 10
kernel-code > /quit
```

---

## Test the Dashboard (free)

```bash
# Generate mock data and launch dashboard
PYTHONPATH=. python -m kernel_code.dashboard.server

# Opens http://localhost:8050 with 10 panels:
# - Optimization trajectory (speedup over iterations)
# - Roofline model
# - Resource utilization gauges
# - Experiment table (filterable)
# - Code diff (side-by-side kernel comparison)
# - 3D optimization landscape (Constellation-style, configurable axes)
# - Strategy tree (treemap)
# - Convergence analysis
# - Cost efficiency frontier
# - Strategy statistics

# Interactive features:
# - Click a point on trajectory → highlights across all panels
# - Use range filters at top to filter by speedup/cost/iteration
# - Toggle linear/log scale on trajectory and roofline
# - Change color-by metric on landscape panel
```

---

## Test the TUI (free)

```bash
PYTHONPATH=. python -c "
from kernel_code.mock_data import generate_mock_session
from kernel_code.tui.app import KernelCodeApp
p = generate_mock_session(25, 'demo')
KernelCodeApp(session_path=p).run()
"

# Shows:
# Left: Optimization feed (summary, intent, critic diagnosis, best kernel, recent activity)
# Right: Trajectory sparkline, profiling gauges, experiment log
# Bottom: Status bar with GPU/backend/model/iteration/cost/time
# Keys: d=dashboard, k=diff, r=roofline, q=quit
```

---

## Test KernelBench Sweep with Mock Problems (free)

```bash
# Run a mini sweep with mock problems (no GPU, no LLM — uses mocks)
PYTHONPATH=. python scripts/run_sweep.py \
  --level 1 \
  --problems 0,1,2 \
  --max-iterations 5 \
  --source mock \
  --output results/sweeps/test/

# Outputs:
# - Scoring summary (fast@1, fast@1.5, fast@2, geomean, correctness)
# - Comparison table vs published baselines
# - Results JSON in results/sweeps/test/

# Generate comparison table from results
PYTHONPATH=. python scripts/publish_results.py \
  --results results/sweeps/test/L1_*.json
```

---

## Test with Real LLM but Mock GPU (cheap, ~$0.01)

Requires: `GROQ_API_KEY` (free tier)

```bash
export GROQ_API_KEY=your-key

# Test single LLM call
PYTHONPATH=. python scripts/test_infra.py --test 4

# Test Generator + Critic
PYTHONPATH=. python scripts/test_infra.py --test 5
```

---

## Test with Real LLM + Real GPU ($0.50-$1.00)

Requires: `GROQ_API_KEY` + Modal setup

```bash
export GROQ_API_KEY=your-key
modal setup  # one-time

# Deploy Modal app (first time builds image ~3-5 min)
python modal_infra/deploy.py

# Run full test ladder
PYTHONPATH=. python scripts/test_infra.py

# Or run single optimization
PYTHONPATH=. python -m openkernel.cli optimize \
  --reference reference.py \
  --config configs/groq_fast.yaml \
  --max-iterations 5
```

---

## Test the Full Shell Experience (mock mode, free)

```bash
PYTHONPATH=. python -m kernel_code.cli

# The full interactive experience:
kernel-code > /help

kernel-code > /config
# Shows settings loaded from .kernel-code/settings.yaml

kernel-code > /skills
# Lists all 10 optimization skills with code templates

kernel-code > /skill:triton_gemm
# Shows the tiled GEMM skill with starter code

kernel-code > /optimize --mock --iterations 15
# [TUI runs optimization with mock data]
# After completion: saves best kernel, prints summary + dashboard link

kernel-code > /show best
# Displays the best kernel with syntax highlighting

kernel-code > /show results
# Summary table: iterations, kept, speedup, cost

kernel-code > /compare
# Side-by-side: baseline vs best

kernel-code > /dashboard
# Opens browser dashboard at localhost:8050

kernel-code > /optimize --mock --parallel
# Runs Triton + CUDA, prints comparison table

kernel-code > /history
# All runs in this session

# Natural language (requires GROQ_API_KEY for LLM):
kernel-code > what should I try next?
kernel-code > explain iteration 9
kernel-code > why did iter 5 fail?

kernel-code > /quit
```

---

## Generate Benchmark Charts (mock data, free)

```bash
PYTHONPATH=. python -c "
from kernel_code.benchmarks.fast_p_chart import create_fast_p_chart
from kernel_code.benchmarks.export import export_figure

# Create fast_p comparison chart with mock results
our_results = {'fast_1': 0.65, 'fast_1_5': 0.40, 'fast_2': 0.20, 'geomean': 1.85, 'correct_pct': 0.90}
fig = create_fast_p_chart(our_results)
export_figure(fig, 'results/charts/fast_p_comparison.png')
print('Chart saved to results/charts/fast_p_comparison.png')
"
```

---

## Full Infrastructure Test Ladder

```bash
# This runs all 8 tests in order (stops on first failure):
PYTHONPATH=. python scripts/test_infra.py

# Test 1: Local imports (free)
# Test 2: Modal deploy + health check (<$0.01)
# Test 3: Single kernel eval on L40S (<$0.05)
# Test 4: Single LLM call (<$0.01)
# Test 5: Generator + Critic agents (<$0.05)
# Test 6: Full optimization run (<$1.00)
# Test 7: Mini sweep, 3 problems (<$5.00)
# Test 8: Publish results (free)

# Run specific test:
PYTHONPATH=. python scripts/test_infra.py --test 3

# Run up to test N:
PYTHONPATH=. python scripts/test_infra.py --up-to 5

# Local only (free):
PYTHONPATH=. python scripts/test_infra.py --local-only
```

---

## What Each Test Validates

| Test | What | Mock/Real | Cost |
|------|------|-----------|------|
| Shell + commands | /optimize, /show, /skills, /config, /compare, /history, /parallel | Mock | Free |
| TUI rendering | All 5 panels, widgets, keybindings | Mock | Free |
| Dashboard | All 10 panels, filters, linked selection, scale toggles | Mock | Free |
| Benchmark charts | fast_p, speedup distribution, scaling curve, cost frontier, heatmap | Mock | Free |
| KernelBench sweep | Problem loading, scoring, comparison table | Mock | Free |
| LLM integration | Generator, Critic, agentic loop, streaming | Real LLM | ~$0.01 |
| GPU eval | Compile, correctness, benchmark, profiling on L40S | Real GPU | ~$0.05 |
| Full pipeline | 3-level hybrid loop end-to-end | Real all | ~$1.00 |

---

## Ready for Real KernelBench Sweep

When all tests pass and you're ready to spend ~$35:

```bash
export MINIMAX_API_KEY=your-key  # or GROQ_API_KEY (free but rate-limited)
modal setup

# Full L1 sweep (100 problems, 50 iterations each)
python scripts/run_sweep.py --level 1 --max-iterations 50 --output results/sweeps/

# Publish results
python scripts/publish_results.py --results results/sweeps/L1_*.json --upload
```
