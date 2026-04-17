# openkernel

Self-recursive GPU kernel optimization engine.

Give it a PyTorch operation and a target GPU -- it produces an optimized CUDA or Triton kernel through structured search with hardware profiler feedback.

## What It Does

- **3-level hybrid search**: Strategy evolution selects *what* to optimize, a world model plans the approach, and a refinement loop implements and validates with profiler feedback. Strategies that work are extracted into a skill library that compounds across problems.
- **Bring your own model (BYOM)**: Any LLM via OpenAI-compatible API. MiniMax M2.5 is the default. Claude, GLM, Kimi, Qwen all work. No lock-in.
- **Cloud-native on Modal**: Compile, benchmark, and profile kernels on cloud GPUs (H100, A100, L40S). No local NVIDIA hardware required.

## Quick Start

```bash
pip install openkernel kernel-code
export MINIMAX_API_KEY=your-key
openkernel optimize --reference my_kernel.py --backend triton
```

Options:

```bash
openkernel optimize --reference my_kernel.py --backend cuda --model claude-sonnet-4-20250514
openkernel evaluate --kernel optimized.py --reference my_kernel.py --eval-mode thorough
openkernel info
```

## kernel code

kernel code is the terminal-native developer tool built on top of openkernel. It wraps the optimization engine with a Textual TUI purpose-built for kernel engineers:

```
+---------------------------------------------------------+
|  kernel code v0.1          [H100]  [Triton]  [L1#23]   |
+----------------------+----------------------------------+
|                      |  Optimization Trajectory          |
|   Chat / Agent       |  ████████████▓▓░░░  1.8x        |
|   Panel              +----------------------------------+
|                      |  Profiling Summary                |
|   > Analyzing        |  Bottleneck: memory_bound         |
|     reference...     |  Bandwidth:  72% of peak          |
|                      |  L2 hit:     45% (poor)           |
|                      +----------------------------------+
|   Critic: "L2 hit    |  Experiment Log                   |
|   rate improved to   |  #1  1.0x  keep   baseline        |
|   78%. Next: try     |  #3  1.3x  keep   shared mem      |
|   register blocking" |  #5  1.8x  keep   vectorized      |
+----------------------+----------------------------------+
|  [d]ashboard  [k]ernel diff  [r]oofline  [q]uit        |
+---------------------------------------------------------+
```

Press `d` to open a web dashboard with roofline plots, 3D optimization landscapes, strategy trees, and side-by-side kernel diffs with performance annotations.

```bash
kernel-code optimize --reference my_kernel.py --backend triton --no-mock
kernel-code dashboard
```

## Supported Models

| Model | Provider | Input / Output (per M tokens) | Recommended For |
|-------|----------|-------------------------------|-----------------|
| **MiniMax M2.5** (default) | MiniMax | $0.30 / $1.20 | General use, sweeps |
| GLM-5.1 | Zhipu AI | $1.40 / $4.40 | Deep optimization, hard problems |
| Kimi K2.5 | Moonshot AI | $0.50 / $2.80 | Speed, parallel exploration |
| Qwen3.5 397B | Alibaba | $0.20 / $0.80 | Budget, local inference |
| Claude Sonnet 4 | Anthropic | $3.00 / $15.00 | Structured output, frontier fallback |

All models are accessed via OpenAI-compatible APIs through litellm. Set the appropriate environment variable (`MINIMAX_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) and pass `--model <model-id>`.

## Supported Backends

openkernel supports **Triton** and **CUDA**. The kernel engineer chooses the backend; the engine applies backend-specific optimization strategies:

- **Triton**: `@triton.autotune` parametric search, Proton profiling, shared memory tiling
- **CUDA**: Warp-level primitives, Tensor Core MMA, CUTLASS CuTe templates, inline PTX

Both backends share strategies for fusion discovery, algorithmic improvements, and memory access pattern optimization.

## Architecture

openkernel uses a 3-level hybrid search:

1. **Outer loop -- Strategy evolution**: Maintains a Pareto frontier of optimization strategies. Strategies that produce results survive; dominated strategies are pruned. Successful strategies are persisted to a skill library for future problems.
2. **Middle loop -- World model search**: An LLM-managed tree of optimization intents. Decouples *what* to optimize from *how* to implement it. If a good strategy produces buggy code, the strategy survives for retry.
3. **Inner loop -- Refinement**: Generator produces kernel code, evaluates on Modal (compile + benchmark + profile), Critic diagnoses bottlenecks from profiler data, Generator produces an improved version. Repeat.

Two LLM roles -- Generator and Critic -- produce structured, inspectable reasoning. The Critic reads hardware profiler output and provides specific diagnoses ("L2 hit rate 45%, restructure to coalesced access with BLOCK_K=64 tiles").

See [docs/openkernel-design.md](docs/openkernel-design.md) for the full system design.

## KernelBench

openkernel is designed to hill-climb [KernelBench](https://github.com/KernelBench/KernelBench) (Stanford, ICML 2025) -- 250 problems across 4 difficulty levels. We benchmark against KernelBench and publish comparison results against other systems.

Metrics tracked: `fast_p` at p={1.0, 1.5, 2.0}, geomean speedup, correctness rate, cost per kernel, iterations to convergence.

Results will be published when available.

## Contributing

openkernel is open-core under the **Apache 2.0** license. Contributions are welcome.

kernel code is the commercial product built on top of openkernel.

## Docs

| File | Description |
|------|-------------|
| [docs/pitch.md](docs/pitch.md) | Project pitch and market thesis |
| [docs/openkernel-design.md](docs/openkernel-design.md) | Full system architecture (5 layers, search engine, memory system) |
| [docs/kernel-code-design.md](docs/kernel-code-design.md) | kernel code product design (TUI, dashboards, trace capture) |
| [docs/visualization-design.md](docs/visualization-design.md) | Dashboard and visualization specifications |
| [docs/research-synthesis.md](docs/research-synthesis.md) | Research survey of kernel optimization systems |
| [docs/five-layer-cake.md](docs/five-layer-cake.md) | Detailed breakdown of the 5-layer architecture |
| [docs/data-and-integrations.md](docs/data-and-integrations.md) | Data pipeline and external integrations |
| [docs/codebase-structure.md](docs/codebase-structure.md) | Repository layout and module organization |
| [docs/build-plan.md](docs/build-plan.md) | Build plan and implementation phases |
| [docs/gtm.md](docs/gtm.md) | Go-to-market strategy |
