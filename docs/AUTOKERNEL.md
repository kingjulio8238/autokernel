# Autokernel: Recursive GPU Kernel Optimization via Autoresearch

*Last updated: March 2026*

---

## Table of Contents

1. [Overview](#1-overview)
2. [Why Fork Autoresearch](#2-why-fork-autoresearch)
3. [Fork Architecture](#3-fork-architecture)
4. [How KernelBench Works](#4-how-kernelbench-works)
5. [How Autokernel Works](#5-how-autokernel-works)
6. [Why Iterative Beats One-Shot](#6-why-iterative-beats-one-shot)
7. [Benchmarking Plan](#7-benchmarking-plan)
8. [Three Levels of Recursive Improvement](#8-three-levels-of-recursive-improvement)
9. [Current SOTA: What We're Beating](#9-current-sota-whats-were-beating)
10. [Implementation Plan](#10-implementation-plan)
11. [Expected Results and Success Criteria](#11-expected-results-and-success-criteria)

---

## 1. Overview

Autokernel applies the autoresearch recursive optimization loop to GPU kernel generation. Instead of modifying `train.py` to minimize validation BPB, an autonomous agent modifies CUDA/Triton kernel code to maximize speedup over reference PyTorch implementations — evaluated on KernelBench (ICML 2025).

**The key insight**: Every published KernelBench result uses **one-shot generation** — one LLM call, one kernel, one score. Autokernel is the first system to apply **iterative, recursive optimization** to this benchmark. The keep/discard loop ensures monotonic improvement, and 100+ iterations per problem should dramatically outperform single-shot approaches.

**The numbers**:

| | Autoresearch (LLM training) | Autokernel (GPU kernels) |
|---|---|---|
| Iteration time | ~5 min (training) | ~5-30 sec (compile + benchmark) |
| Experiments per hour | ~12 | ~100-500 |
| Experiments overnight | ~100 | ~1,000-5,000 |
| Metric | val_bpb (lower is better) | Speedup over PyTorch (higher is better) |
| Correctness | N/A | Hard gate — must match reference within tolerance |

---

## 2. Why Fork Autoresearch

Autokernel is a **fork** of autoresearch (not an environment plugin), hosted at [`kingjulio8238/autokernel`](https://github.com/kingjulio8238/autokernel). The reasons:

### What's Shared (the loop)

The core philosophy is identical:

```
modify code → run evaluation → measure metric → keep if improved, discard if not → repeat forever
```

Both systems:
- Use git commits to track every experiment
- Maintain a `results.tsv` log
- Run an autonomous agent indefinitely until interrupted
- Use a fixed evaluation harness the agent cannot modify

### What's Different (everything else)

| Dimension | Autoresearch | Autokernel |
|---|---|---|
| **What the agent modifies** | `train.py` (model architecture, hyperparameters, optimizer) | `kernel.py` (CUDA/Triton kernel code, tiling, memory patterns) |
| **Evaluation** | Train for 5 min, compute val_bpb | Compile kernel, check correctness, benchmark runtime |
| **Time budget** | Fixed 5-minute wall-clock training | Seconds per iteration (compile + bench); budget is iteration count, not time |
| **Metric** | Bits per byte (lower is better) | Speedup = ref_runtime / kernel_runtime (higher is better) |
| **Correctness** | Always produces output | Must pass allclose against reference (hard gate before perf measurement) |
| **Dependencies** | PyTorch, Flash Attention, rustbpe, data shards | KernelBench, CUDA toolkit, Triton, NVIDIA GPU |
| **Data** | HuggingFace text corpus (~GB) | Problem specifications (KB) — no data pipeline needed |
| **Hardware** | GPU for training | GPU for benchmarking (same hardware, different use) |

### Why Not an Environment Plugin

The environment abstraction in `PLAN.md` assumes the core loop structure (train → evaluate → keep/discard) stays the same, with different data/tokenizer/metric. Autokernel changes the loop itself:

- No training step — compilation replaces training
- Two-phase evaluation (correctness gate → performance measurement) instead of single metric
- Iteration time is seconds, not minutes — the agent loop cadence is fundamentally different
- The agent modifies fundamentally different code (low-level GPU kernels vs ML model architecture)
- KernelBench provides a complete external evaluation framework — no need for `prepare.py`'s data loading, tokenizer, or dataloader

A clean fork preserves the philosophy while allowing the codebase to diverge where needed.

### Fork Status

The fork is **done** — repo lives at `https://github.com/kingjulio8238/autokernel.git`. The autoresearch files (`train.py`, `prepare.py`, `program.md`) still need to be replaced with their autokernel equivalents:

```
autoresearch (upstream)              autokernel (this repo)
├── prepare.py  (data/eval)     →   ├── prepare.py  (wraps KernelBench eval)
├── train.py    (agent modifies)→   ├── kernel.py   (agent modifies)
├── program.md  (agent rules)   →   ├── program.md  (agent rules, adapted)
├── analysis.ipynb              →   ├── analysis.ipynb (speedup trajectories)
└── results.tsv                 →   └── results.tsv
```

The fork keeps the same file naming conventions and loop structure, making it immediately familiar to anyone who's used autoresearch.

---

## 3. Fork Architecture

### Directory Structure

```
autokernel/
├── prepare.py              # FIXED — wraps KernelBench eval_kernel_against_ref()
├── kernel.py               # AGENT MODIFIES — the CUDA/Triton kernel (ModelNew class)
├── reference.py            # READ-ONLY — the KernelBench problem (PyTorch Model class)
├── program.md              # Agent instructions for the iterative kernel optimization loop
├── analysis.ipynb          # Visualization of optimization trajectories
├── results.tsv             # Experiment log
├── pyproject.toml          # Dependencies (kernelbench, torch, triton)
└── scripts/
    ├── setup_problem.py    # Load a KernelBench problem into reference.py
    ├── sweep.py            # Run autokernel across all problems in a level
    └── compare.py          # Compare autokernel results vs one-shot baselines
```

### `prepare.py` — The Fixed Evaluation Harness

```python
"""
Fixed evaluation harness for autokernel.
Wraps KernelBench's eval_kernel_against_ref().
Agent CANNOT modify this file.
"""
import sys
import time
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

# Constants — agent cannot change these
# NOTE: KernelBench defaults are num_correct_trials=1, num_perf_trials=10.
# We override to match the paper's evaluation protocol (5 correctness, 100 perf).
CORRECTNESS_TRIALS = 5       # Number of random inputs for correctness check
PERF_TRIALS = 100            # Number of timing trials
PRECISION = "fp32"           # Match KernelBench default
TIMING_METHOD = "cuda_event" # CUDA event timing

def load_source(path):
    with open(path) as f:
        return f.read()

def evaluate():
    reference_src = load_source("reference.py")
    kernel_src = load_source("kernel.py")

    start = time.time()
    result = eval_kernel_against_ref(
        original_model_src=reference_src,
        custom_model_src=kernel_src,
        measure_performance=True,
        num_correct_trials=CORRECTNESS_TRIALS,
        num_perf_trials=PERF_TRIALS,
        precision=get_torch_dtype_from_string(PRECISION),
        timing_method=TIMING_METHOD,
        verbose=True,
        # check_for_excessive_speedup=True is the default — populates ref_runtime
        # and flags speedups > 10x (excessive_speedup_threshold default)
    )
    elapsed = time.time() - start

    # eval_kernel_against_ref can return None on lock file errors (retry-able)
    if result is None:
        print("status:error")
        print("speedup:0.00")
        sys.exit(1)

    if not result.compiled:
        print("status:compile_error")
        print("speedup:0.00")
        error = result.metadata.get("compilation_error", "unknown")
        print(f"error:{error}")
        sys.exit(1)

    if not result.correctness:
        print("status:incorrect")
        print("speedup:0.00")
        issue = result.metadata.get("correctness_issue", "unknown")
        print(f"error:{issue}")
        if "max_difference" in result.metadata:
            print(f"max_diff:{result.metadata['max_difference']}")
        sys.exit(1)

    # Correct — report performance
    # ref_runtime is populated because check_for_excessive_speedup=True (default)
    speedup = result.ref_runtime / result.runtime
    print(f"status:correct")
    print(f"speedup:{speedup:.4f}")
    print(f"runtime_us:{result.runtime:.2f}")
    print(f"ref_runtime_us:{result.ref_runtime:.2f}")
    print(f"eval_seconds:{elapsed:.1f}")

    if speedup > 10:
        print(f"WARNING:excessive_speedup_{speedup:.1f}x")

if __name__ == "__main__":
    evaluate()
```

### `reference.py` — The Problem (Read-Only)

This is a KernelBench problem copied verbatim. For example, Level 1 Problem 1:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []
```

### `kernel.py` — Agent-Modifiable

The agent writes and iteratively improves this file. It must define a `ModelNew` class with the same interface:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Agent writes the kernel and ModelNew class here.
# Must be functionally equivalent to Model in reference.py.
# Evaluated by prepare.py via KernelBench's eval harness.

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,
                  stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # ... agent writes tiling, memory access, etc.
    pass

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        K, N = B.shape
        C = torch.empty(M, N, device=A.device, dtype=A.dtype)
        # ... launch kernel
        return C
```

### `program.md` — Agent Instructions

```markdown
# Autokernel: Autonomous GPU Kernel Optimization

You are an autonomous agent optimizing GPU kernels. Your goal is to maximize
speedup over the PyTorch reference implementation, as measured by KernelBench.

## Setup
1. Create git branch: autokernel/<tag>
2. Read reference.py (the KernelBench problem — DO NOT MODIFY)
3. Read prepare.py (the evaluation harness — DO NOT MODIFY)
4. Read kernel.py (your starting point — THIS IS WHAT YOU MODIFY)
5. Initialize results.tsv with header

## Loop (repeat forever)
1. Modify kernel.py — change tiling, memory access, block dims, pipeline staging,
   tensor core usage, or any other optimization
2. git commit
3. Run: uv run python prepare.py > run.log 2>&1
4. Parse run.log:
   - grep "^status:" → compile_error, incorrect, or correct
   - grep "^speedup:" → the speedup value
5. Decision:
   - If status=compile_error → debug, fix, retry (max 3 attempts per approach)
   - If status=incorrect → debug numerical issue, fix, retry
   - If status=correct AND speedup > previous best → KEEP (advance branch)
   - If status=correct AND speedup <= previous best → DISCARD (git reset)
6. Log to results.tsv (never committed):
   commit  speedup  runtime_us  status  description
7. Repeat

## Strategy
- Start with a working kernel (even if slow). Correctness first, speed second.
- Make incremental changes. One optimization per commit.
- Track what works: tiling dimensions, shared memory patterns, vectorization.
- Explore aggressively — the keep/discard mechanism prevents regression.
- When stuck, try fundamentally different approaches (Triton vs CUDA C++,
  different tiling strategy, different memory hierarchy usage).

## Constraints
- Do NOT modify reference.py or prepare.py
- Do NOT install new packages
- kernel.py must define a ModelNew class compatible with KernelBench
- Correctness is a hard gate — an incorrect kernel is always worse than a slow one
- Never stop. Run forever until interrupted.
```

---

## 4. How KernelBench Works

### Problem Structure

Each KernelBench problem is a PyTorch `nn.Module` with:
- `Model` class with a `forward()` method (the reference implementation)
- `get_inputs()` — generates random input tensors
- `get_init_inputs()` — constructor arguments for `Model(*init_inputs)`

The LLM-generated kernel must define `class ModelNew(nn.Module)` as a drop-in replacement with the same `forward()` signature.

### Difficulty Levels

| Level | Problems | Description | Example |
|---|---|---|---|
| **L1** | 100 | Single-kernel operators | GEMM, Softmax, LayerNorm, Conv2D |
| **L2** | 100 | Fusion patterns (multiple ops) | Conv2D + ReLU + BiasAdd, GEMM + Scale + Sigmoid |
| **L3** | 50 | Full model architectures | MLP, ResNet, VGG, MobileNet, MiniGPT |
| **L4** | 20 | HuggingFace models | GPT-2, BART, BigBird, OPT-1.3B |

### Verified KernelBench API (from source)

#### Core function: `eval_kernel_against_ref()`

```python
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

def eval_kernel_against_ref(
    original_model_src: str,                    # reference source code as STRING (not path)
    custom_model_src: str,                      # kernel source code as STRING (not path)
    seed_num: int = 42,
    num_correct_trials: int = 1,                # ⚠️ default is 1 (paper uses 5)
    num_perf_trials: int = 10,                  # ⚠️ default is 10 (paper uses 100)
    measure_performance: bool = False,          # ⚠️ must explicitly enable
    timing_method: str = "cuda_event",
    verbose: bool = False,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = ...,
    backend: str = "cuda",                      # also supports "hip" (AMD)
    precision: torch.dtype = torch.float32,
    check_for_excessive_speedup: bool = True,   # populates ref_runtime
    excessive_speedup_threshold: float = 10,
) -> KernelExecResult                           # can return None on lock file errors
```

#### Result object: `KernelExecResult` (Pydantic BaseModel)

```python
class KernelExecResult(BaseModel):
    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}              # error messages, max_difference, etc.
    runtime: float = -1.0            # custom kernel mean time (microseconds)
    runtime_stats: dict = {}
    ref_runtime: float = -1.0        # reference PyTorch mean time (microseconds)
    ref_runtime_stats: dict = {}
```

**Important**: `ref_runtime` is only populated when `check_for_excessive_speedup=True` (the default). Speedup = `ref_runtime / runtime`.

#### Dataset API (for loading problems)

```python
from kernelbench.dataset import construct_kernelbench_dataset

dataset = construct_kernelbench_dataset(
    level=1,                          # 1, 2, 3, or 4
    source="local",                   # "local" or "huggingface"
    problem_ids=None,                 # optional: list of specific IDs
)
problem = dataset.get_problem_by_id(1)
# problem.code  -> raw Python source with Model, get_inputs(), get_init_inputs()
# problem.name  -> e.g. "1_Square_matrix_multiplication_"
# problem.level -> int
```

#### Scoring API

```python
from kernelbench.score import fastp

# fastp(is_correct, baseline_speed, actual_speed, n, p) -> float
# Returns fraction of problems that are correct AND speedup > p
```

### Evaluation Pipeline (internal flow)

```
1. exec() reference source → get Model, get_init_inputs, get_inputs
2. Instantiate: Model(*get_init_inputs())
3. Compile/load custom kernel source → get ModelNew
4. Instantiate: ModelNew(*get_init_inputs())
5. Correctness check (num_correct_trials trials):
   - Generate random inputs via get_inputs() (seeded)
   - Run both models on same inputs (on CUDA)
   - torch.allclose(output, output_new, atol=tolerance, rtol=tolerance)
   - FP32 tolerance: 1e-4, FP16/BF16 tolerance: 1e-2
   - Must pass ALL trials
6. Performance measurement (num_perf_trials trials, 3 warmup):
   - CUDA event timing
   - Report mean runtime in microseconds
   - Also times reference model when check_for_excessive_speedup=True
7. Cleanup: graceful_eval_cleanup() frees GPU memory
```

**Note**: For Triton/TileLang/CuTe backends, the function uses `load_custom_model_with_tempfile()` which writes to a temp file before importing. For raw CUDA/HIP it uses `load_custom_model()` directly.

### Official Metrics

```python
# fast_p: fraction of problems that are correct AND speedup > p
# fast_p = (1/N) Σ 𝟙(correct_i ∧ speedup_i > p)
fast_p(p=1.0)  # correct and faster than PyTorch
fast_p(p=1.5)  # correct and 1.5x faster
fast_p(p=2.0)  # correct and 2x faster

# Geometric mean speedup across correct-and-faster kernels
geomean_speedup = (∏ speedup_i) ^ (1/n)
```

### Hardware

All official benchmarks run on **NVIDIA L40S** (Ada Lovelace, 48GB, 300W). The paper notes results vary across GPUs — DeepSeek-R1 gets 36% fast₁ on L40S but 47% on A10G.

---

## 5. How Autokernel Works

### The Fundamental Difference

**Standard KernelBench approach** (what every published result uses):

```
LLM(prompt + reference_code) → one kernel → one evaluation → final score
```

**Autokernel approach**:

```
LLM generates initial kernel
       ↓
   ┌── evaluate ──────────────────────────┐
   │   compiled?  → NO  → agent fixes     │
   │   correct?   → NO  → agent fixes     │
   │   faster?    → NO  → discard, try    │
   │                       different       │
   │              → YES → KEEP, push       │
   │                       further         │
   │              ↓                        │
   │   iterate (100+ times)                │
   └───────────────────────────────────────┘
       ↓
   final optimized kernel (monotonically best)
```

### The Keep/Discard Mechanic

This is the core innovation. Every iteration either improves the kernel or leaves it unchanged:

1. **Compile error** → agent reads error, fixes code, retries. No regression.
2. **Incorrect output** → agent sees shape mismatch or numerical diff, fixes. No regression.
3. **Correct but slower** → discard via `git reset`. Speedup stays at previous best.
4. **Correct and faster** → keep. Speedup advances.

**Monotonic improvement is guaranteed.** The best kernel is always preserved. The agent can take risks — try exotic optimizations, PTX intrinsics, unusual tiling — because failure is free.

### A Typical Optimization Trajectory

```
Iter  Speedup  Status    What Happened
────  ───────  ────────  ─────────────────────────────────────────
  1   0.00     crash     Initial naive Triton kernel, compile error (wrong grid)
  2   0.00     crash     Fixed grid, new compile error (type mismatch)
  3   0.00     incorrect Compiles, but output shape wrong
  4   0.72     discard   Correct! But slower than PyTorch (bad tiling)
  5   1.08     keep      Tuned BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
  6   1.08     discard   Tried BLOCK_K=64, register pressure too high
  7   1.08     incorrect Tried shared memory, introduced race condition
  8   1.23     keep      Fixed race condition, shared memory working
  9   1.23     discard   Attempted double buffering, no gain
 10   1.41     keep      Switched to tensor core WMMA instructions
 ...
 50   1.89     keep      Optimized memory access coalescing
 ...
100   2.34     keep      Pipeline staging + vectorized loads
```

The first ~5 iterations are typically debugging (getting correctness). Iterations 5-20 capture low-hanging fruit. Iterations 20-100+ discover progressively subtler optimizations.

---

## 6. Why Iterative Beats One-Shot

### One-Shot Failure Modes (All Solved by Iteration)

| One-Shot Failure | Frequency | Autokernel Solution |
|---|---|---|
| **Compilation error** | ~20-40% of L1 attempts | Agent reads error message, fixes code, retries |
| **Incorrect output** | ~10-20% of compilable kernels | Agent sees max_diff, adjusts algorithm |
| **Functional but slow** | ~30% of correct kernels | Agent iteratively optimizes with benchmark feedback |
| **Suboptimal tiling** | Most kernels | Agent tries many tile sizes empirically |
| **Missing hardware features** | Common | Agent discovers tensor cores, async copy, etc. through iteration |
| **Conservative approach** | LLMs default to safe, generic code | Keep/discard enables aggressive optimization attempts |

### The Feedback Signal Advantage

One-shot generation is **blind** — the LLM has no feedback. Autokernel gives the agent:

- **Compilation errors**: exact error message and line number
- **Correctness failures**: shape mismatch details, max/avg numerical difference
- **Performance data**: exact runtime in microseconds, speedup ratio
- **History**: what worked and what didn't in previous iterations (results.tsv + git log)

This is analogous to why AlphaGo beats a chess engine that only looks one move ahead. Each iteration compounds small improvements that a single generation cannot capture.

### Compound Improvements

Individual optimizations are often small (5-15% each). But they compound:

```
Shared memory tiling:     1.0x → 1.15x  (+15%)
Coalesced memory access:  1.15x → 1.32x (+15%)
Tensor core usage:        1.32x → 1.72x (+30%)
Pipeline staging:         1.72x → 1.98x (+15%)
Vectorized loads:         1.98x → 2.28x (+15%)
Register optimization:    2.28x → 2.41x (+6%)
```

No single LLM call produces all six optimizations simultaneously. But iterative refinement discovers them one at a time.

---

## 7. Benchmarking Plan

### Phase 1: Single-Problem Deep Dives (Proof of Concept)

**Goal**: Demonstrate that iterative optimization dramatically outperforms one-shot on individual problems.

Pick 5 representative problems:

| Problem | Level | Type | Why It's Interesting |
|---|---|---|---|
| L1 #1: Square GEMM (4096x4096) | L1 | Compute-bound | Canonical kernel optimization; cuBLAS is the bar |
| L1 #23: Softmax | L1 | Memory-bound | Different optimization strategy than GEMM |
| L1 #26: GELU | L1 | Elementwise | Should be easy — tests how far past PyTorch we can go |
| L2 #1: Conv2D + ReLU + BiasAdd | L2 | Fusion | Tests whether agent discovers fusion opportunity |
| L2 #12: Gemm + Multiply + LeakyReLU | L2 | Fusion | GEMM + elementwise fusion |

For each:
- Run autokernel for 100 iterations
- Plot speedup vs iteration number (optimization trajectory)
- Compare final speedup to one-shot baselines (GPT-4, Claude, Gemini on same problem)
- Measure total wall-clock time and tokens spent

**Expected output**: Optimization trajectory plots showing monotonic improvement, with final speedup significantly exceeding one-shot baselines.

### Phase 2: Full Level 1 Sweep (Benchmark Comparison)

**Goal**: Produce a directly comparable KernelBench score that demonstrates autokernel's advantage over one-shot approaches.

- Run autokernel on all 100 L1 problems
- For each problem, run N iterations (5, 10, 20, 50, 100)
- Compute official KernelBench metrics at each iteration budget:

```
Approach               fast_1.0    fast_1.5    fast_2.0    Geomean Speedup
──────────────────────────────────────────────────────────────────────────
One-shot (baseline)    X%          Y%          Z%          A.Bx
Autokernel (5 iter)    ?           ?           ?           ?
Autokernel (10 iter)   ?           ?           ?           ?
Autokernel (20 iter)   ?           ?           ?           ?
Autokernel (50 iter)   ?           ?           ?           ?
Autokernel (100 iter)  ?           ?           ?           ?
```

**Key analysis**: Plot fast_p and geomean speedup vs iteration budget. Show where diminishing returns begin. Compute cost-efficiency (speedup gained per token spent).

### Phase 3: Level Progression (L1 → L2 → L3)

**Goal**: Show that autokernel scales to harder problems and that knowledge transfers across problems.

- Run L2 (100 fusion problems) with autokernel
- Run L3 (50 full models) with autokernel
- Analyze: do techniques discovered on L1 (tiling strategies, memory patterns) transfer to L2/L3?
- Compare the agent's performance on later problems vs earlier ones (does it "warm up"?)

### Phase 4: Scaling Analysis

**Goal**: Understand the compute-performance tradeoff.

Measure:
- **Iteration efficiency**: speedup improvement per iteration (does it follow a power law?)
- **Token efficiency**: speedup improvement per 1K tokens consumed
- **Time efficiency**: speedup improvement per wall-clock minute
- **Problem difficulty**: which problems benefit most from iteration? (hypothesis: harder problems benefit more)
- **Diminishing returns**: at what iteration count does improvement plateau?

---

## 8. Three Levels of Recursive Improvement

### Level 1: Single-Kernel Optimization (Direct Analogue)

The agent starts with a naive kernel and iteratively improves it through the standard loop. This is the MVP and most direct comparison to one-shot approaches.

**What the agent modifies per iteration**:
- Tile dimensions (BLOCK_M, BLOCK_N, BLOCK_K)
- Number of warps and pipeline stages
- Shared memory allocation and access patterns
- Memory coalescing and vectorized loads
- Tensor core usage (WMMA/MMA instructions)
- Register pressure management
- Pipeline staging (cp.async, TMA)

### Level 2: Parameterized Kernel Templates (Meta-Optimization)

Instead of writing a single kernel, the agent writes a **kernel generator** — a parameterized template plus a search space definition:

```python
# kernel.py — agent writes a generator, not just one kernel
def make_kernel(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages):
    @triton.jit
    def matmul_kernel(...):
        # Parameterized by BLOCK_M, BLOCK_N, BLOCK_K
        ...
    return matmul_kernel

SEARCH_SPACE = {
    'BLOCK_M': [64, 128, 256],
    'BLOCK_N': [64, 128, 256],
    'BLOCK_K': [32, 64],
    'num_warps': [4, 8],
    'num_stages': [2, 3, 4],
}
```

`prepare.py` benchmarks all configurations in the search space. The agent's job is to write better templates and better search spaces. This is **meta-autotuning** — the agent designs the tuning space, not just a single kernel.

The keep/discard metric becomes: **best speedup achievable from any configuration in the agent's search space**.

### Level 3: Kernel Fusion Search (The Combinatorial Frontier)

For L2 and L3 problems, the agent searches for optimal fusion boundaries:

```
L2 Problem: Conv2D + ReLU + BiasAdd

Option A: Three separate kernels (no fusion)
Option B: Fuse Conv2D + ReLU, separate BiasAdd
Option C: Fuse ReLU + BiasAdd, separate Conv2D
Option D: Fuse all three into one kernel

Each option creates a different kernel to write and optimize.
The agent explores both the fusion topology AND per-kernel optimization.
```

The combinatorial space of possible fusions makes exhaustive manual tuning impractical (as Standard Kernel notes). An autonomous agent running 1,000+ iterations overnight can systematically explore it.

---

## 9. Current SOTA: What We're Beating

All published KernelBench results are on **NVIDIA L40S** (Ada Lovelace, 48GB). The official metric is:

```
fast_p = (1/N) Σ 𝟙(correct_i ∧ speedup_i > p)
```

Where speedup = ref_runtime / kernel_runtime, measured via CUDA events over 100 trials with 3 warmup iterations. Correctness = `torch.allclose` over 5 random inputs (fp32 atol=1e-4).

### One-Shot Baselines (KernelBench paper, Table 1)

Single LLM call, greedy decoding, no feedback:

| Model | L1 fast₁ | L2 fast₁ | L3 fast₁ |
|---|---|---|---|
| GPT-4o | 4% | 5% | 0% |
| OpenAI o1 | 10% | 24% | 12% |
| DeepSeek V3 | 6% | 4% | 8% |
| **DeepSeek R1** | **12%** | **36%** | 2% |
| Claude 3.5 Sonnet | 10% | 7% | 2% |
| Llama 3.1-70B | 3% | 0% | 0% |
| Llama 3.1-405B | 3% | 0% | 2% |

### Iterative Refinement SOTA (KernelBench paper, 10 turns with execution + profiler feedback)

| Model | L1 fast₁ | L2 fast₁ | L3 fast₁ |
|---|---|---|---|
| **DeepSeek R1 (10 turns, E+P)** | **43%** | **72%** | **18%** |

This is the current published SOTA for iterative approaches. Only 10 turns of refinement — autokernel targets 100+.

### Repeated Sampling (best-of-k)

DeepSeek-V3 at k=100 samples: **37% fast₁ on L2** (vs 4% one-shot). Brute force sampling without the keep/discard loop.

### Leaderboard Best Solutions (per-problem, any method)

From the [Stanford leaderboard](https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/), best across all submissions:

| Level | Problems with any solution >1.0x | Best model overall |
|---|---|---|
| L1 | ~77/87 submitted (88%) | gpt-o1: 29 problems >1.0x, geomean ~0.51x |
| L2 | ~53/64 submitted (83%) | gpt-o1: 36 problems >1.0x, geomean ~0.96x |
| L3 | ~19/23 submitted (83%) | gpt-o1: 15 problems >1.0x, geomean ~0.84x |

**Key takeaway**: even the best one-shot models have geomean speedup **below 1.0x** — most generated kernels are *slower* than PyTorch. The bar is low.

---

## 10. Implementation Plan

### Stage 1: Fork and Setup ✅ DONE

Fork created at `https://github.com/kingjulio8238/autokernel.git`. The repo currently contains the original autoresearch files that need to be replaced in Stage 2.

**What to keep from autoresearch**:
- `program.md` structure (adapt for kernels)
- `results.tsv` format and logging pattern
- `analysis.ipynb` structure (adapt for speedup trajectories)
- Git-based keep/discard workflow
- The overall file conventions and agent loop design

**What to replace**:
- `train.py` → `kernel.py` (agent-modifiable kernel code)
- `prepare.py` → new `prepare.py` wrapping KernelBench eval
- Data pipeline (not needed — problems are small PyTorch files)
- Tokenizer (not needed)
- Model architecture (replaced by kernel code)

### Stage 2: Core Implementation (MVP) — 🖥️ OFF-POD ✅ DONE

All file writing done off-pod. Files replaced:

| File | Status |
|---|---|
| `pyproject.toml` | ✅ Stripped autoresearch deps, added kernelbench + torch + triton |
| `prepare.py` | ✅ Wraps `eval_kernel_against_ref()` with stdout metrics |
| `program.md` | ✅ Agent instructions for kernel optimization loop |
| `reference.py` | ✅ L1 #1 square GEMM (4096x4096) |
| `kernel.py` | ✅ Starter `ModelNew` (torch.matmul passthrough) |
| `scripts/setup_problem.py` | ✅ Load any KernelBench problem by level+id |
| `train.py` | ✅ Deleted |
| `analysis.ipynb` | ✅ Adapted for speedup trajectories |

#### Stage 2 Gate: On-Pod Validation

##### Pod Requirements

| Resource | Requirement | Notes |
|---|---|---|
| **GPU** | NVIDIA with 24GB+ VRAM | L40S for leaderboard-comparable results. A10G/A100/H100 all work for development. |
| **Container image** | `nvidia/cuda:12.8.0-devel-ubuntu22.04` or PyTorch NGC `nvcr.io/nvidia/pytorch:25.01-py3` | Needs CUDA 12.8 toolkit + dev headers (for Triton/CUDA compilation). Must include `gcc`, `git`. |
| **Python** | 3.10+ | Set via `.python-version` |
| **Disk** | 30GB minimum | ~15-20GB for PyTorch + CUDA + Triton + KernelBench installed. 30GB gives headroom. |
| **Network volume** | Not needed | KernelBench problems are KB-sized. No data downloads. |
| **Agent model** | Claude (Opus/Sonnet) | The LLM running the keep/discard optimization loop. Not a model being trained — it *writes* kernel code. |

##### Setup Steps (run on pod)

```bash
# 1. Clone the repo
git clone https://github.com/kingjulio8238/autokernel.git
cd autokernel

# 2. Install uv (if not in container image)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # or restart shell

# 3. Install dependencies (creates .venv, resolves everything)
uv sync

# 4. Verify CUDA is visible
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 5. Verify KernelBench is installed
uv run python -c "from kernelbench.eval import eval_kernel_against_ref; print('KernelBench OK')"
```

##### Run the Gate

```bash
# Run the evaluation harness on the starter kernel
uv run python prepare.py
```

##### Verify Gate Success

The output should look like:
```
status:correct
speedup:~1.0000       # ≈1.0x because kernel.py is just torch.matmul
runtime_us:<some_number>
ref_runtime_us:<some_number>
eval_seconds:<some_number>
```

**Gate passes if:**
- [ ] `status:correct` (kernel compiles and passes correctness check)
- [ ] `speedup:` line is present with a positive number (any value — even 0.5x is fine)
- [ ] No Python tracebacks or crashes

**If it fails**, check:
- `status:compile_error` → KernelBench or Triton install issue, check CUDA toolkit
- `status:error` → lock file issue, retry or check GPU visibility
- Python traceback → dependency issue, re-run `uv sync`

### Stage 3: First Autonomous Run — 🔥 ON-POD

Requires GPU. Run the agent loop on a single problem (L1 #1 square GEMM, 4096x4096).

**Gates** (benchmarked against one-shot SOTA on GEMM):
- [ ] Agent completes 20+ iterations unattended without crashing
- [ ] Keep/discard mechanic works — speedup is monotonically non-decreasing
- [ ] `results.tsv` logs correctly, git history tracks each experiment
- [ ] Achieve speedup >1.0x on GEMM (beat PyTorch eager). One-shot SOTA on this problem: **0.17x** (Claude 3.5 Sonnet). Beating 1.0x on GEMM with iteration would already exceed every one-shot result on this problem by ~6x.
- [ ] Stretch: achieve speedup >2.0x on GEMM (approach cuBLAS territory)

### Stage 4: Phase 1 Benchmarks (5 Deep Dives) — 🔥 ON-POD

Requires GPU. Run autokernel for 100 iterations on 5 representative problems. Compare against one-shot and 10-turn iterative baselines.

| Problem | One-shot SOTA | 10-turn SOTA (est.) | Autokernel gate (100 iter) |
|---|---|---|---|
| L1 #1: Square GEMM | 0.17x | ~0.5x | **>2.0x** |
| L1 #23: Softmax | ~0.5x | ~1.0x | **>1.5x** |
| L1 #26: GELU | ~1.0x | ~1.5x | **>2.0x** |
| L2 #1: Conv2D+ReLU+BiasAdd | ~0.3x | ~1.0x | **>1.5x** |
| L2 #12: Gemm+Multiply+LeakyReLU | ~0.5x | ~1.2x | **>1.5x** |

**Gates**:
- [ ] All 5 problems achieve speedup >1.0x (beat PyTorch). This alone would match/exceed the best one-shot on most of these problems.
- [ ] At least 3/5 problems achieve speedup >1.5x
- [ ] Optimization trajectory plots show clear improvement curve (not flat after iteration 5)
- [ ] Beat the 10-turn iterative SOTA (DeepSeek R1 + execution + profiler feedback) on at least 4/5 problems. Their approach stops at 10 turns — we run 100+.

### Stage 5: Full L1 Sweep + Benchmark Comparison — 🔥 ON-POD (analysis off-pod)

GPU for running 100 problems. Analysis/plotting can be done off-pod from `results.tsv` files.

**Gates** (compared to published SOTA):

| Metric | One-shot SOTA | 10-turn iterative SOTA | Autokernel gate | Stretch |
|---|---|---|---|---|
| L1 fast₁ (correct + faster) | 12% (DeepSeek R1) | 43% (DeepSeek R1, 10 turns) | **>50%** | >80% |
| L1 fast₁.₅ | ~3% (est.) | ~15% (est.) | **>30%** | >50% |
| L1 fast₂ | ~1% (est.) | ~5% (est.) | **>15%** | >30% |
| L1 geomean speedup | <0.5x (all models) | ~0.8x (est.) | **>1.2x** | >1.5x |
| L1 correctness rate | ~30-45% (one-shot) | ~90% (10-turn) | **>90%** | >95% |

**The headline number**: if autokernel achieves >50% fast₁ on L1, it beats the published 10-turn iterative SOTA (43%) and is **4x better** than the best one-shot (12%). This is the paper-worthy result.

- [ ] fast₁ >50% on L1 at 100 iterations (beat 10-turn SOTA)
- [ ] fast₁ >43% on L1 at ≤50 iterations (match 10-turn SOTA with fewer turns)
- [ ] geomean speedup >1.0x (every published model is below 1.0x)
- [ ] Produce scaling curve: fast₁ vs iteration count at {5, 10, 20, 50, 100} iterations
- [ ] Demonstrate diminishing-returns curve — where does fast₁ plateau?

### Stage 6: L2/L3 + Advanced Features — 🔥 ON-POD

Extend to fusion problems (L2) and full models (L3). This is where the combinatorial advantage of iteration should be largest.

**Gates**:

| Metric | 10-turn iterative SOTA | Autokernel gate | Stretch |
|---|---|---|---|
| L2 fast₁ | 72% (DeepSeek R1, 10 turns) | **>75%** | >85% |
| L3 fast₁ | 18% (DeepSeek R1, 10 turns) | **>25%** | >40% |
| L2 geomean speedup | ~1.0x (est.) | **>1.3x** | >1.5x |

- [ ] L2 fast₁ >72% (beat 10-turn SOTA — this is the hardest gate since their L2 number is already high)
- [ ] L3 fast₁ >25% (beat 10-turn SOTA by >7pp)
- [ ] Demonstrate fusion discovery: agent finds non-trivial operator fusions on L2 problems
- [ ] Cross-problem knowledge transfer: agent performance on problem N+1 improves after solving problems 1..N

### Stage 7: Paper-Ready Results — 🖥️ OFF-POD (writing) + 🔥 ON-POD (hardware generalization runs)

**Gates for a publishable result**:
- [ ] Beat all published SOTA on at least 2 of 3 levels (L1/L2/L3)
- [ ] Scaling curve showing log-linear or power-law improvement with iteration count
- [ ] At least 3 examples of non-obvious kernel optimizations discovered by the agent (documented via git history)
- [ ] Cost-efficiency analysis: tokens spent vs speedup gained, showing iteration is more efficient than repeated sampling (k=100)
- [ ] Hardware generalization: results on at least 2 GPU architectures (L40S + one other)

### Timeline

```
Stage 1:  Fork ✅ DONE
Stage 2:  Core implementation OFF-POD — replace autoresearch files, write all new code
          Validate ON-POD — uv run python prepare.py works end-to-end
Stage 3:  ON-POD — first autonomous run, 20+ iterations on GEMM, beat 1.0x
Stage 4:  ON-POD — 5 deep dives, 100 iterations each, beat 10-turn SOTA per-problem
Stage 5:  ON-POD — full L1 sweep, fast₁ >50%, beat all published SOTA
Stage 6:  ON-POD — L2/L3, beat 72%/18% iterative SOTA
Stage 7:  MIXED — paper writeup (off-pod), hardware generalization runs (on-pod)
```

---

## 11. Expected Results and Success Criteria

### Hypotheses

1. **Correctness recovery**: Autokernel will achieve >90% correctness rate on L1 (vs ~30-45% for one-shot) because the agent can fix compilation and correctness errors iteratively. The 10-turn approach already reaches ~90% — we should match or exceed this.

2. **Speedup improvement**: Autokernel at 100 iterations will push geomean speedup above 1.0x on L1. No published approach achieves this — every model's geomean is below 1.0x (most kernels are slower than PyTorch).

3. **fast_p improvement**: Autokernel will dramatically increase fast₁ scores because:
   - More kernels become correct (fixing compilation/correctness errors)
   - Correct kernels are iteratively optimized past the speedup thresholds
   - 100+ iterations vs 10 turns gives 10x more optimization budget

4. **Diminishing returns**: Most gains will come in the first 20-50 iterations. Iterations 50-100+ will show diminishing but non-zero returns. Some hard problems will show breakthroughs at high iteration counts.

5. **L2 benefit**: Fusion problems (L2) already show the highest SOTA (72% fast₁ at 10 turns). The gap between one-shot (36%) and 10-turn (72%) is massive, suggesting iteration helps most here. Autokernel at 100+ iterations should push further.

### What Would Make This a Paper

A compelling result would be:

1. **Clear improvement over all baselines**: Side-by-side on KernelBench showing autokernel exceeds both one-shot (12% L1 fast₁) and 10-turn iterative (43% L1 fast₁) approaches.

2. **First geomean >1.0x**: No published approach has geomean speedup above 1.0x on any level. Breaking this barrier would be significant.

3. **Scaling curve**: Plot showing fast₁ vs iteration count with a clear trend, demonstrating that more compute → better kernels. Compare against repeated sampling (best-of-k) to show the keep/discard loop is more sample-efficient.

4. **Novel kernel discoveries**: Examples where the agent found non-obvious optimizations that a human kernel engineer would find impressive — documented via git history.

5. **Cost-efficiency**: Analysis showing that iterative refinement with keep/discard is more efficient than brute-force repeated sampling (DeepSeek-V3 at k=100 achieves 37% fast₁ on L2 — can we beat this with fewer total tokens?).

---

## Appendix: Mapping to Standard Kernel's Rubric

Standard Kernel defines a five-dimensional evaluation framework for kernel generation systems. Here's how autokernel maps:

| Dimension | Standard Kernel's Scale | Autokernel's Position |
|---|---|---|
| **K (Kernel Complexity)** | K1 (simple) to K4 (frontier) | Starts at K1-K2 (KernelBench L1), progresses to K3-K4 (L2-L3) through level progression |
| **R (Representation Level)** | Library composition to instruction-level | Agent naturally escalates — starts with Triton (high-level DSL), drops to CUDA C++ when hitting Triton's ceiling, potentially inline PTX for peak performance |
| **H (Hardware Specialization)** | Portable to frontier features | Agent discovers hardware features empirically through iteration — tensor cores, async copy, TMA — without needing documentation |
| **P (Performance Target)** | Functional to state-of-the-art | Keep/discard loop drives toward peak performance. 100+ iterations should approach or match expert hand-tuned kernels on simple ops |
| **A (Automation Level)** | Expert co-design to fully autonomous | Fully autonomous (A4) — no human in the loop once started |

The key advantage over Standard Kernel's current approach: **autokernel doesn't need expert knowledge baked in.** It discovers what works through empirical iteration. This means it can adapt to new hardware without retraining or reprogramming — just run the loop on the new GPU and let the agent rediscover optimal patterns.
