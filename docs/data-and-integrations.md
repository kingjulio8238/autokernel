# Data, Storage, Hugging Face Integration & Trace Pipeline

*April 2026*

---

## Overview

openkernel produces three categories of data that need storage, versioning, and distribution:

1. **Optimization traces** — every prompt, response, kernel variant, eval result, profile data from optimization runs (feeds kernelgen-1 training)
2. **Optimized kernels** — the best kernels produced, organized by problem and hardware
3. **Models** — kernelgen-1 and future iterations, plus recommended third-party model configs

Hugging Face Hub is the primary platform for all three.

---

## 1. Trace Pipeline (for kernelgen-1 Training)

### What We Capture (Modeled on Cursor's Real-Time RL)

Cursor's Composer 2 pipeline:
- Client-side instrumentation translates user interactions into signals
- Backend data pipelines aggregate signals into the training loop
- New checkpoints deployed as often as every 5 hours
- Reward signals: accept/reject, whether code "sticks around," model switching, user churn

**Our equivalent for kernel optimization:**

| Signal | What It Captures | Reward Mapping |
|--------|-----------------|----------------|
| **Kernel accepted (keep)** | User/system kept this kernel as new best | Positive reward proportional to speedup delta |
| **Kernel rejected (discard)** | Kernel was correct but slower | Weak negative (tried but didn't improve) |
| **Compile error** | Generated invalid code | Negative reward |
| **Incorrect output** | Code compiles but wrong results | Strong negative reward |
| **Speedup magnitude** | How much faster than reference | Continuous positive signal |
| **Profiler diagnosis quality** | Did critic's recommendation lead to improvement? | Reward for critic accuracy |
| **Iterations to convergence** | How quickly did the system find a good kernel? | Efficiency signal |
| **Backend choice** | Triton vs CUDA, which produced better results | Backend-conditional signal |
| **Strategy effectiveness** | Which strategies from the Pareto frontier worked? | Strategy-level reward |

### Trace Schema

```python
@dataclass
class OptimizationTrace:
    # Session metadata
    session_id: str
    timestamp: str
    problem_id: str              # KernelBench level+id or custom
    problem_source: str          # "kernelbench" | "custom" | "production"
    hardware: str                # "H100" | "A100" | "L40S"
    backend: str                 # "triton" | "cuda"
    model_id: str                # which LLM was used
    openkernel_version: str
    
    # Results
    final_speedup: float
    final_correct: bool
    total_iterations: int
    total_tokens: int
    total_cost_usd: float        # LLM + Modal compute
    total_time_seconds: float
    
    # Iteration-level detail
    iterations: list[IterationTrace]
    
    # Strategy-level detail
    strategies_tried: list[str]
    strategies_succeeded: list[str]
    skills_retrieved: list[str]
    skills_created: list[str]

@dataclass
class IterationTrace:
    iteration: int
    intent: str                  # optimization intent from world model
    
    # LLM interaction
    generator_prompt: str
    generator_response: str
    critic_prompt: str | None
    critic_response: str | None
    
    # Generated code
    kernel_code: str
    
    # Evaluation
    eval_status: str             # correct | compile_error | incorrect | error
    speedup: float
    runtime_us: float
    ref_runtime_us: float
    
    # Profiling
    profile_data: dict           # full profiler output
    bottleneck_type: str | None
    critic_diagnosis: str | None
    
    # Decision
    decision: str                # keep | discard | retry
    
    # Cost
    tokens_used: int
    llm_cost_usd: float
    modal_cost_usd: float
    latency_seconds: float
```

### Storage Format

**Parquet** — columnar, compressed, efficient for ML training pipelines. Hugging Face's native format.

```
traces/
├── raw/                         # Raw traces as they come in
│   ├── 2026-04/
│   │   ├── session_abc123.parquet
│   │   ├── session_def456.parquet
│   │   └── ...
│   └── 2026-05/
├── processed/                   # Filtered + formatted for training
│   ├── training_pairs_v1.parquet    # (prompt, response, reward) tuples
│   ├── strategy_rewards_v1.parquet  # strategy effectiveness data
│   └── critic_accuracy_v1.parquet   # critic diagnosis vs actual outcome
└── metadata/
    ├── schema.json              # Trace schema version
    └── stats.json               # Aggregate statistics
```

### Hugging Face Dataset Upload

```python
from datasets import Dataset
import pyarrow.parquet as pq

# Upload traces as HF dataset
dataset = Dataset.from_parquet("traces/processed/training_pairs_v1.parquet")
dataset.push_to_hub(
    "openkernel/optimization-traces",
    private=True,  # Private until we decide on open-sourcing
    token=hf_token
)
```

Dataset organization on HF Hub:
```
openkernel/optimization-traces          # Full traces (private)
openkernel/kernelbench-results          # Public KernelBench results
openkernel/optimized-kernels            # Best kernels per problem (public)
openkernel/skill-library                # Optimization skills (public)
```

### Privacy & Opt-In

Following Cursor's model:
- **kernel code product**: Opt-in telemetry. Users explicitly consent to trace sharing.
- **No kernel source code shared without consent** — traces can be anonymized (strip custom kernel code, keep only KernelBench problem IDs + metrics)
- **Aggregated signals always collected**: speedup, correctness rate, iterations — no PII
- **Full traces**: opt-in, used for kernelgen training

---

## 2. Optimized Kernels

Best kernels produced by openkernel, organized for reuse and comparison.

### Storage on Hugging Face

```
openkernel/optimized-kernels/
├── kernelbench/
│   ├── L1/
│   │   ├── problem_001/
│   │   │   ├── triton/
│   │   │   │   ├── best_kernel.py       # Best Triton kernel
│   │   │   │   ├── metadata.json        # speedup, hardware, model, iterations
│   │   │   │   └── optimization_log.json # abbreviated optimization history
│   │   │   └── cuda/
│   │   │       ├── best_kernel.cu
│   │   │       └── metadata.json
│   │   ├── problem_002/
│   │   └── ...
│   ├── L2/
│   ├── L3/
│   └── L4/
└── custom/                      # User-submitted optimized kernels (future)
```

### Download for reuse

```python
from huggingface_hub import hf_hub_download

kernel_path = hf_hub_download(
    repo_id="openkernel/optimized-kernels",
    filename="kernelbench/L1/problem_023/triton/best_kernel.py"
)
```

This enables:
- Kernel engineers can browse optimized kernels as starting points
- Comparison between openkernel's results and other systems
- Community contributions (future: submit your own optimized kernels)

---

## 3. Models (kernelgen-1+)

### Hugging Face Model Hub

```
openkernel/kernelgen-1                  # First post-trained model
openkernel/kernelgen-1-triton           # Triton-specialized variant
openkernel/kernelgen-1-cuda             # CUDA-specialized variant
```

### Training Pipeline (Future)

Following Cursor's real-time RL pattern:

```
1. Collect traces from kernel code users (opt-in)
2. Filter for quality: top 10% speedup traces, correct kernels only
3. Format as instruction-following pairs:
   - Input: reference code + hardware + profiler feedback
   - Output: optimized kernel code
   - Reward: speedup achieved
4. Training phases:
   a. Continued pretraining on kernel code corpus (KernelBook 18K pairs + our traces)
   b. SFT on best (input, output) pairs
   c. GRPO/DPO with speedup as verifiable reward
5. Deploy new checkpoint → test on KernelBench → if better, ship
6. Repeat (target: new checkpoint every 1-2 weeks with enough data)
```

### Model Download in openkernel

```python
from openkernel.llm import load_model

# BYOM — user's own model
model = load_model("claude-sonnet-4")

# Or use kernelgen-1 (once available)
model = load_model("openkernel/kernelgen-1", source="huggingface")
```

---

## 4. Skill Library Distribution

Optimization skills are lightweight JSON — stored in the repo and also on HF Hub for versioning.

```python
from huggingface_hub import snapshot_download

skills_dir = snapshot_download(
    repo_id="openkernel/skill-library",
    repo_type="dataset"
)
```

Skills are versioned — as the skill library grows from KernelBench runs and user contributions, new versions are pushed to HF.

---

## 5. KernelBench Results Dataset

Public dataset of all our KernelBench sweep results:

```
openkernel/kernelbench-results/
├── sweeps/
│   ├── L1_v1.parquet            # First L1 sweep results
│   ├── L2_v1.parquet
│   └── ...
├── comparisons/
│   ├── vs_cudaforge.json
│   ├── vs_kernel_smith.json
│   └── ...
└── README.md                    # Methodology, hardware, model configs
```

Columns: `problem_id, level, backend, model, iterations, final_speedup, correctness, cost_usd, time_seconds, fast_p_1, fast_p_1_5, fast_p_2`

This is the credibility dataset — public, reproducible, comparable.
