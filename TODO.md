# TODO

## High Priority

### ~~Deterministic Gating Module (from KernelMem)~~ IMPLEMENTED
`kernel_code/optimization_gate.py` — ports KernelMem's `machine_check_ver2.py` rules. Includes `is_removable_kernel()` (identity/memcpy/view-only detection), `validate_smem_budget()` (tile sizing vs GPU SMEM limits), `check_coalescing_first()` (streaming kernels must coalesce before SMEM tiling), `classify_headroom()` (Tier-H/M/L), `get_allowed_methods()` (bottleneck+structure-aware method allowlists). Main entry: `run_optimization_gate()`. Wired into `kernel_agent_bridge.py` — removable kernels are skipped before optimization starts.

### ~~One-Method Discipline (from KernelMem)~~ IMPLEMENTED
`validate_single_method()` in `optimization_gate.py` enforces exactly one optimization method per round. `OptimizationPrescription.primary_method` is a `str` (not list). Detects multi-method signals ("additionally", "also apply") and history de-duplication (same method relabeling).

### ~~Inline Profiling Coupling (from KernelMem)~~ IMPLEMENTED
`auto_optimizer.py` now creates `ProfileMetrics` from Modal eval profile data after each round and sets `OPENKERNEL_INLINE_PROFILE` env var with fresh metrics (BW%, Compute%, Occupancy, OI) for the next round. `AutoResult.round_history` uses structured `OptimizationLog.to_dict_list()`.

### ~~Hardware-Specific Skill Templates (from HF Skills)~~ IMPLEMENTED
Three GPU-specific skill JSON files in `data/skills/`: `hw_h100_optimization.json` (132 SMs, 192KB SMEM, 3350 GB/s HBM3), `hw_a100_optimization.json` (108 SMs, 164KB SMEM, HBM2e), `hw_l40s_optimization.json` (142 SMs, 100KB SMEM, GDDR6X). Each includes concrete block sizing, grid params, bandwidth specs, and pitfalls.

### ~~Integration Pattern Library (from HF Skills)~~ IMPLEMENTED
Two integration skill JSON files: `integration_diffusers.json` (module-patching closures for diffusers pipelines with code template) and `integration_transformers.json` (patching LlamaRMSNorm/Qwen2RMSNorm/etc. with variance_epsilon handling). Both include pitfalls (e.g., use `type()` not `isinstance()`, inject before CPU offload).

### ~~Pitfall Field on Skills (from HF Skills)~~ IMPLEMENTED
Added `pitfalls: list[str]` to `OptimizationSkill` dataclass in `skill_library.py`. `to_context_string()` renders pitfalls as bulleted warnings in LLM prompts. All new skill files include populated pitfalls.

### ~~Formalized Optimization Log (from Arxiv 2509.07506)~~ IMPLEMENTED
`kernel_code/optimization_log.py` — typed dataclasses replacing ad-hoc dict-based round history. `OptimizationRound` with `to_context_string()` for LLM injection and `to_dict()` for serialization. `OptimizationLog` accumulates rounds, tracks best speedup, provides `get_method_history()` for deduplication. `ProfileMetrics` with `from_modal_profile()` factory. `RoundStatus` enum. Integrated into `MetaOptimizer`.

### ~~Explicit Planning Agent (from Arxiv 2509.07506)~~ IMPLEMENTED
`openkernel/agents/planner.py` — sits between Critic (bottleneck ID) and Generator (kernel code). `Planner.plan()` produces `PlanDiagnostic` with single-method optimization plan, numbered code action checklist, evidence, expected improvements. Enforces one-method discipline, takes allowed_methods from deterministic gate, deduplicates against history.

### ~~Fix Plot A: worker speedups not captured in round JSON~~ RESOLVED
Workers using Modal eval (OPENKERNEL_USE_MODAL=1) write speedup data to round JSON stdout field via `_run_remote_eval()`. The bridge's `_poll_workers()` now scans ALL round files for best speedup (not just latest), correctly parsing "Speedup: X.XXx" from stdout. Plot A data flow is functional when Modal eval is enabled.

### ~~Fix chart persistence across autopilot rounds~~ VERIFIED WORKING
LiveOptimizationDisplay is created once per /autopilot session and passed through all rounds via MetaOptimizer → KernelAgentBridge. Worker speedup data accumulates in `_worker_speedups` dict across rounds. Not an issue.

### ~~Plot A X-axis time intervals~~ RESOLVED
X-axis now shows intermediate markers (30s, 60s, 90s, etc.) for runs > 60s with adaptive intervals.

### ~~Dynamic GPU selection for Modal eval~~ RESOLVED
Deployed separate Modal functions per GPU type: `eval_kernel_on_gpu` (L40S default), `eval_kernel_h100`, `eval_kernel_a100_80gb`, `eval_kernel_a100_40gb`. Bridge routes to correct function via `_GPU_FUNCTION_MAP`. Workers read `OPENKERNEL_GPU_TYPE` env var. KernelBench standardizes on L40S — non-L40S results include a note about comparability.

### ~~Real operational intensity for /roofline --me~~ IMPLEMENTED
Modal's `_collect_basic_profile()` now uses `with_flops=True` and `profile_memory=True` to measure actual FLOPs and bytes. Computes `operational_intensity = total_flops / total_bytes`. Profile dict includes `total_flops`, `total_bytes`, `operational_intensity`. Roofline view accepts `profile` param for measured OI placement. Hardware specs added for H100, A100-80GB, A100-40GB. Shows "measured OI" vs "estimated OI" label.

### ~~Wire KernelAgent optimization pipeline (6-agent loop)~~ SCAFFOLDED
Integration module at `kernel_code/integration/opt_pipeline.py`. Uses `create_optimizer(engine="kernel-agent-opt")` to select the hardware-aware pipeline with auto-fallback to Direct path when deps are unavailable. Requires kernel_perf_agent + local GPU + NCU. Settings: `engine: "kernel-agent-opt"`.

### ~~Fuser pipeline for complex multi-kernel problems~~ STUB CREATED
`FuserPipeline` stub + `needs_fusion()` detector in `opt_pipeline.py`. Raises `NotImplementedError` with instructions to copy Fuser/ from upstream KernelAgent. Detects Level 2+ problems via multi-module/multi-op heuristics.

## Medium Priority

### A/B Testing Framework (validate adoption changes with data)
See `AB_TESTING.md` for full experiment designs. Four experiments planned:
1. **Reflection vs Fixed Rounds** — 50 runs, measure best_speedup / rounds_to_success / cost_per_speedup_unit
2. **Evidence-Injected Skills vs Minimal Skills** — 30 runs, measure first_pass_correctness / avg_rounds_to_success
3. **Hardware-Specific Templates** — 30 runs across H100/A100/L40S, measure speedup / bandwidth utilization
4. **Deterministic Gating** — measure prescription_validity_rate / convergence_speed / false_optimization_rate
Phase rollout: Experiments 1 & 4 (Week 1), then 2 & 3 (Week 2).

### SOL Score Computation (from Cursor/NVIDIA)
Create `kernel_code/sol_metrics.py` — logarithmic 0-1 score where 0.5 = optimized baseline, 1.0 = theoretical hardware limit. `compute_sol_score()` uses hardware peak specs (from `optimization_gate.GPU_SPECS`) + profiling data (FLOPs, bytes, runtime). Add `sol_score` field to `ProfileMetrics` in `optimization_log.py`. This is the single most impactful metric change — raw speedup doesn't contextualize performance against hardware ceiling.

### Problem Classifier + Strategy Templates (from Cursor/NVIDIA)
Create `kernel_code/problem_classifier.py` — detect problem types (BF16 GQA, NVFP4 MoE, GEMM, Reduction, Fused ops, Attention) from reference code. Create per-type strategy templates in `data/strategies/`. Cursor showed that different problem types need fundamentally different optimization strategies (GQA: hardware instructions → scheduling → shape specialization; GEMM: general → arch-specific → R/W overlap → M-dim specialization).

### Round-Level Checkpointing (from Cursor/NVIDIA)
Save `best_kernel`, `best_speedup`, `round_history` to disk after each round. Resume from last checkpoint on restart. Cursor ran autonomously for 3 weeks — we need fault recovery for multi-hour/multi-day runs.

### Annotated SOL Trajectory Visualization (from Cursor/NVIDIA)
Extend Plot A: SOL score on Y-axis (not raw speedup), horizontal baseline (0.5) and ceiling (1.0) lines, annotated strategy inflection points from MetaOptimizer pivots. Cursor's trajectory plots are the most compelling visualization of optimization progress.

### Batch Optimizer for Multi-Problem Evaluation (from Cursor/NVIDIA)
Create `kernel_code/batch_optimizer.py` — run MetaOptimizer across a list of problem files, aggregate geomean speedup, % beating baseline, % exceeding 2x, median SOL. Cursor ran 235 problems; we need the same capability for evaluation.

### Wire Planner agent into optimization pipeline
`openkernel/agents/planner.py` is implemented but not yet called from the main optimization path. Wire Planner between Critic and Generator in the opt_pipeline flow: Critic.analyze() → Planner.plan() → Generator.generate(). Pass `allowed_methods` from `optimization_gate.run_optimization_gate()` into Planner.

### Wire deterministic gate into opt_pipeline (full integration)
`optimization_gate.run_optimization_gate()` is called for removable kernel detection in the bridge. Full integration into the optimization pipeline (SMEM budget checks, coalescing-first, allowed methods gating) requires wiring into `OptimizationWorker.optimize_kernel()` where NCU profiling data is available.

### ~~KernelLLM integration~~ IMPLEMENTED
OllamaProvider at `kernel_agent/ka_utils/providers/ollama_provider.py`. Connects to local Ollama server (configurable via `ollama_base_url` setting). Registered models: `KernelLLM`, `ollama/KernelLLM`, `ollama/codellama`, `ollama/deepseek-coder-v2`. Usage: `ollama pull hf.co/facebook/KernelLLM && /config set default_model KernelLLM`.

### popcorn-cli integration
GPU Mode's `popcorn-cli` supports scriptable competition submission. Add `/submit` command to submit optimized kernels to GPU Mode leaderboards programmatically.

### ~~Few-shot from KernelBook~~ IMPLEMENTED
`kernel_code/kernelbook.py` loads from `GPUMODE/KernelBook` via HuggingFace datasets library. Keyword-based matching finds relevant Triton examples. Cached locally at `~/.kernel-code/cache/kernelbook/`. Wired as Level 2 in bridge's `_find_few_shot_example()` (between exact solution and skill template match). Graceful fallback when datasets library unavailable.

### Plot A wall-time X-axis
Workers that start late (staggered spawn) begin mid-axis. Consider a synchronized-start mode or relative-to-worker-start time.

### Blackwell B200 GPU Support (from Cursor Kernels blog)
Add B200 to `optimization_gate.GPU_SPECS`: 148 SMs, 227KB SMEM per SM, ~8 TB/s HBM3e bandwidth, 4500 TFLOPS FP8, 2250 TFLOPS BF16. Add Blackwell-specific skill template (`hw_b200_optimization.json`) covering TMEM (128x512 tensor memory), tcgen05.mma, TMA (cp.async.bulk.tensor), 2-CTA clustering, warp specialization patterns. Cursor achieved 2750 TFLOPS MXFP8 and 1550 TFLOPS BF16 on B200 — these are the ceiling benchmarks to target.

### MXFP8 Quantization-Aware Optimization (from Cursor Kernels blog)
Add quantization-aware skills and gating rules. Key insight: MXFP8 quantization overhead can be 76% of matmul time if not fused properly. Add skill templates for: (1) fused quantization+kernel patterns (avoid separate quant pass), (2) 32-block scaling with FP8E4M3+FP8E8M0, (3) scale factor layout for hardware MMA instructions, (4) dequantization cost analysis (CUDA cores are 1/56th the speed of tensor cores on Blackwell). Add "Quant" problem tier to problem classifier.

### CUDA/PTX Generation Path (from Cursor Kernels blog)
Cursor wrote pure CUDA C + inline PTX with zero library dependencies for peak Blackwell performance. Our Triton-only path cannot reach hardware limits on Blackwell (TMEM, tcgen05.mma, TMA are PTX-level). Add CUDA C backend alongside Triton: (1) create `cuda_c` backend in problem.py, (2) add CUDA/PTX skill templates with warp specialization patterns, (3) enable A/B comparison: Triton vs CUDA C on same kernel.

### MoE-Specific Strategy Templates (from Cursor Kernels + Warp Decode blogs)
MoE layers are 53% of forward-pass time in large models. Add MoE-specific optimization strategies covering BOTH training and inference:

**Training path** (from Kernels blog): (1) grouped GEMM patterns (Fprop/Dgrad/Wgrad), (2) expert-wise L2 cache supergrouping (ThunderKittens pattern), (3) persistent grid patterns (one threadblock per SM), (4) quantization fusion into SwiGLU/GEGLU epilogues. Cursor achieved 3.5x MoE layer speedup.

**Inference/decode path** (from Warp Decode blog): (1) parallelism axis flip — organize by outputs not experts for small-batch decode, (2) warp-per-output pattern (each warp owns one neuron, streams weight rows, accumulates all top-k experts in registers), (3) stage elimination (8 stages → 3: route → warp compute → write), (4) buffer elimination (no gather buffer, no per-expert output buffer — everything in registers), (5) `__shfl_xor_sync` butterfly reduction (bypasses SMEM entirely). Cursor achieved 1.84x decode throughput + 1.4x accuracy improvement. Key insight: the agent must detect workload type (small-batch decode vs prefill/large-batch) and select the right parallelism axis automatically.

## Low Priority

### Delete dead code
- `_cmd_autopilot` method in shell.py (replaced by unified `/optimize`)
- `_step_quick_demo` and `_step_whats_next` in onboarding.py (absorbed by welcome)
- ~~`_print_post_optimization_summary` in shell.py~~ REMOVED
- ~~`_print_welcome` in shell.py~~ REMOVED

### ~~Episode-based context (Slate-inspired)~~ PARTIALLY ADDRESSED
`OptimizationRound` in `optimization_log.py` now stores structured per-round data (kernel code, profile metrics, status, strategy, bottleneck, method). `OptimizationLog.to_context_string()` formats trajectory for LLM injection. Remaining: pass full `optimization_log.to_context_string()` into generator prompts (currently only env var carries inline profile).

### Cross-model routing (Slate-inspired)
Use cheap model for test generation and profiling analysis, strong model for kernel code generation. Auto-select per task.
