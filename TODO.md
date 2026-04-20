# TODO

## High Priority

### Dynamic GPU selection for Modal eval
Currently Modal's `@app.function(gpu="L40S")` is hardcoded at deploy time. The KE's `default_gpu` setting only affects LLM prompts (context files, hardware specs), NOT the actual GPU the kernel runs on. To support H100/A100 eval:
- Option A: Deploy separate Modal functions per GPU (`eval_kernel_h100`, `eval_kernel_a100`)
- Option B: Use Modal's parameterized deployments (if supported)
- Option C: Single container with GPU auto-detection at runtime
- KernelBench standardizes on L40S — any non-L40S results aren't comparable to the leaderboard

### Real operational intensity for /roofline --me
Currently hinted from kernel filename (gemm→high OI, softmax→low OI). Accurate OI requires profiling bytes-moved ÷ FLOPs during eval. Could add `torch.profiler` instrumentation to Modal eval to capture this.

### Wire KernelAgent optimization pipeline (6-agent loop)
The forked KernelAgent has `opt_worker.py` + `opt_worker_component/` with Profiler→Judge→Analyzer→Orchestrator→OptManager→Benchmarking agents. Currently unused — only the Direct path (TritonKernelAgent) is wired. The optimization pipeline requires local GPU for ncu profiling.

### Fuser pipeline for complex multi-kernel problems
KernelAgent's `Fuser/` directory (not copied into our fork) handles complex PyTorch modules by decomposing them into fusable subgraphs. Needed for Level 2+ KernelBench problems.

## Medium Priority

### KernelLLM integration
facebook/KernelLLM (Llama 3.1 8B fine-tuned on 25K Triton pairs) scores 20.2 pass@1 on KernelBench L1 — better than GPT-4o (15). Self-hosted only (Ollama/llama.cpp). Add as model option with Ollama provider.

### popcorn-cli integration
GPU Mode's `popcorn-cli` supports scriptable competition submission. Add `/submit` command to submit optimized kernels to GPU Mode leaderboards programmatically.

### Few-shot from KernelBook
HuggingFace dataset `GPUMODE/KernelBook` has 25K PyTorch→Triton pairs. Pull relevant examples at optimization time for richer few-shot context than our 10 skills.

### Plot A wall-time X-axis
Workers that start late (staggered spawn) begin mid-axis. Consider a synchronized-start mode or relative-to-worker-start time.

## Low Priority

### Delete dead code
- `_cmd_autopilot` method in shell.py (replaced by unified `/optimize`)
- `_step_quick_demo` and `_step_whats_next` in onboarding.py (absorbed by welcome)
- `_print_post_optimization_summary` in shell.py (replaced by kernel profile)

### Episode-based context (Slate-inspired)
Instead of lossy meta-reflect summaries between rounds, pass structured "episodes" (kernel code + profiling data + error traces) forward. Would improve carry-forward quality.

### Cross-model routing (Slate-inspired)
Use cheap model for test generation and profiling analysis, strong model for kernel code generation. Auto-select per task.
