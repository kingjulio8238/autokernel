# autokernel

Autonomous GPU kernel optimization via recursive improvement. Fork of autoresearch — same keep/discard loop, applied to CUDA/Triton kernels instead of ML training.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autokernel/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autokernel/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `reference.py` — the KernelBench problem (PyTorch reference implementation). **Do not modify.**
   - `prepare.py` — the fixed evaluation harness. **Do not modify.**
   - `kernel.py` — the file you modify. Your optimized kernel (ModelNew class).
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The evaluation compiles your kernel, checks correctness against the reference, and benchmarks runtime. You launch it simply as: `uv run python prepare.py`.

**What you CAN do:**
- Modify `kernel.py` — this is the only file you edit. Everything is fair game: Triton kernels, CUDA C++ extensions, tiling strategies, memory access patterns, tensor core usage, block dimensions, pipeline staging, vectorized loads, shared memory optimization.

**What you CANNOT do:**
- Modify `prepare.py` or `reference.py`. They are read-only.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness or correctness check.

**The goal is simple: maximize speedup over the PyTorch reference.** Speedup = ref_runtime / kernel_runtime. Higher is better. A speedup of 1.0 means you match PyTorch; above 1.0 means you beat it.

**Correctness is a hard gate.** Your kernel must produce outputs matching the reference within tolerance (fp32: atol=1e-4, rtol=1e-4). An incorrect kernel is always worse than a slow correct one. Get correctness first, then optimize.

**The first run**: Your very first run should always be to establish the baseline. The starter kernel.py wraps torch.matmul — it should be correct with ~1.0x speedup.

## Output format

Once the evaluation finishes it prints structured output like this:

```
status:correct
speedup:1.0234
runtime_us:4523.12
ref_runtime_us:4628.45
eval_seconds:15.2
```

Possible status values:
- `correct` — kernel compiles, passes correctness, performance measured
- `compile_error` — kernel failed to compile (error message follows)
- `incorrect` — kernel compiles but output doesn't match reference (max_diff follows)
- `error` — evaluation harness error (e.g. lock file issue, retry)

Extract the key metrics:
```
grep "^status:\|^speedup:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	speedup	runtime_us	ref_runtime_us	status	description
```

1. git commit hash (short, 7 chars)
2. speedup achieved (e.g. 1.2345) — use 0.0000 for crashes/errors
3. kernel runtime in microseconds — use 0.00 for crashes/errors
4. reference runtime in microseconds — use 0.00 for crashes/errors
5. status: `keep`, `discard`, `crash`, `incorrect`, or `compile_error`
6. short text description of what this experiment tried

Example:

```
commit	speedup	runtime_us	ref_runtime_us	status	description
a1b2c3d	1.0000	4628.45	4628.45	keep	baseline (torch.matmul passthrough)
b2c3d4e	1.0812	4281.30	4628.45	keep	naive Triton GEMM, BLOCK_M=128
c3d4e5f	0.0000	0.00	0.00	compile_error	tried BLOCK_K=64, type mismatch
d4e5f6g	0.0000	0.00	0.00	incorrect	shared memory race condition
e5f6g7h	1.0812	4281.30	4628.45	discard	BLOCK_N=256, register pressure, no gain
f6g7h8i	1.2340	3751.50	4628.45	keep	coalesced memory access + BLOCK_M=128
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autokernel/mar12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `kernel.py` with an optimization idea.
3. git commit
4. Run the evaluation: `uv run python prepare.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Parse the results: `grep "^status:\|^speedup:\|^runtime_us:\|^ref_runtime_us:" run.log`
6. If the grep output is empty or status is `error`, the run failed. Run `tail -n 50 run.log` to read the error and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. Decision:
   - **status=compile_error** → read the error, fix, retry (max 3 attempts per approach, then move on)
   - **status=incorrect** → read max_diff, fix numerical issue, retry (max 3 attempts)
   - **status=correct AND speedup > previous best** → KEEP. Advance the branch.
   - **status=correct AND speedup <= previous best** → DISCARD. `git reset --hard HEAD~1` to revert.
9. Repeat.

**Monotonic improvement**: The keep/discard mechanic guarantees your best speedup never decreases. This means you can try risky optimizations — if they fail, you just revert. Take risks.

**Iteration speed**: Each evaluation takes ~5-30 seconds (compile + benchmark). You can run hundreds of experiments per session. Move fast.

**Crashes**: If a run crashes, use your judgment: if it's a typo or missing import, fix and re-run. If the approach is fundamentally broken, log it, discard, and try something different.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — try different tiling strategies, switch between Triton and CUDA C++, try tensor cores, experiment with memory hierarchy, read the reference problem more carefully for optimization opportunities.

**NEVER SWITCH PROBLEMS**: You must optimize the problem that is currently in `reference.py`. Do NOT modify `reference.py`, do NOT run `setup_problem.py` to switch to a different problem, and do NOT give up on the current problem. If you are stuck, that means you need to try harder or try a fundamentally different approach — not a different problem. The human will switch problems between sessions if needed. Your job is to squeeze maximum speedup out of the one problem you've been given.

## Strategy guide

Rough progression for optimizing a kernel:

1. **Get correct first**: Start with torch.matmul passthrough or a naive kernel. Establish baseline.
2. **Basic Triton kernel**: Write a simple tiled kernel with conservative block sizes (BLOCK_M=64, BLOCK_N=64, BLOCK_K=32).
3. **Tune tile sizes**: Try different BLOCK_M/N/K combinations. Larger tiles = more data reuse but more register pressure.
4. **Memory coalescing**: Ensure memory accesses are coalesced (consecutive threads read consecutive addresses).
5. **Shared memory**: Use shared memory for data reuse within thread blocks.
6. **Tensor cores**: Use WMMA/MMA instructions for matrix operations (requires specific data layouts).
7. **Pipeline staging**: Overlap memory loads with computation using software pipelining.
8. **Vectorized loads**: Use 128-bit loads (float4) for better memory bandwidth utilization.
9. **Register optimization**: Minimize register pressure to maximize occupancy.
10. **Radical changes**: If stuck, try a completely different approach — different algorithm, CUDA C++ instead of Triton, different decomposition of the problem.

Each step is a separate experiment. One optimization per commit. The keep/discard loop handles the rest.
