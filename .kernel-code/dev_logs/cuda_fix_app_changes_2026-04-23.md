# CUDA contamination fix — `modal_infra/app.py` changes

**Date:** 2026-04-23
**Owner:** modal-fix (team `phase-1-cuda-contamination`, task #1)
**Target:** stop intra-container CUDA-context poisoning from leaking across
unrelated KernelBench evals; must be generalizable across all 208 problems.

## Summary of changes

Three generalizable changes to `modal_infra/app.py`. No op-type special
casing. Hot path (no `cuda_error`) is unchanged in cost.

### Change A — Conditional container death on `cuda_error`

- **New module-level helper `_maybe_schedule_container_death(result)`**
  (`modal_infra/app.py:81-101`). If
  `OPENKERNEL_DIE_ON_CUDA_ERROR != "0"` (default on) AND the result dict
  has `status == "cuda_error"`, schedule `os._exit(1)` ~2s later via
  `threading.Timer(2.0, lambda: os._exit(1)).start()`. The timer lets
  Modal finish serializing the result dict back to the client before the
  container dies; Modal transparently spins a fresh container for the
  next call.
- Called in every GPU entry point right before `return result`:
  - `eval_kernel_on_gpu` — L40S (`modal_infra/app.py:436-448`, call at line 447)
  - `eval_kernel_h100` — H100 (`modal_infra/app.py:452-464`, call at line 463)
  - `eval_kernel_a100_80gb` — A100-80GB (`modal_infra/app.py:468-480`, call at line 479)
  - `eval_kernel_a100_40gb` — A100-40GB (`modal_infra/app.py:484-496`, call at line 495)
  - `EvalWorker.run` — class variant (`modal_infra/app.py:1358-1379`, call at line 1378)
- **Kill switch:** set `OPENKERNEL_DIE_ON_CUDA_ERROR=0` in the container
  env to disable the behavior for debugging. Default is "1" (on).
- `_eval_kernel_impl` was deliberately **not** modified — it can be
  called outside a Modal container (tests, local dev), so killing the
  parent there would terminate the wrong process. Only the GPU-function
  entry points (which always run inside a Modal container) schedule the
  death.

### Change B — Triton JIT cache isolation per call

- **Parent side** (`_eval_kernel_impl`, `modal_infra/app.py:335-339`):
  before spawning the subprocess, compute a unique per-call Triton cache
  dir `/tmp/triton_cache_<pid>_<ns>` and set
  `os.environ["TRITON_CACHE_DIR"]` to it. The spawn child inherits the
  env, so its first `import torch` / `import triton` sees the isolated
  dir.
- **Child side** (`_eval_kernel_core`,
  `modal_infra/app.py:163-166` + cleanup in finally at
  `modal_infra/app.py:237-238`): reads `os.environ["TRITON_CACHE_DIR"]`
  and `shutil.rmtree`s it in the finally block after eval completes. No
  stale JIT state can leak into the next eval.

### Change C — Tempdir cleanup

- `_eval_kernel_core` previously called `tempfile.mkdtemp(prefix=
  "openkernel_eval_")` but never removed it, so `/tmp` grew
  monotonically for the life of the container.
- Wrapped the body of `_eval_kernel_core` in `try/finally`
  (`modal_infra/app.py:183-238`) and added
  `shutil.rmtree(tmpdir, ignore_errors=True)` (line 236) + the
  Triton-cache rmtree (lines 237-238) in the finally block.

## Supporting edits

- Added `shutil` (line 15) and `threading` (line 17) to the top-level
  imports in `modal_infra/app.py`.

## Deploy result

```
$ modal deploy modal_infra/app.py
✓ Created objects.
├── 🔨 Created mount /Users/juliansaks/Desktop/code/autokernel/modal_infra/app.py
├── 🔨 Created function eval_kernel_on_gpu.
├── 🔨 Created function eval_kernel_a100_40gb.
├── 🔨 Created function health_check.
├── 🔨 Created function EvalWorker.*.
├── 🔨 Created function eval_kernel_a100_80gb.
└── 🔨 Created function eval_kernel_h100.
✓ App deployed in 4.144s! 🎉

View Deployment: https://modal.com/apps/juliansaks/main/deployed/openkernel-eval
```

App: `openkernel-eval` (workspace `juliansaks`, environment `main`).
Deployment URL: <https://modal.com/apps/juliansaks/main/deployed/openkernel-eval>.

## Failure-mode test plan (for task #2 / #4)

1. **Happy-path regression:** send a known-good kernel through
   `eval_kernel_on_gpu` twice. Both calls succeed; container should be
   reused (timer never fires on `status=correct`).
2. **CUDA-poisoning isolation test:** craft a minimal kernel that
   triggers illegal-memory-access (e.g. out-of-bounds global write),
   followed immediately by a trivially-correct kernel. With the fix
   deployed, call #2 should return `status=correct` (fresh container),
   not a wedged `cuda_error` on unrelated code.
3. **Kill-switch behavior:** set `OPENKERNEL_DIE_ON_CUDA_ERROR=0` in
   the container env and re-run the poisoning test; container should
   NOT die, and subsequent evals may still see the poisoned context
   (confirming the gate works).
4. **Tempdir/Triton-cache cleanup:** exec into a warm container after
   a few evals; verify `/tmp/openkernel_eval_*` and
   `/tmp/triton_cache_*` are gone.
5. **Serialization timing:** confirm `cuda_error` result dicts reach
   the client (bridge.py) BEFORE the container dies. The 2.0s timer
   provides ample headroom; lower it only if cold-start cost becomes a
   concern and Modal's serialization proves faster.

## Constraints honored

- No op-type special-casing (fix applies to all 208 problems).
- Hot path unchanged: the `Timer` is only scheduled when
  `status=="cuda_error"`; successful/incorrect/compile-error/timeout
  paths never trigger container death.
- Env-var kill switch in place (`OPENKERNEL_DIE_ON_CUDA_ERROR=0`).
- `threading.Timer(2.0, os._exit)` guarantees `os._exit` fires AFTER
  Modal has serialized the return value back to the caller.
- No client-side (`bridge.py`) changes — that audit is task #3.
- No concurrency-knob changes.
