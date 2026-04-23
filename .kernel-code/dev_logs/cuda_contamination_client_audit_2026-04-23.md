# CUDA Container-Death Client-Side Audit (2026-04-23)

## Audit Context

After the modal-fix ships (container-death scheduling in `modal_infra/app.py`), when a child subprocess returns `cuda_error`:
1. The FIRST eval call returns a valid dict with `status=cuda_error`
2. A Timer schedules `os._exit(1)` ~2s later
3. The NEXT call transparently spins up a new Modal container

This audit verifies the client-side bridge correctly handles this behavior.

---

## Audit Question 1: Does `_modal_eval` correctly NOT retry on `cuda_error`?

**FINDING: ✓ CORRECT**

The function correctly avoids retrying `cuda_error` results. Here's the flow:

```python
# kernel_agent_bridge.py, lines 192-214
last_result: dict = {}
for attempt in range(_INFRA_RETRY_LIMIT + 1):  # _INFRA_RETRY_LIMIT=2
    try:
        result = eval_fn.remote(...)  # line 194
    except Exception as exc:
        result = {"status": "process_crashed", ...}  # line 203-210
    
    last_result = result if isinstance(result, dict) else {}
    if not _is_infra_error(last_result):  # line 213
        return last_result  # EARLY EXIT on non-infra
```

The `_is_infra_error()` check at line 156-167 is the gatekeeper:

```python
def _is_infra_error(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    return result.get("status") in _INFRA_STATUSES  # line 167
```

Where `_INFRA_STATUSES = frozenset({"timeout", "process_crashed"})` (line 149).

**Behavior**: When `cuda_error` is returned:
- Line 213: `_is_infra_error()` returns False (cuda_error ∉ {timeout, process_crashed})
- Line 214: Function IMMEDIATELY returns the cuda_error dict **without retrying**
- The error feeds to the critic as a kernel bug, not an infra issue ✓

**Correct by design**: The subprocess isolation in `modal_infra/app.py` ensures each fresh eval gets a clean CUDA context, so `cuda_error` genuinely indicates a kernel OOB bug, not infra contamination.

---

## Audit Question 2: Can Modal-level exceptions escape the try/except?

**FINDING: ✓ SAFE**

All Modal-layer exceptions are caught and wrapped. Line 201-210:

```python
except Exception as exc:  # Catches ALL exceptions
    result = {
        "status": "process_crashed",
        "correct": False,
        "speedup": 0.0,
        "runtime_us": 0.0,
        "ref_runtime_us": 0.0,
        "error": f"modal remote call failed: {type(exc).__name__}: {exc}",
    }
```

This catches:
- Container cold-start failures (Timeout, ConnectionError, Modal RPC errors)
- Connection resets mid-call (any exception from `eval_fn.remote()`)
- Network transients

The wrapped result has `status=process_crashed`, which IS in `_INFRA_STATUSES`, so it triggers a retry (up to 2 more times, then returns the error). ✓

**No gaps here**: The try/except is appropriately broad.

---

## Audit Question 3: Are there other `.remote()` call sites that bypass the retry wrapper?

**FINDING: 1 bypass site identified**

Grep found **2 total `.remote()` sites**:

1. **kernel_code/integration/kernel_agent_bridge.py:194** — wrapped in `_modal_eval()` with retries ✓
2. **kernel_code/shell.py:2483** — **BYPASS** — no retry wrapper

### shell.py bypass (lines 2479–2493):

```python
import modal
eval_fn = modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")
for i in range(5):
    r = eval_fn.remote(  # line 2483 — DIRECT CALL, NO RETRY WRAPPER
        kernel_source=kernel_source,
        reference_source=reference_source,
        eval_mode="fast",
        problem_format=pf,
    )
    if r.get("ref_runtime_us", 0) > 0:
        trials.append(r)
except Exception as exc:
    self._console.print(f"  [#ff6b80]Profile failed: {exc}[/#ff6b80]")
    return
```

**Gap**: The outer try/except (line 2479–2493) catches exceptions ONLY on loop entry/exit errors, not per-call failures. If any of the 5 `.remote()` calls raises (container rotation lag, network glitch), the exception stops the entire profile loop with a generic "Profile failed" message instead of retrying.

**Proposed patch**:

```python
def _safe_modal_eval_for_profiling(
    kernel_source: str,
    reference_source: str,
    problem_format: str,
    gpu_type: str = "L40S",
    max_retries: int = 2,
) -> dict | None:
    """Evaluate with infra retries, matching _modal_eval logic."""
    import modal
    from kernel_code.integration.kernel_agent_bridge import _is_infra_error
    
    fn_name = _GPU_FUNCTION_MAP.get(gpu_type, "eval_kernel_on_gpu")
    eval_fn = modal.Function.from_name("openkernel-eval", fn_name)
    
    for attempt in range(max_retries + 1):
        try:
            result = eval_fn.remote(
                kernel_source=kernel_source,
                reference_source=reference_source,
                eval_mode="fast",
                problem_format=problem_format,
                gpu_type=gpu_type,
            )
        except Exception as exc:
            result = {
                "status": "process_crashed",
                "correct": False,
                "speedup": 0.0,
                "runtime_us": 0.0,
                "ref_runtime_us": 0.0,
                "error": f"modal remote call failed: {type(exc).__name__}: {exc}",
            }
        
        result = result if isinstance(result, dict) else {}
        if not _is_infra_error(result):
            return result
    
    return None  # All retries exhausted

# Then in shell.py:
trials = []
try:
    from kernel_code.integration.kernel_agent_bridge import (
        _GPU_FUNCTION_MAP, _is_infra_error
    )
    for i in range(5):
        r = _safe_modal_eval_for_profiling(
            kernel_source=kernel_source,
            reference_source=reference_source,
            problem_format=pf,
            gpu_type="L40S",
        )
        if r and r.get("ref_runtime_us", 0) > 0:
            trials.append(r)
except Exception as exc:
    self._console.print(f"  [#ff6b80]Profile failed: {exc}[/#ff6b80]")
    return
```

---

## Audit Question 4: Consecutive cuda_error short-circuit?

**FINDING: Partial coverage — stopping module exists but doesn't track CUDA errors specifically**

### Stopping module (kernel_code/stopping.py):

A `StoppingController` already tracks `consecutive_errors` (line 104, default max=5):

```python
# lines 119–122
if status in ("compile_error", "error", "incorrect"):
    self._consecutive_errors += 1
else:
    self._consecutive_errors = 0

# lines 171–176
if self._consecutive_errors >= s.max_consecutive_errors:
    return StopDecision(
        stop=True,
        reason=f"{self._consecutive_errors} consecutive errors — likely a systematic issue",
        gate="errors",
    )
```

**However**: This catches "error" and "incorrect" status, but the current code path has two issues:

1. **KernelAgent doesn't use StoppingController** — The `KernelAgentBridge` (kernel_agent_bridge.py) does NOT call the stopping module. KernelAgent's internal worker loop (in `kernel_agent/worker.py`, `kernel_agent/manager.py`) has its own iteration counting but no CUDA-specific short-circuit.

2. **`cuda_error` is deliberately NOT treated as an error** — By design, `cuda_error` is returned to the critic as a kernel bug signal, not wrapped as an "error" status. So even if the stopping module were used, cuda_error results would NOT increment `consecutive_errors`.

### Check KernelAgent's worker loop:

```bash
grep -n "cuda_error\|consecutive" kernel_agent/worker.py kernel_agent/manager.py
```

→ No CUDA-error or consecutive-failure tracking in KernelAgent itself. KernelAgent assumes each eval result is independent; no short-circuit for repeated container rotations.

**Recommendation** (do NOT implement unless team-lead asks):

If K consecutive `cuda_error` on the same problem suggest a systemic kernel flaw, a threshold of **3 cuda_errors → skip problem** would save ~60s per bad problem (20s per container rotation × 3). This would go in `KernelAgentBridge.run()` or as a worker-side check in KernelAgent's loop.

Example threshold (pseudo-code):

```python
# In KernelAgentBridge or KernelAgent worker
max_consecutive_cuda_errors = 3
consecutive_cuda_errors = 0

for round in range(max_rounds):
    eval_result = _modal_eval(...)
    if eval_result.get("status") == "cuda_error":
        consecutive_cuda_errors += 1
        if consecutive_cuda_errors >= max_consecutive_cuda_errors:
            logger.warning("3+ consecutive cuda_errors on problem %s — giving up", problem_id)
            break  # Skip to next problem
    else:
        consecutive_cuda_errors = 0
```

Currently no such logic exists, so each worker blindly continues even after 10 container rotations on the same problem.

---

## Summary

| # | Finding | Status |
|---|---------|--------|
| 1 | `_modal_eval` does NOT retry on `cuda_error` | ✓ Correct |
| 2 | Modal-level exceptions are caught and retried | ✓ Safe |
| 3 | **shell.py profile path bypasses retry wrapper** | ⚠️ **Gap** |
| 4 | No consecutive-cuda-error short-circuit in KernelAgent | ℹ️ Documented (design choice) |

---

## Recommended Action

**Ship the modal-fix as-is** (container-death scheduling is safe on the client side). The `_modal_eval` wrapper is solid.

**For future work** (non-blocking, team-lead discretion):
1. Wrap shell.py's profiling `.remote()` calls with the same retry logic as `_modal_eval` (minor code reuse win).
2. Optional: Add consecutive-cuda-error short-circuit in KernelAgent worker if benchmarks show container-rotation churn on broken problems.
