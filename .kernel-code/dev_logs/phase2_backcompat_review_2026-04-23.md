# Phase 2 Backward-Compatibility Audit Report
**Date:** 2026-04-23  
**Auditor:** review-backcompat (read-only)  
**Status:** ✅ CLEAN — All 6 questions pass

---

## Q1: Old `.kernel-code/runs/*.log` files parse via `run_log.py` parsers?

**Finding:** ✅ **PASS**

Pre-Phase-2 run logs parse cleanly and retain their original structure. Tested with actual log file `2026-04-19_12-40-48_optimize.log`:

```json
{
  "command": "",
  "timestamp": "2026-04-19T12:40:48.873251",
  "config": {...},
  "iterations": [],
  "best_speedup": 0.0,
  "total_cost": 0.0,
  "elapsed_seconds": 0.4,
  "stop_reason": "no correct kernel found"
  // Note: no best_sol field
}
```

- `run_log.py::end_run()` writes `best_sol` to the JSON summary as a **new additive field** (`kernel_code/run_log.py:174`)
- The `best_sol` field defaults to `0.0` when not explicitly passed (`kernel_code/run_log.py:130`)
- Old logs simply omit this field — they remain valid JSON and load without crashing
- `run_analysis.py` can ingest old logs via `_parse_run_log()` which uses regex + `json.loads()` — no schema validation that would reject missing keys

**Verdict:** No compatibility issue. Old logs remain readable.

---

## Q2: Checkpoints with speedup-only state resume cleanly?

**Finding:** ✅ **PASS**

`CheckpointState.from_dict()` handles missing fields gracefully with `.get()` defaults:

```python
# kernel_code/checkpoint.py:66-77
@classmethod
def from_dict(cls, d: dict[str, Any]) -> "CheckpointState":
    return cls(
        round_num=d.get("round_num", 0),
        best_speedup=d.get("best_speedup", 0.0),
        best_kernel=d.get("best_kernel", ""),
        total_cost_usd=d.get("total_cost_usd", 0.0),
        total_iterations=d.get("total_iterations", 0),
        current_strategy=d.get("current_strategy", "general optimization"),
        round_history=d.get("round_history", []),
        optimization_log=d.get("optimization_log", []),
        exploratory_round_done=d.get("exploratory_round_done", False),
    )
```

All fields use safe `.get(key, default)` — no crashes on missing keys. Pre-Phase-2 checkpoint dicts (without any SOL-related fields) load and restore without error.

**Verdict:** Pre-Phase-2 checkpoints resume perfectly.

---

## Q3: GoalSpec JSON/dict round-trip tolerates old data?

**Finding:** ✅ **PASS**

Three paths all work:

### 3a. `GoalSpec(**dict)` initialization
Old dict missing `target_sol`:
```python
old_dict = {
    "target_speedup": 2.0,
    "max_budget_usd": 5.00,
    # ...
    "iterations_per_round": 5,
    # Note: no target_sol
}
spec = GoalSpec(**old_dict)  # ✓ Uses default: target_sol=0.80
```

`GoalSpec` is a dataclass with a default: `target_sol: float = 0.80` (`kernel_code/goal_spec.py:32`). Python dataclass init handles missing kwargs gracefully using defaults.

### 3b. `asdict(GoalSpec)` round-trip
```python
from dataclasses import asdict
spec = GoalSpec(target_speedup=2.0, target_sol=0.80)
d = asdict(spec)  # ✓ Includes target_sol
spec2 = GoalSpec(**d)  # ✓ Loads cleanly
```

### 3c. No pickle/JSON serialization of GoalSpec itself
No `pickle.dumps(goal)` or `json.dumps(asdict(goal))` found in the codebase for GoalSpec. The checkpoint system (`checkpoint.py`) does **not** serialize GoalSpec; it only stores `CheckpointState`, which is orthogonal.

**Verdict:** GoalSpec dict round-trip is fully backward-compatible.

---

## Q4: Evidence tracker — pre-SOL skill-library records?

**Finding:** ✅ **PASS**

The evidence gate in `_passes_gate()` explicitly handles missing SOL via fallback:

```python
# kernel_code/evidence_tracker.py:178-192
def _passes_gate(round_dict: dict, prior_best_sol: float) -> bool:
    """Decide whether a winning round should be promoted to evidence.
    
    SOL-first: a SOL >= max(floor, prior_best) is a winner. If SOL is
    unavailable on this round, fall back to the legacy speedup floor so
    pre-SOL logs still produce evidence during the rollout.
    """
    sol = _extract_sol(round_dict)
    if sol > 0.0:
        required = max(_SOL_FLOOR, prior_best_sol)
        return sol >= required
    # SOL missing -> speedup fallback.
    speedup = round_dict.get("speedup", 0.0) or 0.0
    return speedup > _SPEEDUP_FLOOR  # _SPEEDUP_FLOOR = 1.02
```

**Pre-SOL record tested:**
```python
pre_sol_round = {
    "speedup": 1.05,
    "profile": {},  # No sol_score
    "status": "success"
}
passes = _passes_gate(pre_sol_round, prior_best=0.0)
# ✓ Result: passes=True (speedup 1.05 > 1.02 floor)
```

The filtering logic in `extract_and_update_evidence()` also handles this:
- Line 99: `winning_rounds = _extract_winning_rounds(optimization_log)` (uses SOL-first, speedup fallback)
- Line 107: Applies `_passes_gate()` which has explicit speedup fallback
- Line 106: `_load_prior_best_sol()` returns `0.0` gracefully on missing problem_id

**Verdict:** Evidence gate cleanly handles pre-SOL records via the speedup fallback path.

---

## Q5: `run_log.end_run()` signature remains additive?

**Finding:** ✅ **PASS**

Signature in `kernel_code/run_log.py:124-131`:
```python
def end_run(
    self,
    best_speedup: float = 0.0,
    best_kernel: str = "",
    stop_reason: str = "",
    total_cost: float = 0.0,
    best_sol: float = 0.0,  # ← NEW, but optional with default
) -> None:
```

All callers in `shell.py` use **positional or keyword arguments without `best_sol`**:
- Line 1901: `end_run(best_speedup=..., best_kernel=..., stop_reason=..., total_cost=...)`
- Line 3200: `end_run(best_speedup=..., best_kernel=..., stop_reason=..., total_cost=...)`
- Line 3795: `end_run(best_speedup=..., best_kernel=..., stop_reason=..., total_cost=...)`
- Line 4097: `end_run(best_speedup=..., best_kernel=..., stop_reason=..., total_cost=...)`

**None pass `best_sol` explicitly.** The new parameter defaults to `0.0`, and old calls work unchanged. The implementation at line 143-148 reconstructs best_sol from iterations if not provided:

```python
if not best_sol:
    for it in self._iterations:
        prof = it.get("profile") or {}
        s = float(prof.get("sol_score", 0.0) or 0.0) if isinstance(prof, dict) else 0.0
        if s > best_sol:
            best_sol = s
```

**Verdict:** Signature is fully backward-compatible (additive, optional, defaults to safe value).

---

## Q6: `MetaOptimizer.run(target_sol=None)` default path?

**Finding:** ✅ **PASS**

Signature and logic in `kernel_code/auto_optimizer.py:202-227`:

```python
def run(self, target_sol: float | None = None) -> AutoResult:
    """Run the autonomous optimization loop.
    
    Args:
        target_sol: Optional per-call override for the SOL stopping target.
            When provided and positive, it takes precedence over
            ``self._goal.target_sol`` for this run. Used by
            ``batch_optimizer.run_suite`` to pass the suite-level
            ``target_sol`` through without mutating the GoalSpec. Leave
            ``None`` to use ``goal.target_sol`` unchanged.
    """
    _clear_per_run_env()
    if target_sol is not None and target_sol > 0.0:
        self._effective_target_sol = float(target_sol)
    else:
        self._effective_target_sol = float(self._goal.target_sol)
```

**Sentinel logic verified:**
- Line 218: `if target_sol is not None and target_sol > 0.0:` — safe guard against `None`, `0.0`, and `False`
- Line 221: Falls back to `self._goal.target_sol` (never `None` — initialized at line 138)
- No caller passes empty dict, bool False, or other ambiguous sentinel values

**Verdict:** Sentinel logic is correct and safe. `target_sol=None` reliably means "use goal default."

---

## Summary

| Question | File:Line | Status | Finding |
|----------|-----------|--------|---------|
| **Q1** Old logs parse | `run_log.py:174` | ✅ PASS | Additive `best_sol` field, old logs valid |
| **Q2** Checkpoints resume | `checkpoint.py:66-77` | ✅ PASS | `.get()` defaults handle missing fields |
| **Q3** GoalSpec round-trip | `goal_spec.py:32` | ✅ PASS | Dataclass defaults; no direct serialization |
| **Q4** Evidence gate (pre-SOL) | `evidence_tracker.py:178-192` | ✅ PASS | Explicit speedup fallback path |
| **Q5** `end_run()` signature | `run_log.py:124-131` | ✅ PASS | Additive, optional, default=0.0 |
| **Q6** `MetaOptimizer.run()` sentinel | `auto_optimizer.py:202-221` | ✅ PASS | Safe `None` + `> 0.0` guard |

---

## Conclusion

✅ **CLEAN — No backward-compatibility issues found.**

All Phase 2 additions are strictly additive:
- New fields have sensible defaults (`best_sol=0.0`, `target_sol=0.80`)
- Old data paths use safe `.get()` + fallback logic
- Sentinel checks are explicit and correct
- No required mutations to pre-Phase-2 state

Pre-Phase-2 checkpoints, run logs, and GoalSpec dicts load and resume without modification.
