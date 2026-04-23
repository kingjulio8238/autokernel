# Phase 2 Dual-Display Surface Catalog — 2026-04-23

Task #4 (`dual-display`). Enumeration of every speedup-mentioning user-facing
surface across the five files named in the brief, classified into:

- **SOL-primary** — headline/primary metric replaced with SOL, speedup kept as
  secondary subtitle or debug column.
- **dual** — surface already shows both; verify formatting and fallbacks.
- **justified-debug-only** — internal state, labels, log-file schema, or
  computations that stay speedup-only because they're not part of the
  user-facing headline (still necessary for Phase 4 removal work).

Convention (applied where both metrics exist on a user-facing surface):

- Primary headline: `SOL 0.42 · 42% of hardware peak · memory-bound`
- Secondary subtitle (dimmed/smaller): `1.8x speedup vs reference`
- Fallback when SOL missing (sol_score absent or 0.0): `1.8x speedup · (SOL unknown)`

---

## kernel_code/summary_card.py

| Line | Current | Classification | Action |
|-----:|---|---|---|
| 52 | `best_speedup: float,` (fn parameter) | justified-debug-only | keep; add new `best_sol: float = 0.0` parameter alongside |
| 73 | `_metric_row("Best Speedup", ...)` | SOL-primary | replace metric row — emit "Best SOL" as primary row with peak-pct + bottleneck context; emit dim secondary "Best Speedup" row below |
| 87–91 | `speedups = [it.get("speedup", 0) ...]` + trajectory chart | justified-debug-only | trajectory chart stays speedup-indexed (visual sparkline already understood by KEs); do not change — Phase 4 will replace with SOL trajectory |
| 153 | Subtitle `{best_speedup:.2f}x` | SOL-primary | switch subtitle to SOL headline; show speedup in parens |

## kernel_code/kernel_profile.py

| Line | Current | Classification | Action |
|-----:|---|---|---|
| 12 | docstring `render_kernel_profile(speedup=2.14, ...)` | dual | update docstring — add `sol_score` example |
| 56 | `speedup: float = 0.0,` (fn parameter) | justified-debug-only | keep; rely on `profile["sol_score"]` inside to drive primary headline |
| 77–99 | Speedup headline block | SOL-primary | replace: emit "SOL {x:.2f} · {pct}% of peak · {bottleneck}" as primary headline when sol_score > 0; emit dim subtitle `{speedup:.2f}x speedup vs reference`; fallback to speedup-only when sol missing |
| 132–142 | Runtime before/after row | justified-debug-only | keep as-is; it's a runtime-delta row, complementary to SOL |

## kernel_code/live_display.py

| Line | Current | Classification | Action |
|-----:|---|---|---|
| 62 | comment "No SOL column" | dual | update comment to reflect new dual format |
| 75, 101 | per-iteration `{speedup:.2f}x` in history log | dual | add SOL prefix before speedup when sol_score > 0 on that iteration: `SOL 0.42 · 1.8x`; fall back to speedup-only when SOL missing |
| 133–144 | internal `_speedups`, `_best_speedup`, `_target_speedup` | justified-debug-only | keep — backing state for legacy target-progress bar (Phase 4 removes `target_speedup`) |
| 172–180 | `update_workers` speedup history | justified-debug-only | per-worker sparkline data, shown in `worker_plots.render_live_lines`; stays speedup-indexed this phase |
| 184–197 | `update_iteration(... sol_score=0.0)` signature | dual | already accepts sol_score; keep |
| 262–274 | BAND 1 status ctx2 line "best {x:.2f}x" | SOL-primary | make primary "best SOL {y:.2f}" when best_sol > 0; dim secondary "{x:.2f}x" line underneath or inline; fall back to speedup when no SOL |
| 276–295 | BAND 1 progress bar (target + SOL bars side-by-side) | dual | already has SOL bar — promote SOL bar to primary position, move target/speedup bar to secondary (right side, dim) |
| 311–314 | worker_plots.render_live_lines (speedup sparkline) | justified-debug-only | external plot module, not in scope |
| 330–332 | worker tick ` ✓{speedup:.2f}x` | justified-debug-only | per-worker completion marker; keep speedup (tight horizontal space, no room for dual) |

## kernel_code/run_log.py

| Line | Current | Classification | Action |
|-----:|---|---|---|
| 4 | docstring "speedups, statuses, intents, errors" | justified-debug-only | no change needed |
| 18, 20 | docstring examples | dual | add `profile={"sol_score": ...}` to example |
| 66, 92 | `speedup: float` parameter + JSON field | justified-debug-only | log schema stays (Phase 5/back-compat concern tracked in Task #5); we only adjust printed line format |
| 101–109 | Two-branch print: SOL+speedup when present, else speedup-only | dual | already correct format. Verify format matches the convention (prefix: `SOL 0.42 (1.8x)`). Only tweak wording for consistency with screens. |
| 126, 136, 153 | `best_speedup` in RESULT block + JSON summary | SOL-primary | promote RESULT block: when any iteration recorded a `sol_score`, print "Best SOL: {s:.2f}" primary + "Best speedup: {x:.2f}x" secondary. Fall back to speedup-only when no iteration had SOL. |

## kernel_code/sol_plots.py

| Line | Current | Classification | Action |
|-----:|---|---|---|
| 54 | docstring field list | dual | already SOL-first; no change |
| 70–72 | `0.5 * speedup` fallback when `sol_score` missing | **SOL-primary correctness** | **remove fallback** per brief: "confirm it reads sol_score from the profile dict, not 0.5 * speedup fallback". Emit 0 for the point (it will just not render) and when best_sol is 0, show a clear "SOL unavailable" note in the header. |
| 177 | label "Reference (1.0× speedup)" | dual | clarify: label the baseline line with what it is ("Reference · 1.0× speedup"); this is a reference-line label, not a primary-metric surface |
| 232, 243–252 | strategy timeline: reads `speedup` + `profile.sol_score` | dual | keep both; the fallback from sol→speedup in label rendering (308–312) is already SOL-primary |
| 284–296 | stage coloring uses both sol and speedup | dual | keep; reasonable heuristic |
| 308–312 | label: "Show SOL if available, otherwise speedup" | dual | already SOL-primary with speedup fallback — matches our convention |
| 330–363 | summary line uses `best_speedup` | SOL-primary | promote: when any stage has sol>0, use best_sol as the "Start/End/Best" primary; speedup becomes secondary. Fall back to speedup-only when no stage has sol. |

---

## Summary

- Surfaces touched (user-facing): **5 files** — summary_card, kernel_profile, live_display, run_log, sol_plots.
- User-facing headline/subtitle locations updated: **7** (one in summary_card panel + subtitle, one in kernel_profile Speedup block, two in live_display BAND 1, one in run_log RESULT block, two in sol_plots summary + fallback-removal).
- Internal/log-schema speedup fields **preserved** (Phase 4 removal, not this phase).
- `0.5 * speedup` fallback in `sol_plots.py` **removed** per brief.
- `target_speedup` in `live_display.py` NOT deprecated — stays as legacy target bar, rendered as secondary.
