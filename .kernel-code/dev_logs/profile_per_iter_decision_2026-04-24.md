# Profile Per-Iteration Decision

## Current State
Only round-winners receive full Modal GPU profile re-evaluation. Non-winning iterations show `profile: {}` (empty) in run logs. This design trades observability for speed.

## Why This Tradeoff Exists
Kernel+ runs up to **24 LLM+eval cycles**: 4 workers × 2 rounds × 3 iterations. Full profile re-eval on every iteration requires one Modal GPU call per iteration (~12s wallclock after input-cache fixes) × 24 = **~5 minutes extra per run**. Running only round-winners on Modal keeps the critical path tight while still gathering the data needed for winner validation.

## Cost of Changing It
- **Wallclock cost:** +~5 min per run (each of 24 iterations triggers Modal profile call)
- **Dollar cost:** Single-iteration GPU jobs at ~$0.50/min ≈ $2.50 extra per full run
- **Observability gain:** Non-winning iterations get real profile data instead of `{}`

## Recommendation
**Do NOT fix** unless a specific observability need emerges that justifies the 5-minute runtime penalty. If users find empty profiles annoying, hide those columns for non-winner rows in live_display.py (~5-line change) instead—cleaner UX without the compute cost.
