# kernel+ Plan

Multi-phase roadmap to take the kernel optimization harness from "works on one problem" to "respectable on KernelBench." Living doc — update phase status and RESULTS column as work completes.

## Terminology

- **kernel+** — the agent harness (meta-optimizer + bridge + orchestrator + skill library). The research/capability layer.
- **kernel_code** — the TUI/CLI product layer that wraps kernel+. User-facing.
- **SOL** — Speed of Light = achieved / hardware peak. Bounded [0, 1]. Hardware-normalized, problem-normalized. The metric kernel engineers use internally.

## North-star metric

**Mean SOL on KernelBench L1** (primary), with secondary cuts by tier, P90, solved-rate @ SOL ≥ 0.3, geomean speedup, correctness rate.

Replaces vs-reference speedup as the primary optimization target. Speedup survives as a debug-column secondary metric during the transitional phase.

## Targets

| Milestone   | Mean SOL (L1) | Solved-rate @ SOL ≥ 0.3 | Notes                                       |
|-------------|---------------|-------------------------|---------------------------------------------|
| Week 1      | 0.25 – 0.35   | ≥ 20%                   | Beats naive baseline; approaches o1 rates   |
| Month 1     | 0.40 – 0.50   | ≥ 40%                   | Respectable; approaches Cursor's 0.56 median |
| Stretch     | ≈ 0.56        | ≥ 55%                   | Matches Cursor median; expand to L2         |

### Baselines (cite when reporting results)

- Cursor multi-agent kernels (H200, 235 proprietary problems): median SOL 0.56, peak 0.97. [blog](https://cursor.com/blog/multi-agent-kernels)
- OpenAI o1 on KernelBench: 10% L1, 24% L2, 12% L3 solved.
- DeepSeek R1 on KernelBench: 12% L1, 36% L2, 2% L3 solved.
- Stanford KernelBench leaderboard SOTA: <20% combined. [leaderboard](https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/)
- Sakana AI (retracted, then rebuilt robustly): [robust-kbench](https://github.com/SakanaAI/robust-kbench). Lesson: strict verification, no sandbox exploits, deterministic inputs, flag absurd speedups as suspicious.

## Cross-cutting execution pattern: work-team + review-team per phase

Every phase runs this loop:

```
1. Plan   → lead decomposes phase into independent tasks
2. Work   → spawn N work agents in parallel, one per task, into a team
3. Review → once work reports in, spawn M review agents in parallel (read-only, Explore type)
4. Gate   → review agents must all report "no blocking issues"
            If blocking issues: spawn fix agents, re-review, repeat
            If clean: phase closed, merge to main, update RESULTS row
5. Next   → advance to next phase
```

Rules:
- **Parallel-by-default**: work and review agents always run concurrently. Serial work only when there's a hard file-level conflict.
- **Read-only reviews**: review agents use the `Explore` subagent type. They produce reports, not edits.
- **Concrete review rubric**: every phase ships a written review rubric (see per-phase sections below) so reviewers aren't guessing what to check.
- **No phase advance without green review**: if reviews flag blocking issues, fix and re-review before moving on. Non-blocking improvements can be deferred with a TODO.
- **RESULTS log**: every completed phase adds a row to the RESULTS table at the bottom with mean SOL before/after, cost, elapsed wall time, and link to the run log.

---

## Phase 0 — SOL Instrumentation

**Status:** not started
**Estimate:** 3 days
**Depends on:** nothing
**Unlocks:** Phase 1, Phase 2

### Goal

SOL flows end-to-end from Modal eval → profile dict → run log → plots → UI. Every iteration record in `.kernel-code/runs/*.log` carries real `sol_score`, `compute_util`, `bandwidth_util`, `occupancy`. Plots stop using the `0.5 × speedup` fallback.

### Root cause from audit

- Modal's `_collect_basic_profile()` returns flops/bytes but not `sol_score`.
- `KernelAgentBridge` computes SOL in-memory for UI display only (never merges back to profile dict).
- `RunLogger.log_iteration()` accepts no profile param.
- JSON summaries and plots fall back to `sol = 0.5 × speedup`.

### Work agents (parallel, into team `phase-0-sol`)

- **`modal-sol`** — `modal_infra/app.py`: compute `sol_score` inside `_collect_basic_profile()` using hardware peaks already defined (L40S 183 TFLOPS / 864 GB/s; H100 989 / 3350; A100-80GB 312 / 2039; B200 2250 / 8000). Include in returned profile dict. Handle missing flops/bytes gracefully (SOL = 0.0, don't crash).
- **`logger-sol`** — `kernel_code/run_log.py`: extend `log_iteration()` to accept optional `profile: dict | None = None`; merge relevant keys (`sol_score`, `compute_util`, `bandwidth_util`, `occupancy`, `cache_efficiency`, `bottleneck_type`) into the iteration entry. Backwards-compatible: existing callers pass nothing, behave as before.
- **`plumbing-sol`** — `kernel_code/auto_optimizer.py` + `kernel_code/integration/kernel_agent_bridge.py`: thread per-iteration profile from `round_result["per_worker"]` (or equivalent) into `log_iteration()`. Ensure the bridge emits profile data (not just speedup) in its round result.

### Review agents (parallel, into team `phase-0-review`)

- **`review-correctness`** — Explore type. Check: does a known memory-bound op (e.g., LayerNorm on 4M elements) produce a SOL value in the expected range (0.3-0.7)? Does a compute-bound op (a GEMM if available) differentiate correctly? Walk the SOL formula in `sol_metrics.py` and verify against a hand-computed roofline.
- **`review-edge-cases`** — Explore type. What happens when:
  - Profile is missing entirely (kernel failed to compile)?
  - `ref_runtime_us` is 0? Division by zero in SOL calc?
  - Hardware unknown to the peak table? Does it fall back gracefully?
  - NaN or negative utilization reaches plots?
- **`review-regression`** — Explore type. Check: does a fresh `/optimize` run on `reference_layernorm.py` (backed-up old reference) still produce the expected output structure? Does `tests/test_workload_spec.py` still pass? Any existing tests in `tests/` for run_log or auto_optimizer — do they still pass?

### Review rubric

| Dimension         | Bar to clear                                                              |
|-------------------|---------------------------------------------------------------------------|
| Functionality     | A fresh run produces non-zero `sol_score` in the run-log JSON             |
| Numerical sanity  | LayerNorm SOL in [0.2, 0.8], GEMM SOL in [0.2, 0.95]                      |
| Edge cases        | Missing profile, zero runtime, unknown hardware all handled gracefully    |
| Backwards-compat  | Old run logs still parse; `log_iteration()` callers without profile work  |
| No regressions    | Phase 2 UX catalog's existing surfaces still render (no template errors)  |

### Gate decision

Once all 3 review agents report no blocking issues, Phase 0 is closed and Phase 1 can begin. Non-blocking issues get logged to the tail of this file as TODOs.

---

## Phase 1 — Benchmark Harness

**Status:** not started
**Estimate:** 4-5 days
**Depends on:** Phase 0
**Unlocks:** Phase 3 (all measurement-driven work)

### Goal

Run a full KernelBench L1 + L2 + GPU MODE sweep on demand / nightly. Store results in a leaderboard. Produce a one-line summary: `Mean SOL 0.31 on KernelBench L1 (42/100 solved, $187 cost, 1h 52m wall).`

### Suite

- KernelBench L1: 100 problems
- KernelBench L2: 100 problems
- GPU MODE: 8 problems (regression check on every commit to `kernel_code/`)
- Total: 208 problems
- Budget: $0.50/problem = $100-150/sweep
- Wall time target: ≤ 2 hours with parallelism

### Sub-phase DAG

Phase 1 is broken into four rounds. Each round gates on clean reviews before advancing. Within a round, all work agents run in parallel.

```
Round A (foundation, 3 parallel)
├── 1.0a kb-smoke
├── 1.0b schema
└── 1.0c problem-spec
    → review (3 agents)

Round B (loaders + storage, 5 parallel)
├── 1.1a kb-l1-loader
├── 1.1b kb-l2-loader
├── 1.1c gpumode-loader
├── 1.2a leaderboard-writer
└── 1.2b leaderboard-reader
    → review (3 agents)

Round C (runner + reporting, 4 parallel)
├── 1.3  batch-runner
├── 1.4a metrics
├── 1.4b markdown-report
└── 1.4c regression-detector
    → review (4 agents)

Round D (integration, sequential)
├── 1.5 dry-run (10-problem subset)
│   → review (2 agents)
└── 1.6 full-sweep (208 problems — the Phase 1 deliverable)
    → review (2 agents)
```

### Round A — Foundation (3 work agents parallel)

| ID    | Agent         | Deliverable | Acceptance |
|-------|---------------|-------------|------------|
| 1.0a  | `kb-smoke`    | `scripts/kb_smoke.py` — runs `kernelbench.dataset[0]` through `shell.py:_cmd_profile_ref`'s kernelbench branch on Modal. | Exit code 0. Prints SOL > 0, speedup, correctness for one L1 problem. |
| 1.0b  | `schema`      | `results/leaderboard/SCHEMA.md` + `results/leaderboard/_example.json`. Fields: `problem_id, problem_name, tier, hardware, date, kernel_hash, model, speedup, sol_score, compute_util, bandwidth_util, bottleneck_type, correct, cost_usd, elapsed_s, stop_reason, config_hash`. | Markdown readable; sample JSON parses; every field documented with type + example. |
| 1.0c  | `problem-spec`| `openkernel/benchmarks/problem_spec.py` — `ProblemSpec` dataclass: `id, name, tier, source, reference_source, workload_spec, expected_dtype`. | Import works; one unit test passes. |

**Round A review rubric:**
- `review-smoke`: can kb-smoke be trusted as a regression gate? Does it catch real breakage or just import errors?
- `review-schema`: field coverage complete? Any future reporting need we can't serve?
- `review-spec`: interface flexible enough to hold kernelbench AND gpumode without forcing awkward coercions?

**Round A gate:** `kb-smoke` passes on Modal with current harness. If it fails, blockers must be fixed before Round B (otherwise batch-runner builds on quicksand).

**Round A CLOSEOUT (2026-04-21):**

Delivered:
- `scripts/kb_smoke.py` — L1 matmul passthrough on L40S returns `status=correct, speedup=0.99, sol_score=0.495`. Generic `ModelNew = Model` aliasing works for any KB op.
- `results/leaderboard/SCHEMA.md` + `_example.json` — 22 fields. Required additions after review: `schema_version` (at record top, `"1.0"`) + `kernel_source_path` (for Sakana-style audits) + reader contract section + Changelog.
- `openkernel/benchmarks/problem_spec.py` + `__init__.py` — frozen dataclass `ProblemSpec(id, name, tier, source, reference_source, workload_spec, expected_dtype)` with `__post_init__` runtime validation (`_VALID_TIERS`, `_VALID_SOURCES`, non-empty id + reference_source).
- `tests/test_problem_spec.py` — 7/7 tests pass (defaults, all-fields, frozen, invalid tier, invalid source, empty id, empty reference_source).

Cleanup (5 additional work agents, from Deferred):
- `openkernel/kernelbench/problems.py` — `load_problem` now uses `dataset.get_problem_by_id(id)` + `source="huggingface"`. Dropped stale `len(dataset)` guard.
- `kernel_code/shell.py:2468` — kernelbench passthrough replaced matmul-hardcode with generic `ModelNew = Model`. Now works for softmax, layernorm, conv, etc.
- `scripts/smoke_test_sol.py` + `scripts/kb_smoke.py` — tightened `bool(profile)` to `profile.get("sol_score", 0.0) > 0.0`. Silent all-zero profiles now fail the smoke test.

Gate outcome: both reviews CLEAN. 3 non-blocking items logged to Deferred (whitespace-only reference_source, workload_spec key conventions, config_hash algorithm). Round B unblocked.

### Round B — Loaders + Storage (5 work agents parallel)

| ID    | Agent               | Deliverable | Acceptance |
|-------|---------------------|-------------|------------|
| 1.1a  | `kb-l1-loader`      | `openkernel/benchmarks/kb_l1.py:load_l1() -> list[ProblemSpec]` | `len(load_l1()) == 100`; each spec has reference_source populated. |
| 1.1b  | `kb-l2-loader`      | `openkernel/benchmarks/kb_l2.py:load_l2() -> list[ProblemSpec]` | `len(load_l2()) == 100`. |
| 1.1c  | `gpumode-loader`    | `openkernel/benchmarks/gpumode_loader.py:load_gpumode() -> list[ProblemSpec]` — the 8 problems under `data/benchmarks/gpumode/`. | `len(load_gpumode()) == 8`; reference.py source captured for each. |
| 1.2a  | `leaderboard-writer`| `openkernel/benchmarks/leaderboard_writer.py:write_record(spec, result) -> Path`. Atomic temp+rename under `results/leaderboard/YYYY-MM-DD/`. | Two concurrent writes don't corrupt; crash mid-write leaves no partial file. |
| 1.2b  | `leaderboard-reader`| `openkernel/benchmarks/leaderboard_reader.py`: `load_all()`, `filter(tier=, date_range=, hardware=)`, `latest_per_problem()`. | 3 unit tests on fake data pass. |

**Round B review rubric:**
- `review-loaders`: all 208 problems across the 3 loaders produce valid `ProblemSpec` instances; no missing `reference_source`, no dupes.
- `review-writer`: atomic under concurrent access; schema-valid records only (validates against 1.0b schema).
- `review-reader`: filter/sort correctness on fake data; handles empty/missing dates gracefully.

**Round B gate:** all 208 loaders + both storage APIs pass their unit tests. Writer produces schema-compliant records.

### Round C — Runner + Reporting (4 work agents parallel)

| ID    | Agent                 | Deliverable | Acceptance |
|-------|-----------------------|-------------|------------|
| 1.3   | `batch-runner`        | `kernel_code/batch_optimizer.py:run_suite(specs, hardware, budget_per_problem, concurrency)`. Parallel execution, per-problem budget, resumable (skip if today's leaderboard row exists), graceful budget overage. | 10-spec fake run produces 10 leaderboard rows; mid-run Ctrl-C leaves clean state; re-run skips completed. |
| 1.4a  | `metrics`             | `scripts/score_suite.py` CLI. Computes mean SOL (correct only), P50, P90, solved-rate @ SOL ≥ 0.3, correctness rate, geomean speedup, per-tier cuts. | `--date YYYY-MM-DD --hardware L40S` emits JSON scores. |
| 1.4b  | `markdown-report`     | `scripts/emit_report.py` → `results/reports/YYYY-MM-DD.md`. Headline + per-tier + baseline comparison (Cursor 0.56 median, o1 10% L1, R1 12% L1, Stanford SOTA <20%). | Report reads well, cites baselines, includes day-over-day trend if available. |
| 1.4c  | `regression-detector` | `scripts/check_regressions.py`. Compares today vs last N days (default 7), flags problems where SOL dropped > 0.05 from N-day max. | Exit code 1 if regressions found (CI-hookable); markdown list of regressions emitted. |

**Round C review rubric:**
- `review-runner`: budget/concurrency/resume correctness — highest-risk component. Ctrl-C mid-suite recoverable? Budget overage graceful?
- `review-metrics`: formulas match KernelBench paper definitions. Per-tier aggregation correct. SOL-only (correct kernels) not polluted by incorrect.
- `review-report-quality`: readable for a KE skimming it. Actionable. Baselines cited.
- `review-regression-alerts`: false-positive rate acceptable on synthetic noise data.

**Round C gate:** batch-runner drives a 10-spec fake suite end-to-end, metrics + report + regression all run off the produced leaderboard without error.

### Round D — Integration (sequential)

| ID   | Agent       | Deliverable | Acceptance |
|------|-------------|-------------|------------|
| 1.5  | `dry-run`   | Run 10-problem subset on Modal L40S (3 L1 + 3 L2 + 4 gpumode), write leaderboard, emit report. | Wall time ≤ 15min; cost ≤ $10; all 10 rows in leaderboard; report markdown committed at `results/reports/`. |
| 1.6  | `full-sweep`| Run all 208 problems on Modal L40S with $0.50/problem cap. Emit report. Update `KERNEL_PLUS_PLAN.md` RESULTS row with Phase 1's first real mean SOL number. | Full sweep completes; report committed; RESULTS row populated. |

**Round D.5 review rubric (before 1.6):**
- `review-dry-run-correctness`: silent failures? Did any problem report "success" but produce no kernel?
- `review-cost-model`: extrapolating from dry-run, full-sweep cost ≤ $300? If higher, reduce concurrency or tighten per-problem budget.

**Round D.6 review rubric (final Phase 1 gate):**
- `review-first-baseline`: is our number defensible? Any reward-hacking risk (Sakana lesson — absurd speedups should be flagged and manually audited)?
- `review-report-citable`: format matches Cursor / KernelBench publication so side-by-side comparison is straightforward.

### Agent count per round

| Round | Work | Review | Total tasks |
|-------|------|--------|-------------|
| A     | 3    | 3      | 6           |
| B     | 5    | 3      | 8           |
| C     | 4    | 4      | 8           |
| D.5   | 1    | 2      | 3           |
| D.6   | 1    | 2      | 3           |
| **Total** | **14** | **14** | **28** |

Never more than 5 work agents concurrent per round → file conflict surface is low.

### Critical path

- **Round A blocker:** 1.0a `kb-smoke`. If KernelBench integration is broken on current harness, all downstream rounds are blocked until fixed. Fast-fail early.
- **Round B blocker:** 1.2a `leaderboard-writer`. Batch-runner can't land without a writer to call.
- **Round C blocker:** 1.3 `batch-runner`. Dry-run can't land without it.

### Gate criteria (Phase 1 closed)

| Dimension          | Bar                                                                    |
|--------------------|------------------------------------------------------------------------|
| Completion         | Full sweep finishes in ≤ 2h, cost ≤ $300                              |
| Reproducibility    | Same-seed re-runs within 5% SOL variance                              |
| No cheating        | Random sample of top-5 kernels audited and clean                      |
| Summary quality    | Scoring script produces ready-to-cite markdown matching baseline format |
| RESULTS row        | Phase 1 row in this doc populated with mean SOL + solved-rate         |

---

## Phase 2 — SOL-as-metric transition

**Status:** not started
**Estimate:** 3 days (parallel with Phase 1)
**Depends on:** Phase 0
**Unlocks:** Phase 4

### Goal

`GoalSpec` gains `target_sol`. User can type "target SOL 0.8" and the run stops on SOL, not speedup. Evidence gate uses SOL. Both metrics display side-by-side during the transitional period.

### Work agents (parallel, into team `phase-2-sol-ux`)

- **`goal-spec`** — `kernel_code/goal_spec.py` + `kernel_code/goal_parser.py`: add `target_sol: float = 0.80` field. Parse `"SOL 0.8"`, `"0.8 SOL"`, `"80% SOL"`. Keep `target_speedup` readable for backwards-compat.
- **`stopping-rule`** — `kernel_code/auto_optimizer.py`: target-reached fires on `sol >= target_sol` when SOL is available; falls back to `speedup >= target_speedup` if not. Overshoot-exploratory logic mirrored for SOL.
- **`evidence-gate`** — `kernel_code/evidence_tracker.py`: replace `speedup > 1.02` filter with `sol > 0.50 AND sol >= prior_best_sol_for_this_problem`. Requires a "prior best" lookup from leaderboard (Phase 1 dependency — can stub initially).
- **`dual-display`** — `kernel_code/summary_card.py`, `kernel_profile.py`, `live_display.py`, `run_log.py`, `sol_plots.py`: dual-metric display for transition period. SOL primary, speedup secondary.

### Review agents (parallel)

- **`review-backcompat`** — old `.kernel-code/runs/*.log` files (from before Phase 0) still readable via current tooling. Checkpoints with only speedup still resume.
- **`review-prompts`** — new CLI prompts read naturally. Regression: user typing `"2x speedup"` still works (parser should handle both).
- **`review-consistency`** — all 21 surfaces from ux-migration's catalog now show SOL primary, no orphaned speedup-only displays.

### Gate criteria

- User types `/optimize target=0.8 sol` → autopilot stops on SOL
- Old logs still parse
- Skill library evidence no longer polluted by sub-baseline runs

---

## Phase 3 — kernel+ iteration

**Status:** not started
**Estimate:** 10-14 days total
**Depends on:** Phase 0, Phase 1, Phase 2
**Unlocks:** Phase 4 (alignment)

### Goal

Drive mean SOL on KernelBench L1 from week-1 baseline (~0.25-0.35) to ≥ 0.40. Each sub-phase is individually measured against the Phase 1 benchmark. Ship only if delta ≥ +0.02 mean SOL, held for 24h.

### Sequenced sub-phases (each with work-team + review-team)

Each sub-phase gets its own work team, review team, and A/B measurement before shipping.

#### 3a — Profiler-in-loop for critic  (2 days, +0.3-0.5 SOL est.)
Inject `compute_util`, `bandwidth_util`, `cache_efficiency`, `bottleneck_type` into the critic prompt after each failing attempt.
**Single biggest gap** in the current prompt flow (per levers-analysis). Critic today sees `"correct/incorrect + speedup"` only; no utilization data.

Work agents: `critic-prompt`, `pipeline-plumbing`, `prompt-template`
Review agents: `review-prompt-quality`, `review-a-b-lift`, `review-cost-delta`

#### 3b — Per-op-type prompt templates  (1.5 days, +0.2-0.3 SOL est.)
Classifier produces `(tier × op_type)` — today used for display only. Route to type-specific generator/pivot prompts:
- `REDUCTION` → warp-shuffle template
- `GEMM` → shared-mem tiling template
- `ATTENTION` → online-softmax (Flash) template
- `HISTOGRAM` → privatized-bins / warp-aggregated atomics template
- `NORM` → fused-mean-var template

Work agents: `template-gemm`, `template-attention`, `template-histogram`, `template-reduction-norm`
Review agents: `review-template-coverage`, `review-pivot-quality`, `review-a-b-lift`

#### 3c — Model routing by tier/op  (2 days, +0.1-0.2 SOL + cost savings)
Route L1 → Haiku (cheap, fast); L2/Quant → Sonnet; MoE/Attention → Opus or GPT-5. Measure cost-per-SOL.

Work agents: `routing-table`, `routing-integration`
Review agents: `review-routing-correctness`, `review-cost-per-sol`, `review-a-b-lift`

#### 3d — Auto-tune sweep on correct kernels  (1 day, +0.1-0.15 SOL est.)
After a correct kernel passes, cartesian-sweep `BLOCK_SIZE × num_warps × num_stages` on small workload, bake best into final.

Work agents: `autotune-sweep`, `autotune-integration`
Review agents: `review-sweep-speed`, `review-a-b-lift`

#### 3e — Retrieval-augmented few-shot  (3 days, +0.15-0.25 SOL est.)
Embed skill library + recent wins; retrieve top-3 similar kernels as generator few-shot.

Work agents: `embed-index`, `retrieval-plumbing`, `few-shot-injection`
Review agents: `review-retrieval-quality`, `review-a-b-lift`, `review-token-budget`

#### 3f — Bandit over strategies  (2.5 days, +0.1-0.2 SOL est.)
Meta-reflect picks pivot strategy via Thompson sampling over skill-library priors.

Work agents: `bandit-core`, `bandit-integration`, `prior-init`
Review agents: `review-bandit-convergence`, `review-a-b-lift`, `review-edge-cases`

#### 3g — Quant correctness relaxation  (1 day, +0.05-0.1 SOL est.)
Dtype-aware tolerance: fp8 → 5e-2, int8 → 1e-1, else → 1e-2.

Work agents: `tolerance-dispatch`
Review agents: `review-false-positive-rate`, `review-a-b-lift`

### Gate per sub-phase

- A/B test: 30-problem subset (split evenly across op types). Old kernel+ vs new kernel+.
- **Ship criterion:** `delta_mean_sol >= 0.02 AND p95(delta) > 0` (Mann-Whitney U for signal check)
- **Rollback if:** any mean metric regresses, OR cost-per-SOL worsens by >20% without an SOL win to justify it.
- 24h stability hold before advancing to next sub-phase.

---

## Phase 4 — kernel_code alignment

**Status:** not started
**Estimate:** 3 days
**Depends on:** Phase 3 (stable SOL numbers)
**Unlocks:** Phase 5 (optional)

### Goal

Drop dual-metric transitional. SOL becomes primary everywhere. `/leaderboard` command exposes Phase 1 results. Model routing transparent in the Optimization Plan. Deprecate `target_speedup`.

### Work agents (parallel, into team `phase-4-alignment`)

- **`headline-sol`** — `kernel_profile.py` primary line becomes SOL: `"SOL 0.42 · 42% of hardware peak · memory-bound"`. Speedup demoted to subtitle.
- **`leaderboard-cmd`** — new `/leaderboard` TUI command. Browse by problem, date, model; show trend vs baseline.
- **`routing-display`** — Optimization Plan reveals which model was chosen per worker.
- **`deprecation`** — speedup prompts warn on use; `target_speedup` marked `@deprecated`; 2-release deprecation window.

### Review agents (parallel)

- **`review-new-user-flow`** — fresh user launches kernel-code, does `/optimize` — what do they see? Is SOL explained? Is the leaderboard discoverable?
- **`review-speedup-orphans`** — every speedup mention either deleted, deprecated, or justified as a debug-only column.
- **`review-leaderboard-ui`** — sort, filter, regression-detection all work.

---

## Phase 5 — Learning loop (optional, post-plan)

**Status:** not started (ideation only)
**Depends on:** Phase 1 running weekly for ≥ 4 weeks, generating ≥ 1000 skill-library wins

### Ideas (rank at the time, not now)

- **SFT on skill library**: fine-tune Qwen2.5-Coder-32B or DeepSeek-Coder on `(reference + profile + classifier_hints) → winning_kernel` pairs
- **RLAIF with Sonnet grader**: cheap reward model to proxy real eval
- **PPO/GRPO on generation**: only if SFT plateaus
- **Process reward models**: score intermediate compile/correctness/perf states as shaping reward
- **MCTS over kernel mutations**: tree search in a small action space

Skip until Phases 0-4 are done and you have a stable data flywheel. Every RL approach is useless without reliable eval, which Phase 1 provides.

---

## Dependencies & critical path

```
Phase 0 (SOL instrumentation) ─┬─> Phase 1 (benchmark harness)
                               └─> Phase 2 (UX transition)
                                         ↓
Phase 1 + Phase 2 ────────────────> Phase 3 (kernel+ iteration, 7 sub-phases)
                                         ↓
                                   Phase 4 (kernel_code alignment)
                                         ↓
                                   Phase 5 (RL, optional)
```

**Critical path:** Phase 0 → Phase 1 → Phase 3a (profiler-in-loop). Everything else branches from there.

**Total effort estimate:** 4-5 engineering weeks to Phase 4 complete.

## Risks

| Risk                                  | Mitigation                                                          |
|---------------------------------------|---------------------------------------------------------------------|
| KernelBench `ModelNew` wrapper breaks | Phase 1 smoke-test (`kb-smoke` agent) is the de-risk; budget +3 days |
| ncu profiler adds >5s/eval            | Profile only best candidate per round, not every failure            |
| Reward hacking (Sakana lesson)        | Deterministic per-seed inputs; flag speedups > 100× as suspicious   |
| Model ceiling on hard ops (histograms)| Phase 3c (routing) makes Sonnet/Opus default for L2+                |
| Benchmark cost blows up               | Per-problem $0.50 cap; graceful cancellation on overage             |
| Workload-spec drift                   | Phase 0 `WORKLOAD_SPEC` covenant: references declare their shape    |

---

## RESULTS log

Update after each phase completes.

| Phase | Status     | Mean SOL (L1) | Solved-rate | Cost | Elapsed | Notes |
|-------|------------|---------------|-------------|------|---------|-------|
| 0     | ✅ closed 2026-04-21 | n/a (pre-benchmark) | n/a | ~$0.01 smoke | ~3h | SOL threads end-to-end. Smoke test on histogram passthrough yields sol_score=0.5 (correct for 1.0x speedup). Torch-profiler-untrackable ops (bincount/histograms) fall back to speedup-relative SOL — expected. Modal redeployed. 1 blocker (`_run_eval_gpumode` skipped `_collect_basic_profile`) caught by review, fixed inline. Deferred: 5 non-blocking items. |
| 1.A   | ✅ closed 2026-04-21 | n/a (pre-benchmark) | n/a | ~$0.02 smoke | ~2h | Foundation + cleanup done. kb-smoke PASS on L1 matmul (sol=0.495, 0.99x passthrough). SCHEMA.md (22 fields with `schema_version` + `kernel_source_path` + reader contract). ProblemSpec w/ `__post_init__` validation (7/7 tests). Cleanup closed 5 Deferred items: `load_problem` fixed, `shell.py` kernelbench passthrough now op-generic, spec validation, tightened smoke assertion, schema versioning. Integration review: READY FOR ROUND B. 3 minor non-blocking items remain in Deferred. |
| 1.B   | ✅ closed 2026-04-21 | n/a (pre-benchmark) | n/a | $0 (static) | ~2h | Loaders + storage done. kb_l1 (100), kb_l2 (100), gpumode (8) → 208 unique valid ProblemSpecs; dtype detection correct (histogram=uint8, matmul=float16). Leaderboard writer atomic temp+rename+fsync, kernel companion file, 4/4 tests. Reader with schema_version contract (skip-lower-major stands), 7/7 tests. 3 blockers fixed inline: reader warnings for missing `correct`/`timestamp`; docstring "zero-indexed" → "1-indexed"; `_LEVEL_PROBLEM_COUNTS[2]` 50→100. 4 non-blocking items to Deferred. |
| 1.C   | ✅ closed 2026-04-21 | n/a (pre-benchmark) | n/a | $0 (static) | ~2h | Execution + reporting done. `run_suite()` orchestrator with budget/concurrency/resume/Ctrl-C/error-isolation (3/3 tests). `scripts/score_suite.py` aggregation CLI (2/2 tests, per-tier + ALL cuts, null vs NaN handled). `scripts/emit_report.py` markdown emitter with citation-backed baselines (3/3 tests). `scripts/check_regressions.py` CI-hookable detector (4/4 tests, correctness regression always flagged). 3 blockers fixed inline: profile-key mismatch (`_pct`-suffix → short-form), test mocks updated to exercise primary path, Cursor citation H200→B200. 6 non-blocking items to Deferred. Ready for Round D. |
| 1.D.6-unblocker | ✅ closed 2026-04-23 | n/a (fix-only) | n/a | ~$0.10 (regression test 3 runs) | ~3h fix + test + audit | **Intra-problem CUDA contamination unblocker.** Root cause: Modal container reused across `eval_kernel_on_gpu` calls retained wedged GPU device state after a kernel's illegal memory access, even though each eval already ran in a `multiprocessing.spawn` subprocess with a fresh CUDA context. The poisoned device state survived the context rebuild on some failure modes (vectoradd, Conv2D). Flagged in commit 68a4678. Fix (generalizable, no op-type special casing): (a) `modal_infra/app.py::_maybe_schedule_container_death` — on child subprocess `status=cuda_error`, schedule `os._exit(1)` via a 2.0s `threading.Timer` from the GPU entry points (L40S/H100/A100-80GB/A100-40GB/EvalWorker.run). The timer lets Modal serialize the result dict back to the client before the container dies; next call transparently gets a fresh container. Gated by `OPENKERNEL_DIE_ON_CUDA_ERROR` (default "1"); hot path unchanged. (b) Per-call `TRITON_CACHE_DIR` at `/tmp/triton_cache_<pid>_<ns>`, wiped in the child's finally — prevents stale JIT cache from one attempt poisoning the next. (c) `_eval_kernel_core` tempdir (`openkernel_eval_*`) now `shutil.rmtree`'d in finally — prevents /tmp growth across container lifetime. Validation: `tests/test_cuda_contamination.py` submits an OOB Triton kernel (huge-index gather into unmapped address space — shallow pool-adjacent OOB doesn't fault under PyTorch's caching allocator) followed by a trivially-correct identity-copy kernel on the same Modal function handle. 3/3 deterministic passes against the deployed fix (~$0.10 total Modal cost). Client-side retry audit (`kernel_agent_bridge._modal_eval`): no changes needed — `_INFRA_STATUSES` correctly excludes `cuda_error` (feeds critic as kernel bug per design), Modal-layer exceptions caught and retried with `process_crashed` status. Targeted 4-problem rerun was attempted via `scripts/run_dry_run.py` but hung silently at 45min (separate known issue: `run_dry_run.py` matmul hang, unrelated to CUDA contamination — deferred as a Phase 1 follow-up). Close-out evidence is the regression test + code-level review; full empirical validation will come with the Phase 1.D.6 sweep. Non-blocking: `shell.py` profile loop bypasses the `_modal_eval` retry wrapper (identified by bridge-audit; tracked as Deferred). |
| 1.D.5-smoke | ✅ closed 2026-04-23 | 0.50 (N=1 ELU parity) | 0/1 @ 0.3 (L1); 0/1 gpumode correct | ~$0.14 (2 rerun + 1 aborted) | ~90min across 2 reruns | **Integrity-fix pass, not a baseline number.** Round D.5 first run (2026-04-22) produced a contaminated `gpumode_prefixsum` record — kernel file was a pristine histogram. Root cause: `os.environ["OPENKERNEL_BEST_KERNEL"]` (set by `auto_optimizer.run()`, read by `kernel_agent_bridge` at line 338) was never cleared between problems, so problem N's kernel leaked into N+1's generator prompt as "PREVIOUS BEST KERNEL". Same Python process + `os.environ` = shared state. 3 generalizable fixes shipped across 3 parallel agents: (a) `auto_optimizer.py` clears 5 env vars on `run()` entry AND in `finally:`; (b) `modal_infra/app.py::_collect_basic_profile` flattens tuple/list/dict inputs via `_sum_bytes()` recursion + always carries `runtime_us`/`ref_runtime_us` + on profile-collection exception the gpumode wrapper still returns a SOL-bearing fallback dict; (c) `openkernel/benchmarks/leaderboard_writer.py` adds 3 write-time integrity guards (placeholder-with-correct=True, op-token docstring mismatch with `OPENKERNEL_SKIP_KERNEL_IDENTITY_CHECK=1` bypass, cross-problem hash reuse). Guard #3 fired a false-positive on the shared failure placeholder in the rerun → 3-line patch to exempt placeholder + case-6 test. Net: 6/6 integrity tests pass, 7/7 ref-path-plumbing tests pass, prefixsum now fails CLEAN (kernel_hash=27c7dcf3ca24 placeholder) instead of being credited a histogram kernel. Fix #1 empirically validated; Fix #3 empirically validated + false-positive caught; Fix #2 parts 2+3 validated via ELU runtime fallback (sol=0.4989 ≈ 0.5×0.9978); Fix #2 part 1 (tuple bytes) unit-reasoned but not yet empirically validated (requires a correct gpumode eval). Full 208-problem sweep blocked on intra-problem CUDA container contamination (commit 68a4678) — known, deferred to Phase 3. |
| 1     | in progress | —             | —           | —    | —       | —     |
| 2     | ✅ closed 2026-04-23 | n/a (UX only, pre-sweep) | n/a | $0 (offline, no Modal) | ~3h across 4 work + 3 review agents | **SOL-as-metric transition.** `GoalSpec` gained `target_sol: float = 0.80` (always set, 0.0<v≤1.0 validated) alongside `target_speedup` (backcompat preserved). `GoalParser` extended for `"SOL 0.8"`, `"0.8 SOL"`, `"80% SOL"`, `"target sol 0.8"` and both-in-one-string (`"target 2x speedup SOL 0.8"`) — ambiguous bare numbers now raise `ValueError`. `MetaOptimizer.run()` accepts `target_sol: float \| None = None` runtime override; target-reached fires on `sol >= target_sol` when profile has SOL, falls back to `speedup >= target_speedup` otherwise, with the criterion logged per-event. Overshoot-exploratory mirrored for SOL. `batch_optimizer.run_suite` now passes `target_sol` straight through — closing Round C Deferred item on the `target_speedup = 1/(1-target_sol)` heuristic. Evidence gate replaced `speedup > 1.02` with `sol >= max(0.50, prior_best)` using `leaderboard_reader.prior_best_sol(id, hw)` (new helper); pre-SOL records gracefully fall back to speedup gate so skill library stays readable. Skill priority scoring now weights SOL×10 primary + speedup×0.5 legacy. Five UX surfaces updated to SOL-primary + speedup-secondary (`summary_card`, `kernel_profile`, `live_display`, `run_log`, `sol_plots`); `sol_plots` now reads `sol_score` directly instead of the `0.5×speedup` fallback; format convention `"SOL 0.42 · 1.80x"`, `"(SOL unknown)"` when profile lacks it. Shell `_smart_optimize` wraps `parse_goal` ValueError with a friendly grammar-listing message (review-prompts catch). Dual-display caught + fixed a preexisting f-string backslash bug in `kernel_profile.py:217` that was blocking imports on Python 3.11. Reviews: `review-prompts` (1 issue → task #9 fix shipped), `review-backcompat` clean (6/6 — old logs + checkpoints parse, GoalSpec round-trip safe, sentinel logic correct), `review-consistency` clean (5/5 catalog surfaces + 0 orphans; LLM-prompt `speedup` vocabulary flagged for Phase 3/4). Tests: 77/77 new tests pass (goal_spec 11, goal_parser 24, shell 3, stopping_sol 7, evidence_tracker 11, summary_card 5, kernel_profile_display 16) in 1.08s. No Modal cost (all offline, unit-testable). Ready for Phase 3. |
| 3a    | not started | —             | —           | —    | —       | —     |
| 3b    | not started | —             | —           | —    | —       | —     |
| 3c    | not started | —             | —           | —    | —       | —     |
| 3d    | not started | —             | —           | —    | —       | —     |
| 3e    | not started | —             | —           | —    | —       | —     |
| 3f    | not started | —             | —           | —    | —       | —     |
| 3g    | not started | —             | —           | —    | —       | —     |
| 4     | not started | —             | —           | —    | —       | —     |

---

## Deferred (non-blocking from phase reviews)

Fill in as reviews surface non-blocking improvements that shouldn't gate phase advancement.

### From Phase 0

- **openkernel bridge also calls `log_iteration`** — `kernel_code/integration/openkernel_bridge.py:450` has a separate `log_iteration` call site not updated with profile plumbing. Audit and add profile threading before Phase 1 benchmark sweeps exercise it heavily. Non-blocking for current workflow (autopilot uses `kernel_agent_bridge`). Reported by `plumbing-sol`.
- **Checkpoint `raw_metrics` JSON serialization risk** — `_collect_basic_profile` populates `raw_metrics` with data from torch profiler events. All current fields are primitives, but if a future profiler-event field contained a torch object or tensor, JSON round-trip would crash. Add a defensive `json.dumps(...)` check in checkpoint save, or explicit key whitelist. Reported by `review-edge-cases`.
- **Old run-log profile backfill** — `kernel_code/run_analysis.py:_parse_run_log` cannot show SOL for pre-Phase-0 logs. Low priority; only matters for cross-history analytics. Reported by `review-edge-cases`.
- **Dead code in `checkpoint.py:127`** — `logger.info(...)` references `state.round_num` but `state` is not in scope at that call site. Not caused by Phase 0; pre-existing. Fix when touching checkpoint next. Reported by `review-edge-cases`.
- **`_run_eval_kernelbench` profiling path not independently audited** — Phase 0 only exercised the gpumode path end-to-end (our current workflow). The kernelbench path also calls `_collect_basic_profile` (app.py:335) and should populate SOL correctly, but first exercise is Phase 1 smoke test. If broken, fix in Phase 1.

### From Phase 1 Round A

- **`openkernel/kernelbench/problems.py:load_problem` is broken** — `dataset[problem_id]` subscript doesn't work on current kernelbench dataset objects. kb-smoke bypassed by calling `construct_kernelbench_dataset(level, source="huggingface").get_problem_by_id(id)` directly. Fix before Round B's `kb_l1_loader` / `kb_l2_loader` land, or have those loaders use the direct-call pattern.
- **kernelbench pins `python==3.10`, project uses 3.11** — regular `pip install kernelbench` rejects; `uv pip install` ignores the pin and installs cleanly. No runtime issues observed so far. Add to dev setup notes; consider proposing a relaxed pin upstream.
- **`shell.py:2463-2472` kernelbench passthrough is matmul-specific** — hardcodes `torch.matmul(...)` in the generated `ModelNew`. Non-matmul KB problems (softmax, layernorm, relu, conv) would fail the `/optimize` flow today. kb-smoke uses the generic `ModelNew = Model` pattern which works for all KB ops. Either port the generic pattern into `shell.py`, or document that `/optimize` against KernelBench is matmul-only until fixed. Orthogonal to Round B (batch-runner builds its own path via ProblemSpec), so not a Round B blocker.
- **GPU MODE workload_spec validation** — `ProblemSpec.workload_spec` accepts arbitrary dicts; loaders should validate that GPU MODE specs include required keys (`size`, `seed`, and op-specific extras like `contention`) at loader boundary in Round B. Not a ProblemSpec concern.
- **Runtime enum validation for ProblemSpec** — `Tier` and `Source` use `typing.Literal` which only type-checks; add `__post_init__` to reject invalid values at runtime. Also add a test for empty `reference_source=""`. Nice-to-have.
- **Classification caching in ProblemSpec** — Phase 3c per-op-type routing will re-classify each spec. Either add a mutable `classification_cache` field (but spec is frozen) or keep the cache external (as `_get_classification()` already does in `auto_optimizer.py`). Lean toward external.
- **kb-smoke zero-profile assertion** — `scripts/kb_smoke.py:106` checks `bool(profile)` which passes even if all fields are 0. Tighten to `any(v for k, v in profile.items() if k != 'error')` or similar.
- **Schema `schema_version` field** — add explicit versioning (`"schema_version": "1.0"` in every record) to let future readers detect pre-migration records. Do as part of Phase 2 transition.
- **Schema `config_hash` opacity** — Phase 3c per-op-type routing may want `backend` / `target_sol` / `budget_usd` as explicit fields rather than hash-rolled-up. Evaluate before Phase 3.

### From Phase 1 Round A cleanup

- **ProblemSpec whitespace-only `reference_source`** — `__post_init__` only rejects `""` / `None`, not `"   "`. Loaders already produce non-whitespace so not a live bug; consider `reference_source.strip()` if a loader ever regresses. Low priority.
- **`workload_spec` keys not documented per source** — GPU MODE conv2d uses `{size, kernelsize, channels, batch, seed}`, other tasks vary. Loaders will discover empirically. If Phase 3 starts relying on specific keys, codify per-op contracts in SCHEMA.md or a sibling doc.
- **`config_hash` algorithm not codified** — Writer will hash `(model, backend, target_sol, budget, seed)` but there's no formal spec. Collisions unlikely at current scale. Add to SCHEMA.md when it matters.

### From Phase 1 Round B

- **Kernel-source bit-identity assumption** — writer's "skip if `kernels/{hash}.py` exists" short-circuit assumes same hash = same bytes (sha256 collision-free at our scale). Document the assumption in the writer docstring when Phase 2 scales up.
- **Writer mutex under high concurrency** — current `rename`-atomicity handles 2-worker contention fine. If Phase 2 pushes >16 concurrent workers against the same (problem, hardware, config, date) tuple, consider adding a file-lock. Not a Phase 1 concern.
- **Reader test gaps** — empty-dir case, nested `kernels/` subdir skip, non-padded ISO dates (e.g. `"2026-4-1"` breaks string compare). All non-blocking; add when touching the reader next.
- **GPU MODE name casing** — `_PROBLEMS` slugs get `.title()`'d for `ProblemSpec.name` → `"Vectoradd"` instead of `"VectorAdd"`. Ugly but harmless. Standardize if ever exposed in UI.

### From Phase 1 Round C

- **`solved_rate_at_sol_0.3` vs KernelBench `fast_p` semantics** — our denominator is tier problem_count (100 for L1), whereas `fast_p` in the Stanford paper uses attempted count. Our framing is "fraction of benchmark solved," theirs is "fraction of attempts that succeeded." Document the difference in SCHEMA.md or in the markdown report's footnote so direct citations to KernelBench numbers aren't misleading.
- **`target_sol` → `target_speedup` mapping is a heuristic** — ~~`run_suite` currently maps `target_speedup = 1/(1-target_sol)` (sol=0.8 → 5×, sol=0.95 → 20×). This is workload-dependent and not invertible in general. For Phase 2 (SOL UX transition), `MetaOptimizer.run()` should natively accept `target_sol` as a stopping criterion, not a derived speedup.~~ **Resolved in Phase 2 (stopping-rule)**: `MetaOptimizer.run()` now takes `target_sol: float | None` and `run_suite` passes it through directly — speedup fallback only fires when the latest profile lacks SOL.
- **Regression-detector relative threshold missing** — `--threshold 0.05` is absolute. A problem at baseline 0.10 dropping to 0.05 (50% relative loss) does NOT trigger with threshold 0.05 absolute. Consider `--threshold-relative` flag for Phase 2 when mean SOL per problem varies more widely.
- **L2 baselines not shown in report** — `scripts/emit_report.py` shows L1-only baseline comparison. o1 has published 24% L2 solved; worth adding. L2 tier has independent importance.
- **Report trend regex fragility** — prior-report mean-SOL extraction via regex silently yields "N/A" in delta column if prior headline changes format. Graceful but opaque. Phase 2 should maintain a sidecar `results/reports/scores/YYYY-MM-DD_{hardware}.json` alongside the markdown so trend extraction is structured, not text-parsed.
- **Metrics `N=1` percentile edge** — works correctly via stdlib but not explicitly tested. Add a one-liner test to lock the behavior against regression.

### From Phase 1.D.6-unblocker

- **`shell.py` profile loop bypasses `_modal_eval` retry wrapper** — `kernel_code/shell.py:2483` calls `eval_fn.remote()` directly inside a 5-iteration profile loop. If any call raises (container rotation lag post container-death, network glitch), the entire profile loop aborts with a generic "Profile failed" message instead of retrying. Proposed patch in `.kernel-code/dev_logs/cuda_contamination_client_audit_2026-04-23.md` — extract a shared retry helper or reuse `_modal_eval`. Low priority; the profile loop is a diagnostic tool, not on the sweep hot path.
- **No consecutive-`cuda_error` short-circuit in KernelAgent** — with container-death enabled, each cuda_error costs ~20s cold-start on the next call. If a kernel repeatedly ooB's (e.g., LLM stuck on a bad idea), KernelAgent keeps paying that cost with no early exit. Stopping module (`kernel_code/stopping.py`) already tracks `consecutive_errors` with max=5, but (a) KernelAgentBridge doesn't use it and (b) `cuda_error` isn't classified as "error" by design. Consider wiring a 3-cuda_error threshold in KernelAgent's worker loop. Non-blocking — cost impact at expected failure rates is minor.
- **`run_dry_run.py` silent matmul hang** — targeted validation rerun hung for 45min with zero output; no Modal call logs, no leaderboard records, no dev_log writes. Unrelated to CUDA contamination (the regression test proves the Modal-side fix works). Likely in the KernelAgent iteration loop or LLM-call layer. Reproduce with: `uv run python scripts/run_dry_run.py --ids kb_l1_0001 --concurrency 1 --force`. Pre-existing — historical same symptom noted in commit 68a4678 messaging. Fix before Phase 1.D.6 full sweep OR scope sweep to known-good subset.
