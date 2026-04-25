# Kernel+ as a Recursive Self-Improving (RSI) Engine

## Vision

> Given `(SOL goal, kernel to optimize, context)`, Kernel+ is a recursive
> self-improving engine that **does not stop until the goal is achieved** —
> within a user-specified envelope and the physical limits of the hardware.

Two products evolve in tandem:

- **Kernel+** is the engine: autonomous, self-improving, envelope-bound.
- **Kernel Code** is the interface: inspectable, trustworthy, dev-loop-native.

This document is the roadmap for aligning the current codebase with that vision.

## Boundary conditions

"Doesn't stop until achieved" is only safe under two boundaries:

1. **Physical reachability** must be pre-verified. HBM bandwidth, compute
   TFLOPs, and memory capacity impose hard ceilings. An engine that runs
   forever against physically unreachable targets is broken.
2. **Economic boundedness** must be user-specified. The envelope is
   `(dollars, wallclock, rounds)` — the engine exhausts every reasonable
   strategy **within** the envelope; it does not spend past it.

Within these boundaries, the engine's natural stops are reduced from the
current five+ to exactly three:

| Natural stop | Meaning |
|---|---|
| `TARGET_MET_AND_VERIFIED` | SOL reached, multi-shape verifier passed — ship |
| `ENVELOPE_EXHAUSTED`      | Every ladder rung tried, budget/time/rounds gone |
| `REACHABILITY_VIOLATED`   | Physics refused the target — refuse before starting |

Everything else (sub-baseline, convergence, max-rounds) becomes an
**escalation trigger**, not a stop.

## Scope of recursion

Recursive self-improvement is **scoped to three artifacts**:

1. **Skill library** — accumulated past-win evidence, queried at run start
2. **Prompt templates** — post-sweep correlation analysis feeds template updates
3. **Hyperparameter policy** — `(max_rounds, num_workers, temperature, model_ladder)` tuned by periodic meta-sweeps

The engine does **not** recursively edit its own `MetaOptimizer`, `KernelAgent`,
or evaluation infrastructure. Those remain human-authored, version-controlled
Python. Limiting the recursion surface to (skills, prompts, hyperparameters)
keeps the attack surface bounded and the system debuggable.

## The 13 steps (strictly dependency-ordered)

### Step 1 — Complete the reachability oracle

**What:** Extend `_hbm_ceiling_check` in `kernel_code/shell.py` to also
check compute-bound ceiling (FLOPs / peak_TFLOPs) and memory-capacity
ceiling (working set vs HBM size). Emit a unified refusal before the
engine starts if the target is physically unreachable.

**Why first:** "Don't stop until achieved" is only safe if we refuse
impossible targets up front. Without this, the engine burns unbounded
budget on problems with no physical solution.

**Success criterion:** `/optimize @reference_relu.py 1.5x` (above the
~1.36× HBM ceiling) refuses with a specific recommendation for the
actual reachable SOL target.

**Scope:** ~100 lines.

---

### Step 2 — Lock down the baseline (full L1 sweep)

**What:** Run `scripts/kb_l1_sweep.py --count 100` against the current
engine. Preserve output as `baseline_l1_<commit_sha>.json`.

**Why second:** Every subsequent change needs measurement against this.
Without a locked baseline, improvements can't be distinguished from noise.

**Success criterion:** A committed file you can point at and say "Kernel+
on this commit hits X% target-SOL rate, Y% beat-baseline rate, on full
L1, at cost Z."

**Scope:** ~$3, overnight. No code changes; configuration only.

---

### Step 3 — Reframe premature stops as escalation triggers

**What:** Rewrite stopping rules in `kernel_code/auto_optimizer.py`:
- `Sub-baseline after N rounds` → `escalate_model` trigger
- `Converged within 2%` → `accept if SOL hit, else pivot_strategy_family`
- `max_rounds=2` → default `None` (unbounded; envelope-only stops)
- Only natural stops remain: `TARGET_MET_AND_VERIFIED`,
  `ENVELOPE_EXHAUSTED`, `REACHABILITY_VIOLATED`

**Why third:** This is the literal implementation of "doesn't stop until
achieved." Steps 1 + 2 made it safe and measurable; this makes it real.

**Success criterion:** Re-run L1 #6 (the 0.20× regression). Engine
escalates (model upgrade or strategy pivot) instead of stopping at
sub-baseline. Target: recover to ≥1.0×.

**Scope:** ~150 lines. Biggest semantic change in the engine.

---

### Step 4 — Escalation ladder (concrete rungs)

**What:** Define and implement the ordered escalation ladder:

| Rung | Move | Cost multiplier |
|---|---|---|
| 0 | Retry same strategy, higher temperature / different seed | 1× |
| 1 | Pivot strategy text (we have this — meta-reflection output) | 1× |
| 2 | Expand worker pool (4 → 8 → 16) | 2–4× |
| 3 | Upgrade model (gpt-4o-mini → o3-mini → o3 → Claude Sonnet) | 2–10× |
| 4 | Switch backend (Triton → CUDA or vice versa) | 1× (regen required) |
| 5 | Relax constraints (fp32→fp16, single→multi-kernel) | 1× (user-gated) |

Engine walks the ladder each time a round underperforms. Budget-aware:
doesn't climb if remaining envelope can't afford the next rung.

**Why fourth:** Step 3 removed premature stops; this step defines what to
do *instead*. Without this, "never stop" collapses into "retry the same
strategy forever."

**Success criterion:** On a problem that previously failed at baseline
model, engine successfully escalates through 2+ rungs and hits target.
Verify in `llm_calls.jsonl` that different rungs actually fired.

**Scope:** ~300 lines in `auto_optimizer.py` + provider-layer additions.

---

### Step 5 — Multi-shape, multi-seed verifier loop

**What:** Before shipping a winning kernel, re-eval on 5+ distinct
`(random_seed, shape)` combinations including edge cases: tiny, huge,
non-power-of-2 dims. Accept as winner only if all combinations pass
correctness AND speedup margin is preserved within 20%.

**Why fifth:** "Achieved" must mean "achieved correctly and robustly."
The optimizer is adversarial to the benchmark — it will eventually
produce kernels that overfit to the test shape. The verifier is the
defense.

**Success criterion:** Artificial overfit kernel (hardcoded
`if dim == 16384: …`) is rejected; legitimate kernels pass.

**Scope:** ~200 lines, post-round-winner-selection in `auto_optimizer.py`.

---

### Step 6 — Problem embedder and indexed memory

**What:** Extract features from each reference problem (op type, tensor
elements, dtype, reduction-dim ratio, memory footprint, hardware). Store
every past run's outcome indexed by feature vector:
`{problem_features, winning_kernel, strategy, achieved_SOL, hardware, model, rationale}`.

**Why sixth:** This is the memory substrate for recursive
self-improvement. Without it, the engine cannot retrieve prior wins when
a new problem arrives — every run starts from scratch.

**Success criterion:** `query_similar(ref, top_k=5)` returns the 5 most
structurally-similar past problems with their winning kernels in
sub-second lookup.

**Scope:** ~300 lines. Classifier features + cosine similarity for v0;
vector embeddings are v2.

---

### Step 7 — Retrieval-augmented generation (RAG)

**What:** At the start of each `/optimize`, query indexed memory for
top-K similar past problems. Inject their winning kernels + rationales as
in-context few-shot exemplars in the first-round generation prompt.

**Why seventh:** Step 6 built the memory; this step connects it to the
generator. This is the first true recursion loop — past wins directly
influence future attempts.

**Success criterion:** A/B-measure: re-run L1 sweep with retrieval ON vs
OFF. Retrieval-ON should show ≥5 percentage-point improvement on
target-SOL rate.

**Scope:** ~200 lines: template changes to `kernel_generation.j2`, bridge
code to inject exemplars.

---

### Step 8 — Post-sweep prompt-outcome correlator

**What:** After each sweep, analyze `llm_calls.jsonl` across all
problems. Extract: which template sections appeared in winning prompts vs
losing prompts, per problem class. Emit a structured report:
`"Section 8 AUTOTUNE correlates +30% with matmul wins, -5% with reduction wins."`

**Why eighth:** Step 7 made past kernels reusable. This step makes
*prompt guidance itself* improveable. The engine starts learning which of
its own instructions work where.

**Success criterion:** Correlator run on baseline sweep produces a
human-readable report with ≥1 actionable finding. Commit as
`prompt_correlations_<sweep_id>.md`.

**Scope:** ~400 lines, offline analysis script. v0 surfaces findings
for human review; auto-modify is a v2 upgrade.

---

### Step 9 — Structured skill-library writes with rationales

**What:** Extend `evidence_tracker.py` to capture not just speedup/SOL
per evidence entry but also:
- **Rationale** — extracted from the winning worker's LLM reasoning trace
- **Failure modes** — "common failure modes we saw on similar problems and how we recovered"

**Why ninth:** Step 7's retrieval is only as good as the memory's
quality. Raw kernels are useful; rationales turn them into teaching
examples that transfer better.

**Success criterion:** `data/skills/*.json` has rationale strings per
evidence entry. Retrieval returns these for injection.

**Scope:** ~150 lines extending the evidence-writer path.

---

### Step 10 — Hyperparameter meta-loop

**What:** Periodic (weekly/monthly) sweep that varies
`(max_rounds, num_workers, temperature, model_ladder_order)` against a
fixed benchmark subset. Compute Pareto-optimal settings per problem
class. Output a learned-policy file that `GoalSpec` reads at
construction time.

**Why tenth:** Steps 7–9 made the engine's *outputs* self-improving.
This step makes the engine's *control surface* self-improving. Recursive
self-improvement is incomplete without it.

**Success criterion:** After running once, at least one default changes
from its hand-set value, and a follow-up measurement sweep confirms the
change is better.

**Scope:** ~500 lines. New script on top of `kb_l1_sweep.py`.

---

### Step 11 — Inspectability: interactive CLI viewers

**What:** CLI commands that make engine state inspectable:
- `/show llm-trace [round] [worker]` — TUI browser over captured prompts + reasoning traces
- `/show skills [query]` — search the skill library, see which entries fired on the last run
- `/explain failure` — synthesize a one-paragraph explanation of why a run didn't hit target, drawing from workers' logs and LLM traces
- `/show ladder` — show which escalation rungs fired and at what cost

**Why eleventh:** An RSI engine that can't be inspected is a black box.
Users won't trust it without inspectability, no matter how good its
outputs. Also prerequisite for debugging the engine when it goes wrong.

**Success criterion:** A user new to Kernel+ can, after 10 minutes of
reading, explain why a specific run stopped where it did.

**Scope:** ~800 lines across CLI + live_display integrations.

---

### Step 12 — Versioned leaderboard + CI gate

**What:**
- Leaderboard artifact format:
  `{commit_sha, model, hardware, backend, problem_class → (target_sol_rate, beat_baseline_rate, median_sol, cost)}`
- GitHub Action: runs a 5-problem mini-sweep on any PR touching
  `kernel_code/`, `kernel_agent/templates/`, or `modal_infra/app.py`.
  Fails the PR if pass rate drops ≥5 percentage points vs baseline.
- Longitudinal view: plot rate-over-time across commits.

**Why twelfth:** Without this, all the recursion gains from steps 7–10
can regress silently. The CI gate is what makes improvements stick.

**Success criterion:** A hand-crafted bad-prompt PR is caught by CI
before merge.

**Scope:** ~300 lines (mostly GH Actions YAML + a sweep variant).

---

### Step 13 — Editor integration (adoption multiplier)

**What:** VSCode/Cursor extension with one command: "Optimize this
function with Kernel+". Extension sends selected function + file context
to Kernel Code, streams back progress, presents diff + SOL + cost in a
webview, lets user accept into source.

**Why last:** This is the multiplier on everything prior. An RSI engine
hidden behind a CLI gets used by the author; an RSI engine in every
kernel engineer's editor gets used by everyone. Needs steps 1–12 to be
meaningful — polish over foundation.

**Success criterion:** A kernel engineer on a different codebase uses
it without reading the README.

**Scope:** ~2000 lines TypeScript + streaming endpoint in Kernel Code.

---

## Phase breakdown (for parallel execution)

The 13 steps group into five phases with clear dependency boundaries.
Within each phase, agents can work in parallel; between phases, there
is a hard barrier.

### Phase A — Foundation (safety + measurement)

**Steps:** 1, 2

**Parallelism:** step 2 (sweep) runs overnight on the engine as it exists;
step 1 (reachability) is code-only. These are independent and can run
in the same team.

**Duration:** ~1 day of code work, overnight sweep.

### Phase B — Unstop the engine

**Steps:** 3, 4, 5

**Parallelism:** step 3 must complete before 4 (4 defines the moves that
3's triggers dispatch to). Step 5 (verifier) is independent of both and
can be built in parallel.

**Duration:** ~1–2 weeks.

### Phase C — Memory substrate

**Steps:** 6, 7, 9

**Parallelism:** step 6 must complete before 7 (7 queries 6's index).
Step 9 (rationales) touches a different module and can run in parallel
with either.

**Duration:** ~1–2 weeks.

### Phase D — Meta-learning

**Steps:** 8, 10

**Parallelism:** both are offline analysis scripts, fully independent.

**Duration:** ~1–2 weeks.

### Phase E — Product surface

**Steps:** 11, 12, 13

**Parallelism:** 11 is CLI-local; 12 is CI-YAML + a sweep variant; 13 is
editor. All three are independent. Largest phase.

**Duration:** ~3–4 weeks.

**Total: ~8 weeks of focused engineering.**

## Agent-team execution pattern

Each phase follows the established workflow:

1. `TeamCreate team_name=rsi-phase-<letter>`
2. `TaskCreate` one task per step (or per clearly-separable sub-step)
3. Add `addBlockedBy` for within-phase dependencies
4. `Agent` spawn with `team_name=rsi-phase-<letter>`, one per parallel lane
5. Teammates report via `SendMessage` on completion, mark tasks via `TaskUpdate`
6. Team lead verifies edits with spot-checks before marking tasks done
7. Redeploy Modal when `modal_infra/app.py` changed
8. `SendMessage shutdown_request` to all teammates when phase complete
9. `TeamDelete`

Agent-type guidance per task:

- **Code edits** (steps 1, 3, 4, 5, 6, 7, 9, 11, 12, 13): `general-purpose`
- **Read-only investigations** (diagnostic steps, correlator analysis in step 8): `Explore`
- **Architecture planning** (if a step's scope is uncertain): `Plan` before spawning the general-purpose agent

**Non-negotiable team rules:**

- Never spawn `Agent` with `run_in_background: true` — breaks cmux visibility
- Every code-edit task must parse-check after the edit (`python -c "import ast; ast.parse(open(F).read())"`)
- Every Modal-touching task must redeploy + health-check before marking done
- Every step must have a **success criterion** (from the step definition above) verified before close

## Success metrics (sweep-based, not vibes-based)

Each phase transitions on measurable criteria:

| Phase | Transition metric | Threshold |
|---|---|---|
| A → B | Baseline L1 sweep committed | file exists, pass rate recorded |
| B → C | Re-sweep with escalation ladder on | ≥10pp improvement vs baseline, or clear "no headroom here" diagnosis |
| C → D | A/B sweep with RAG on vs off | ≥5pp improvement |
| D → E | Hyperparameter policy-driven sweep | at least one policy default changed and verified better |
| E → ✓ | Editor extension used by one external engineer | qualitative — user success report |

## What we will not build

Explicitly ruled out, each with its reason:

- **Cloud-hosted Kernel+ service** — competes with local-first strength; dilutes dev-loop fit
- **Our own foundation model** — model commoditization is a tailwind, not a risk
- **A DSL for kernels** — Python + Triton + CUDA already won; no parallel universe
- **Agents editing arbitrary code** — single-function scope is the credibility anchor
- **Auto-applying prompt edits (v0 of step 8)** — human review gates prompt changes until we've seen the correlator report ≥3 times and it's consistently right
- **Editing the MetaOptimizer from within a run** — recursion is scoped to skills + prompts + hyperparameters; control-flow stays human-authored

## What makes this path differentiated

Competitors (NVIDIA R1 inference-time-scaling, closed-source research loops):
- **Their strength:** raw speedup numbers, huge compute budgets
- **Our differentiation:**
  1. **Open and inspectable** — full `llm_calls.jsonl`, documented stop rules, released skill library
  2. **Reachability-aware** — refuses impossible targets, doesn't burn budget against physics
  3. **Cost-transparent** — $ per success tracked, not hidden in infra costs
  4. **Multi-provider** — OpenAI, Anthropic, NVIDIA NIM, Ollama all work
  5. **Dev-loop native** — CLI-first, git-native, editor-integrated (step 13)
  6. **Longitudinally accountable** — leaderboard + CI gate (step 12) prevent regression

## The minimum viable RSI

If the full 8-week plan is too long, the MVP is steps 1 + 3 + 4 + 5 + 6 + 7 + 11:

- **1, 3, 4, 5** → "doesn't stop until achieved" (physically + escalation + verified)
- **6, 7** → "recursive self-improving" at its basic form (retrieval-augmented)
- **11** → inspectability so users trust it

That's ~3 weeks of focused work and gets to ~80% of the target.

Steps 2, 8, 9, 10, 12, 13 are the difference between "working" and "the best."

## Appendix — current-state audit

What aligns with RSI thesis (keep):

- SOL as first-class target (`target_sol=0.80` default, SOL-overshoot stop)
- HBM speed-of-light pre-check
- Meta-reflection between rounds (Loop 2)
- Worker parallelism (Kernel+ 4-worker mode)
- Skill library (writes evidence; retrieval not yet wired)
- Full LLM trace capture (`llm_calls.jsonl` with reasoning tokens + cached tokens)
- Problem classification (`classify_problem`)
- Cost tracking per iteration
- Auto-model routing (elementwise → gpt-4o-mini)

What misaligns (needs rework in steps 3, 5):

- Sub-baseline-after-2-rounds stop (→ escalation trigger)
- Converged-no-headroom stop (→ pivot or accept)
- `max_rounds=2` default (→ `None`)
- Time-limit default (already moved to `None` as of commit `f441024`)
- Correctness = 3 trials on one shape (→ multi-shape verifier)

What's structurally missing:

- Reachability oracle beyond HBM (step 1)
- Escalation ladder (step 4)
- Memory retrieval path (steps 6, 7)
- Rationale-bearing skill entries (step 9)
- Prompt-outcome correlator (step 8)
- Hyperparameter meta-loop (step 10)
- Interactive inspectability commands (step 11)
- CI gate (step 12)
- Editor integration (step 13)

## Source of truth

This document is the authoritative roadmap. Any divergence between a
running sprint and this doc should either update the doc or correct
the sprint. Phase-transition metrics are non-negotiable — no phase
closes without its metric being verified.
