# A/B Testing Framework for Validation

Every architectural change should be grounded in data. Proposed experiments:

---

## Experiment 1: Reflection vs Fixed Rounds

- **Control**: MetaOptimizer with 5 fixed rounds, no reflection
- **Treatment**: MetaOptimizer with 10 rounds + LLM reflection
- **Metrics**: `best_speedup`, `rounds_to_success`, `cost_per_speedup_unit`
- **Sample**: 50 runs (25 easy, 15 medium, 10 hard problems)

---

## Experiment 2: Evidence-Injected Skills vs Minimal Skills

- **Control**: Generator gets problem + reference + minimal skill metadata
- **Treatment**: Full skill metadata including hardware, evidence, pitfalls, code_template
- **Metrics**: `first_pass_correctness`, `avg_rounds_to_success`

---

## Experiment 3: Hardware-Specific Templates

- **Control**: Generic Triton/CUDA patterns
- **Treatment**: H100-specific block sizing, shared mem, grid params
- **Run on**: H100, A100, L40S (10 runs each)
- **Metrics**: `speedup achieved`, `bandwidth utilization %`

---

## Experiment 4: Deterministic Gating

- **Control**: Current LLM-only prescription validation
- **Treatment**: Add gating rules (SMEM budget, coalescing-first, one-method)
- **Metrics**: `prescription_validity_rate`, `convergence_speed`, `false_optimization_rate`

---

## Statistical Plan

- T-tests on speedup distributions (Control vs Treatment)
- Cohen's d effect sizes
- Convergence curves (speedup vs round number)

## Phase Rollout

- **Week 1**: Experiments 1 & 4 (Reflection + Gating)
- **Week 2**: Experiments 2 & 3 (Skills + Hardware Templates)
