"""Evidence compounding for persistent cross-run learning.

After each optimization run, extracts winning strategies and updates
skill evidence. Skills with high evidence scores get prioritized in
future runs. This is autokernel's structural advantage over Cursor —
they start fresh each run.

Usage::

    from kernel_code.evidence_tracker import extract_and_update_evidence

    # After an optimization run completes:
    extract_and_update_evidence(
        optimization_log=result.round_history,
        reference_code=reference_source,
        hardware=goal.hardware,
        skills_dir="data/skills",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

from openkernel.memory.skill_library import OptimizationSkill, SkillLibrary
from kernel_code.problem_classifier import classify_problem

logger = logging.getLogger(__name__)

# SOL floor: a run whose SOL is below this should never enter the skill
# library, even if there is no prior evidence. Prevents sub-baseline
# kernels from polluting future generations.
_SOL_FLOOR = 0.50
# Fallback speedup floor for runs where SOL is unavailable (legacy
# profiles or missing telemetry). Matches the prior gate value.
_SPEEDUP_FLOOR = 1.02


def _load_prior_best_sol(problem_id: str | None, hardware: str) -> float:
    """Return the best SOL ever recorded on the leaderboard for this problem+hardware.

    Returns ``0.0`` when ``problem_id`` is missing or the leaderboard has
    no matching record. Import-scoped and failure-tolerant so evidence
    extraction never crashes due to leaderboard I/O.
    """
    if not problem_id:
        return 0.0
    try:
        from openkernel.benchmarks.leaderboard_reader import prior_best_sol
        return prior_best_sol(problem_id, hardware)
    except Exception as exc:  # noqa: BLE001 — keep gate resilient
        logger.debug("prior_best_sol lookup failed: %s", exc)
        return 0.0


def extract_and_update_evidence(
    optimization_log: list[dict],
    reference_code: str,
    hardware: str,
    skills_dir: str | Path = "data/skills",
    problem_id: str | None = None,
) -> int:
    """Extract winning strategies from a completed run and update skill evidence.

    Looks at rounds where SOL improved (falling back to speedup when SOL
    is unavailable), identifies which strategies led to gains, and
    records evidence entries in matching skills.

    Args:
        optimization_log: List of round dicts from AutoResult.round_history.
        reference_code: Original reference code (for problem classification).
        hardware: GPU type used (e.g., "L40S", "H100").
        skills_dir: Path to skill JSON files.
        problem_id: Optional leaderboard problem identifier used to look
            up the prior best SOL. When provided, runs whose SOL is
            worse than the leaderboard's best are skipped (prevents
            stale evidence promotion).

    Returns:
        Number of evidence entries added.
    """
    if not optimization_log:
        return 0

    # Classify the problem
    classification = classify_problem(reference_code)
    problem_desc = f"{classification.tier.value}/{classification.op_type.value}"

    # Load skill library
    lib = SkillLibrary(skills_dir)
    lib.load()

    if not lib.all_skills:
        logger.info("No skills loaded — skipping evidence update")
        return 0

    # Extract winning rounds (SOL improved; falls back to speedup).
    winning_rounds = _extract_winning_rounds(optimization_log)

    # SOL-first promotion gate. Runs below the 0.50 SOL floor are
    # rejected outright (sub-baseline). Runs whose SOL is no better
    # than the leaderboard's prior best are also rejected (stale).
    # When SOL is missing (0.0), we fall back to the legacy speedup
    # floor so legacy profiles keep working during the SOL rollout.
    prior_best = _load_prior_best_sol(problem_id, hardware)
    winning_rounds = [w for w in winning_rounds if _passes_gate(w, prior_best)]
    if not winning_rounds:
        logger.debug("No winning rounds beat SOL/speedup gate — no evidence to record")
        return 0

    # Match winning strategies to skills and update evidence
    evidence_count = 0
    for win in winning_rounds:
        matched_skills = _match_to_skills(win, classification, lib)
        for skill in matched_skills:
            evidence = {
                "problem": problem_desc,
                "speedup": win["speedup"],
                "hardware": hardware,
                "round": win["round"],
                "strategy": win.get("strategy", ""),
                "bottleneck": win.get("bottleneck", ""),
                "sol_score": _extract_sol(win),
            }
            try:
                lib.update_evidence(skill.id, evidence)
                evidence_count += 1
                logger.debug(
                    "Evidence added to skill '%s': %.2fx on %s (%s)",
                    skill.name, win["speedup"], hardware, problem_desc,
                )
            except KeyError:
                continue

    # Persist updated skills
    if evidence_count > 0:
        lib.save()
        logger.debug("Updated %d evidence entries across skills", evidence_count)

    return evidence_count


def _extract_winning_rounds(rounds: list[dict]) -> list[dict]:
    """Find rounds that improved over the previous best in-run.

    Ranking prefers SOL when any round in the log exposes a non-zero
    SOL; otherwise falls back to speedup so legacy runs without SOL
    telemetry still produce evidence.
    """
    use_sol = any(_extract_sol(r) > 0.0 for r in rounds)

    winners = []
    best_so_far = 0.0
    for r in rounds:
        status = r.get("status", "")
        if status not in ("success", "SUCCESS"):
            continue
        metric = _extract_sol(r) if use_sol else r.get("speedup", 0.0)
        if metric > best_so_far and metric > 0:
            best_so_far = metric
            winners.append(r)
    return winners


def _extract_sol(round_dict: dict) -> float:
    """Extract SOL score from a round dict."""
    profile = round_dict.get("profile", {})
    if isinstance(profile, dict):
        sol = profile.get("sol_score", 0.0)
        try:
            return float(sol) if sol is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    return 0.0


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
    return speedup > _SPEEDUP_FLOOR


def _match_to_skills(
    winning_round: dict,
    classification: "ProblemClassification",
    lib: SkillLibrary,
) -> list[OptimizationSkill]:
    """Match a winning round to relevant skills via keyword overlap.

    Uses the round's strategy, bottleneck, and problem classification
    to find skills that could have contributed to the win.
    """
    from kernel_code.problem_classifier import ProblemClassification

    # Build a query from the round's context
    strategy = winning_round.get("strategy", "")
    bottleneck = winning_round.get("bottleneck", "")
    method = winning_round.get("method_applied", "")

    query_parts = [
        strategy,
        bottleneck,
        method,
        classification.op_type.value,
        classification.tier.value,
    ]
    query = " ".join(p for p in query_parts if p)

    if not query.strip():
        return []

    # Search skills library — return top 2 matches
    matches = lib.search_skills(query, top_k=2)
    return matches


def compute_skill_priority(skill: OptimizationSkill) -> float:
    """Compute a priority score for a skill based on accumulated evidence.

    Higher priority = more evidence of success. Used to rank skills
    when injecting into generator prompts.

    Scoring (SOL-driven):
    - Base: number of evidence entries (breadth)
    - Quality: average SOL (primary) + small speedup debug weight
    - Bonus for diversity of hardware/problem types (generality)

    The avg_speedup number is retained for human readability (shell
    surfaces still display it) but selection weight is on SOL.

    Returns:
        Priority score (higher is better). Range: 0.0 to ~100.0
    """
    if not skill.evidence:
        return 0.0

    # Breadth: number of evidence entries (capped at 20)
    breadth = min(len(skill.evidence), 20)

    # Quality: SOL first (primary), speedup retained for debug/readability
    sols = [e.get("sol_score", 0.0) for e in skill.evidence if (e.get("sol_score") or 0.0) > 0]
    avg_sol = sum(sols) / len(sols) if sols else 0.0

    speedups = [e.get("speedup", 0.0) for e in skill.evidence if e.get("speedup", 0.0) > 0]
    avg_speedup_debug = sum(speedups) / len(speedups) if speedups else 0.0

    # Generality: number of unique hardware+problem combos
    combos = set()
    for e in skill.evidence:
        hw = e.get("hardware", "")
        prob = e.get("problem", "")
        if hw or prob:
            combos.add(f"{hw}:{prob}")
    generality = min(len(combos), 10)  # cap at 10

    # SOL-driven quality with a small readability weight on speedup.
    # avg_sol is [0..1]; scaling by 10 lines it up with the prior
    # avg_speedup scale (~1x..10x). avg_speedup_debug gets a small
    # weight (0.5) so legacy entries without SOL still contribute but
    # don't dominate.
    score = breadth * 2.0 + avg_sol * 10.0 + avg_speedup_debug * 0.5 + generality * 3.0
    return score
