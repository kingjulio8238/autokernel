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


def extract_and_update_evidence(
    optimization_log: list[dict],
    reference_code: str,
    hardware: str,
    skills_dir: str | Path = "data/skills",
) -> int:
    """Extract winning strategies from a completed run and update skill evidence.

    Looks at rounds where speedup improved, identifies which strategies
    and bottleneck types led to gains, and records evidence entries in
    matching skills.

    Args:
        optimization_log: List of round dicts from AutoResult.round_history.
        reference_code: Original reference code (for problem classification).
        hardware: GPU type used (e.g., "L40S", "H100").
        skills_dir: Path to skill JSON files.

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

    # Extract winning rounds (speedup improved over previous best)
    winning_rounds = _extract_winning_rounds(optimization_log)
    # Filter out rounds that didn't beat the baseline (2% noise margin).
    # Without this, a run whose best was 0.85x would still pollute the
    # skill library with "evidence" for losing strategies.
    winning_rounds = [w for w in winning_rounds if w.get("speedup", 0.0) > 1.02]
    if not winning_rounds:
        logger.debug("No winning rounds beat baseline — no evidence to record")
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
    """Find rounds where speedup improved over the previous best."""
    winners = []
    best_so_far = 0.0
    for r in rounds:
        speedup = r.get("speedup", 0.0)
        status = r.get("status", "")
        if speedup > best_so_far and speedup > 0 and status in ("success", "SUCCESS"):
            best_so_far = speedup
            winners.append(r)
    return winners


def _extract_sol(round_dict: dict) -> float:
    """Extract SOL score from a round dict."""
    profile = round_dict.get("profile", {})
    if isinstance(profile, dict):
        return profile.get("sol_score", 0.0)
    return 0.0


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

    Scoring:
    - Base: number of evidence entries (breadth)
    - Weighted by average speedup achieved (quality)
    - Bonus for diversity of hardware/problem types (generality)

    Returns:
        Priority score (higher is better). Range: 0.0 to ~100.0
    """
    if not skill.evidence:
        return 0.0

    # Breadth: number of evidence entries (capped at 20)
    breadth = min(len(skill.evidence), 20)

    # Quality: average speedup across evidence
    speedups = [e.get("speedup", 0.0) for e in skill.evidence if e.get("speedup", 0.0) > 0]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

    # Generality: number of unique hardware+problem combos
    combos = set()
    for e in skill.evidence:
        hw = e.get("hardware", "")
        prob = e.get("problem", "")
        if hw or prob:
            combos.add(f"{hw}:{prob}")
    generality = min(len(combos), 10)  # cap at 10

    # Weighted score
    score = breadth * 2.0 + avg_speedup * 10.0 + generality * 3.0
    return score
