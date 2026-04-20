"""Automatic skill triggering based on bottleneck detection.

Suggests relevant optimization skills when a bottleneck type is identified
during profiling. Uses lightweight keyword matching against the skill library
and per-skill auto_trigger metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

from openkernel.memory.skill_library import OptimizationSkill

# Project root for loading skills
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Mapping: bottleneck_type -> relevant skill trigger keywords.
# A skill matches if any of its tags, trigger text, or approach text contains
# one of the keywords associated with the detected bottleneck type.
_BOTTLENECK_SKILL_MAP: dict[str, list[str]] = {
    "memory_bound": [
        "reduction",
        "elementwise",
        "layernorm",
        "softmax",
        "coalescing",
        "vectorize",
        "bandwidth",
    ],
    "compute_bound": [
        "gemm",
        "matmul",
        "conv",
        "tensor_core",
        "tiling",
        "attention",
    ],
    "latency_bound": [
        "fusion",
        "persistent",
        "epilogue",
        "pipeline",
    ],
}


def _load_skill_dicts(skills_dir: Path | None = None) -> list[dict]:
    """Load raw skill dicts from JSON files."""
    directory = skills_dir or (_PROJECT_ROOT / "data" / "skills")
    if not directory.exists():
        return []
    skills: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            with open(path) as f:
                skills.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return skills


def _skill_corpus(skill: dict) -> str:
    """Build a lowercase text corpus from a skill dict for keyword matching."""
    parts = [
        skill.get("trigger", ""),
        skill.get("approach", ""),
        skill.get("name", ""),
        " ".join(skill.get("tags", [])),
    ]
    return " ".join(parts).lower()


def _compute_avg_speedup(skill: dict) -> float:
    """Compute average speedup from a skill's evidence entries."""
    evidence = skill.get("evidence", [])
    speedups = [e.get("speedup", 0) for e in evidence if e.get("speedup")]
    if not speedups:
        return 0.0
    return sum(speedups) / len(speedups)


def suggest_skills(
    bottleneck_type: str,
    problem_description: str = "",
    skill_library: list[dict] | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Suggest relevant skills based on bottleneck and problem type.

    Scores each skill by counting keyword hits from the bottleneck-specific
    keyword list and from the problem description. Skills with an
    ``auto_trigger`` field get a bonus when their declared bottleneck types
    match and their ``min_speedup_evidence`` threshold is met.

    Args:
        bottleneck_type: One of ``"memory_bound"``, ``"compute_bound"``,
            ``"latency_bound"``, or ``"unknown"``.
        problem_description: Free-text description of the optimization
            problem (e.g. ``"softmax over 4096-dim vectors"``).
        skill_library: Optional pre-loaded list of skill dicts. If *None*,
            skills are loaded from ``data/skills/*.json``.
        top_k: Maximum number of suggestions to return.

    Returns:
        List of dicts with keys: ``skill_id``, ``name``, ``reason``,
        ``evidence_count``, ``avg_speedup``.
    """
    if bottleneck_type not in _BOTTLENECK_SKILL_MAP:
        return []

    skills = skill_library if skill_library is not None else _load_skill_dicts()
    if not skills:
        return []

    keywords = _BOTTLENECK_SKILL_MAP[bottleneck_type]
    problem_tokens = [w.lower() for w in problem_description.split() if len(w) > 2]

    scored: list[tuple[float, dict, str]] = []

    for skill in skills:
        corpus = _skill_corpus(skill)
        score = 0.0
        reasons: list[str] = []

        # Score from bottleneck-keyword matches
        matched_keywords = [kw for kw in keywords if kw in corpus]
        if matched_keywords:
            score += len(matched_keywords)
            reasons.append(f"matches {bottleneck_type} keywords: {', '.join(matched_keywords[:3])}")

        # Score from problem description tokens
        matched_problem = [tok for tok in problem_tokens if tok in corpus]
        if matched_problem:
            score += len(matched_problem) * 0.5
            reasons.append(f"matches problem: {', '.join(matched_problem[:3])}")

        # Bonus from auto_trigger metadata
        auto_trigger = skill.get("auto_trigger")
        if auto_trigger:
            trigger_bottlenecks = auto_trigger.get("bottleneck_types", [])
            if bottleneck_type in trigger_bottlenecks:
                score += 2.0
                reasons.append("auto_trigger: bottleneck match")

            # Check problem keywords from auto_trigger
            trigger_keywords = auto_trigger.get("problem_keywords", [])
            problem_lower = problem_description.lower()
            trigger_matched = [kw for kw in trigger_keywords if kw in problem_lower]
            if trigger_matched:
                score += len(trigger_matched)
                reasons.append(f"auto_trigger keywords: {', '.join(trigger_matched[:3])}")

            # Check min_speedup_evidence threshold
            min_speedup = auto_trigger.get("min_speedup_evidence", 0)
            avg_spd = _compute_avg_speedup(skill)
            if avg_spd > 0 and avg_spd < min_speedup:
                score *= 0.5  # penalise if evidence doesn't meet threshold

        if score > 0:
            reason = "; ".join(reasons) if reasons else "keyword match"
            scored.append((score, skill, reason))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[dict] = []
    for _score, skill, reason in scored[:top_k]:
        avg_speedup = _compute_avg_speedup(skill)
        results.append({
            "skill_id": skill.get("id", ""),
            "name": skill.get("name", ""),
            "reason": reason,
            "evidence_count": len(skill.get("evidence", [])),
            "avg_speedup": avg_speedup,
        })

    return results


def format_skill_suggestions(suggestions: list[dict], bottleneck_type: str = "") -> str:
    """Format suggestions for display in shell/optimization feed.

    Args:
        suggestions: Output of :func:`suggest_skills`.
        bottleneck_type: The bottleneck type string (for the header).

    Returns:
        A multi-line string ready for console output.
    """
    if not suggestions:
        return ""

    label = bottleneck_type.upper() if bottleneck_type else "DETECTED"
    lines = [f"Suggested skills based on {label} bottleneck:"]

    for s in suggestions:
        speedup_info = ""
        if s["avg_speedup"] > 0:
            speedup_info = f" ({s['avg_speedup']:.1f}x avg on similar problems)"
        elif s["evidence_count"] > 0:
            speedup_info = f" ({s['evidence_count']} evidence entries)"

        lines.append(f"  -> /skill:{s['skill_id']}{speedup_info}")

    return "\n".join(lines)
