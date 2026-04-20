"""Skill library for optimization pattern retrieval.

Stores OptimizationSkill entries (pre-seeded from data/skills/*.json and
accumulated during optimization runs) and provides keyword-based retrieval
for seeding Generator prompts.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class OptimizationSkill:
    """A reusable optimization pattern that can seed kernel generation."""

    id: str
    name: str
    trigger: str
    approach: str
    backend: str  # "triton", "cuda", or "any"
    evidence: list[dict] = field(default_factory=list)
    code_template: str | None = None
    tags: list[str] = field(default_factory=list)


class SkillLibrary:
    """In-memory skill library backed by JSON files on disk.

    Usage::

        lib = SkillLibrary()
        lib.load()  # reads data/skills/*.json
        matches = lib.search_skills("softmax reduction memory-bound", backend="triton")
    """

    def __init__(self, skills_dir: str | Path = "data/skills") -> None:
        self._skills_dir = Path(skills_dir)
        self._skills: dict[str, OptimizationSkill] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all .json files from *skills_dir* into memory."""
        if not self._skills_dir.exists():
            return
        # Known dataclass fields -- filter out extra keys (e.g. auto_trigger)
        _known_fields = {f.name for f in OptimizationSkill.__dataclass_fields__.values()}
        for path in sorted(self._skills_dir.glob("*.json")):
            with open(path) as f:
                data = json.load(f)
            filtered = {k: v for k, v in data.items() if k in _known_fields}
            skill = OptimizationSkill(**filtered)
            self._skills[skill.id] = skill

    def save(self, skills_dir: str | Path | None = None) -> None:
        """Persist every skill as an individual JSON file."""
        out_dir = Path(skills_dir) if skills_dir is not None else self._skills_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for skill in self._skills.values():
            path = out_dir / f"{skill.id}.json"
            with open(path, "w") as f:
                json.dump(asdict(skill), f, indent=2)
                f.write("\n")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_skill(self, skill: OptimizationSkill) -> None:
        """Add (or replace) a skill in the library."""
        if not skill.id:
            skill.id = uuid.uuid4().hex[:12]
        self._skills[skill.id] = skill

    def update_evidence(self, skill_id: str, evidence: dict) -> None:
        """Append an evidence entry to an existing skill.

        Args:
            skill_id: ID of the skill to update.
            evidence: Dict with keys like ``problem``, ``speedup``,
                ``hardware``, ``iteration``, ``intent``.

        Raises:
            KeyError: If *skill_id* does not exist in the library.
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            raise KeyError(f"Skill {skill_id!r} not found in library")
        skill.evidence.append(evidence)

    @property
    def all_skills(self) -> list[OptimizationSkill]:
        """Return every skill currently in memory."""
        return list(self._skills.values())

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search_skills(
        self,
        problem_description: str,
        backend: str | None = None,
        top_k: int = 5,
    ) -> list[OptimizationSkill]:
        """Keyword-match skills against *problem_description*.

        Scoring: each skill gets +1 for every query keyword that appears
        (case-insensitively) in its trigger, approach, name, or tags.  Skills
        whose backend does not match are excluded unless their backend is
        ``"any"``.  Results are returned in descending relevance order, capped
        at *top_k*.
        """
        query_tokens = _tokenize(problem_description)
        if not query_tokens:
            return []

        scored: list[tuple[float, OptimizationSkill]] = []
        for skill in self._skills.values():
            # Backend filter
            if backend and skill.backend not in (backend, "any"):
                continue

            # Build the text corpus for this skill
            corpus = " ".join(
                [
                    skill.trigger,
                    skill.approach,
                    skill.name,
                    " ".join(skill.tags),
                ]
            ).lower()

            score = sum(1 for token in query_tokens if token in corpus)
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [skill for _, skill in scored[:top_k]]

    # ------------------------------------------------------------------
    # LLM context formatting
    # ------------------------------------------------------------------

    @staticmethod
    def to_context_string(skills: list[OptimizationSkill]) -> str:
        """Format a list of skills into a string suitable for LLM prompts."""
        if not skills:
            return "No relevant optimization skills found."

        parts: list[str] = []
        for i, skill in enumerate(skills, 1):
            tags_str = ", ".join(skill.tags) if skill.tags else "none"
            block = (
                f"### Skill {i}: {skill.name}\n"
                f"**Trigger**: {skill.trigger}\n"
                f"**Approach**: {skill.approach}\n"
                f"**Backend**: {skill.backend}\n"
                f"**Tags**: {tags_str}"
            )
            if skill.code_template:
                block += f"\n**Code template**:\n```\n{skill.code_template}\n```"
            parts.append(block)
        return "\n\n".join(parts)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase split, dropping very short noise words."""
    return [w for w in text.lower().split() if len(w) > 2]
