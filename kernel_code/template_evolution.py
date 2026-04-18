"""Template evolution — closes the optimization flywheel.

Winning kernels feed back into improved skill templates so the next
optimization starts from a stronger baseline.  The flow:

1. ``record_win()`` is called from the ``post_keep`` hook whenever a
   kernel beats the current best.
2. Optimization patterns are detected in the kernel source via
   simple keyword matching (no heavy NLP).
3. After a skill accumulates enough wins (default: 5), an LLM is
   asked to merge winning patterns into an improved template.
4. The operator reviews and approves the evolution via ``/evolve approve``.

State is persisted under ``.kernel-code/evolution/{skill_id}.json``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum wins before proposing an evolution.
_EVOLUTION_THRESHOLD = 5

# Pattern must appear in >= this fraction of wins to be incorporated.
_PATTERN_ADOPTION_RATIO = 0.60


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvolutionRecord:
    """Tracks a winning kernel for a skill."""

    skill_id: str
    speedup: float
    hardware: str
    backend: str
    kernel_code: str
    patterns_detected: list[str]  # e.g., ["float4_loads", "shared_memory"]
    timestamp: str


@dataclass
class SkillEvolutionState:
    """Tracks evolution state for a single skill."""

    skill_id: str
    wins: list[EvolutionRecord] = field(default_factory=list)
    pattern_frequency: dict[str, int] = field(default_factory=dict)
    avg_speedup: float = 0.0
    evolved_template: str | None = None  # proposed new template
    evolution_approved: bool = False


# ---------------------------------------------------------------------------
# Pattern detection rules
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, pattern name).
# Order does not matter — all patterns are checked independently.
_PATTERN_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"tl\.load\s*\(.*mask\s*=", re.DOTALL), "masked_loads"),
    (re.compile(r"float4|vectorized|vec_load", re.IGNORECASE), "vectorized_loads"),
    (re.compile(r"tl\.dot\b"), "tensor_cores"),
    (re.compile(r"@triton\.autotune"), "autotuning"),
    (re.compile(r"\bshared\b|tl\.zeros\s*\(\s*\[.*(?:128|256|512|1024)", re.DOTALL), "shared_memory"),
    (re.compile(r"\bpipeline\b|\basync\b", re.IGNORECASE), "pipeline_staging"),
    (re.compile(r"\bshuffle\b|\bwarp\b", re.IGNORECASE), "warp_primitives"),
    (re.compile(r"\bpersistent\b|\bwhile\b.*\bnum_blocks\b|\bwhile\b.*\bblock_id\b", re.DOTALL | re.IGNORECASE), "persistent_kernel"),
    (re.compile(r"\bonline\b|running.?max|running.?sum", re.IGNORECASE), "online_algorithm"),
    (re.compile(r"\bsplit\b.*\bK\b|\bsplit.?k\b", re.IGNORECASE), "split_k"),
    (re.compile(r"\bregister\b.*\bblock\b|\bREG_BLOCK\b", re.IGNORECASE), "register_blocking"),
    (re.compile(r"coalesced|coalesce", re.IGNORECASE), "coalesced_access"),
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TemplateEvolver:
    """Tracks winning kernels and evolves skill templates over time."""

    def __init__(self, evolution_dir: str | Path = ".kernel-code/evolution") -> None:
        self._dir = Path(evolution_dir)
        self._states: dict[str, SkillEvolutionState] = {}
        self.load()

    # ------------------------------------------------------------------
    # Recording wins
    # ------------------------------------------------------------------

    def record_win(
        self,
        skill_id: str,
        kernel_code: str,
        speedup: float,
        hardware: str,
        backend: str,
    ) -> None:
        """Record a winning kernel for a skill.  Called from the post_keep hook."""
        patterns = self.detect_patterns(kernel_code)

        record = EvolutionRecord(
            skill_id=skill_id,
            speedup=speedup,
            hardware=hardware,
            backend=backend,
            kernel_code=kernel_code,
            patterns_detected=patterns,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        state = self._states.setdefault(
            skill_id, SkillEvolutionState(skill_id=skill_id)
        )
        state.wins.append(record)

        # Update pattern frequency
        for pattern in patterns:
            state.pattern_frequency[pattern] = state.pattern_frequency.get(pattern, 0) + 1

        # Update average speedup
        total = sum(w.speedup for w in state.wins)
        state.avg_speedup = total / len(state.wins)

        self.save()

        if self.should_evolve(skill_id):
            logger.info(
                "Skill %r has %d wins (avg %.2fx) — ready for template evolution",
                skill_id,
                len(state.wins),
                state.avg_speedup,
            )

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def detect_patterns(self, kernel_code: str) -> list[str]:
        """Detect optimization patterns in kernel code via keyword matching."""
        found: list[str] = []
        for pattern_re, pattern_name in _PATTERN_RULES:
            if pattern_re.search(kernel_code):
                found.append(pattern_name)
        return found

    # ------------------------------------------------------------------
    # Evolution readiness
    # ------------------------------------------------------------------

    def should_evolve(self, skill_id: str) -> bool:
        """Check if a skill has enough wins to propose evolution."""
        state = self._states.get(skill_id)
        if state is None:
            return False
        return len(state.wins) >= _EVOLUTION_THRESHOLD

    # ------------------------------------------------------------------
    # LLM-based evolution
    # ------------------------------------------------------------------

    async def propose_evolution(
        self, skill_id: str, llm_provider=None, original_template: str | None = None
    ) -> str:
        """Generate an evolved template using the LLM.

        Sends the LLM:
        - Original template
        - Top 3 winning kernels (by speedup)
        - Pattern frequency data
        - Instructions to merge common winning patterns

        Returns:
            The proposed new template code.

        Raises:
            ValueError: If the skill has no recorded wins or llm_provider is None.
        """
        state = self._states.get(skill_id)
        if state is None or not state.wins:
            raise ValueError(f"No wins recorded for skill {skill_id!r}")
        if llm_provider is None:
            raise ValueError("An LLM provider is required to propose evolution")

        # Top 3 wins by speedup
        top_wins = sorted(state.wins, key=lambda w: w.speedup, reverse=True)[:3]

        # Determine which patterns appear in >= 60% of wins
        n_wins = len(state.wins)
        dominant_patterns = [
            p for p, count in state.pattern_frequency.items()
            if count / n_wins >= _PATTERN_ADOPTION_RATIO
        ]

        prompt = self._build_evolution_prompt(
            skill_id=skill_id,
            original_template=original_template or "(no original template available)",
            top_wins=top_wins,
            pattern_frequency=state.pattern_frequency,
            dominant_patterns=dominant_patterns,
            n_wins=n_wins,
            avg_speedup=state.avg_speedup,
        )

        evolved_code = await llm_provider.generate(prompt)

        # Extract code block if the LLM wrapped it
        evolved_code = self._extract_code_block(evolved_code)

        state.evolved_template = evolved_code
        self.save()

        return evolved_code

    def _build_evolution_prompt(
        self,
        skill_id: str,
        original_template: str,
        top_wins: list[EvolutionRecord],
        pattern_frequency: dict[str, int],
        dominant_patterns: list[str],
        n_wins: int,
        avg_speedup: float,
    ) -> str:
        """Build the LLM prompt for template evolution."""
        wins_section = ""
        for i, win in enumerate(top_wins, 1):
            wins_section += (
                f"\n--- Winning kernel #{i} ({win.speedup:.2f}x on {win.hardware}) ---\n"
                f"Patterns: {', '.join(win.patterns_detected)}\n"
                f"```python\n{win.kernel_code}\n```\n"
            )

        freq_lines = "\n".join(
            f"  {p}: {c}/{n_wins} wins ({c / n_wins:.0%})"
            for p, c in sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)
        )

        return f"""\
You are an expert GPU kernel optimization engineer. Your task is to evolve
a kernel template by incorporating patterns that consistently win.

## Original template for skill "{skill_id}"
```python
{original_template}
```

## Winning kernel data ({n_wins} total wins, avg {avg_speedup:.2f}x speedup)
{wins_section}

## Pattern frequency across all wins
{freq_lines}

## Dominant patterns (appearing in 60%+ of wins)
{', '.join(dominant_patterns) if dominant_patterns else 'None yet'}

## Instructions
1. Start from the original template structure.
2. Incorporate the dominant patterns that appear in 60%+ of wins.
3. Use the top winning kernels as guidance for HOW to incorporate each pattern.
4. Keep the code compilable and correct — it must be a valid Triton/CUDA kernel.
5. The evolved template should achieve ~1.5x speedup without further optimization.
6. Preserve the ModelNew class interface (forward method with same signature).
7. Return ONLY the evolved Python code, no explanation.

Respond with the complete evolved template code:"""

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extract code from a markdown code block, if present."""
        # Match ```python ... ``` or ``` ... ```
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    def approve_evolution(self, skill_id: str, skill_library) -> bool:
        """Apply the evolved template to the skill library.

        Updates the skill's ``code_template`` with the evolved version
        and persists the change.

        Args:
            skill_id: The skill to evolve.
            skill_library: A :class:`~openkernel.memory.skill_library.SkillLibrary`.

        Returns:
            True if the template was applied, False if there was nothing to apply.
        """
        state = self._states.get(skill_id)
        if state is None or state.evolved_template is None:
            return False

        # Find the skill in the library and update its template
        skill = None
        for s in skill_library.all_skills:
            if s.id == skill_id:
                skill = s
                break

        if skill is None:
            logger.warning("Skill %r not found in library — cannot apply evolution", skill_id)
            return False

        skill.code_template = state.evolved_template
        state.evolution_approved = True
        skill_library.save()
        self.save()

        logger.info("Evolved template applied to skill %r", skill_id)
        return True

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_evolution_status(self) -> list[dict]:
        """Get status of all skills' evolution.

        Returns:
            List of dicts with keys: skill_id, wins, avg_speedup,
            ready_to_evolve, has_proposal, approved, top_patterns.
        """
        results: list[dict] = []
        for skill_id, state in sorted(self._states.items()):
            # Top patterns (up to 5, sorted by frequency)
            top_patterns = sorted(
                state.pattern_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            results.append({
                "skill_id": skill_id,
                "wins": len(state.wins),
                "avg_speedup": round(state.avg_speedup, 2),
                "ready_to_evolve": self.should_evolve(skill_id),
                "has_proposal": state.evolved_template is not None,
                "approved": state.evolution_approved,
                "top_patterns": [
                    f"{name} ({count})" for name, count in top_patterns
                ],
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load evolution state from .kernel-code/evolution/."""
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                skill_id = data["skill_id"]
                wins = [EvolutionRecord(**w) for w in data.get("wins", [])]
                state = SkillEvolutionState(
                    skill_id=skill_id,
                    wins=wins,
                    pattern_frequency=data.get("pattern_frequency", {}),
                    avg_speedup=data.get("avg_speedup", 0.0),
                    evolved_template=data.get("evolved_template"),
                    evolution_approved=data.get("evolution_approved", False),
                )
                self._states[skill_id] = state
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load evolution state from %s: %s", path, exc)

    def save(self) -> None:
        """Save evolution state to .kernel-code/evolution/."""
        self._dir.mkdir(parents=True, exist_ok=True)
        for skill_id, state in self._states.items():
            path = self._dir / f"{skill_id}.json"
            data = {
                "skill_id": state.skill_id,
                "wins": [asdict(w) for w in state.wins],
                "pattern_frequency": state.pattern_frequency,
                "avg_speedup": state.avg_speedup,
                "evolved_template": state.evolved_template,
                "evolution_approved": state.evolution_approved,
            }
            path.write_text(json.dumps(data, indent=2) + "\n")
