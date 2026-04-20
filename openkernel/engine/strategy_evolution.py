"""Strategy evolution system (Phase D outer loop).

Maintains a Pareto frontier of optimization strategies that evolve across
problems via GEPA-style reflection. Strategies are persisted as JSON files
and retrieved by keyword matching (same pattern as SkillLibrary).

Pareto dominance is computed over three dimensions:
  1. avg_speedup          — higher is better
  2. correctness_rate     — higher is better
  3. avg_iterations       — lower is better
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, field
from pathlib import Path

from openkernel.engine.world_model import Strategy

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Keyword tokenizer (matches SkillLibrary._tokenize)
# ------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase split, dropping very short noise words."""
    return [w for w in text.lower().split() if len(w) > 2]


# ------------------------------------------------------------------
# LLM prompt builders
# ------------------------------------------------------------------

def prompt_evolve_strategies(current_frontier_json: str, results_summary: str) -> str:
    """Build a prompt asking the LLM to propose new strategy variants.

    The LLM should reflect on what worked and what didn't, then propose
    2-3 new strategies via mutation (tweaking one strategy) or crossover
    (combining aspects of two strategies).
    """
    return f"""\
You are an expert GPU kernel optimization strategist managing a Pareto frontier
of optimization strategies. Your job is to reflect on recent results and propose
new strategy variants.

## Current Pareto Frontier
```json
{current_frontier_json}
```

## Recent Results Summary
{results_summary}

## Instructions

Analyze the current frontier and recent results. Propose 2-3 new strategy
variants using one of these approaches:

1. **Mutation**: Take an existing strategy and modify it — change the approach,
   add/remove techniques, adjust the target problem types.
2. **Crossover**: Combine successful aspects of two existing strategies into a
   new hybrid.
3. **Novel**: Propose an entirely new strategy targeting an underserved problem
   type or backend.

For each new strategy, explain the rationale and what improvement you expect.

## Response Format

Return ONLY valid JSON with this structure:
```json
{{
  "reasoning": "<analysis of what worked and what to try next>",
  "new_strategies": [
    {{
      "description": "<high-level description of the optimization approach>",
      "problem_types": ["<problem type 1>", "<problem type 2>"],
      "backend": "<triton|cuda|any>",
      "hardware_targets": ["<target 1>"],
      "rationale": "<why this strategy should be effective>"
    }}
  ]
}}
```

Respond with ONLY the JSON object, no other text."""


def prompt_select_strategies(problem_description: str, frontier_json: str) -> str:
    """Build a prompt asking the LLM to pick the best strategies for a problem.

    The LLM should analyze the problem and select the most promising strategies
    from the current Pareto frontier.
    """
    return f"""\
You are an expert GPU kernel optimization strategist. Given a problem
description and the current Pareto frontier of strategies, select the most
promising strategies to try.

## Problem Description
{problem_description}

## Available Strategies (Pareto Frontier)
```json
{frontier_json}
```

## Instructions

Analyze the problem and select 1-3 strategies from the frontier that are most
likely to succeed. Consider:

1. Does the problem type match the strategy's target problem types?
2. Does the backend match?
3. What is the strategy's track record (success/failure history)?
4. Would combining aspects of multiple strategies be beneficial?

## Response Format

Return ONLY valid JSON with this structure:
```json
{{
  "selected_strategy_ids": ["<id1>", "<id2>"],
  "reasoning": "<why these strategies were selected>",
  "suggested_intents": [
    {{
      "description": "<specific optimization intent derived from the strategy>",
      "priority": <float 0-1>,
      "source_strategy_id": "<which strategy this intent comes from>"
    }}
  ]
}}
```

Respond with ONLY the JSON object, no other text."""


# ------------------------------------------------------------------
# StrategyEvolution
# ------------------------------------------------------------------

class StrategyEvolution:
    """Manages a Pareto frontier of optimization strategies.

    Strategies are persisted as individual JSON files, loaded on startup,
    and evolved after optimization runs via LLM reflection.

    Usage::

        evo = StrategyEvolution("data/strategies")
        evo.load()
        relevant = evo.retrieve_strategies("GEMM compute-bound", backend="triton")
        # ... run optimization ...
        evo.update_strategy(strategy_id, {"speedup": 2.1, "correct": True, "iterations": 3})
        new_strats = evo.evolve(results_summary, llm_response)
        evo.save()
    """

    def __init__(self, strategies_dir: str | Path = "data/strategies") -> None:
        self._strategies_dir = Path(strategies_dir)
        self._strategies: dict[str, Strategy] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all .json files from *strategies_dir* into memory."""
        if not self._strategies_dir.exists():
            return
        for path in sorted(self._strategies_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                strategy = self._strategy_from_dict(data)
                self._strategies[strategy.id] = strategy
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load strategy from %s: %s", path, exc)

    def save(self, strategies_dir: str | Path | None = None) -> None:
        """Persist every strategy as an individual JSON file."""
        out_dir = Path(strategies_dir) if strategies_dir is not None else self._strategies_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for strategy in self._strategies.values():
            path = out_dir / f"{strategy.id}.json"
            with open(path, "w") as f:
                json.dump(self._strategy_to_dict(strategy), f, indent=2)
                f.write("\n")

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strategy_to_dict(strategy: Strategy) -> dict:
        """Convert a Strategy dataclass to a JSON-serializable dict."""
        return asdict(strategy)

    @staticmethod
    def _strategy_from_dict(data: dict) -> Strategy:
        """Reconstruct a Strategy from a plain dict."""
        return Strategy(
            id=data["id"],
            description=data.get("description", ""),
            problem_types=list(data.get("problem_types", [])),
            backend=data.get("backend", "any"),
            hardware_targets=list(data.get("hardware_targets", [])),
            success_history=list(data.get("success_history", [])),
            failure_history=list(data.get("failure_history", [])),
            pareto_scores=dict(data.get("pareto_scores", {})),
        )

    def serialize(self) -> list[dict]:
        """Serialize all strategies to a list of dicts."""
        return [self._strategy_to_dict(s) for s in self._strategies.values()]

    def deserialize(self, data: list[dict]) -> None:
        """Replace all strategies from a list of dicts."""
        self._strategies.clear()
        for item in data:
            strategy = self._strategy_from_dict(item)
            self._strategies[strategy.id] = strategy

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_strategy(self, strategy: Strategy) -> None:
        """Add (or replace) a strategy in the collection."""
        if not strategy.id:
            strategy.id = uuid.uuid4().hex[:12]
        self._strategies[strategy.id] = strategy

    @property
    def all_strategies(self) -> list[Strategy]:
        """Return every strategy currently in memory."""
        return list(self._strategies.values())

    @property
    def frontier(self) -> list[Strategy]:
        """Return all non-dominated strategies (the Pareto frontier).

        A strategy is on the frontier if no other strategy dominates it
        on all three Pareto dimensions.
        """
        strategies = self.all_strategies
        if not strategies:
            return []

        non_dominated: list[Strategy] = []
        for s in strategies:
            dominated = False
            for other in strategies:
                if other.id != s.id and self.is_dominated(s, other):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(s)
        return non_dominated

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_strategies(
        self,
        problem_description: str,
        backend: str | None = None,
        top_k: int = 3,
    ) -> list[Strategy]:
        """Keyword-match strategies against *problem_description*.

        Scoring: each strategy gets +1 for every query keyword that appears
        (case-insensitively) in its description, problem_types, or backend.
        Strategies whose backend does not match are excluded unless their
        backend is ``"any"``. Results are returned in descending relevance
        order, capped at *top_k*.
        """
        query_tokens = _tokenize(problem_description)
        if not query_tokens:
            return []

        scored: list[tuple[float, Strategy]] = []
        for strategy in self._strategies.values():
            # Backend filter
            if backend and strategy.backend not in (backend, "any"):
                continue

            # Build corpus for matching
            corpus = " ".join([
                strategy.description,
                " ".join(strategy.problem_types),
                strategy.backend,
                " ".join(strategy.hardware_targets),
            ]).lower()

            score = sum(1 for token in query_tokens if token in corpus)
            if score > 0:
                scored.append((score, strategy))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [strategy for _, strategy in scored[:top_k]]

    # ------------------------------------------------------------------
    # Strategy updates
    # ------------------------------------------------------------------

    def update_strategy(self, strategy_id: str, result: dict) -> None:
        """Update a strategy's success/failure history and recompute Pareto scores.

        Parameters
        ----------
        strategy_id : str
            The ID of the strategy to update.
        result : dict
            Must contain keys:
            - ``speedup`` (float): achieved speedup
            - ``correct`` (bool): whether the kernel was correct
            - ``iterations`` (int): inner-loop iterations consumed

            May optionally contain:
            - ``problem`` (str): description of the problem
            - ``backend`` (str): backend used
        """
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            logger.warning("update_strategy: unknown strategy_id %r", strategy_id)
            return

        entry = {
            "speedup": result.get("speedup", 0.0),
            "correct": result.get("correct", False),
            "iterations": result.get("iterations", 0),
            "problem": result.get("problem", ""),
            "backend": result.get("backend", ""),
        }

        if entry["correct"] and entry["speedup"] > 0:
            strategy.success_history.append(entry)
        else:
            strategy.failure_history.append(entry)

        # Recompute Pareto scores from full history
        self._recompute_pareto_scores(strategy)

    def _recompute_pareto_scores(self, strategy: Strategy) -> None:
        """Recompute a strategy's Pareto scores from its history.

        Dimensions:
          - avg_speedup: average speedup across successful runs (higher is better)
          - correctness_rate: successes / (successes + failures) (higher is better)
          - avg_iterations: average iterations across all runs (lower is better)
        """
        total_runs = len(strategy.success_history) + len(strategy.failure_history)
        if total_runs == 0:
            strategy.pareto_scores = {
                "avg_speedup": 0.0,
                "correctness_rate": 0.0,
                "avg_iterations": float("inf"),
            }
            return

        # Average speedup from successes only
        if strategy.success_history:
            avg_speedup = sum(
                e.get("speedup", 0.0) for e in strategy.success_history
            ) / len(strategy.success_history)
        else:
            avg_speedup = 0.0

        # Correctness rate
        correctness_rate = len(strategy.success_history) / total_runs

        # Average iterations across ALL runs (successes + failures)
        all_iterations = [
            e.get("iterations", 0)
            for e in strategy.success_history + strategy.failure_history
        ]
        avg_iterations = sum(all_iterations) / len(all_iterations) if all_iterations else float("inf")

        strategy.pareto_scores = {
            "avg_speedup": round(avg_speedup, 4),
            "correctness_rate": round(correctness_rate, 4),
            "avg_iterations": round(avg_iterations, 4),
        }

    # ------------------------------------------------------------------
    # Pareto dominance
    # ------------------------------------------------------------------

    def is_dominated(self, s1: Strategy, s2: Strategy) -> bool:
        """Return True if *s2* dominates *s1* on all Pareto dimensions.

        Dominance means s2 is at least as good as s1 on every dimension
        AND strictly better on at least one dimension.

        Dimensions:
          - avg_speedup: higher is better
          - correctness_rate: higher is better
          - avg_iterations: lower is better

        Strategies without Pareto scores (no history) are never dominated
        and never dominate — they remain on the frontier until evaluated.
        """
        p1 = s1.pareto_scores
        p2 = s2.pareto_scores

        # If either strategy has no scores, we can't determine dominance
        if not p1 or not p2:
            return False

        speedup_1 = p1.get("avg_speedup", 0.0)
        speedup_2 = p2.get("avg_speedup", 0.0)
        correct_1 = p1.get("correctness_rate", 0.0)
        correct_2 = p2.get("correctness_rate", 0.0)
        iter_1 = p1.get("avg_iterations", float("inf"))
        iter_2 = p2.get("avg_iterations", float("inf"))

        # s2 must be at least as good on all dimensions
        at_least_as_good = (
            speedup_2 >= speedup_1
            and correct_2 >= correct_1
            and iter_2 <= iter_1  # lower is better for iterations
        )

        if not at_least_as_good:
            return False

        # s2 must be strictly better on at least one dimension
        strictly_better = (
            speedup_2 > speedup_1
            or correct_2 > correct_1
            or iter_2 < iter_1
        )

        return strictly_better

    def prune_dominated(self) -> list[str]:
        """Remove strategies that are dominated on all Pareto dimensions.

        Returns a list of IDs that were pruned. Strategies with no Pareto
        scores (no history yet) are kept.
        """
        frontier_ids = {s.id for s in self.frontier}
        to_remove: list[str] = []

        for sid, strategy in self._strategies.items():
            # Keep strategies with no scores (untested)
            if not strategy.pareto_scores:
                continue
            if sid not in frontier_ids:
                to_remove.append(sid)

        for sid in to_remove:
            logger.info("Pruning dominated strategy: %s", sid)
            del self._strategies[sid]

        return to_remove

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(self, results_summary: str, llm_response: str) -> list[Strategy]:
        """Parse LLM-proposed new strategies and add non-dominated ones.

        Parameters
        ----------
        results_summary : str
            Human-readable summary of recent optimization results.
        llm_response : str
            Raw LLM response text (expected JSON with ``new_strategies`` key).

        Returns
        -------
        list[Strategy]
            The newly added strategies.
        """
        try:
            data = _parse_json_from_response(llm_response)
        except ValueError as exc:
            logger.warning("evolve: failed to parse LLM response: %s", exc)
            return []

        new_strategies_data = data.get("new_strategies", [])
        if not new_strategies_data:
            logger.info("evolve: LLM proposed no new strategies")
            return []

        added: list[Strategy] = []
        for item in new_strategies_data:
            strategy = Strategy(
                id=uuid.uuid4().hex[:12],
                description=item.get("description", ""),
                problem_types=list(item.get("problem_types", [])),
                backend=item.get("backend", "any"),
                hardware_targets=list(item.get("hardware_targets", [])),
                success_history=[],
                failure_history=[],
                pareto_scores={},
            )

            if not strategy.description:
                continue

            # New strategies with no history are always non-dominated
            # (they have no scores to compare), so we add them.
            # They'll be pruned later once they have scores and
            # are shown to be dominated.
            self.add_strategy(strategy)
            added.append(strategy)
            logger.info(
                "evolve: added new strategy %r (%s)",
                strategy.description[:60],
                strategy.id,
            )

        return added

    # ------------------------------------------------------------------
    # LLM context formatting
    # ------------------------------------------------------------------

    def frontier_to_json_str(self) -> str:
        """Format the current frontier as a JSON string for LLM prompts."""
        frontier = self.frontier
        items = []
        for s in frontier:
            items.append({
                "id": s.id,
                "description": s.description,
                "problem_types": s.problem_types,
                "backend": s.backend,
                "hardware_targets": s.hardware_targets,
                "pareto_scores": s.pareto_scores,
                "successes": len(s.success_history),
                "failures": len(s.failure_history),
            })
        return json.dumps(items, indent=2)

    @staticmethod
    def strategies_to_context_string(strategies: list[Strategy]) -> str:
        """Format a list of strategies into a string suitable for LLM prompts."""
        if not strategies:
            return "No relevant strategies found."

        parts: list[str] = []
        for i, s in enumerate(strategies, 1):
            types_str = ", ".join(s.problem_types) if s.problem_types else "any"
            targets_str = ", ".join(s.hardware_targets) if s.hardware_targets else "any"
            scores = s.pareto_scores
            scores_str = (
                f"speedup={scores.get('avg_speedup', 'N/A')}, "
                f"correctness={scores.get('correctness_rate', 'N/A')}, "
                f"avg_iters={scores.get('avg_iterations', 'N/A')}"
            ) if scores else "no history yet"

            block = (
                f"### Strategy {i}: {s.description[:80]}\n"
                f"**ID**: {s.id}\n"
                f"**Problem types**: {types_str}\n"
                f"**Backend**: {s.backend}\n"
                f"**Hardware**: {targets_str}\n"
                f"**Track record**: {len(s.success_history)} successes, "
                f"{len(s.failure_history)} failures\n"
                f"**Pareto scores**: {scores_str}"
            )
            parts.append(block)
        return "\n\n".join(parts)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_json_from_response(response: str) -> dict:
    """Extract a JSON object from an LLM response (light version).

    Handles raw JSON, fenced code blocks, and embedded JSON.
    Raises ValueError if no valid JSON can be extracted.
    """
    import re

    text = response.strip()

    # Strategy 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from code fences
    for pattern in [r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    # Strategy 3: find outermost { ... }
    brace_start = text.find("{")
    if brace_start != -1:
        brace_end = text.rfind("}")
        if brace_end > brace_start:
            candidate = text[brace_start:brace_end + 1]
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                # Try removing trailing commas
                cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    result = json.loads(cleaned)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass

    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")
