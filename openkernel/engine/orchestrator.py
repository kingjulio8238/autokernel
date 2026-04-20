"""Orchestrator: drives the K-Search-style world model loop.

Ties together the intent tree (world model), LLM-based tree operations,
and the inner refinement loop into a complete optimization engine.

Key classes:
- InnerLoopInterface: Protocol for any refinement loop implementation
- RefinementResult: outcome of a single inner-loop run on one intent
- Orchestrator: the main search driver
- OptimizationResult: final output of a full optimization run
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import re

from openkernel.engine.strategy_evolution import StrategyEvolution
from openkernel.engine.world_model import IntentNode, IntentStatus, IntentTree
from openkernel.engine.world_model_prompts import (
    prompt_propose_intents,
    prompt_prune_decision,
    prompt_update_priorities,
)
from openkernel.eval.types import EvalResult, EvalStatus
from openkernel.traces.capture import TraceCapture
from openkernel.traces.storage import save_trace
from openkernel.traces.types import OptimizationTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inner loop protocol + result
# ---------------------------------------------------------------------------


@dataclass
class RefinementResult:
    """Outcome of one inner-loop refinement cycle for a single intent."""

    status: str  # "succeeded" | "failed" | "error"
    best_kernel: str  # best kernel source code produced (empty on failure)
    best_speedup: float  # best speedup achieved (0.0 on failure)
    iterations: int  # number of generate-eval-diagnose iterations used
    critic_feedback: str  # last CriticDiagnosis summary text
    total_tokens: int = 0  # LLM tokens consumed in this refinement
    total_cost_usd: float = 0.0  # estimated LLM cost for this refinement


@runtime_checkable
class InnerLoopInterface(Protocol):
    """Protocol for inner refinement loops.

    Any implementation must accept an intent description, the reference code,
    backend, and config dict, and return a RefinementResult.
    """

    def refine(
        self,
        intent: IntentNode,
        reference_code: str,
        backend: str,
        config: dict,
    ) -> RefinementResult: ...


# ---------------------------------------------------------------------------
# Optimization result
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Final output of a complete orchestrated optimization run."""

    final_kernel: str  # best kernel source code found
    final_speedup: float  # best speedup achieved
    tree_history: list[dict]  # snapshots of the tree at each iteration
    iterations_total: int  # total inner-loop iterations consumed
    wall_time_seconds: float = 0.0
    intents_explored: int = 0
    intents_succeeded: int = 0
    intents_failed: int = 0
    stagnation_triggered: bool = False
    trace: OptimizationTrace | None = None  # full trace if capture was enabled
    total_tokens: int = 0  # total LLM tokens consumed across all refinements
    total_cost_usd: float = 0.0  # total estimated cost (LLM + compute)
    budget_exceeded: bool = False  # True if optimization stopped early due to budget


# ---------------------------------------------------------------------------
# LLM caller abstraction (pluggable)
# ---------------------------------------------------------------------------


class LLMCaller(Protocol):
    """Protocol for calling an LLM with a prompt and getting text back."""

    def call(self, prompt: str) -> str: ...


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output that may include markdown fences.

    Handles common LLM response patterns:
    - Pure JSON
    - ```json ... ```
    - ``` ... ```
    - Text before/after the JSON block
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise json.JSONDecodeError("No valid JSON found in LLM response", text, 0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Drives the K-Search-style world model optimization loop.

    Flow:
    1. Initialize the intent tree with a root node
    2. Ask LLM to propose initial intents (children of root)
    3. Select highest-priority pending intent
    4. Run the inner loop on that intent
    5. Update the tree with results
    6. Ask LLM to update priorities
    7. Check for stagnation; if detected, ask LLM about pruning
    8. Repeat until max iterations or convergence

    Non-monotonic path support: the orchestrator tracks children of nodes
    that regressed — if a child improves beyond the global best, it is
    still accepted. Strategy quality is judged by subtree outcomes, not
    just immediate results.
    """

    def __init__(
        self,
        inner_loop: InnerLoopInterface,
        config: dict,
        llm: LLMCaller | None = None,
        strategy_evolution: StrategyEvolution | None = None,
        trace_capture: TraceCapture | None = None,
    ) -> None:
        self._inner_loop = inner_loop
        self._config = config
        if llm is None:
            raise ValueError(
                "An LLMCaller instance is required. "
                "Pass an LLMProvider or, for tests, a MockLLMCaller from tests.mocks."
            )
        self._llm = llm
        self._strategy_evolution = strategy_evolution
        self._trace_capture = trace_capture

        # Defaults from config, with fallbacks
        self._max_iterations: int = config.get("max_iterations", 100)
        self._stagnation_threshold: int = config.get("stagnation_threshold", 7)
        self._max_retries_per_intent: int = config.get("max_retries_per_intent", 5)
        self._max_budget_usd: float | None = config.get("max_budget_usd")

    def optimize(
        self,
        reference_code: str,
        backend: str = "triton",
        hardware: str = "L40S",
    ) -> OptimizationResult:
        """Run the full world-model-guided optimization loop.

        Returns an OptimizationResult with the best kernel found, the final
        speedup, and the full tree history.
        """
        start_time = time.monotonic()
        tree = IntentTree(root_description=f"Optimize kernel for {hardware} ({backend})")
        tree_history: list[dict] = [tree.serialize()]

        # Start trace capture if enabled
        if self._trace_capture is not None:
            model_id = self._config.get("model_id", "unknown")
            problem_id = self._config.get("problem_id", "unknown")
            self._trace_capture.start_session(
                problem_id=problem_id,
                hardware=hardware,
                backend=backend,
                model_id=model_id,
            )

        best_kernel = ""
        best_speedup = 0.0
        total_iterations = 0
        total_tokens = 0
        total_llm_cost = 0.0
        total_eval_count = 0
        intents_explored = 0
        intents_succeeded = 0
        intents_failed = 0
        stagnation_triggered = False
        budget_exceeded = False

        # Track which strategies were used (for post-optimization updates)
        active_strategy_ids: list[str] = []

        # Step 0: If strategy evolution is available, seed the tree with
        # strategy-derived intents before the normal LLM proposal step.
        if self._strategy_evolution is not None:
            problem_desc = f"Optimize kernel for {hardware} ({backend}): {reference_code[:200]}"
            relevant_strategies = self._strategy_evolution.retrieve_strategies(
                problem_description=problem_desc,
                backend=backend,
                top_k=3,
            )
            if relevant_strategies:
                for strategy in relevant_strategies:
                    active_strategy_ids.append(strategy.id)
                    # Create an intent from each strategy
                    tree.add_node(
                        parent_id=tree.root.id,
                        description=strategy.description,
                        priority=0.8,  # high priority — these are proven strategies
                    )
                    logger.info(
                        "Seeded tree with strategy: %s (%s)",
                        strategy.description[:60],
                        strategy.id,
                    )
                tree_history.append(tree.serialize())

        # Step 1: Propose initial intents (LLM adds more beyond strategy seeds)
        self._propose_intents(tree, reference_code, hardware, backend)
        tree_history.append(tree.serialize())

        # Step 2: Main loop
        for _ in range(self._max_iterations):
            # Select the highest-priority pending node
            node = tree.get_highest_priority_pending()
            if node is None:
                logger.info("No more pending intents — search exhausted.")
                break

            # Mark as active
            tree.update_node(node.id, status=IntentStatus.ACTIVE)
            node.attempts += 1
            intents_explored += 1
            logger.info(
                f"Exploring intent: {node.description!r} "
                f"(priority={node.priority:.2f}, attempt={node.attempts})"
            )

            # Run the inner loop
            result = self._inner_loop.refine(
                intent=node,
                reference_code=reference_code,
                backend=backend,
                config={
                    "max_retries_per_intent": self._max_retries_per_intent,
                },
            )
            total_iterations += result.iterations
            total_tokens += result.total_tokens
            total_llm_cost += result.total_cost_usd
            total_eval_count += result.iterations

            # Update the tree with results
            if result.status == "succeeded" and result.best_speedup > 0:
                tree.update_node(
                    node.id,
                    status=IntentStatus.SUCCEEDED,
                    best_speedup=result.best_speedup,
                    profiler_summary=result.critic_feedback,
                )
                intents_succeeded += 1

                # Track global best kernel
                if result.best_speedup > best_speedup:
                    best_speedup = result.best_speedup
                    best_kernel = result.best_kernel
                    logger.info(
                        f"New global best: {best_speedup:.3f}x "
                        f"(intent: {node.description!r})"
                    )

                # Propose new child intents building on this success
                self._propose_intents(tree, reference_code, hardware, backend)
            else:
                # Failed — check if we should retry or mark as failed
                if node.attempts < node.max_attempts:
                    # Allow retry: put back to PENDING with reduced priority
                    tree.update_node(
                        node.id,
                        status=IntentStatus.PENDING,
                        best_speedup=result.best_speedup,
                        profiler_summary=result.critic_feedback,
                    )
                    node.priority = max(0.05, node.priority * 0.7)
                else:
                    tree.update_node(
                        node.id,
                        status=IntentStatus.FAILED,
                        best_speedup=result.best_speedup,
                        profiler_summary=result.critic_feedback,
                    )
                    intents_failed += 1

            # Record trace iteration if capture is enabled
            if self._trace_capture is not None:
                # Determine the decision string from the result status
                if result.status == "succeeded" and result.best_speedup > 0:
                    trace_decision = "keep"
                elif node.attempts < node.max_attempts:
                    trace_decision = "retry"
                else:
                    trace_decision = "discard"

                # Build a synthetic EvalResult from the available summary data
                trace_eval = EvalResult(
                    status=EvalStatus.CORRECT if result.status == "succeeded" else EvalStatus.ERROR,
                    correct=(result.status == "succeeded"),
                    speedup=result.best_speedup,
                )

                self._trace_capture.record_iteration(
                    iteration=intents_explored,
                    intent=node.description,
                    generator_prompt="",  # not available at orchestrator level
                    generator_response="",  # not available at orchestrator level
                    kernel_code=result.best_kernel,
                    eval_result=trace_eval,
                    critic_diagnosis=None,  # detailed diagnosis not available here
                    decision=trace_decision,
                    tokens_used=result.total_tokens,
                    latency_seconds=0.0,  # per-iteration latency not available
                )

            # Update priorities based on latest result
            self._update_priorities(tree, result)

            # Check stagnation
            if tree.stagnation_detected(threshold=self._stagnation_threshold):
                logger.warning(
                    f"Stagnation detected after {tree.failed_count_streak} "
                    f"consecutive non-improvements."
                )
                stagnation_triggered = True
                # Ask LLM about pruning
                self._handle_stagnation(tree)
                # If still stagnating after pruning attempt and no pending nodes, stop
                if tree.get_highest_priority_pending() is None:
                    logger.info("No pending intents after stagnation handling — stopping.")
                    break

            # Check budget — estimate cumulative cost so far and stop if exceeded
            if self._max_budget_usd is not None:
                _GPU_RATES_CHECK = {
                    "H100": 3.95,
                    "A100-80GB": 2.50,
                    "A100-40GB": 2.10,
                    "L40S": 2.00,
                }
                elapsed = time.monotonic() - start_time
                gpu_rate_check = _GPU_RATES_CHECK.get(hardware, 3.95)
                avg_eval_s = (elapsed / total_eval_count) if total_eval_count > 0 else 0.0
                running_compute = total_eval_count * avg_eval_s * (gpu_rate_check / 3600)
                running_total = total_llm_cost + running_compute
                if running_total > self._max_budget_usd:
                    logger.warning(
                        "Budget exceeded: $%.2f spent of $%.2f limit — stopping early.",
                        running_total,
                        self._max_budget_usd,
                    )
                    budget_exceeded = True
                    break

            tree_history.append(tree.serialize())

        wall_time = time.monotonic() - start_time

        # Post-optimization: update strategy history if strategy evolution is active
        if self._strategy_evolution is not None and active_strategy_ids:
            result_entry = {
                "speedup": best_speedup,
                "correct": best_speedup > 0 and best_kernel != "",
                "iterations": total_iterations,
                "problem": f"kernel optimization ({hardware}, {backend})",
                "backend": backend,
            }
            for sid in active_strategy_ids:
                self._strategy_evolution.update_strategy(sid, result_entry)
            logger.info(
                "Updated %d strategies with optimization result (speedup=%.3f)",
                len(active_strategy_ids),
                best_speedup,
            )

        # End trace capture and persist if enabled
        optimization_trace: OptimizationTrace | None = None
        if self._trace_capture is not None:
            self._trace_capture.end_session(
                final_speedup=best_speedup,
                final_correct=(best_speedup > 0 and best_kernel != ""),
            )
            optimization_trace = self._trace_capture.get_trace()
            traces_dir = self._config.get("traces_dir", "traces/raw")
            try:
                trace_path = save_trace(optimization_trace, output_dir=traces_dir)
                logger.info("Trace saved to %s", trace_path)
            except Exception as exc:
                logger.warning("Failed to save trace: %s", exc)

        # Approximate Modal compute cost.
        # Each eval iteration runs on a GPU; estimate cost from wall time
        # and GPU hourly rate.
        _GPU_HOURLY_RATES = {
            "H100": 3.95,
            "A100-80GB": 2.50,
            "A100-40GB": 2.10,
            "L40S": 2.00,
        }
        gpu_rate = _GPU_HOURLY_RATES.get(hardware, 3.95)
        # Rough estimate: average ~15s per eval, but use actual wall time
        # divided across evals as a proxy when available.
        avg_eval_seconds = (wall_time / total_eval_count) if total_eval_count > 0 else 0.0
        compute_cost = total_eval_count * avg_eval_seconds * (gpu_rate / 3600)
        total_cost = total_llm_cost + compute_cost

        return OptimizationResult(
            final_kernel=best_kernel,
            final_speedup=best_speedup,
            tree_history=tree_history,
            iterations_total=total_iterations,
            wall_time_seconds=wall_time,
            intents_explored=intents_explored,
            intents_succeeded=intents_succeeded,
            intents_failed=intents_failed,
            stagnation_triggered=stagnation_triggered,
            trace=optimization_trace,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            budget_exceeded=budget_exceeded,
        )

    # -- LLM-driven tree operations ------------------------------------------

    def _propose_intents(
        self,
        tree: IntentTree,
        reference_code: str,
        hardware: str,
        backend: str,
    ) -> None:
        """Ask the LLM to propose new intents and add them to the tree."""
        prompt = prompt_propose_intents(
            reference_code=reference_code,
            hardware=hardware,
            current_tree_json=tree.to_summary_json(),
            current_best_speedup=tree.global_best_speedup,
            backend=backend,
        )
        try:
            response_text = self._llm.call(prompt)
            response = _extract_json(response_text)
            intents = response.get("intents", [])
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM intent proposals: {e}")
            return

        for intent_data in intents:
            parent_id = intent_data.get("parent_id", tree.root.id)
            # Handle sentinel for mock LLM
            if parent_id == "__ROOT__":
                parent_id = tree.root.id
            # Validate parent exists
            if tree.get_node(parent_id) is None:
                parent_id = tree.root.id  # fallback to root

            description = intent_data.get("description", "")
            priority = float(intent_data.get("priority", 0.5))

            if description:
                tree.add_node(
                    parent_id=parent_id,
                    description=description,
                    priority=priority,
                )
                logger.debug(f"Added intent: {description!r} (priority={priority:.2f})")

    def _update_priorities(self, tree: IntentTree, result: RefinementResult) -> None:
        """Ask the LLM to re-estimate priorities after seeing a result."""
        prompt = prompt_update_priorities(
            tree_json=tree.to_summary_json(),
            latest_eval_result_summary=(
                f"Status: {result.status}, "
                f"Best speedup: {result.best_speedup:.3f}x, "
                f"Iterations used: {result.iterations}, "
                f"Feedback: {result.critic_feedback}"
            ),
        )
        try:
            response_text = self._llm.call(prompt)
            response = _extract_json(response_text)
            updates = response.get("priority_updates", {})
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM priority updates: {e}")
            return

        if updates:
            tree.update_priorities(updates)

    def _handle_stagnation(self, tree: IntentTree) -> None:
        """Handle stagnation by asking the LLM about pruning.

        If the LLM recommends pruning, prune the indicated subtrees and
        propose new intents to replace them.
        """
        # Find the most recently failed nodes for context
        failed_nodes = [
            n for n in tree.all_nodes if n.status == IntentStatus.FAILED
        ]
        if not failed_nodes:
            return

        # Build a summary of failures
        recent_failures = failed_nodes[-3:]  # last 3 failures
        failure_summary = "\n".join(
            f"- Node {n.id}: '{n.description}' — {n.attempts} attempts, "
            f"best speedup {n.best_speedup:.3f}x. Feedback: {n.profiler_summary}"
            for n in recent_failures
        )

        prompt = prompt_prune_decision(
            tree_json=tree.to_summary_json(),
            failed_node_summary=failure_summary,
        )
        try:
            response_text = self._llm.call(prompt)
            response = _extract_json(response_text)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM prune decision: {e}")
            return

        if response.get("prune", False):
            node_ids = response.get("node_ids_to_prune", [])
            for nid in node_ids:
                if tree.get_node(nid) is not None:
                    tree.prune_subtree(nid)
                    logger.info(f"Pruned subtree rooted at {nid}")
