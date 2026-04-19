"""Bridge between the kernel-code TUI and the live openkernel engine.

The TUI reads JSON cache files on a 2-second timer.  The bridge writes to
those same files after each orchestrator iteration, keeping the two systems
decoupled.

Usage::

    bridge = OpenKernelBridge(config, session_id="abc123")
    result = bridge.run_optimization(reference_source)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernel_code.file_cache import FileStateCache
    from kernel_code.hooks import HookRegistry
    from kernel_code.progress import OptimizationProgress

from openkernel.config import OpenKernelConfig
from openkernel.engine.factory import create_engine
from openkernel.engine.orchestrator import (
    InnerLoopInterface,
    Orchestrator,
    OptimizationResult,
    RefinementResult,
)
from openkernel.engine.world_model import IntentNode, IntentStatus, IntentTree

logger = logging.getLogger(__name__)

# Project root -- cache lives at repo root, not inside the kernel_code package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SESSIONS_DIR = _PROJECT_ROOT / "cache" / "sessions"


class _CallbackOrchestrator(Orchestrator):
    """Orchestrator subclass that fires a callback after each inner-loop step.

    The base ``Orchestrator.optimize`` is synchronous and doesn't expose
    per-iteration hooks.  Rather than monkey-patching, we duplicate the
    main loop with a single added call to ``self._on_iteration(...)`` so the
    bridge can flush JSON to disk.
    """

    def __init__(
        self,
        inner_loop: InnerLoopInterface,
        config: dict,
        llm=None,
        on_iteration=None,
    ) -> None:
        super().__init__(inner_loop=inner_loop, config=config, llm=llm)
        self._on_iteration = on_iteration

    # Override optimize to inject the callback ---------------------------

    def optimize(
        self,
        reference_code: str,
        backend: str = "triton",
        hardware: str = "H100",
    ) -> OptimizationResult:
        start_time = time.monotonic()
        tree = IntentTree(root_description=f"Optimize kernel for {hardware} ({backend})")
        tree_history: list[dict] = [tree.serialize()]

        best_kernel = ""
        best_speedup = 0.0
        total_iterations = 0
        intents_explored = 0
        intents_succeeded = 0
        intents_failed = 0
        stagnation_triggered = False

        # Step 1: Propose initial intents
        self._propose_intents(tree, reference_code, hardware, backend)
        tree_history.append(tree.serialize())

        # Step 2: Main loop
        for _ in range(self._max_iterations):
            node = tree.get_highest_priority_pending()
            if node is None:
                logger.info("No more pending intents -- search exhausted.")
                break

            tree.update_node(node.id, status=IntentStatus.ACTIVE)
            node.attempts += 1
            intents_explored += 1
            logger.info(
                "Exploring intent: %r (priority=%.2f, attempt=%d)",
                node.description,
                node.priority,
                node.attempts,
            )

            result = self._inner_loop.refine(
                intent=node,
                reference_code=reference_code,
                backend=backend,
                config={"max_retries_per_intent": self._max_retries_per_intent},
            )
            total_iterations += result.iterations

            if result.status == "succeeded" and result.best_speedup > 0:
                tree.update_node(
                    node.id,
                    status=IntentStatus.SUCCEEDED,
                    best_speedup=result.best_speedup,
                    profiler_summary=result.critic_feedback,
                )
                intents_succeeded += 1

                if result.best_speedup > best_speedup:
                    best_speedup = result.best_speedup
                    best_kernel = result.best_kernel
                    logger.info(
                        "New global best: %.3fx (intent: %r)",
                        best_speedup,
                        node.description,
                    )

                self._propose_intents(tree, reference_code, hardware, backend)
            else:
                if node.attempts < node.max_attempts:
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

            self._update_priorities(tree, result)

            if tree.stagnation_detected(threshold=self._stagnation_threshold):
                logger.warning("Stagnation detected.")
                stagnation_triggered = True
                self._handle_stagnation(tree)
                if tree.get_highest_priority_pending() is None:
                    logger.info("No pending intents after stagnation handling -- stopping.")
                    break

            tree_history.append(tree.serialize())

            # ---- CALLBACK: notify bridge after each iteration ----
            if self._on_iteration is not None:
                self._on_iteration(
                    iteration=intents_explored,
                    node=node,
                    result=result,
                    best_speedup=best_speedup,
                    best_kernel=best_kernel,
                )

        wall_time = time.monotonic() - start_time

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
        )


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class OpenKernelBridge:
    """Glue between the kernel-code TUI and the openkernel engine.

    Creates the engine, runs optimization, and writes intermediate results
    to a JSON cache file that the TUI polls on a timer.
    """

    def __init__(
        self,
        config: OpenKernelConfig,
        session_id: str | None = None,
        *,
        problem_label: str = "custom kernel",
        hardware: str = "H100",
        backend: str = "triton",
        hooks: "HookRegistry | None" = None,
        progress: "OptimizationProgress | None" = None,
        file_cache: "FileStateCache | None" = None,
    ) -> None:
        self._config = config
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._problem_label = problem_label
        self._hardware = hardware
        self._backend = backend
        self._hooks = hooks
        self._progress = progress
        self._file_cache = file_cache

        # Where the TUI looks for data
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_path = _SESSIONS_DIR / f"{self._session_id}.json"

        # Accumulated iterations (TUI schema)
        self._iterations: list[dict] = []
        self._best_speedup: float = 0.0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_optimization(self, reference_source: str) -> OptimizationResult:
        """Run the full optimization and return the final result.

        Intermediate state is flushed to ``cache/sessions/{session_id}.json``
        after every orchestrator iteration so the TUI can display progress.
        """
        from openkernel.engine.factory import InnerLoopAdapter
        from openkernel.engine.inner_loop import InnerLoop
        from openkernel.eval import create_eval_fn
        from openkernel.agents.critic import Critic
        from openkernel.agents.generator import Generator
        from openkernel.backends.triton_backend import TritonBackend
        from openkernel.backends.cuda_backend import CudaBackend
        from openkernel.config import Backend
        from openkernel.llm.models import get_default_model
        from openkernel.llm.provider import LLMProvider

        # --- Replicate create_engine() but use _CallbackOrchestrator ---
        model_config = self._config.model or get_default_model()
        llm = LLMProvider(model_config)

        if self._config.backend == Backend.CUDA:
            be = CudaBackend()
        else:
            be = TritonBackend()

        generator = Generator(llm, be)
        critic = Critic(llm)

        inner_loop = InnerLoop(generator, critic, self._config, on_phase=self._on_phase)
        eval_fn = create_eval_fn(self._config, file_cache=self._file_cache)
        adapter = InnerLoopAdapter(inner_loop, eval_fn, critic)

        orch_config = {
            "max_iterations": self._config.max_iterations,
            "stagnation_threshold": self._config.stagnation_threshold,
            "max_retries_per_intent": self._config.max_retries_per_intent,
        }

        orchestrator = _CallbackOrchestrator(
            inner_loop=adapter,
            config=orch_config,
            llm=llm,
            on_iteration=self._on_iteration,
        )

        # Write initial empty session so TUI can start
        self._flush_cache()

        result = orchestrator.optimize(
            reference_code=reference_source,
            backend=self._backend,
            hardware=self._hardware,
        )

        # Final flush
        self._best_speedup = result.final_speedup
        self._flush_cache()

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_phase(self, message: str) -> None:
        """Called by the inner loop at each phase (generate, eval, analyze).

        Updates the spinner with the current status message.
        """
        if self._progress is not None:
            self._progress._update_status(message)

    def _on_iteration(
        self,
        iteration: int,
        node: IntentNode,
        result: RefinementResult,
        best_speedup: float,
        best_kernel: str,
    ) -> None:
        """Called after each orchestrator iteration to update the cache."""
        # Report per-step progress if a progress reporter is configured
        if self._progress is not None:
            self._progress.start_iteration(iteration, node.description)

        # Map orchestrator result -> TUI iteration schema (matches mock_data.py)
        if result.status == "failed" or result.status == "error":
            status = "compile_error" if "compile" in result.critic_feedback.lower() else "error"
            decision = "error"
            speedup = 0.0
        elif result.best_speedup > self._best_speedup:
            status = "keep"
            decision = "keep"
            speedup = result.best_speedup
            self._best_speedup = result.best_speedup
        else:
            status = "discard"
            decision = "discard"
            speedup = result.best_speedup

        # Report outcome via progress reporter
        if self._progress is not None:
            if decision == "keep":
                self._progress.kept(speedup, is_new_best=(speedup == self._best_speedup))
            elif decision == "discard":
                self._progress.discarded(speedup, self._best_speedup)
            elif decision == "error":
                error_msg = result.critic_feedback[:80] if result.critic_feedback else "unknown"
                self._progress.error(status, error_msg)

        iteration_data = {
            "iteration": iteration,
            "speedup": speedup,
            "status": status,
            "runtime_us": 0.0,  # not available from RefinementResult directly
            "ref_runtime_us": 0.0,
            "profile": {
                "bandwidth_util": 0.0,
                "compute_util": 0.0,
                "cache_efficiency": 0.0,
                "occupancy": 0.0,
                "bottleneck_type": "unknown",
            },
            "kernel_code_snippet": result.best_kernel[:200] if result.best_kernel else "",
            "intent": node.description,
            "decision": decision,
            "error": result.critic_feedback if decision == "error" else None,
        }

        self._iterations.append(iteration_data)
        self._flush_cache()

        # Fire lifecycle hooks
        if self._hooks is not None:
            from kernel_code.hooks import HookRegistry

            self._hooks.fire(
                HookRegistry.POST_ITERATE,
                iteration=iteration,
                speedup=speedup,
                status=status,
                intent=node.description,
            )
            if decision == "keep":
                self._hooks.fire(
                    HookRegistry.POST_KEEP,
                    speedup=speedup,
                    iteration=iteration,
                    intent=node.description,
                    problem=self._problem_label,
                    hardware=self._hardware,
                )
            elif decision == "discard":
                self._hooks.fire(
                    HookRegistry.POST_DISCARD,
                    speedup=speedup,
                    best_speedup=self._best_speedup,
                    intent=node.description,
                )

    def _flush_cache(self) -> None:
        """Write current state to the JSON cache file (TUI schema)."""
        session = {
            "session_id": self._session_id,
            "problem": self._problem_label,
            "hardware": self._hardware,
            "backend": self._backend,
            "model": self._config.model.model_id if self._config.model else "unknown",
            "ref_runtime_us": 0.0,
            "best_speedup": self._best_speedup,
            "num_iterations": len(self._iterations),
            "iterations": self._iterations,
        }

        try:
            self._cache_path.write_text(json.dumps(session, indent=2))
        except OSError:
            logger.warning("Failed to write cache file: %s", self._cache_path)
