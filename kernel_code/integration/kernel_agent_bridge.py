"""Bridge between kernel-code shell and Meta's KernelAgent.

Replaces our native inner loop with KernelAgent's proven pipeline:
- 4 parallel workers at varied temperatures
- 10-round error feedback loop
- Battle-tested Jinja2 prompt templates
- Structured error feedback

Adds our Modal remote eval and progress callbacks on top.

Usage::

    bridge = KernelAgentBridge(
        reference_source=open("reference.py").read(),
        model_name="claude-sonnet-4-20250514",
        hardware="L40S",
        live_display=display,
    )
    result = bridge.run()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernel_code.live_display import LiveOptimizationDisplay
    from kernel_code.run_log import RunLogger

logger = logging.getLogger(__name__)


def _modal_eval(kernel_code: str, reference_code: str) -> dict:
    """Evaluate a kernel via Modal remote GPU."""
    import modal

    eval_fn = modal.Function.from_name("openkernel-eval", "eval_kernel_on_gpu")
    result = eval_fn.remote(
        kernel_source=kernel_code,
        reference_source=reference_code,
        eval_mode="fast",
    )
    return result


class KernelAgentBridge:
    """Runs KernelAgent with Modal eval and our UX hooks."""

    def __init__(
        self,
        reference_source: str,
        model_name: str = "gpt-4o",
        num_workers: int = 4,
        max_rounds: int = 8,
        hardware: str = "L40S",
        live_display: "LiveOptimizationDisplay | None" = None,
        run_logger: "RunLogger | None" = None,
        use_modal: bool = True,
    ) -> None:
        self._reference = reference_source
        self._model_name = model_name
        self._num_workers = num_workers
        self._max_rounds = max_rounds
        self._hardware = hardware
        self._live_display = live_display
        self._run_logger = run_logger
        self._use_modal = use_modal

        self._best_speedup: float = 0.0
        self._best_kernel: str = ""
        self._iteration_count: int = 0

    def run(self) -> dict[str, Any]:
        """Run KernelAgent optimization.

        Returns dict with: success, kernel_code, speedup, rounds, worker_id
        """
        from kernel_agent.agent import TritonKernelAgent

        start_time = time.time()

        # Suppress KernelAgent's verbose logging (raw LLM responses, worker details)
        for logger_name in [
            "TritonKernelAgent", "kernel_agent.manager", "kernel_agent.worker",
            "kernel_agent.agent", "kernel_agent.prompt_manager", "httpx",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        # Ensure API keys are in env — KernelAgent's providers read from os.environ
        from kernel_code.settings import load_settings, inject_api_keys
        settings = load_settings()
        inject_api_keys(settings)

        # Build problem description from reference
        problem_desc = (
            f"Optimize the following PyTorch code into a fast Triton kernel "
            f"for {self._hardware} GPU.\n\n"
            f"Reference implementation:\n```python\n{self._reference}\n```\n\n"
            f"The kernel must be correct (torch.allclose with rtol=1e-2, atol=1e-2)."
        )

        # Create agent
        agent = TritonKernelAgent(
            num_workers=self._num_workers,
            max_rounds=self._max_rounds,
            model_name=self._model_name,
        )

        # Configure workers with Modal eval and callbacks
        self._configure_agent(agent)

        # Report phase
        if self._live_display:
            self._live_display.update_phase(
                f"KernelAgent: {self._num_workers} workers × {self._max_rounds} rounds "
                f"({self._model_name})"
            )

        # Run
        try:
            result = agent.generate_kernel(
                problem_description=problem_desc,
            )
        except Exception as exc:
            logger.error("KernelAgent failed: %s", exc)
            return {
                "success": False,
                "kernel_code": "",
                "speedup": 0.0,
                "error": str(exc),
                "elapsed": time.time() - start_time,
            }

        elapsed = time.time() - start_time

        # If successful, eval on Modal for speedup measurement
        kernel_code = result.get("kernel_code", "")
        speedup = 0.0

        if result.get("success") and kernel_code and self._use_modal:
            try:
                # Wrap in ModelNew if needed
                from kernel_agent.model_wrapper import wrap_in_model_new
                wrapped = wrap_in_model_new(kernel_code, self._reference)

                eval_result = _modal_eval(wrapped, self._reference)
                if eval_result.get("correct"):
                    speedup = eval_result.get("speedup", 0.0)

                if self._live_display:
                    self._live_display.update_iteration(
                        num=self._iteration_count + 1,
                        speedup=speedup,
                        status="keep" if speedup > 1.0 else "discard",
                        intent=f"KernelAgent ({self._model_name})",
                    )
                    self._iteration_count += 1
            except Exception as exc:
                logger.warning("Modal eval of final kernel failed: %s", exc)

        return {
            "success": result.get("success", False),
            "kernel_code": kernel_code,
            "speedup": speedup,
            "worker_id": result.get("worker_id"),
            "rounds": result.get("rounds", 0),
            "elapsed": elapsed,
        }

    def _configure_agent(self, agent: Any) -> None:
        """Configure agent's workers with Modal eval via env vars.

        Workers are spawned via multiprocessing.Process — we can't pass
        Python callbacks or function objects. Instead, we set env vars
        that workers read at startup:
        - OPENKERNEL_USE_MODAL=1 — tells workers to use Modal for eval
        - OPENKERNEL_REFERENCE_CODE — the reference source code
        """
        if self._use_modal:
            os.environ["OPENKERNEL_USE_MODAL"] = "1"
            os.environ["OPENKERNEL_REFERENCE_CODE"] = self._reference
